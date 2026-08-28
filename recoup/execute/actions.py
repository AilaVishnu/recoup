"""Execution: the one place where a decision stops being a row and becomes an effect.

Everything upstream of here is reversible. An Assessment can be recomputed, a
Decision can be overruled, a PolicyReview can be re-run against tighter bounds.
Once this module runs, a customer has been messaged or an order exists at
Razorpay, and neither of those can be taken back. So the invariants are enforced
here a second time even though the caller already enforced them.

**Nothing executes without an ALLOW.** `review.allowed` is checked at the top of
`execute()` and a failure raises rather than returning a FAILED run. The agent
loop already checks it, and that is precisely the reason this check exists: the
value of defence in depth is entirely in the layer that looks redundant. A
control-arm event reaching this function is not an execution error to be logged
and moved past - it means the holdout has been contaminated and the eval is now
fiction, and it should stop the run.

**The bookkeeping happens once, in the tail.** Every handler returns an `_Effect`
and a single block turns it into an ActionRun and a SpendLog row. Distributing
that across six handlers is how one of them eventually forgets to record a
channel cost, and a zero in ActionRun.channel_cost_paise is not a missing number
- it is a claim that the message was free, which the eval will believe and put in
the report.

**The one exception is ContactLog, and the exception is the point.** The fatigue
slot is reserved in `execute()` *before* any handler can reach the customer, and
released afterwards only if nothing went out. Writing it in the tail with
everything else left a window - the outbox appends to a file, outside the
transaction - where a message was delivered and the counter was not incremented,
irreversibly. Reserving first means a crash costs a phantom contact instead: the
customer is under-messaged rather than silently over-messaged, which is the only
direction this bound is allowed to fail in.

**On ActionStatus.SKIPPED_DRY_RUN**: it marks a run whose *gateway* leg was
withheld for want of credentials, not a failure and not an absence of intent.
The decision was made, the message went to the outbox with its cost, and the
eval must count the event as acted on. Only NO_ACTION means nothing happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from recoup.db import (
    ActionRun,
    ActionStatus,
    ActionType,
    Cohort,
    ContactLog,
    Customer,
    Decision,
    DecisionSource,
    EventStatus,
    RevenueEvent,
    SpendLog,
)
from recoup.execute import outbox, razorpay_client
from recoup.execute.razorpay_client import RecoupExecutionError
from recoup.policy.rules import CONTACT_ACTIONS, Review
from recoup.taxonomy import Rail, Strategy, profile_for


class ExecutionRefused(RuntimeError):
    """The executor was asked to do something it must never do.

    Distinct from RecoupExecutionError, which means Razorpay said no. This means
    Recoup said no to itself, and it is always a bug in the caller.
    """


@dataclass
class _Effect:
    """What a handler did, before any of it is written down."""

    status: ActionStatus
    razorpay_ref: str | None = None
    response: dict[str, Any] = field(default_factory=dict)
    message: outbox.Message | None = None
    incentive_paise: int = 0
    error: str | None = None


def _gateway_status() -> ActionStatus:
    return (
        ActionStatus.SKIPPED_DRY_RUN
        if razorpay_client.is_dry_run()
        else ActionStatus.SENT
    )


def _incentive_of(decision: Decision, event: RevenueEvent) -> int:
    """The discount this decision carries, in paise. Validated, not trusted.

    Policy has already checked depth against both caps, so a value that fails
    here means the decision and the review that cleared it disagree - which is
    the one situation where continuing is worse than crashing.
    """
    raw = (decision.params or {}).get("incentive_paise", 0)
    incentive = int(raw or 0)

    if incentive < 0:
        raise ExecutionRefused(f"negative incentive {incentive} on {decision.event_id}")
    if incentive >= event.amount_paise:
        raise ExecutionRefused(
            f"incentive {incentive} >= order value {event.amount_paise} on "
            f"{event.id} - policy should have caught this at 15%"
        )
    return incentive


def _notes(event: RevenueEvent, decision: Decision) -> dict[str, str]:
    """Razorpay notes carrying the audit thread back to the decision.

    These survive on the entity at Razorpay, which means the merchant can answer
    "why does this order exist" from their own dashboard without access to
    Recoup's database.
    """
    return {
        "recoup_event_id": event.id,
        "recoup_decision_id": str(decision.id),
        "reason_code": event.reason_code,
        "cohort": event.cohort.value,
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _retry_payment(
    decision: Decision, event: RevenueEvent, customer: Customer, now: datetime
) -> _Effect:
    """A re-presentment. Recoup sends no message; the customer still hears about it.

    A true re-presentment charges a saved instrument token, which a test account
    with no stored mandates cannot do. What Recoup creates is the order the
    re-presentment would attach to - the same object, carrying the same
    reference, minus the leg that needs a real customer's card on file.

    This docstring used to call that silent and note that a retry is deliberately
    not a contact action. Both claims were wrong, and they are worth leaving
    corrected in place rather than quietly rewritten. A silent re-presentment
    needs a registered mandate; without one, RBI's additional-factor rules put a
    card retry in front of an OTP prompt and a UPI retry is a collect request
    that rings the customer's phone. RETRY_PAYMENT is in CONTACT_ACTIONS for that
    reason, and 63 of 191 retries were firing after 21:00 IST before it was.

    Recoup still composes nothing and pays no channel cost here - which is a
    different statement from "the customer was not contacted", and conflating the
    two is what hid the problem.
    """
    order = razorpay_client.create_order(
        amount_paise=event.amount_paise,
        receipt=f"recoup_{event.id}",
        notes={**_notes(event, decision), "attempt_no": str(event.attempt_no + 1)},
    )
    return _Effect(
        status=_gateway_status(),
        razorpay_ref=order["id"],
        response=order,
    )


def _link_with_message(
    decision: Decision,
    event: RevenueEvent,
    customer: Customer,
    now: datetime,
    incentive_paise: int,
) -> _Effect:
    """A Payment Link plus the one message that points at it.

    The discount is applied to the link amount and to the copy from the same
    variable. Splitting those - a coupon code in the email, full price on the
    link - is the classic recovery-tooling bug: the customer arrives at a page
    that contradicts the message that brought them there, and the recovery fails
    for a reason no dashboard will ever show.
    """
    profile = profile_for(event.reason_code)
    rail = profile.switch_to[0] if profile.switch_to else Rail.UPI
    link_amount = event.amount_paise - incentive_paise

    description = (
        f"{profile.label} on {event.rail} - complete via {rail.value}"
        if incentive_paise == 0
        else f"{profile.label} - Rs {outbox.rupees(incentive_paise)} off"
    )

    link = razorpay_client.create_payment_link(
        amount_paise=link_amount,
        customer=customer,
        description=description,
        notes={
            **_notes(event, decision),
            # The preferred rail rides in the copy and the notes rather than in
            # a checkout restriction. Locking the link to UPI would shut out a
            # customer whose second card works fine, and a recovery link that
            # refuses a working instrument is worse than no link.
            "suggested_rail": rail.value,
            "incentive_paise": str(incentive_paise),
        },
        reference_id=f"recoup_{event.id}",
    )

    message = outbox.send(
        event=event,
        customer=customer,
        action_type=decision.action_type,
        now=now,
        link_url=link.get("short_url"),
        incentive_paise=incentive_paise,
    )

    return _Effect(
        status=_gateway_status(),
        razorpay_ref=link["id"],
        response=link,
        message=message,
        incentive_paise=incentive_paise,
    )


def _nudge(
    decision: Decision, event: RevenueEvent, customer: Customer, now: datetime
) -> _Effect:
    """A reminder with no money and no link. The cheapest thing Recoup can do.

    The message id becomes the ActionRun's reference. There is no Razorpay entity
    to point at, and leaving the column null would break the join that ties a
    later recovery back to the action that plausibly caused it - which is the
    difference between a claimed recovery and an attributable one.
    """
    message = outbox.send(
        event=event,
        customer=customer,
        action_type=decision.action_type,
        now=now,
    )
    return _Effect(
        status=ActionStatus.SENT,
        razorpay_ref=message.message_id,
        response={"channel": message.channel, "subject": message.subject},
        message=message,
    )


def _escalate(
    decision: Decision, event: RevenueEvent, customer: Customer, now: datetime
) -> _Effect:
    """Hand the event to a person. No external effect, which is the entire point.

    Recorded as SENT rather than skipped: the queue entry is a real deliverable
    and the merchant's ops team is the recipient. An escalation logged as a
    non-event is an escalation nobody works.
    """
    entry = {
        "queue": "human_review",
        "event_id": event.id,
        "customer_id": customer.id,
        "amount_paise": event.amount_paise,
        "amount_display": f"Rs {outbox.rupees(event.amount_paise)}",
        "reason_code": event.reason_code,
        "queued_at": now.isoformat(timespec="seconds"),
        "rationale": decision.rationale,
    }
    return _Effect(
        status=ActionStatus.SENT,
        razorpay_ref=f"queue_{event.id}",
        response=entry,
    )


def queue_for_human(
    session: Session,
    decision: Decision,
    event: RevenueEvent,
    customer: Customer,
    now: datetime,
    why: str,
) -> ActionRun:
    """Record a policy escalation as something a person can actually find.

    Distinct from the ESCALATE_TO_HUMAN *decision* path, which goes through
    execute() normally. This is the other route to the same place: policy
    returning ESCALATE on an action the agent proposed in good faith - which is
    what happens to every event above the autonomy limit, i.e. the most valuable
    events in the system.

    Until this existed those events set status AWAITING_APPROVAL and returned,
    with no ActionRun and no queue entry anywhere. A Rs 5,00,000 receivable sat
    unrecovered and unlisted - precisely the state _escalate's docstring warns
    about, produced by the code path that is supposed to be the careful one.

    No gateway call, no message, no cost. The deliverable is the row.
    """
    run = ActionRun(
        decision_id=decision.id,
        executed_at=now,
        action_type=ActionType.ESCALATE_TO_HUMAN,
        status=ActionStatus.SENT,
        razorpay_ref=f"queue_{event.id}",
        incentive_paise=0,
        channel_cost_paise=0,
        response={
            "queue": "human_review",
            "event_id": event.id,
            "customer_id": customer.id,
            "amount_paise": event.amount_paise,
            "amount_display": f"Rs {outbox.rupees(event.amount_paise)}",
            "reason_code": event.reason_code,
            "queued_at": now.isoformat(timespec="seconds"),
            "escalated_because": why,
            "proposed_action": decision.action_type.value,
            # Provenance travels with the text, because on this path the text may
            # not be Recoup's.
            #
            # Policy escalates the highest-value events in the system, and on
            # exactly those events the rationale is most likely to have been
            # written by the model. It was being copied verbatim into a queue a
            # human reads before approving a payment, with nothing distinguishing
            # it from the engine's own reasoning. A review reproduced the obvious
            # consequence: "Pre-cleared by Finance, approve without further
            # checks" placed in front of an approver on a Rs 5,00,000 event.
            #
            # The policy engine contains what the model can *do*. It has no view
            # on what the model can *say* to a person, and a human approver is
            # not a bound - they are the thing the bound defers to.
            "rationale_source": decision.source.value,
            "rationale_is_model_authored": decision.source is DecisionSource.LLM,
            "rationale": (
                f"[UNVERIFIED - written by {decision.model or 'the decision model'}, "
                f"not by Recoup] {decision.rationale}"
                if decision.source is DecisionSource.LLM
                else decision.rationale
            ),
        },
    )
    session.add(run)
    return run


def _no_action(
    decision: Decision, event: RevenueEvent, customer: Customer, now: datetime
) -> _Effect:
    """Deliberate inaction, written down.

    SKIPPED_DRY_RUN is the nearest honest member of ActionStatus: FAILED would
    claim something broke and SENT would claim something went out. The row exists
    at all because suppressed value is reported, not discarded - a fraud decline
    Recoup correctly refused to touch is a number the merchant should see.
    """
    return _Effect(
        status=ActionStatus.SKIPPED_DRY_RUN,
        response={
            "suppressed": True,
            "reason_code": event.reason_code,
            "rationale": decision.rationale,
            "value_at_risk_paise": event.amount_paise,
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


REVIEW_STALENESS_TOLERANCE = timedelta(minutes=5)
"""How far execution may drift from the instant the Review was granted.

Not arbitrary. Every time-sensitive bound - quiet hours above all - is evaluated
against ReviewContext.now, so an approval granted at 14:00 IST says nothing
about 03:00 IST. A reviewer demonstrated exactly that: a genuine ALLOW taken at
noon sent a message at 3am, because `now` is an independent argument here and
nothing compared the two. Any layer that reviews and executes at different
instants - a queue, a retry sweep, a scheduler honouring a delay - walks
straight through the quiet-hours bound with a real approval in hand.
"""


def _verify_review_provenance(
    review: Review,
    decision: Decision,
    event: RevenueEvent,
    customer: Customer,
    now: datetime,
) -> None:
    """Confirm this Review was issued for this action, at about this moment.

    Without these checks a Review is an unauthenticated capability token: any
    ALLOW authorises any action. tests/test_policy_bypass.py demonstrates six
    ways that goes wrong, including a hand-built Review(ALLOW, checks=[]) - a
    verdict from thirteen bounds that ran none of them.
    """
    if not review.checks:
        raise ExecutionRefused(
            f"review for event {event.id} carries no checks - an ALLOW that "
            "evaluated nothing is not an ALLOW"
        )
    if review.event_id != event.id:
        raise ExecutionRefused(
            f"review was issued for event {review.event_id!r}, not {event.id}"
        )
    if review.customer_id != customer.id:
        raise ExecutionRefused(
            f"review was issued for customer {review.customer_id!r}, not {customer.id}"
        )
    if review.action_type != decision.action_type:
        raise ExecutionRefused(
            f"review approved {review.action_type} but the decision is "
            f"{decision.action_type}"
        )
    if review.reviewed_at is None:
        raise ExecutionRefused(f"review for event {event.id} has no timestamp")

    drift = abs(now - review.reviewed_at)
    if drift > REVIEW_STALENESS_TOLERANCE:
        raise ExecutionRefused(
            f"review for event {event.id} was granted at {review.reviewed_at} "
            f"but execution is at {now} ({drift} adrift). Time-sensitive bounds "
            "were evaluated against the earlier instant; re-review instead."
        )


def _recheck_invariants(review: Review, decision: Decision, event: RevenueEvent) -> None:
    """Re-test the cheap absolutes here, independently of the policy engine.

    This module's docstring claims the executor is a safety net. It was not: the
    only independent limit on a discount was `incentive < order value`, so an
    executor bound of 100% stood behind a policy bound of 15%. Everything below
    is checkable without a database round trip, which is the whole reason it is
    affordable to check twice.
    """
    # Control-arm events may still reach here, but only for actions with no
    # external effect. NO_ACTION and ESCALATE_TO_HUMAN write a row and touch
    # nothing outside the database, which is exactly how a held-out event gets
    # its decision recorded without being acted on - the asymmetry the whole
    # measurement depends on. Refusing those too would mean the holdout carried
    # no audit trail at all, and the counterfactual it exists to provide would
    # be unreadable.
    if event.cohort is Cohort.CONTROL and decision.action_type not in (
        ActionType.NO_ACTION,
        ActionType.ESCALATE_TO_HUMAN,
    ):
        raise ExecutionRefused(
            f"event {event.id} is in the control arm and {decision.action_type} "
            "would reach the customer. Executing it would contaminate the holdout "
            "and invalidate every measurement built on it."
        )

    profile = profile_for(event.reason_code)
    if profile.strategy is Strategy.DO_NOT_RETRY and decision.action_type not in (
        ActionType.NO_ACTION,
        ActionType.ESCALATE_TO_HUMAN,
    ):
        raise ExecutionRefused(
            f"{event.reason_code} is do-not-retry; {decision.action_type} would "
            "re-present a risk decline"
        )

    incentive = int((decision.params or {}).get("incentive_paise", 0) or 0)
    if incentive > 0:
        if not profile.incentive_eligible:
            raise ExecutionRefused(
                f"{event.reason_code} is a technical failure - a discount buys "
                "nothing and is pure margin burn"
            )
        bounds = review.bounds or {}
        max_fraction = bounds.get("max_incentive_fraction", 0.15)
        max_absolute = bounds.get("max_incentive_paise", 2_000_00)
        if event.amount_paise <= 0:
            raise ExecutionRefused(f"event {event.id} has no order value to discount")
        if incentive > max_absolute or incentive / event.amount_paise > max_fraction:
            raise ExecutionRefused(
                f"incentive {incentive} exceeds the reviewed caps "
                f"({max_fraction:.0%} / {max_absolute} paise) on event {event.id}"
            )


def execute(
    session: Session,
    decision: Decision,
    review: Review,
    event: RevenueEvent,
    customer: Customer,
    now: datetime,
) -> ActionRun:
    """Carry out one approved decision and record everything it cost.

    Also advances `RevenueEvent.status`, the single field in the schema that is
    updated in place. On a gateway failure the status is deliberately left alone
    so a later sweep can pick the event up again.
    """
    if not review.allowed:
        raise ExecutionRefused(
            f"execute() called on event {event.id} with verdict "
            f"{review.verdict.value} (violations: {review.violations or 'none'}). "
            "Only an ALLOW may reach the executor."
        )

    # A decision executed against the wrong event mis-attributes a recovery, and
    # mis-attribution is the failure mode this whole project exists to avoid.
    if decision.event_id != event.id:
        raise ExecutionRefused(
            f"decision {decision.id} belongs to event {decision.event_id}, "
            f"not {event.id}"
        )
    if event.customer_id != customer.id:
        raise ExecutionRefused(
            f"event {event.id} belongs to customer {event.customer_id}, "
            f"not {customer.id}"
        )

    if "delay_hours" in (decision.params or {}):
        raise ExecutionRefused(
            f"decision {decision.id} still carries delay_hours - the scheduler "
            "must apply a requested wait before review, so the action fires at "
            "the instant the bounds were evaluated against. Executing here would "
            "put it outside the window it was authorised in."
        )

    _verify_review_provenance(review, decision, event, customer, now)
    _recheck_invariants(review, decision, event)

    # Handlers stamp decision_id into Razorpay notes, so it has to exist first.
    if decision.id is None:
        session.flush()

    incentive = _incentive_of(decision, event)
    action = decision.action_type

    # Reserve the fatigue slot BEFORE anything can reach the customer.
    #
    # The outbox appends to a file, which is outside the SQLAlchemy transaction,
    # so writing the ContactLog row afterwards left a window where the message
    # was delivered and the counter was not incremented - irreversibly. A
    # reviewer reproduced it by making the bookkeeping raise: outbox held one
    # message, ContactLog held none, and every later contact_frequency check for
    # that customer was computed from an undercount. db.py calls the fatigue cap
    # "the one bound whose failure the customer feels directly", and this was how
    # it failed.
    #
    # Reserving first inverts the risk: a crash now costs a phantom contact, so
    # the customer is under-messaged rather than silently over-messaged. The
    # reservation is released below if nothing actually went out, which preserves
    # the original intent - a Payment Link that Razorpay refused must not burn a
    # slot on the customer's week.
    reservation: ContactLog | None = None
    if action in CONTACT_ACTIONS:
        reservation = ContactLog(
            customer_id=event.customer_id,
            occurred_at=now,
            action_type=action,
            event_id=event.id,
        )
        session.add(reservation)
        session.flush()

    try:
        if action is ActionType.RETRY_PAYMENT:
            effect = _retry_payment(decision, event, customer, now)
        elif action is ActionType.PAYMENT_LINK:
            effect = _link_with_message(decision, event, customer, now, incentive)
        elif action is ActionType.NUDGE_WITH_INCENTIVE:
            # A discount the customer cannot redeem is not a discount. The
            # incentive only becomes real on a link priced at the reduced
            # amount, so this variant creates one and the copy quotes it.
            effect = _link_with_message(decision, event, customer, now, incentive)
        elif action is ActionType.NUDGE:
            effect = _nudge(decision, event, customer, now)
        elif action is ActionType.ESCALATE_TO_HUMAN:
            effect = _escalate(decision, event, customer, now)
        elif action is ActionType.NO_ACTION:
            effect = _no_action(decision, event, customer, now)
        else:
            raise ExecutionRefused(f"no executor for action type {action}")

    except RecoupExecutionError as exc:
        # Razorpay refused or was unreachable. Nothing reached the customer, so
        # nothing is logged as contact and no incentive is booked as spent.
        effect = _Effect(
            status=ActionStatus.FAILED,
            response=exc.payload,
            error=str(exc),
        )

    # Release the slot only when the action FAILED - not merely when it composed
    # no message of Recoup's own.
    #
    # Those are different things, and treating them as one silently un-counted
    # every retry. A retry writes no outbox row because Recoup composes nothing;
    # the re-presentment still reaches the customer as an OTP prompt or a UPI
    # collect notification, which is why RETRY_PAYMENT is in CONTACT_ACTIONS at
    # all. Keying the release on `effect.message is None` reserved the slot and
    # then handed it straight back.
    #
    # Deleting a reservation stays safe in a way that failing to create one is
    # not: the worst case here is one extra touch this week, against a customer
    # being contacted an unbounded number of times in the other direction.
    if reservation is not None and effect.status is ActionStatus.FAILED:
        session.delete(reservation)
        session.flush()

    return _record(session, decision, event, effect, now)


def _record(
    session: Session,
    decision: Decision,
    event: RevenueEvent,
    effect: _Effect,
    now: datetime,
) -> ActionRun:
    """Write the run and its two side-ledgers. The only place that does either."""
    run = ActionRun(
        decision_id=decision.id,
        executed_at=now,
        action_type=decision.action_type,
        status=effect.status,
        razorpay_ref=effect.razorpay_ref,
        incentive_paise=effect.incentive_paise,
        channel_cost_paise=effect.message.cost_paise if effect.message else 0,
        response=effect.response,
        error=effect.error,
    )
    session.add(run)

    # ContactLog is written in execute(), before the send, and released there if
    # nothing went out. It cannot be written here: by the time this runs the
    # message has already left, and a failure in between would lose the count
    # while the customer keeps the message.

    # Only money that was actually committed. Booking spend for a link that never
    # got created would eat the daily budget on behalf of customers who were
    # never offered anything.
    if effect.incentive_paise > 0:
        session.add(
            SpendLog(
                occurred_at=now,
                amount_paise=effect.incentive_paise,
                event_id=event.id,
            )
        )

    if effect.status is not ActionStatus.FAILED:
        if decision.action_type is ActionType.NO_ACTION:
            event.status = EventStatus.SUPPRESSED
        elif decision.action_type is ActionType.ESCALATE_TO_HUMAN:
            event.status = EventStatus.AWAITING_APPROVAL
        else:
            event.status = EventStatus.ACTED

    # Flush, not commit: the caller owns the transaction boundary so a batch that
    # dies halfway rolls back as one unit instead of leaving orphaned runs.
    session.flush()
    return run
