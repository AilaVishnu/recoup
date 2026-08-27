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
and a single block turns it into an ActionRun, a ContactLog row and a SpendLog
row. Distributing that across six handlers is how one of them eventually forgets
to record a channel cost, and a zero in ActionRun.channel_cost_paise is not a
missing number - it is a claim that the message was free, which the eval will
believe and put in the report.

**On ActionStatus.SKIPPED_DRY_RUN**: it marks a run whose *gateway* leg was
withheld for want of credentials, not a failure and not an absence of intent.
The decision was made, the message went to the outbox with its cost, and the
eval must count the event as acted on. Only NO_ACTION means nothing happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from recoup.db import (
    ActionRun,
    ActionStatus,
    ActionType,
    ContactLog,
    Customer,
    Decision,
    EventStatus,
    RevenueEvent,
    SpendLog,
)
from recoup.execute import outbox, razorpay_client
from recoup.execute.razorpay_client import RecoupExecutionError
from recoup.policy.rules import CONTACT_ACTIONS, Review
from recoup.taxonomy import Rail, profile_for


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
    """Silent re-presentment. The customer is not told, because nothing is asked of them.

    A true re-presentment charges a saved instrument token, which a test account
    with no stored mandates cannot do. What Recoup creates is the order the
    re-presentment would attach to - the same object, carrying the same
    reference, minus the leg that needs a real customer's card on file. Notably
    this is *not* a contact action: an issuer outage does not become recoverable
    by emailing about it, and counting a silent retry against the fatigue cap
    would make Recoup refuse to retry a bank outage because it sent two nudges
    last week.
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

    # Handlers stamp decision_id into Razorpay notes, so it has to exist first.
    if decision.id is None:
        session.flush()

    incentive = _incentive_of(decision, event)
    action = decision.action_type

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

    # Gated on membership *and* on a message having actually gone out. Membership
    # alone would burn a slot on the customer's weekly fatigue cap for a payment
    # link whose creation failed - punishing the customer for Razorpay's outage.
    if decision.action_type in CONTACT_ACTIONS and effect.message is not None:
        session.add(
            ContactLog(
                customer_id=event.customer_id,
                occurred_at=now,
                action_type=decision.action_type,
                event_id=event.id,
            )
        )

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
