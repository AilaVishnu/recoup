"""Adversarial tests: each one names a thing Recoup must never do, and does it.

These are written to FAIL against the current tree. Every test asserts the
invariant the project claims to hold; the failure output shows the bypass that
breaks it. A test here going green means the hole is closed.

Numbering matches the findings report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from recoup import pipeline
from recoup.agent import brain, providers
from recoup.db import (
    ActionStatus,
    ActionType,
    Assessment,
    Base,
    Cohort,
    ContactLog,
    Customer,
    Decision,
    DecisionSource,
    EventKind,
    EventStatus,
    PolicyReview,
    PolicyVerdict,
    RevenueEvent,
    SpendLog,
)
from recoup.detect.features import is_quiet_hours
from recoup.execute import actions, outbox, razorpay_client
from recoup.execute.actions import ExecutionRefused, execute
from recoup.policy.rules import Bounds, Check, Review, ReviewContext, review

# 14:00 IST on 3 Mar 2026 - the same instant the existing suites use, so nothing
# here is incidentally suppressed by quiet hours.
NOON_IST_UTC = datetime(2026, 3, 3, 8, 30)
# 03:00 IST on 4 Mar 2026. Quiet hours. Nobody may be messaged at this instant.
THREE_AM_IST_UTC = datetime(2026, 3, 3, 21, 30)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class StubSettings:
    """Keeps a developer's real .env from turning these tests into network calls."""

    dry_run: bool = True
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""

    @property
    def razorpay_configured(self) -> bool:
        return bool(self.razorpay_key_id and self.razorpay_key_secret)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, future=True)()
    yield s
    s.close()


@pytest.fixture
def outbox_file(tmp_path, monkeypatch):
    path = tmp_path / "outbox.jsonl"
    monkeypatch.setattr(outbox, "OUTBOX_PATH", path)
    return path


@pytest.fixture
def offline(monkeypatch):
    """No Razorpay, no Anthropic. Nothing in this file may touch a network."""
    monkeypatch.setattr(razorpay_client, "get_settings", lambda: StubSettings())
    monkeypatch.setattr(
        razorpay_client,
        "_require_client",
        lambda: pytest.fail("dry run reached the network"),
    )
    monkeypatch.setattr(providers, "key_available", lambda: False)


def ctx(**overrides) -> ReviewContext:
    base = dict(
        event_id="evt_test",
        customer_id="cust_test",
        cohort=Cohort.TREATMENT,
        reason_code="payment_cancelled",
        amount_paise=2_000_00,
        expected_value_paise=400_00,
        recoverability=0.20,
        earliest_action_at=NOON_IST_UTC - timedelta(hours=1),
        attempts_so_far=0,
        action_type=ActionType.NUDGE,
        incentive_paise=0,
        now=NOON_IST_UTC,
        bounds=Bounds(),
    )
    base.update(overrides)
    return ReviewContext(**base)


def named(result, name):
    return next(c for c in result.checks if c.name == name)


def make(
    session,
    *,
    reason_code="card_expired",
    amount_paise=2_000_00,
    cohort=Cohort.TREATMENT,
    action_type=ActionType.PAYMENT_LINK,
    params=None,
    attempt_no=1,
    event_id="evt_1",
    customer_id="cust_1",
    occurred_at=NOON_IST_UTC,
    status=EventStatus.DECIDED,
):
    """One customer / event / decision chain, flushed and ready to execute."""
    customer = session.get(Customer, customer_id)
    if customer is None:
        customer = Customer(
            id=customer_id,
            name="Ananya Iyer",
            email="ananya@example.com",
            contact="+919812345678",
            created_at=occurred_at - timedelta(days=400),
            prior_success_count=8,
            prior_failure_count=0,
            prior_recovery_count=0,
            lifetime_value_paise=40_000_00,
            preferred_rail="upi",
        )
        session.add(customer)

    event = RevenueEvent(
        id=event_id,
        kind=EventKind.PAYMENT_FAILED,
        customer_id=customer.id,
        amount_paise=amount_paise,
        occurred_at=occurred_at,
        reason_code=reason_code,
        rail="card",
        cohort=cohort,
        status=status,
        attempt_no=attempt_no,
        extra={},
    )
    decision = Decision(
        event_id=event.id,
        created_at=occurred_at,
        source=DecisionSource.RULES,
        action_type=action_type,
        params=params or {},
        rationale="test fixture",
    )
    session.add_all([event, decision])
    session.flush()
    return customer, event, decision


def counts(session):
    return (
        session.scalar(select(func.count()).select_from(ContactLog)) or 0,
        session.scalar(select(func.count()).select_from(SpendLog)) or 0,
    )


# ===========================================================================
# FINDING 1 - the bounds are a caller-supplied argument, and the audit trail
# does not record which ones were in force.
#
# pipeline.run(bounds=...) / process_event(bounds=...) threads an arbitrary
# Bounds into every ReviewContext. Nothing pins it to DEFAULT_BOUNDS, the
# PolicyReview row does not record it, and api/read.bounds_table() renders
# DEFAULT_BOUNDS regardless of what the run actually used.
# ===========================================================================


def test_the_autonomy_limit_cannot_be_raised_by_the_caller(
    offline, outbox_file, session
):
    """A Rs 5,00,000 event must reach a human. Here it is acted on autonomously."""
    customer, event, _ = make(
        session,
        reason_code="card_expired",
        amount_paise=5_00_000_00,  # Rs 5,00,000 - twenty times the autonomy limit
        status=EventStatus.OPEN,
    )
    session.flush()

    stats = pipeline.RunStats()
    pipeline.process_event(
        session,
        event,
        customer,
        stats,
        None,
        Bounds(human_approval_above_paise=10**14),
    )

    assert event.status is EventStatus.AWAITING_APPROVAL, (
        "BYPASS: a Rs 5,00,000 event executed with no human in the loop. "
        f"status={event.status}, executed={stats.executed}, "
        f"messages_sent={len(outbox.read_all(outbox_file))}. "
        "pipeline.process_event(bounds=...) accepts any Bounds object and the "
        "policy engine applies it without complaint."
    )


def test_the_audit_trail_records_which_bounds_were_applied(
    offline, outbox_file, session
):
    """A PolicyReview that cannot say what limit it applied cannot be audited.

    Re-pointed after the fix. This originally asserted the *loosened* limit
    appeared in the row, which was correct against code that applied whatever
    bounds it was handed. Bounds are now clamped to the defaults, so the loose
    limit is never applied and recording it would itself be a lie. The property
    that survives - and the one the name asks for - is that the row states the
    limit which actually governed the decision, instead of leaving a reader to
    assume the defaults were in force.
    """
    customer, event, _ = make(
        session,
        reason_code="card_expired",
        amount_paise=5_00_000_00,
        status=EventStatus.OPEN,
    )
    session.flush()

    pipeline.process_event(
        session,
        event,
        customer,
        pipeline.RunStats(),
        None,
        Bounds(human_approval_above_paise=10**14),
    )

    stored = session.scalars(select(PolicyReview)).one()

    assert stored.bounds, (
        "BYPASS: the PolicyReview records no bounds at all. A check reports its "
        "verdict and never the threshold behind it, so without this the row "
        "cannot answer what limit was applied."
    )
    assert stored.bounds["human_approval_above_paise"] == 25_000_00, (
        "the caller's loosened autonomy limit must be clamped to the default, "
        f"but the row records {stored.bounds['human_approval_above_paise']}"
    )
    assert event.status is EventStatus.AWAITING_APPROVAL


def test_the_incentive_caps_cannot_be_raised_by_the_caller(
    offline, outbox_file, session
):
    """15% / Rs 2,000. A caller must not be able to book a 50% discount.

    Re-pointed after the fix. The original asserted an intermediate step - that
    the engine says ALLOW under relaxed bounds - as its demonstration of the
    bypass. The engine now denies, so that line would assert the bug rather than
    the guarantee. The final assertion, which is the actual safety property, is
    unchanged and still the point of the test.
    """
    customer, event, decision = make(
        session,
        reason_code="payment_cancelled",
        amount_paise=20_000_00,
        action_type=ActionType.NUDGE_WITH_INCENTIVE,
        params={"incentive_paise": 10_000_00},
    )

    relaxed = Bounds(
        max_incentive_fraction=1.0,
        max_incentive_paise=10**12,
        daily_incentive_budget_paise=10**12,
        min_incremental_ev_ratio=0.0,
    )
    result = review(
        ctx(
            event_id=event.id,
            customer_id=customer.id,
            reason_code="payment_cancelled",
            amount_paise=event.amount_paise,
            expected_value_paise=4_000_00,
            action_type=ActionType.NUDGE_WITH_INCENTIVE,
            incentive_paise=10_000_00,
            earliest_action_at=NOON_IST_UTC,
            bounds=relaxed,
        ),
        session,
    )

    assert result.verdict is PolicyVerdict.DENY, (
        "relaxed bounds must be clamped to the defaults before any rule reads them"
    )
    assert "incentive_depth" in result.violations

    # Belt and braces: handed this review anyway, the executor must still refuse.
    with pytest.raises(ExecutionRefused):
        execute(session, decision, result, event, customer, NOON_IST_UTC)

    spent = session.scalar(select(func.sum(SpendLog.amount_paise))) or 0
    assert spent <= 2_000_00, (
        f"BYPASS: Rs {spent / 100:,.0f} of discount booked on a Rs "
        f"{event.amount_paise / 100:,.0f} order against a 15% / Rs 2,000 cap"
    )


def test_quiet_hours_cannot_be_switched_off(session):
    """A bound with an off switch is not a bound.

    Re-pointed after the fix. The original passed
    Bounds(quiet_hours_enforced=False) to prove the switch worked. The field is
    now deleted outright, so this asserts both that it cannot come back and that
    03:00 IST is still refused.
    """
    assert "quiet_hours_enforced" not in Bounds.__dataclass_fields__, (
        "Bounds must not carry a flag that disables a customer-protection rule"
    )

    result = review(
        ctx(
            now=THREE_AM_IST_UTC,
            earliest_action_at=THREE_AM_IST_UTC,
            action_type=ActionType.NUDGE,
        ),
        session,
    )
    assert result.verdict is PolicyVerdict.DENY, (
        "BYPASS: a nudge at 03:00 IST was allowed. quiet_hours reports "
        f"{named(result, 'quiet_hours').detail!r}"
    )


def test_the_weekly_contact_cap_cannot_be_raised_by_the_caller(session):
    for i in range(9):
        session.add(
            ContactLog(
                customer_id="cust_test",
                occurred_at=NOON_IST_UTC - timedelta(hours=i + 1),
                action_type=ActionType.NUDGE,
                event_id=f"prior_{i}",
            )
        )
    session.flush()

    result = review(ctx(bounds=Bounds(max_contacts_per_customer_per_week=50)), session)
    assert result.verdict is PolicyVerdict.DENY, (
        "BYPASS: a tenth contact in one day was allowed. "
        f"{named(result, 'contact_frequency').detail}"
    )


# ===========================================================================
# FINDING 2 - the control-arm rule is the only rule that compares by identity,
# and it fails OPEN.
#
#   if ctx.cohort is not Cohort.CONTROL:   ->   "treatment arm", ALLOW
#
# Cohort is a str-Enum, so `"control" == Cohort.CONTROL` is True but
# `"control" is Cohort.CONTROL` is False. Every other rule in the file uses
# `in (...)`, which is `==`-based and string-tolerant. The one rule the module
# docstring says "must never be relaxed for a demo" is the one that silently is.
# ===========================================================================


def test_control_arm_is_suppressed_when_the_cohort_is_a_plain_string(session):
    result = review(ctx(cohort="control"), session)
    assert result.verdict is PolicyVerdict.DENY, (
        "BYPASS: a CONTROL event was cleared for execution. "
        f"control_arm_suppression says {named(result, 'control_arm_suppression').detail!r}. "
        "`'control' == Cohort.CONTROL` is True but `is` is False, and the rule "
        "uses `is not`. Every number in the eval is computed against this holdout."
    )


def test_control_arm_is_suppressed_when_the_cohort_is_missing(session):
    result = review(ctx(cohort=None), session)
    assert result.verdict is PolicyVerdict.DENY, (
        "BYPASS: cohort=None was classified as 'treatment arm' and allowed. "
        "An unknown cohort must fail closed, not default to treatment."
    )


def test_control_arm_rule_matches_on_value_not_identity(session):
    """The narrow unit behind the two tests above: the rule function itself.

    `Cohort.CONTROL.value == Cohort.CONTROL` is True while `is` is False, so a
    value-equal cohort walks past an identity test. Every other rule in
    policy/rules.py compares with `in (...)`, which is `==`-based and would have
    matched. This one uses `is not`.
    """
    from recoup.policy.rules import _rule_control_arm_is_never_executed

    check = _rule_control_arm_is_never_executed(ctx(cohort=Cohort.CONTROL.value), session)
    assert not check.passed and check.verdict is PolicyVerdict.DENY, (
        f"BYPASS: the rule returned passed={check.passed} / {check.detail!r} for a "
        "cohort that compares equal to Cohort.CONTROL. Rewrite the guard as "
        "`if ctx.cohort != Cohort.CONTROL` (or coerce in ReviewContext) so an "
        "equal-but-not-identical value cannot read as the treatment arm."
    )


# ===========================================================================
# FINDING 3 - review() raises instead of denying.
#
# policy/rules.py opens with: "Fail closed. A rule that cannot evaluate -
# missing data, unknown reason code, arithmetic it cannot complete - denies."
# It does not. There is no try/except anywhere in review(), and the rules do
# bare arithmetic on caller-supplied fields. pipeline.process_event() wraps
# execute() in `except Exception` but leaves review() unguarded, so a single
# unevaluable event takes down the whole batch run.
# ===========================================================================


@pytest.mark.parametrize(
    "overrides,what",
    [
        ({"amount_paise": None}, "order value missing"),
        ({"expected_value_paise": None}, "expected value missing"),
        ({"recoverability": None, "incentive_paise": 100_00,
          "action_type": ActionType.NUDGE_WITH_INCENTIVE}, "recoverability missing"),
        ({"incentive_paise": None}, "incentive missing"),
    ],
)
def test_unevaluable_input_denies_rather_than_raising(session, overrides, what):
    try:
        result = review(ctx(**overrides), session)
    except Exception as exc:  # noqa: BLE001 - that is precisely the finding
        pytest.fail(
            f"BYPASS ({what}): review() raised {type(exc).__name__}: {exc}. "
            "The module docstring guarantees these deny. There is no exception "
            "boundary in review(), so no PolicyReview row is written at all - and "
            "pipeline.process_event() does not guard review(), so this ends the run."
        )
    assert result.verdict is PolicyVerdict.DENY, f"{what} was allowed: {result.violations}"


def test_a_non_utc_clock_is_rejected_rather_than_silently_shifted():
    """`is_quiet_hours` adds 5h30m to whatever it is handed and reads .hour.

    Handed a tz-aware IST timestamp - correct, unambiguous data - it shifts a
    second time and reports 03:00 IST as business hours.
    """
    ist = timezone(timedelta(hours=5, minutes=30))
    three_am_ist = datetime(2026, 3, 4, 3, 0, tzinfo=ist)

    assert is_quiet_hours(three_am_ist), (
        "BYPASS: is_quiet_hours() says 03:00 IST is outside quiet hours. "
        "to_ist() blindly adds IST_OFFSET, so any datetime that is not naive-UTC "
        "is shifted twice; 02:30-08:00 IST all report as business hours. The "
        "function accepts any datetime and asserts nothing about its timezone."
    )


# ===========================================================================
# FINDING 4 - execute() does not check that the Review it was handed belongs
# to this decision, this event, this customer, or this instant.
#
# actions.py claims: "the invariants are enforced here a second time even
# though the caller already enforced them". They are not. The only re-checks
# are `review.allowed`, decision.event_id == event.id and event.customer_id ==
# customer.id. `Review` carries no event id, no decision id, no timestamp and
# no bounds - it is an unauthenticated capability token.
# ===========================================================================


def _genuine_allow(session, customer, at=NOON_IST_UTC) -> Review:
    """A real ALLOW from the real engine - for a completely different event."""
    result = review(
        ctx(
            event_id="evt_unrelated",
            customer_id=customer.id,
            reason_code="payment_cancelled",
            action_type=ActionType.NUDGE,
            now=at,
            earliest_action_at=at,
        ),
        session,
    )
    assert result.verdict is PolicyVerdict.ALLOW
    return result


def test_a_review_from_another_event_cannot_authorise_a_fraud_retry(
    offline, outbox_file, session
):
    """payment_risk_check_failed is the hardest stop in the taxonomy. Here it is retried."""
    customer, event, decision = make(
        session,
        reason_code="payment_risk_check_failed",
        action_type=ActionType.RETRY_PAYMENT,
        event_id="evt_fraud",
    )
    borrowed = _genuine_allow(session, customer)

    with pytest.raises(ExecutionRefused):
        execute(session, decision, borrowed, event, customer, NOON_IST_UTC)


def test_a_review_from_another_event_cannot_authorise_a_control_arm_action(
    offline, outbox_file, session
):
    """One contaminated holdout event makes every reported number fiction."""
    customer, event, decision = make(
        session,
        reason_code="payment_cancelled",
        cohort=Cohort.CONTROL,
        action_type=ActionType.NUDGE,
        event_id="evt_control",
    )
    borrowed = _genuine_allow(session, customer)

    with pytest.raises(ExecutionRefused):
        execute(session, decision, borrowed, event, customer, NOON_IST_UTC)

    assert outbox.read_all(outbox_file) == [], (
        "BYPASS: a CONTROL-arm customer was messaged. execute() never looks at "
        "event.cohort; it trusts the Review object it is handed."
    )


def test_a_review_from_another_event_cannot_discount_a_technical_failure(
    offline, outbox_file, session
):
    """A discount on a bank outage is the exact margin burn the taxonomy exists to stop."""
    customer, event, decision = make(
        session,
        reason_code="bank_not_available",
        amount_paise=20_000_00,
        action_type=ActionType.NUDGE_WITH_INCENTIVE,
        params={"incentive_paise": 19_000_00},  # 95%, on a technical failure
        event_id="evt_outage",
    )
    borrowed = _genuine_allow(session, customer)

    with pytest.raises(ExecutionRefused):
        execute(session, decision, borrowed, event, customer, NOON_IST_UTC)

    spent = session.scalar(select(func.sum(SpendLog.amount_paise))) or 0
    assert spent == 0, (
        f"BYPASS: Rs {spent / 100:,.0f} discounted on bank_not_available. execute()'s only "
        "independent incentive bound is `incentive < order value` (_incentive_of), "
        "i.e. 100% - not 15%, not Rs 2,000, and not eligibility."
    )


def test_an_allow_taken_at_noon_cannot_be_executed_at_3am(
    offline, outbox_file, session
):
    """`now` is a separate argument to execute() and must be checked against the review.

    Re-pointed only in mechanics: execute() now raises instead of proceeding, so
    the call is wrapped. The assertion that matters - that nothing left the
    outbox - is unchanged.
    """
    customer, event, decision = make(
        session,
        reason_code="payment_cancelled",
        action_type=ActionType.NUDGE,
        event_id="evt_deferred",
    )
    at_noon = _genuine_allow(session, customer, at=NOON_IST_UTC)

    with pytest.raises(ExecutionRefused):
        execute(session, decision, at_noon, event, customer, THREE_AM_IST_UTC)

    sent = outbox.read_all(outbox_file)
    assert sent == [], (
        f"BYPASS: message sent at {sent[0]['sent_at'] if sent else '?'} "
        "(03:00 IST) on the strength of a review taken at 14:00 IST. Any layer "
        "that reviews and executes at different instants - a queue, a retry "
        "sweep, a scheduler honouring a delay - walks straight through quiet hours."
    )


def test_a_fabricated_review_cannot_authorise_anything(offline, outbox_file, session):
    """Review is a plain dataclass with no provenance. Anyone can mint one."""
    customer, event, decision = make(
        session,
        reason_code="payment_risk_check_failed",
        cohort=Cohort.CONTROL,
        action_type=ActionType.RETRY_PAYMENT,
        event_id="evt_forged",
    )
    forged = Review(verdict=PolicyVerdict.ALLOW, checks=[])

    with pytest.raises(ExecutionRefused):
        execute(session, decision, forged, event, customer, NOON_IST_UTC)


def test_an_allow_with_zero_checks_is_not_an_allow(offline, session):
    """Thirteen bounds are advertised. execute() accepts a Review that ran none."""
    customer, event, decision = make(session, action_type=ActionType.RETRY_PAYMENT)
    empty = Review(verdict=PolicyVerdict.ALLOW, checks=[])

    with pytest.raises(ExecutionRefused):
        execute(session, decision, empty, event, customer, NOON_IST_UTC)


def test_a_review_of_a_different_customer_cannot_message_this_one(
    offline, outbox_file, session
):
    """The fatigue cap is per customer. A borrowed review moves the count to someone else."""
    victim, event, decision = make(
        session,
        reason_code="payment_cancelled",
        action_type=ActionType.NUDGE,
        event_id="evt_victim",
        customer_id="cust_victim",
    )
    for i in range(3):
        session.add(
            ContactLog(
                customer_id="cust_victim",
                occurred_at=NOON_IST_UTC - timedelta(days=i + 1),
                action_type=ActionType.NUDGE,
                event_id=f"earlier_{i}",
            )
        )
    session.flush()

    # A genuine ALLOW - computed for a customer who has never been contacted.
    fresh = review(
        ctx(customer_id="cust_never_contacted", now=NOON_IST_UTC), session
    )
    assert fresh.verdict is PolicyVerdict.ALLOW

    with pytest.raises(ExecutionRefused):
        execute(session, decision, fresh, event, victim, NOON_IST_UTC)

    contacts = session.scalar(
        select(func.count())
        .select_from(ContactLog)
        .where(ContactLog.customer_id == "cust_victim")
    )
    assert contacts <= 3, (
        f"BYPASS: cust_victim now has {contacts} contacts in 7 days against a cap "
        "of 3. execute() checks event.customer_id == customer.id but never checks "
        "either against the review that authorised the send."
    )


# ===========================================================================
# FINDING 5 - the per-reason attempt ceiling never binds.
#
# ReviewContext.attempts_so_far is filled from pipeline._attempts_so_far(),
# which counts ActionRun rows for the event. pipeline.run() only selects events
# with status == OPEN and every processed event leaves OPEN, so an event is
# processed exactly once and the count is always 0. RevenueEvent.attempt_no -
# the attempts that actually happened on the order, which the router reads - is
# invisible to policy.
# ===========================================================================


def test_the_attempt_cap_sees_attempts_that_already_happened(offline, session):
    """card_expired permits exactly 1 attempt. This order is on its 4th."""
    customer, event, _ = make(
        session,
        reason_code="card_expired",
        attempt_no=4,
        status=EventStatus.OPEN,
    )
    session.flush()

    seen = pipeline._attempts_so_far(session, event.id)
    assert seen >= event.attempt_no - 1, (
        f"BYPASS: policy is told attempts_so_far={seen} for an event on attempt "
        f"{event.attempt_no}. card_expired's ceiling is 1, so the cap passes and a "
        "fifth attempt is authorised. attempt_cap cannot deny anything in a "
        "single pipeline run - run() never revisits an event, so the count it "
        "reads is structurally always 0."
    )


def test_an_exhausted_order_is_refused_end_to_end(offline, outbox_file, session):
    customer, event, _ = make(
        session,
        reason_code="card_expired",
        attempt_no=4,
        status=EventStatus.OPEN,
    )
    session.flush()

    stats = pipeline.RunStats()
    pipeline.process_event(session, event, customer, stats, None, Bounds())

    assert stats.executed == 0, (
        "BYPASS: a 5th recovery attempt executed on a card_expired order whose "
        "taxonomy ceiling is 1 attempt."
    )


# ===========================================================================
# FINDING 6 - an ESCALATE verdict queues nothing for the human it names.
#
# actions._escalate() writes the queue entry, but it only runs when the
# *decision* is ESCALATE_TO_HUMAN and the verdict is ALLOW. A policy ESCALATE
# returns from process_event() before execute() is ever called, so the event is
# marked AWAITING_APPROVAL with no ActionRun and nothing in any queue - the
# state actions.py calls "an escalation nobody works".
# ===========================================================================


def test_an_escalated_event_produces_something_a_human_can_work(
    offline, outbox_file, session
):
    customer, event, _ = make(
        session,
        reason_code="card_expired",
        amount_paise=5_00_000_00,  # over the Rs 25,000 autonomy limit
        status=EventStatus.OPEN,
    )
    session.flush()

    stats = pipeline.RunStats()
    pipeline.process_event(session, event, customer, stats, None, Bounds())
    assert event.status is EventStatus.AWAITING_APPROVAL
    assert stats.escalated == 1

    from recoup.db import ActionRun

    queued = session.scalar(select(func.count()).select_from(ActionRun)) or 0
    assert queued == 1, (
        f"BYPASS: Rs {event.amount_paise / 100:,.0f} is parked in AWAITING_APPROVAL "
        "with no ActionRun and no human_review queue entry. Nothing anywhere lists "
        "it for a person, so the value sits unrecovered and unnoticed."
    )


# ===========================================================================
# FINDING 7 - the outbox write is not transactional.
#
# outbox.send() appends to data/outbox.jsonl before _record() writes the
# ContactLog row. Any failure between the two leaves the customer messaged and
# the fatigue counter unincremented - permanently, because the file is outside
# the transaction and a rollback cannot reach it. process_event() swallows the
# exception and moves on.
# ===========================================================================


def test_a_sent_message_is_always_counted_against_the_fatigue_cap(
    offline, outbox_file, session, monkeypatch
):
    customer, event, decision = make(
        session, reason_code="payment_cancelled", action_type=ActionType.NUDGE
    )

    real_record = actions._record

    def explode(*args, **kwargs):
        raise RuntimeError("database went away after the message was sent")

    monkeypatch.setattr(actions, "_record", explode)

    with pytest.raises(RuntimeError):
        execute(session, decision, _genuine_allow(session, customer), event, customer, NOON_IST_UTC)

    monkeypatch.setattr(actions, "_record", real_record)
    session.rollback()

    messages = len(outbox.read_all(outbox_file))
    contacts, _ = counts(session)
    assert messages == contacts, (
        f"BYPASS: {messages} message(s) on disk, {contacts} ContactLog row(s). "
        "The customer was contacted and the weekly cap does not know it. Every "
        "later review for this customer is computed from an undercount."
    )


# ===========================================================================
# FINDING 8 - Decision.params['delay_hours'] is validated, stored, and never read.
#
# brain.py exposes delay_hours to the model (0-168h), clamps it, and writes it
# to the Decision row. No rule reads it and the executor does not apply it. The
# action fires at action_time regardless, so a wait the model asked for - or
# that the taxonomy implied - is silently discarded.
# ===========================================================================


def test_a_requested_delay_is_either_applied_or_refused(offline, outbox_file, session):
    """delay_hours must not be a persisted field that nothing enforces.

    Re-pointed to the second half of its own title. Applying the wait inside
    execute() - which is what the original asserted - would fire the action at an
    instant no bound was evaluated against, handing the model a field it could
    set to move its own execution outside the window it was authorised in. The
    delay is applied by the scheduler *before* review instead, so the delayed
    instant is the reviewed one, and the executor refuses anything still carrying
    an unapplied delay. That makes "the scheduler applied it" checkable rather
    than conventional.
    """
    customer, event, decision = make(
        session,
        reason_code="payment_cancelled",
        action_type=ActionType.NUDGE,
        params={"incentive_paise": 0, "delay_hours": 72.0},
    )
    result = _genuine_allow(session, customer)

    with pytest.raises(ExecutionRefused, match="delay_hours"):
        execute(session, decision, result, event, customer, NOON_IST_UTC)

    assert outbox.read_all(outbox_file) == [], (
        "BYPASS: a decision asking to wait 72h sent immediately"
    )


def test_the_scheduler_consumes_a_requested_delay(offline, session):
    """The other half of the invariant: the field must not survive scheduling."""
    customer, event, _ = make(
        session,
        reason_code="payment_cancelled",
        action_type=ActionType.NUDGE,
        status=EventStatus.OPEN,
    )
    session.flush()

    pipeline.process_event(session, event, customer, pipeline.RunStats())
    stored = session.scalars(select(Decision)).all()[-1]

    assert "delay_hours" not in (stored.params or {}), (
        "the scheduler must consume delay_hours, so a stale one stays detectable"
    )




def test_an_aware_clock_is_converted_not_denied_and_not_double_shifted(session):
    """Well-formed input in another timezone is correct data, not bad data.

    This replaces an "aware clock" case in the unevaluable-input parametrisation
    above, which asserted a DENY. Denying it was one defensible fix and the worse
    one: db.utcnow() returns aware values while every column stores naive, so
    refusing them would reject timestamps the codebase itself produces.

    What must never happen is the silent double shift - adding the IST offset to
    a value already carrying one, which reported 03:00 IST as business hours and
    broke the enforcing rule and the deferring scheduler in the same direction at
    once. So: convert, and prove both spellings of the same instant agree.
    """
    aware_3am_ist = datetime(2026, 3, 3, 21, 30, tzinfo=timezone.utc)
    naive_3am_ist = datetime(2026, 3, 3, 21, 30)

    as_aware = review(
        ctx(now=aware_3am_ist, earliest_action_at=aware_3am_ist,
            action_type=ActionType.NUDGE),
        session,
    )
    as_naive = review(
        ctx(now=naive_3am_ist, earliest_action_at=naive_3am_ist,
            action_type=ActionType.NUDGE),
        session,
    )

    assert as_aware.verdict is as_naive.verdict is PolicyVerdict.DENY
    assert not named(as_aware, "quiet_hours").passed, (
        "03:00 IST expressed as an aware UTC timestamp must still read as quiet "
        "hours - a double shift would report it as 08:30 and allow the send"
    )
