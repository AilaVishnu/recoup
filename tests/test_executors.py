"""Execution is the irreversible step. These tests name what it must never do.

Two properties dominate: the executor cannot act without an ALLOW, and it cannot
under-report what an action cost. Everything else here is a way one of those two
gets quietly broken - a contact logged for a message that never sent, a discount
booked against a link that failed, a 4xx retried into a second charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

from dataclasses import asdict

import pytest
from razorpay.errors import BadRequestError, ServerError
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from recoup.db import (
    ActionStatus,
    ActionType,
    Base,
    Cohort,
    ContactLog,
    Customer,
    Decision,
    DecisionSource,
    EventKind,
    EventStatus,
    PolicyVerdict,
    RevenueEvent,
    SpendLog,
)
from recoup.execute import actions, outbox, razorpay_client
from recoup.execute.actions import ExecutionRefused, execute
from recoup.execute.razorpay_client import RecoupExecutionError
from recoup.policy.rules import Bounds, Check, Review

# 14:00 IST - outside quiet hours, so nothing here is incidentally suppressed.
NOW = datetime(2026, 3, 3, 8, 30)


@dataclass
class StubSettings:
    """Stands in for recoup.config.Settings so a developer's real .env - which
    may well hold working test keys - cannot make these tests hit the network."""

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
def no_keys(monkeypatch):
    """No credentials at all - the state a reviewer is in five minutes after clone."""
    monkeypatch.setattr(razorpay_client, "get_settings", lambda: StubSettings())
    monkeypatch.setattr(
        razorpay_client,
        "_require_client",
        lambda: pytest.fail("dry run reached the network"),
    )


def allowed(event=None, customer=None, decision=None, now=None) -> Review:
    """An ALLOW carrying the provenance execute() now insists on.

    A bare Review(ALLOW) is no longer accepted, and deliberately so: without
    provenance any approval authorises any action, which tests/test_policy_bypass.py
    demonstrates six ways. Fixtures have to mint a Review for the thing they are
    actually executing, exactly as review() does.
    """
    return Review(
        verdict=PolicyVerdict.ALLOW,
        checks=[Check("stub", True, PolicyVerdict.ALLOW, "test fixture")],
        event_id=event.id if event is not None else None,
        customer_id=customer.id if customer is not None else None,
        action_type=decision.action_type if decision is not None else None,
        reviewed_at=now,
        bounds=asdict(Bounds()),
    )


def denied(name: str = "control_arm_suppression") -> Review:
    return Review(
        verdict=PolicyVerdict.DENY,
        checks=[Check(name, False, PolicyVerdict.DENY, "test fixture")],
    )


def make(session, reason_code="card_expired", amount_paise=2_000_00, **kwargs):
    """One customer, one event, one decision, flushed and ready to execute."""
    customer = Customer(
        id="cust_1",
        name="Ananya Iyer",
        email="ananya@example.com",
        contact="+919812345678",
        created_at=NOW,
        preferred_rail="card",
    )
    event = RevenueEvent(
        id="evt_1",
        kind=kwargs.pop("kind", EventKind.PAYMENT_FAILED),
        customer_id=customer.id,
        amount_paise=amount_paise,
        occurred_at=NOW,
        reason_code=reason_code,
        rail="card",
        cohort=Cohort.TREATMENT,
        status=EventStatus.DECIDED,
        extra={},
    )
    decision = Decision(
        event_id=event.id,
        source=DecisionSource.RULES,
        action_type=kwargs.pop("action_type", ActionType.PAYMENT_LINK),
        params=kwargs.pop("params", {}),
        rationale="test fixture",
    )
    session.add_all([customer, event, decision])
    session.flush()
    return customer, event, decision


def counts(session):
    return (
        session.scalar(select(func.count()).select_from(ContactLog)) or 0,
        session.scalar(select(func.count()).select_from(SpendLog)) or 0,
    )


# --- running without credentials -------------------------------------------


def test_dry_run_needs_no_keys(no_keys, outbox_file, session):
    """The whole pipeline has to run on a clean clone, or nobody will run it."""
    assert razorpay_client.get_client() is None
    assert razorpay_client.is_dry_run() is True

    customer, event, decision = make(session)
    run = execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)

    assert run.status is ActionStatus.SKIPPED_DRY_RUN
    assert run.razorpay_ref.startswith("plink_dry_")
    assert run.response["_dry_run"] is True
    assert "recoup.invalid" in run.response["short_url"]


def test_dry_run_ids_are_stable_across_runs(no_keys, session):
    """A replay must reproduce the same references or the eval's join drifts."""
    customer, event, _ = make(session)
    first = razorpay_client.create_payment_link(
        1_000_00, customer, "d", reference_id="recoup_evt_1"
    )
    second = razorpay_client.create_payment_link(
        1_000_00, customer, "d", reference_id="recoup_evt_1"
    )
    assert first["id"] == second["id"]


def test_missing_keys_are_treated_as_dry_run(monkeypatch):
    monkeypatch.setattr(
        razorpay_client, "get_settings", lambda: StubSettings(dry_run=False)
    )
    assert razorpay_client.is_dry_run() is True


# --- the executor refuses -------------------------------------------------


def test_execution_without_an_allow_raises(no_keys, outbox_file, session):
    """A control event arriving here means the holdout is already contaminated."""
    customer, event, decision = make(session)
    with pytest.raises(ExecutionRefused, match="Only an ALLOW"):
        execute(session, decision, denied(), event, customer, NOW)

    assert counts(session) == (0, 0)
    assert outbox.read_all(outbox_file) == []


def test_escalate_verdict_is_not_an_allow(no_keys, outbox_file, session):
    customer, event, decision = make(session)
    review = Review(verdict=PolicyVerdict.ESCALATE, checks=[])
    with pytest.raises(ExecutionRefused):
        execute(session, decision, review, event, customer, NOW)


def test_decision_from_another_event_is_refused(no_keys, session):
    """Executing against the wrong event mis-attributes the recovery."""
    customer, event, decision = make(session)
    decision.event_id = "evt_somewhere_else"
    with pytest.raises(ExecutionRefused, match="belongs to event"):
        execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)


def test_an_incentive_bigger_than_the_order_is_refused(no_keys, session):
    """Unreachable past the 15% cap - which is why reaching it must stop the run.

    Uses payment_cancelled because it is one of only two incentive-eligible
    reason codes. On an ineligible reason the executor now refuses earlier and
    for a stronger reason, which would make this test pass without ever
    exercising the size check it is named for.
    """
    customer, event, decision = make(
        session,
        reason_code="payment_cancelled",
        action_type=ActionType.NUDGE_WITH_INCENTIVE,
        params={"incentive_paise": 5_000_00},
        amount_paise=2_000_00,
    )
    with pytest.raises(ExecutionRefused, match="exceeds the reviewed caps"):
        execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)


def test_a_discount_on_a_technical_failure_is_refused_by_the_executor(no_keys, session):
    """Defence in depth on the bound that costs real margin when it fails.

    Policy already refuses this. The executor refuses it independently, because
    a discount on a bank outage pays a customer to do what they were going to do
    anyway, and the only layer that can still stop it is the last one.
    """
    customer, event, decision = make(
        session,
        reason_code="issuer_down",
        action_type=ActionType.NUDGE_WITH_INCENTIVE,
        params={"incentive_paise": 100_00},
        amount_paise=2_000_00,
    )
    with pytest.raises(ExecutionRefused, match="pure margin burn"):
        execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)


# --- ContactLog is written for contact, and only for contact ---------------


def test_silent_retry_writes_no_contact_log(no_keys, outbox_file, session):
    """A retry is not a message. Counting it against the fatigue cap would make
    Recoup refuse to retry a bank outage because it emailed twice last week."""
    customer, event, decision = make(
        session, reason_code="issuer_down", action_type=ActionType.RETRY_PAYMENT
    )
    run = execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)

    assert run.razorpay_ref.startswith("order_dry_")
    assert counts(session) == (0, 0)
    assert outbox.read_all(outbox_file) == []
    assert run.channel_cost_paise == 0


def test_nudge_writes_exactly_one_contact_log(no_keys, outbox_file, session):
    customer, event, decision = make(
        session, reason_code="payment_cancelled", action_type=ActionType.NUDGE
    )
    execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)

    contacts, spends = counts(session)
    assert (contacts, spends) == (1, 0)
    assert len(outbox.read_all(outbox_file)) == 1


def test_escalation_is_not_customer_contact(no_keys, outbox_file, session):
    customer, event, decision = make(
        session, action_type=ActionType.ESCALATE_TO_HUMAN, amount_paise=40_000_00
    )
    run = execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)

    assert run.status is ActionStatus.SENT
    assert run.razorpay_ref == "queue_evt_1"
    assert run.response["queue"] == "human_review"
    assert counts(session) == (0, 0)
    assert event.status is EventStatus.AWAITING_APPROVAL


def test_no_action_sends_nothing_at_all(no_keys, outbox_file, session):
    customer, event, decision = make(
        session, reason_code="fraud_suspected", action_type=ActionType.NO_ACTION
    )
    run = execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)

    assert run.status is ActionStatus.SKIPPED_DRY_RUN
    assert run.razorpay_ref is None
    assert run.response["suppressed"] is True
    assert counts(session) == (0, 0)
    assert outbox.read_all(outbox_file) == []
    assert event.status is EventStatus.SUPPRESSED


def test_a_failed_link_logs_no_contact(monkeypatch, outbox_file, session):
    """The customer never heard from us, so their weekly cap must not shrink."""
    monkeypatch.setattr(
        razorpay_client, "get_settings", lambda: StubSettings(dry_run=False)
    )
    monkeypatch.setattr(
        actions.razorpay_client,
        "create_payment_link",
        lambda **kw: (_ for _ in ()).throw(
            RecoupExecutionError(
                "create_payment_link", "boom", code="SERVER_ERROR", retryable=True
            )
        ),
    )
    customer, event, decision = make(
        session,
        action_type=ActionType.NUDGE_WITH_INCENTIVE,
        params={"incentive_paise": 200_00},
        reason_code="payment_cancelled",
    )
    run = execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)

    assert run.status is ActionStatus.FAILED
    assert run.incentive_paise == 0
    assert counts(session) == (0, 0)
    assert event.status is EventStatus.DECIDED  # left for a later sweep


# --- money and cost are recorded honestly ---------------------------------


def test_spend_log_only_when_an_incentive_was_granted(no_keys, outbox_file, session):
    customer, event, decision = make(
        session, reason_code="payment_cancelled", action_type=ActionType.NUDGE
    )
    execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)
    assert counts(session)[1] == 0


def test_incentive_is_booked_and_discounts_the_link(no_keys, outbox_file, session):
    customer, event, decision = make(
        session,
        reason_code="payment_cancelled",
        action_type=ActionType.NUDGE_WITH_INCENTIVE,
        params={"incentive_paise": 200_00},
        amount_paise=2_000_00,
    )
    run = execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)

    contacts, spends = counts(session)
    assert (contacts, spends) == (1, 1)
    assert run.incentive_paise == 200_00
    assert session.scalar(select(SpendLog.amount_paise)) == 200_00

    # The link is priced at the discount, and the copy quotes the same number.
    assert run.response["amount"] == 1_800_00
    body = outbox.read_all(outbox_file)[0]["body"]
    assert "Rs 200 off" in body and "Rs 1,800" in body


def test_channel_costs_are_real_numbers(no_keys, outbox_file, session):
    """A zero here is not a missing number - it is a claim the message was free."""
    customer, event, decision = make(
        session, reason_code="card_expired", action_type=ActionType.PAYMENT_LINK
    )
    run = execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)

    assert run.channel_cost_paise == outbox.CHANNEL_COST_PAISE["sms"] == 25
    assert outbox.read_all(outbox_file)[0]["cost_paise"] == 25


def test_a_nudge_costs_email_not_zero(no_keys, outbox_file, session):
    customer, event, decision = make(
        session, reason_code="payment_cancelled", action_type=ActionType.NUDGE
    )
    run = execute(session, decision, allowed(event, customer, decision, NOW), event, customer, NOW)
    assert run.channel_cost_paise == outbox.CHANNEL_COST_PAISE["email"] == 10


def test_indian_digit_grouping(no_keys):
    assert outbox.rupees(1_29_900_00) == "1,29,900"
    assert outbox.rupees(999_00) == "999"
    assert outbox.rupees(1_50_00) == "150"
    assert outbox.rupees(12_34) == "12.34"


def test_a_link_is_never_created_and_then_left_out_of_the_copy(no_keys, session):
    """Recoup pays for every link it mints. One the customer never sees is pure loss.

    Runs across the whole taxonomy so a reason code added later cannot silently
    fall through to a template with no call to action.
    """
    from recoup.taxonomy import all_codes

    customer, event, _ = make(session)
    for code in all_codes():
        event.reason_code = code
        for action in (ActionType.PAYMENT_LINK, ActionType.NUDGE_WITH_INCENTIVE):
            _, _, body = outbox.compose(
                event,
                customer,
                action,
                link_url="https://recoup.invalid/dry-run/plink_dry_1",
                incentive_paise=200_00 if action is ActionType.NUDGE_WITH_INCENTIVE else 0,
            )
            assert "plink_dry_1" in body, f"{code}/{action.value} drops the link"


def test_copy_differs_by_strategy(no_keys, session):
    """A bank-outage notice must not read like an abandoned-cart nudge."""
    customer, event, _ = make(session, reason_code="issuer_down")
    outage = outbox.compose(event, customer, ActionType.NUDGE)

    event.reason_code = "payment_cancelled"
    cart = outbox.compose(event, customer, ActionType.NUDGE)

    assert outage[1] != cart[1]
    assert "no action needed" in outage[2]
    assert "no action needed" not in cart[2]


# --- retry policy ---------------------------------------------------------


def live_client(monkeypatch, resource: str, exc: Exception):
    """A configured account whose gateway always fails, counting the attempts."""
    monkeypatch.setattr(
        razorpay_client,
        "get_settings",
        lambda: StubSettings(dry_run=False, razorpay_key_id="rzp_test_x", razorpay_key_secret="y"),
    )
    monkeypatch.setattr(razorpay_client, "RETRY_BACKOFF_SECONDS", (0.0, 0.0))

    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise exc

    monkeypatch.setattr(
        razorpay_client,
        "_require_client",
        lambda: SimpleNamespace(**{resource: SimpleNamespace(create=boom, fetch=boom)}),
    )
    return calls


def test_a_4xx_is_never_retried(monkeypatch):
    """Retrying a rejected request is how a recovery agent double-charges."""
    calls = live_client(monkeypatch, "order", BadRequestError("amount is invalid"))

    with pytest.raises(RecoupExecutionError) as caught:
        razorpay_client.create_order(1_000_00, "recoup_evt_1")

    assert calls["n"] == 1
    assert caught.value.retryable is False
    assert caught.value.payload["error"]["code"] == "BAD_REQUEST_ERROR"


def test_a_5xx_is_retried_twice_then_gives_up(monkeypatch):
    calls = live_client(monkeypatch, "order", ServerError("upstream unavailable"))

    with pytest.raises(RecoupExecutionError) as caught:
        razorpay_client.create_order(1_000_00, "recoup_evt_1")

    assert calls["n"] == razorpay_client.MAX_RETRIES + 1 == 3
    assert caught.value.retryable is True
    assert caught.value.payload["attempts"] == 3


def test_sdk_exceptions_never_escape_raw(monkeypatch):
    """Every caller should be able to catch one type, not import razorpay.errors."""
    live_client(monkeypatch, "payment_link", ServerError("nope"))
    with pytest.raises(RecoupExecutionError):
        razorpay_client.fetch_payment_link("plink_123")


def test_razorpay_is_never_allowed_to_notify_the_customer(monkeypatch, session):
    """The single most dangerous default in the Payment Links API.

    Left on, Razorpay SMSes and emails the link itself - bypassing quiet hours,
    the fatigue cap and the channel-cost ledger, and firing at every generated
    address in the seed set. Recoup owns customer contact or it owns none of it.
    """
    monkeypatch.setattr(
        razorpay_client,
        "get_settings",
        lambda: StubSettings(
            dry_run=False, razorpay_key_id="rzp_test_x", razorpay_key_secret="y"
        ),
    )
    sent = {}
    monkeypatch.setattr(
        razorpay_client,
        "_require_client",
        lambda: SimpleNamespace(
            payment_link=SimpleNamespace(
                create=lambda data: sent.update(data) or {"id": "plink_1"}
            )
        ),
    )
    customer, event, _ = make(session)
    razorpay_client.create_payment_link(1_000_00, customer, "recovery")

    assert sent["notify"] == {"sms": False, "email": False}
    assert sent["reminder_enable"] is False
