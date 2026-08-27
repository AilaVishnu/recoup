"""The dashboard has to survive being looked at.

Two failure modes matter more than the rest, because both surface in front of an
audience rather than in CI:

1. **An empty database.** A fresh clone, a dropped table, a seed that has not
   run yet. Every route renders something calm and instructive, never a stack
   trace.
2. **A page that quietly stops showing the passes.** The detail page's claim is
   that all thirteen bounds were evaluated and recorded. If a template change
   started rendering only the violations, every screenshot would still look
   plausible - so the count is asserted, not eyeballed.

The fixtures build the audit trail by hand and run the real policy engine over
it, so the checks these tests assert on are the ones the engine actually writes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from recoup.api import read
from recoup.api.main import app, db
from recoup.db import (
    ActionRun,
    ActionStatus,
    ActionType,
    Assessment,
    Attribution,
    Base,
    Cohort,
    ContactLog,
    Customer,
    Decision,
    DecisionSource,
    EventKind,
    EventStatus,
    Outcome,
    PolicyReview,
    RevenueEvent,
)
from recoup.policy.rules import RULES, ReviewContext, review

# 14:00 IST on the 3rd: inside business hours, inside a liquidity window - the
# same instant tests/test_policy.py uses, so a quiet-hours denial here would be
# a real one rather than an artefact of when the suite happened to run.
NOW = datetime(2026, 3, 3, 8, 30)


@pytest.fixture
def sessionmaker_for_empty_db(tmp_path):
    """A real file-backed SQLite database with the schema and nothing in it."""
    engine = create_engine(f"sqlite:///{tmp_path / 'recoup.db'}", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)


def _customer(idx: int) -> Customer:
    return Customer(
        id=f"cust_{idx:04d}",
        name=f"Customer {idx}",
        email=f"c{idx}@example.in",
        contact=f"+9198000000{idx:02d}",
        created_at=NOW - timedelta(days=400),
        prior_success_count=8,
        prior_failure_count=4,
        prior_recovery_count=2,
        lifetime_value_paise=40_000_00,
        preferred_rail="upi",
    )


def _chain(
    session,
    *,
    event_id: str,
    customer: Customer,
    reason_code: str,
    amount_paise: int,
    cohort: Cohort,
    action_type: ActionType,
    incentive_paise: int = 0,
    source: DecisionSource = DecisionSource.RULES,
    recovered: bool = False,
) -> RevenueEvent:
    """One event carried the whole way through the pipeline, policy included."""
    revenue_event = RevenueEvent(
        id=event_id,
        kind=EventKind.PAYMENT_FAILED,
        customer_id=customer.id,
        amount_paise=amount_paise,
        occurred_at=NOW - timedelta(hours=6),
        reason_code=reason_code,
        rail="card",
        cohort=cohort,
        status=EventStatus.ACTED,
        attempt_no=1,
        extra={"checkout_stage": "payment", "items": 2},
    )
    session.add(revenue_event)
    session.flush()

    expected_value = amount_paise // 2
    session.add(
        Assessment(
            event_id=event_id,
            features={
                "reason_code": reason_code,
                "amount_inr": amount_paise / 100,
                "prior_failure_count": 4,
                "rail_is_preferred": False,
                "event_age_hours": 6.0,
            },
            recoverability=0.5,
            expected_value_paise=expected_value,
            recommended_strategy="retry_now",
            earliest_action_at=NOW - timedelta(hours=1),
            scorer_version="v1/prior",
        )
    )

    decision = Decision(
        event_id=event_id,
        source=source,
        action_type=action_type,
        params={"rail": "upi", "incentive_paise": incentive_paise},
        rationale=(
            "Correctable in-session failure on a rail the customer does not "
            "normally use; re-present once."
        ),
        model="claude-opus-5" if source is DecisionSource.LLM else None,
        input_tokens=1200 if source is DecisionSource.LLM else 0,
        output_tokens=180 if source is DecisionSource.LLM else 0,
    )
    session.add(decision)
    session.flush()

    verdict = review(
        ReviewContext(
            event_id=event_id,
            customer_id=customer.id,
            cohort=cohort,
            reason_code=reason_code,
            amount_paise=amount_paise,
            expected_value_paise=expected_value,
            recoverability=0.5,
            earliest_action_at=NOW - timedelta(hours=1),
            attempts_so_far=0,
            action_type=action_type,
            incentive_paise=incentive_paise,
            now=NOW,
        ),
        session,
    )
    session.add(
        PolicyReview(
            decision_id=decision.id,
            verdict=verdict.verdict,
            checks=[c.as_dict() for c in verdict.checks],
            violations=verdict.violations,
        )
    )

    if verdict.allowed and cohort is Cohort.TREATMENT:
        session.add(
            ActionRun(
                decision_id=decision.id,
                executed_at=NOW,
                action_type=action_type,
                status=ActionStatus.SENT,
                razorpay_ref="plink_TEST00000001",
                incentive_paise=incentive_paise,
                channel_cost_paise=25,
                response={"id": "plink_TEST00000001", "status": "created"},
            )
        )
        session.add(
            ContactLog(
                customer_id=customer.id,
                occurred_at=NOW,
                action_type=action_type,
                event_id=event_id,
            )
        )

    session.add(
        Outcome(
            event_id=event_id,
            resolved_at=NOW + timedelta(days=3),
            recovered=recovered,
            recovered_paise=amount_paise if recovered else 0,
            attribution=Attribution.AGENT if recovered else Attribution.ORGANIC,
            hours_to_recovery=31.5 if recovered else None,
            note="resolved inside the 7-day window",
        )
    )
    return revenue_event


@pytest.fixture
def sessionmaker_for_seeded_db(sessionmaker_for_empty_db):
    Session = sessionmaker_for_empty_db
    session = Session()

    customers = [_customer(i) for i in range(3)]
    session.add_all(customers)
    session.flush()

    # A treated event that cleared every bound, executed, and recovered.
    _chain(
        session,
        event_id="evt_treatment_allow",
        customer=customers[0],
        reason_code="incorrect_otp",
        amount_paise=8_400_00,
        cohort=Cohort.TREATMENT,
        action_type=ActionType.RETRY_PAYMENT,
        recovered=True,
    )
    # Above the autonomy limit, so policy escalates rather than denies - and the
    # 12,34,567 is load-bearing twice over: it is the digit grouping the page has
    # to get right, and it is what puts this event over the bound.
    _chain(
        session,
        event_id="evt_escalated_high_value",
        customer=customers[0],
        reason_code="issuer_down",
        amount_paise=12_34_567_00,
        cohort=Cohort.TREATMENT,
        action_type=ActionType.RETRY_PAYMENT,
    )
    # The holdout twin: same decision, suppressed at the last step.
    _chain(
        session,
        event_id="evt_control_suppressed",
        customer=customers[1],
        reason_code="payment_cancelled",
        amount_paise=4_500_00,
        cohort=Cohort.CONTROL,
        action_type=ActionType.NUDGE_WITH_INCENTIVE,
        incentive_paise=200_00,
        source=DecisionSource.LLM,
    )
    # A risk decline that policy must refuse outright.
    _chain(
        session,
        event_id="evt_fraud_denied",
        customer=customers[2],
        reason_code="fraud_suspected",
        amount_paise=9_000_00,
        cohort=Cohort.TREATMENT,
        action_type=ActionType.RETRY_PAYMENT,
    )

    # Enough bare events to fill a page and force a second one - the list must
    # tolerate rows with no assessment, no decision and no outcome.
    for i in range(60):
        session.add(
            RevenueEvent(
                id=f"evt_bare_{i:03d}",
                kind=EventKind.CHECKOUT_ABANDONED,
                customer_id=customers[i % 3].id,
                amount_paise=1_000_00 + i,
                occurred_at=NOW - timedelta(days=2, minutes=i),
                reason_code="checkout_abandoned",
                rail="upi",
                cohort=Cohort.TREATMENT if i % 3 else Cohort.CONTROL,
                status=EventStatus.OPEN,
                attempt_no=1,
                extra={},
            )
        )

    session.commit()
    session.close()
    return Session


def make_client(Session) -> TestClient:
    def _session():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[db] = _session
    return TestClient(app)


@pytest.fixture
def empty_client(sessionmaker_for_empty_db):
    client = make_client(sessionmaker_for_empty_db)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def client(sessionmaker_for_seeded_db):
    client = make_client(sessionmaker_for_seeded_db)
    yield client
    app.dependency_overrides.clear()


ROUTES = ["/", "/events", "/policy", "/healthz"]


# ---------------------------------------------------------------------------
# Every route, both states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ROUTES)
def test_routes_render_on_a_seeded_database(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ROUTES)
def test_routes_render_on_an_empty_database(empty_client, path):
    """No rows anywhere. The pages must still be pages."""
    response = empty_client.get(path)
    assert response.status_code == 200


def test_empty_database_says_what_to_run(empty_client):
    assert "scripts/seed.py" in empty_client.get("/").text


def test_missing_event_is_a_page_not_a_traceback(client):
    response = client.get("/events/evt_does_not_exist")
    assert response.status_code == 404
    assert "No such event" in response.text


def test_detail_renders_on_an_empty_database(empty_client):
    assert empty_client.get("/events/evt_anything").status_code == 404


# ---------------------------------------------------------------------------
# The money page
# ---------------------------------------------------------------------------


def test_detail_renders_every_policy_check(client):
    """All thirteen bounds, passes included - that is the claim being made."""
    body = client.get("/events/evt_treatment_allow").text
    checks = [
        "unknown_reason_fails_closed",
        "never_retry_risk_declines",
        "attempt_cap",
        "timing_floor",
        "quiet_hours",
        "contact_frequency",
        "incentive_eligibility",
        "incentive_depth",
        "incentive_ev_positive",
        "daily_budget",
        "minimum_expected_value",
        "high_value_needs_human",
        "control_arm_suppression",
    ]
    for name in checks:
        assert name in body, f"policy check {name} missing from the replay"
    assert body.count("PASS") == len(RULES)
    assert f"{len(RULES)} of {len(RULES)} checks cleared" in body


def test_detail_shows_the_full_chain(client):
    body = client.get("/events/evt_treatment_allow").text
    for fragment in (
        "incorrect_otp",
        "Rs 8,400",
        "expected value",
        "plink_TEST00000001",
        "taxonomy prior",
        "Recovered",
    ):
        assert fragment in body


def test_escalation_is_shown_as_escalation_not_denial(client):
    """A bound that escalates is not a bound that refused - the page says which."""
    body = client.get("/events/evt_escalated_high_value").text
    assert "escalate" in body
    assert "high_value_needs_human" in body
    assert "exceeds the Rs 25,000 autonomy limit" in body


def test_control_event_is_banded_as_a_holdout(client):
    body = client.get("/events/evt_control_suppressed").text
    assert "Held out" in body
    assert "Suppressed for the holdout" in body
    assert "control_arm_suppression" in body
    assert "FAIL" in body


def test_denied_event_shows_the_rule_that_denied_it(client):
    body = client.get("/events/evt_fraud_denied").text
    assert "FAIL" in body
    assert "do-not-retry" in body
    assert "deny" in body


def test_money_uses_indian_digit_grouping(client):
    body = client.get("/events").text
    assert "Rs 12,34,567" in body
    assert "Rs 1,234,567" not in body


# ---------------------------------------------------------------------------
# The list
# ---------------------------------------------------------------------------


def test_events_paginate_at_fifty(client):
    first = client.get("/events").text
    assert first.count('href="/events/evt_') == 50
    assert "page 1 of 2" in first
    assert client.get("/events?page=2").status_code == 200

    # A page past the end lands on the last one. An empty table would read as a
    # broken filter, which is the wrong thing to be debugging on camera.
    beyond = client.get("/events?page=999").text
    assert "page 2 of 2" in beyond
    assert 'href="/events/evt_' in beyond


def test_events_sort_by_expected_value(client):
    """The first screen of the table is the money, not the most recent thing."""
    body = client.get("/events").text
    assert body.index("evt_escalated_high_value") < body.index("evt_fraud_denied")
    assert body.index("evt_fraud_denied") < body.index("evt_treatment_allow")
    assert body.index("evt_treatment_allow") < body.index("evt_bare_")


def test_filters_narrow_the_table(client):
    control = client.get("/events?cohort=control").text
    assert "evt_control_suppressed" in control
    assert "evt_treatment_allow" not in control

    denied = client.get("/events?verdict=deny").text
    assert "evt_fraud_denied" in denied
    assert "evt_treatment_allow" not in denied

    by_reason = client.get("/events?reason=incorrect_otp").text
    assert "evt_treatment_allow" in by_reason
    assert "evt_bare_000" not in by_reason


def test_unparseable_filters_are_ignored_not_fatal(client):
    response = client.get("/events?cohort=banana&status=&verdict=nonsense")
    assert response.status_code == 200
    assert "evt_treatment_allow" in response.text


def test_events_page_does_not_issue_a_query_per_row(sessionmaker_for_seeded_db):
    """Fifty rows, each with five related tables, is 250 queries done naively.

    The exact count is not the point and is allowed to drift with SQLAlchemy's
    eager-loading strategy; the order of magnitude is the point.
    """
    engine = sessionmaker_for_seeded_db.kw["bind"]
    statements: list[str] = []

    def record(conn, cursor, statement, params, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        client = make_client(sessionmaker_for_seeded_db)
        assert client.get("/events").status_code == 200
    finally:
        event.remove(engine, "before_cursor_execute", record)
        app.dependency_overrides.clear()

    assert len(statements) < 20, "\n".join(statements)


# ---------------------------------------------------------------------------
# Overview and policy
# ---------------------------------------------------------------------------


def test_overview_reports_incremental_next_to_gross(client):
    body = client.get("/").text
    assert "gross recovery" in body
    assert "incremental recovery" in body
    assert "cannibalised" in body
    assert "treatment" in body and "control" in body


def test_overview_without_an_eval_report_says_which_script_to_run(client, monkeypatch):
    monkeypatch.setattr("recoup.api.main.read.latest_report", lambda *a, **k: None)
    body = client.get("/").text
    assert "run_eval.py" in body


def _write_report(directory, payload) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "eval_42.json").write_text(json.dumps(payload), encoding="utf-8")


def test_overview_reads_the_eval_report_by_key_not_by_path(client, monkeypatch, tmp_path):
    """The grader owns that file's shape. Nesting it deeper must not blank the page."""
    monkeypatch.setattr(read, "REPORTS_DIR", tmp_path)
    _write_report(
        tmp_path,
        {
            "provenance": "Rupee figures are assumption output.",
            "metrics": {
                "overall": {
                    "gross_recovered_paise": 42_119_00,
                    "incremental_recovered_paise": 18_304_00,
                    "cannibalised_paise": 23_815_00,
                    "cost_paise": 2_105_00,
                    "net_paise": 16_199_00,
                }
            },
            "sweep": {
                "points": [
                    {
                        "label": "pessimistic",
                        "incremental_recovered_paise": 9_021_00,
                        "net_paise": 7_000_00,
                        "lift": {"absolute": -0.02, "ci_low": -0.10, "ci_high": 0.03},
                    },
                    {
                        "label": "optimistic",
                        "incremental_recovered_paise": 31_208_00,
                        "net_paise": 29_000_00,
                        "lift": {"absolute": 0.06, "ci_low": 0.01, "ci_high": 0.11},
                    },
                ]
            },
        },
    )

    body = client.get("/").text
    assert "Rs 18,304" in body
    assert "Rupee figures are assumption output." in body
    assert "sensitivity sweep" in body
    assert "pessimistic" in body and "optimistic" in body
    # The pessimistic interval spans zero and has to say so.
    assert "-10.0% to 3.0% *" in body


def test_a_corrupt_report_is_ignored_rather_than_fatal(client, monkeypatch, tmp_path):
    monkeypatch.setattr(read, "REPORTS_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "eval_broken.json").write_text("{not json", encoding="utf-8")

    response = client.get("/")
    assert response.status_code == 200
    assert "run_eval.py" in response.text


def test_policy_page_renders_the_bounds_from_source(client):
    body = client.get("/policy").text
    for bound in (
        "max_contacts_per_customer_per_week",
        "daily_incentive_budget_paise",
        "human_approval_above_paise",
        "min_incremental_ev_ratio",
    ):
        assert bound in body
    # The justifications are read out of the dataclass source, not restated here.
    assert "Message fatigue is the cost customers actually feel" in body
    assert "Rs 25,000" in body


def test_policy_page_lists_every_rule_even_on_an_empty_database(empty_client):
    body = empty_client.get("/policy").text
    assert "control_arm" in body
    assert "never_retry_risk_declines" in body


def test_policy_page_counts_what_actually_bound(client):
    body = client.get("/policy").text
    assert "never_retry_risk_declines" in body
    assert "control_arm_suppression" in body


def test_healthz_is_json(client):
    assert client.get("/healthz").json() == {"ok": True, "database": True}
