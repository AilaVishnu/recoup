"""The bounds are the product. They get tested like it.

Each test names a way the system could lose money, annoy a customer, or corrupt
its own evaluation, and asserts that the policy engine stops it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recoup.db import (
    ActionType,
    Base,
    Cohort,
    ContactLog,
    PolicyVerdict,
    SpendLog,
)
from recoup.policy.rules import Bounds, ReviewContext, review
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 14:00 IST on the 3rd - inside business hours, inside a liquidity window.
NOON_IST_UTC = datetime(2026, 3, 3, 8, 30)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, future=True)()
    yield s
    s.close()


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


# --- the expensive mistakes ------------------------------------------------


def test_fraud_declines_are_never_retried(session):
    r = review(ctx(reason_code="fraud_suspected", action_type=ActionType.RETRY_PAYMENT), session)
    assert r.verdict is PolicyVerdict.DENY
    assert not named(r, "never_retry_risk_declines").passed


def test_fraud_declines_may_still_be_escalated(session):
    r = review(
        ctx(reason_code="fraud_suspected", action_type=ActionType.ESCALATE_TO_HUMAN),
        session,
    )
    assert named(r, "never_retry_risk_declines").passed


def test_unknown_reason_codes_fail_closed(session):
    r = review(ctx(reason_code="some_new_code_from_2027", action_type=ActionType.NUDGE), session)
    assert r.verdict is PolicyVerdict.DENY
    assert not named(r, "unknown_reason_fails_closed").passed


def test_discounting_a_technical_failure_is_refused(session):
    """The single most common way recovery tooling burns margin."""
    r = review(
        ctx(
            reason_code="issuer_down",
            action_type=ActionType.NUDGE_WITH_INCENTIVE,
            incentive_paise=200_00,
        ),
        session,
    )
    assert r.verdict is PolicyVerdict.DENY
    assert not named(r, "incentive_eligibility").passed


# --- spending bounds -------------------------------------------------------


def test_incentive_capped_by_fraction(session):
    r = review(
        ctx(action_type=ActionType.NUDGE_WITH_INCENTIVE, incentive_paise=500_00),
        session,
    )  # 25% of Rs 2,000
    assert not named(r, "incentive_depth").passed


def test_incentive_capped_in_absolute_terms_on_large_orders(session):
    """A percentage cap alone is unbounded on a big enough order."""
    r = review(
        ctx(
            amount_paise=100_000_00,
            action_type=ActionType.NUDGE_WITH_INCENTIVE,
            incentive_paise=5_000_00,  # only 5%, but Rs 5,000
        ),
        session,
    )
    assert not named(r, "incentive_depth").passed


def test_incentive_must_clear_the_ev_hurdle(session):
    r = review(
        ctx(
            recoverability=0.92,  # almost certain to recover anyway
            action_type=ActionType.NUDGE_WITH_INCENTIVE,
            incentive_paise=250_00,
        ),
        session,
    )
    assert not named(r, "incentive_ev_positive").passed


def test_daily_budget_is_enforced_across_events(session):
    session.add(
        SpendLog(occurred_at=NOON_IST_UTC - timedelta(hours=2), amount_paise=24_900_00, event_id="e0")
    )
    session.commit()
    r = review(
        ctx(action_type=ActionType.NUDGE_WITH_INCENTIVE, incentive_paise=200_00),
        session,
    )
    assert not named(r, "daily_budget").passed


def test_tiny_expected_value_is_not_worth_chasing(session):
    r = review(ctx(expected_value_paise=10_00), session)
    assert not named(r, "minimum_expected_value").passed


# --- customer experience ---------------------------------------------------


def test_no_contact_during_quiet_hours(session):
    three_am_ist = datetime(2026, 3, 3, 21, 30)  # 03:00 IST next day
    r = review(ctx(now=three_am_ist, earliest_action_at=three_am_ist), session)
    assert not named(r, "quiet_hours").passed


def test_silent_retries_are_allowed_during_quiet_hours(session):
    """A retry is not a message. Blocking it overnight would be pointless."""
    three_am_ist = datetime(2026, 3, 3, 21, 30)
    r = review(
        ctx(
            reason_code="issuer_down",
            action_type=ActionType.RETRY_PAYMENT,
            now=three_am_ist,
            earliest_action_at=three_am_ist,
        ),
        session,
    )
    assert named(r, "quiet_hours").passed


def test_contact_cap_counts_across_all_events_not_per_event(session):
    for i in range(3):
        session.add(
            ContactLog(
                customer_id="cust_test",
                occurred_at=NOON_IST_UTC - timedelta(days=i + 1),
                action_type=ActionType.NUDGE,
                event_id=f"other_evt_{i}",
            )
        )
    session.commit()
    r = review(ctx(), session)
    assert not named(r, "contact_frequency").passed


def test_contacts_outside_the_window_do_not_count(session):
    session.add(
        ContactLog(
            customer_id="cust_test",
            occurred_at=NOON_IST_UTC - timedelta(days=20),
            action_type=ActionType.NUDGE,
            event_id="old",
        )
    )
    session.commit()
    r = review(ctx(), session)
    assert named(r, "contact_frequency").passed


# --- timing ----------------------------------------------------------------


def test_acting_before_the_window_opens_is_refused(session):
    r = review(ctx(earliest_action_at=NOON_IST_UTC + timedelta(hours=30)), session)
    assert not named(r, "timing_floor").passed


def test_attempt_cap_comes_from_the_taxonomy(session):
    r = review(
        ctx(reason_code="card_expired", action_type=ActionType.PAYMENT_LINK, attempts_so_far=1),
        session,
    )  # card_expired allows exactly 1
    assert not named(r, "attempt_cap").passed


# --- autonomy and evaluation integrity -------------------------------------


def test_high_value_escalates_rather_than_denies(session):
    r = review(ctx(amount_paise=40_000_00, expected_value_paise=8_000_00), session)
    assert r.verdict is PolicyVerdict.ESCALATE


def test_control_arm_is_never_executed(session):
    r = review(ctx(cohort=Cohort.CONTROL), session)
    assert r.verdict is PolicyVerdict.DENY
    assert not named(r, "control_arm_suppression").passed


def test_control_arm_decision_is_still_fully_recorded(session):
    """Suppression happens at execution, not at reasoning - that is the point."""
    r = review(ctx(cohort=Cohort.CONTROL), session)
    assert len(r.checks) == 13


# --- the engine's own contract --------------------------------------------


def test_every_rule_runs_even_after_a_failure(session):
    """No short-circuiting: the record must show everything that was wrong."""
    r = review(
        ctx(
            reason_code="fraud_suspected",
            action_type=ActionType.NUDGE_WITH_INCENTIVE,
            incentive_paise=9_000_00,
            expected_value_paise=1_00,
            now=datetime(2026, 3, 3, 21, 30),
        ),
        session,
    )
    assert len(r.checks) == 13
    assert len(r.violations) >= 4


def test_a_clean_action_is_allowed(session):
    r = review(ctx(), session)
    assert r.verdict is PolicyVerdict.ALLOW, r.violations
    assert r.violations == []
