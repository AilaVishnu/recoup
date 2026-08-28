"""The deterministic layer, which had no tests at all.

A coverage audit found `recoup/detect/` effectively unexercised: liquidity_window()
at 0% and the do-not-retry branch of the scorer never executed. Both are
load-bearing. The liquidity window is the timing decision the pitch spends
fifteen seconds on, and the do-not-retry branch is what stops a risk decline from
being scored as an opportunity.

Neither would have failed loudly if broken. A liquidity window that returned the
wrong month still returns a datetime, the pipeline still schedules against it,
and the only symptom is that insufficient-funds retries stop working - which
looks like the strategy being wrong rather than the clock being wrong.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recoup.detect.features import (
    IST_OFFSET,
    extract,
    is_quiet_hours,
    liquidity_window,
    to_ist,
)
from recoup.detect.scorer import score_event
from recoup.db import Cohort, Customer, EventKind, RevenueEvent
from recoup.taxonomy import Strategy


def at_ist(y: int, m: int, d: int, hour: int = 12, minute: int = 0) -> datetime:
    """A naive-UTC instant corresponding to the given IST wall clock."""
    return datetime(y, m, d, hour, minute) - IST_OFFSET


# ---------------------------------------------------------------------------
# liquidity_window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("day", [6, 12, 18, 22, 25])
def test_a_mid_month_failure_waits_for_the_next_salary_window(day):
    """The whole point. Retrying on the 22nd fails again for the same reason."""
    occurred = at_ist(2026, 3, day)
    target = to_ist(liquidity_window(occurred))
    assert (target.year, target.month, target.day) == (2026, 4, 1)


@pytest.mark.parametrize("day", [26, 28, 30, 31])
def test_a_late_month_failure_waits_for_the_first_rather_than_days_more(day):
    """Payday is imminent; the answer is the 1st, not another mid-month attempt."""
    occurred = at_ist(2026, 3, day)
    target = to_ist(liquidity_window(occurred))
    assert (target.year, target.month, target.day) == (2026, 4, 1)


@pytest.mark.parametrize("day", [1, 2, 3, 4, 5])
def test_a_failure_inside_the_window_retries_the_next_day(day):
    """In-window means the balance excuse does not apply.

    Salary has landed and the payment still failed, so something else is going
    on - waiting a full month for the next window would be the wrong answer to
    the wrong question.
    """
    occurred = at_ist(2026, 3, day)
    target = to_ist(liquidity_window(occurred))
    assert (target - to_ist(occurred)).days <= 1


def test_the_year_rolls_over():
    """December's mid-month failures wait for January, not month thirteen."""
    occurred = at_ist(2026, 12, 18)
    target = to_ist(liquidity_window(occurred))
    assert (target.year, target.month, target.day) == (2027, 1, 1)


def test_a_late_december_failure_rolls_the_year_too():
    occurred = at_ist(2026, 12, 29)
    target = to_ist(liquidity_window(occurred))
    assert (target.year, target.month, target.day) == (2027, 1, 1)


def test_the_retry_always_lands_in_business_hours():
    """10:00 IST, never 03:00. The window decides the day; this decides the hour."""
    for day in (2, 9, 17, 23, 28):
        target = to_ist(liquidity_window(at_ist(2026, 5, day, hour=3)))
        assert target.hour == 10, f"day {day} scheduled a retry at {target.hour}:00 IST"
        assert not is_quiet_hours(liquidity_window(at_ist(2026, 5, day, hour=3)))


def test_the_window_is_never_before_the_failure():
    """A retry scheduled into the past would fire immediately, defeating the wait."""
    for month in range(1, 13):
        for day in (1, 5, 14, 26, 28):
            occurred = at_ist(2026, month, day, hour=23)
            assert liquidity_window(occurred) > occurred, f"{month}-{day} went backwards"


def test_a_later_now_moves_the_window_forward():
    """Resolving an old event today must not schedule a retry into last month."""
    occurred = at_ist(2026, 3, 14)
    later = at_ist(2026, 6, 14)
    assert liquidity_window(occurred, now=later) >= later


# ---------------------------------------------------------------------------
# quiet hours, at the boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hour, minute, quiet",
    [
        (20, 59, False),
        (21, 0, True),
        (23, 59, True),
        (0, 30, True),
        (7, 59, True),
        (8, 0, False),
        (8, 1, False),
    ],
)
def test_quiet_hours_boundaries(hour, minute, quiet):
    """Off-by-one here means messaging someone at 20:59 or refusing at 08:00."""
    assert is_quiet_hours(at_ist(2026, 3, 3, hour, minute)) is quiet


def test_an_aware_timestamp_is_not_shifted_twice():
    """to_ist adds an offset blindly; an already-aware value must be normalised.

    A tz-aware 03:00 IST double-shifted reads as 08:30 - outside quiet hours,
    which is how a 3am message gets approved by the rule that exists to stop it.
    """
    naive_utc = at_ist(2026, 3, 4, 3, 0)
    aware = naive_utc.replace(tzinfo=timezone.utc)
    assert is_quiet_hours(naive_utc) is True
    assert is_quiet_hours(aware) is True, "aware input was double-shifted"


# ---------------------------------------------------------------------------
# the scorer's do-not-retry branch
# ---------------------------------------------------------------------------


def _event(reason: str, amount: int = 5_000_00) -> tuple[RevenueEvent, Customer]:
    now = datetime(2026, 3, 10, 9, 0)
    customer = Customer(
        id="cust_detect",
        name="Test",
        email="t@example.com",
        contact="+919000000000",
        created_at=now - timedelta(days=500),
        prior_success_count=20,
        prior_failure_count=3,
        prior_recovery_count=2,
        lifetime_value_paise=50_000_00,
        preferred_rail="card",
    )
    event = RevenueEvent(
        id=f"evt_{reason}",
        kind=EventKind.PAYMENT_FAILED,
        customer_id=customer.id,
        amount_paise=amount,
        occurred_at=now,
        reason_code=reason,
        rail="card",
        cohort=Cohort.TREATMENT,
        extra={},
    )
    return event, customer


@pytest.mark.parametrize(
    "reason", ["payment_risk_check_failed", "compliance_violation", "payment_amount_tampered"]
)
def test_a_do_not_retry_reason_scores_zero_however_good_the_customer(reason):
    """No customer history may rescue a reason the taxonomy refuses to act on.

    This branch had never executed in the test suite. Broken, it would price a
    risk decline as an opportunity worth several thousand rupees, and the ranked
    worklist would put it at the top.
    """
    event, customer = _event(reason)
    s = score_event(event, customer, now=event.occurred_at)

    assert s.recoverability == 0.0
    assert s.expected_value_paise == 0
    assert s.strategy is Strategy.DO_NOT_RETRY


def test_an_unrecognised_reason_also_scores_zero():
    """Fail closed. An unknown code is not an average code."""
    event, customer = _event("some_code_razorpay_added_last_tuesday")
    s = score_event(event, customer, now=event.occurred_at)
    assert s.recoverability == 0.0
    assert s.expected_value_paise == 0


def test_a_recoverable_reason_is_not_zeroed():
    """Guards the test above: if everything scored zero these would pass vacuously."""
    event, customer = _event("incorrect_otp")
    s = score_event(event, customer, now=event.occurred_at)
    assert s.recoverability > 0.2
    assert s.expected_value_paise > 0


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------


def test_extracted_features_never_leak_the_cohort_or_the_outcome():
    """The scorer must not be able to see which arm an event is in.

    Cohort in the feature dict would let the model - and any fitted scorer -
    condition on the holdout, which would invalidate every number the project
    reports.
    """
    event, customer = _event("insufficient_funds")
    f = extract(event, customer, now=event.occurred_at)
    forbidden = {"cohort", "treatment", "control", "recovered", "outcome", "roll", "organic_p"}
    leaked = forbidden & {k.lower() for k in f}
    assert not leaked, f"feature dict exposes {leaked}"
    assert "control" not in str(f).lower()


def test_the_liquidity_flag_is_set_only_for_liquidity_failures():
    liquidity, customer = _event("insufficient_funds")
    other, _ = _event("incorrect_otp")
    assert extract(liquidity, customer, now=liquidity.occurred_at)["needs_liquidity_wait"]
    assert not extract(other, customer, now=other.occurred_at)["needs_liquidity_wait"]


def test_a_null_column_does_not_crash_the_scorer():
    """The bug this file found on its first run.

    An event whose attempt_no is None - unflushed, or written by a path that did
    not set it - reached `if f["attempt_no"] > 1` and raised TypeError. `assess()`
    is called outside the pipeline's try/except, because the orchestrator guards
    execution rather than scoring, so one such row ended a 600-event run with a
    NoneType error several frames from anything explaining it.

    The policy engine was hardened against exactly this shape of input months
    earlier; the scorer was not, and nothing had ever handed it a null.
    """
    event, customer = _event("incorrect_otp")
    event.attempt_no = None
    event.amount_paise = None
    customer.prior_success_count = None
    customer.prior_failure_count = None
    customer.prior_recovery_count = None
    customer.lifetime_value_paise = None

    s = score_event(event, customer, now=event.occurred_at)

    assert 0.0 <= s.recoverability <= 1.0
    assert s.expected_value_paise == 0, "no amount means no expected value, not a crash"
    assert s.features["attempt_no"] == 1, "coerced to the schema default"
