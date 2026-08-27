"""Deterministic feature extraction. No model, no randomness, no network.

Everything the scorer and the agent are allowed to reason from is produced here
and recorded verbatim on the Assessment row, which is what makes a score
reproducible months later.

Features are observable-only. Nothing here derives from the outcome, the cohort
assignment, or anything the simulator knows - see tests/test_no_oracle_leak.py.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

from recoup.db import Customer, RevenueEvent
from recoup.taxonomy import Strategy, profile_for

IST_OFFSET = timedelta(hours=5, minutes=30)
"""Events are stored in UTC; customer-facing timing decisions are made in IST.

Getting this wrong is not a rounding error - it is the difference between a
payment reminder at 9am and one at 3:30am.
"""


def to_ist(dt: datetime) -> datetime:
    return dt + IST_OFFSET


def liquidity_window(occurred_at: datetime, now: datetime | None = None) -> datetime:
    """The next instant at which an insufficient-funds retry is worth attempting.

    Salaried accounts in India are credited around the last working day of the
    month, with the balance surviving into the first few days of the next one.
    So the recoverable window for a liquidity failure is roughly the 1st to the
    5th, and a retry on the 22nd is close to guaranteed waste - not because the
    strategy is wrong, but because the money is not there yet.

    Returns the start of the next such window, always at 10:00 IST, converted
    back to UTC. If the failure already happened inside a window, the customer
    has funds and something else is going on - retry after a short cooling-off
    rather than waiting a full month.
    """
    now = now or occurred_at
    ist = to_ist(max(occurred_at, now))
    day = ist.day

    if 1 <= day <= 5:
        # In-window already. The balance excuse does not apply; give it a day.
        target = ist + timedelta(days=1)
    elif day >= 26:
        # Payday is imminent - wait for the 1st of next month.
        year = ist.year + (1 if ist.month == 12 else 0)
        month = 1 if ist.month == 12 else ist.month + 1
        target = datetime(year, month, 1)
    else:
        # Mid-month trough. Nothing changes until the next cycle.
        year = ist.year + (1 if ist.month == 12 else 0)
        month = 1 if ist.month == 12 else ist.month + 1
        target = datetime(year, month, 1)

    target = datetime.combine(target.date(), time(hour=10, minute=0))
    return target - IST_OFFSET


def is_quiet_hours(dt: datetime) -> bool:
    """True between 21:00 and 08:00 IST, when a merchant should not be messaging."""
    hour = to_ist(dt).hour
    return hour >= 21 or hour < 8


def extract(event: RevenueEvent, customer: Customer, now: datetime) -> dict[str, Any]:
    """Build the feature dictionary for one event."""
    profile = profile_for(event.reason_code)
    age_hours = max(0.0, (now - event.occurred_at).total_seconds() / 3600.0)

    attempts = customer.prior_success_count + customer.prior_failure_count
    historical_success_rate = (
        customer.prior_success_count / attempts if attempts else 0.5
    )
    historical_recovery_rate = (
        customer.prior_recovery_count / customer.prior_failure_count
        if customer.prior_failure_count
        else 0.0
    )

    ist = to_ist(event.occurred_at)

    return {
        # --- failure semantics, straight from the taxonomy ---
        "reason_code": event.reason_code,
        "strategy": profile.strategy.value,
        "failure_source": profile.source,
        "failure_step": profile.step,
        "base_recoverability": profile.base_recoverability,
        "incentive_eligible": profile.incentive_eligible,
        "max_attempts": profile.max_attempts,
        "retry_after_minutes": profile.retry_after_minutes,
        "switch_rails": [r.value for r in profile.switch_to],
        # --- the money ---
        "amount_paise": event.amount_paise,
        "amount_inr": round(event.amount_paise / 100, 2),
        "is_high_value": event.amount_paise >= 25_000_00,
        # --- who this is ---
        "customer_tenure_days": max(0, (now - customer.created_at).days),
        "prior_success_count": customer.prior_success_count,
        "prior_failure_count": customer.prior_failure_count,
        "prior_recovery_count": customer.prior_recovery_count,
        "historical_success_rate": round(historical_success_rate, 3),
        "historical_recovery_rate": round(historical_recovery_rate, 3),
        "lifetime_value_inr": round(customer.lifetime_value_paise / 100, 2),
        "is_repeat_customer": customer.prior_success_count > 3,
        # --- the attempt ---
        "rail": event.rail,
        "preferred_rail": customer.preferred_rail,
        "rail_is_preferred": event.rail == customer.preferred_rail,
        "attempt_no": event.attempt_no,
        # --- timing ---
        "event_age_hours": round(age_hours, 2),
        "occurred_hour_ist": ist.hour,
        "occurred_day_of_month": ist.day,
        "occurred_is_weekend": ist.weekday() >= 5,
        "in_quiet_hours_now": is_quiet_hours(now),
        "needs_liquidity_wait": profile.strategy is Strategy.RETRY_ON_LIQUIDITY,
        # --- context ---
        "event_kind": event.kind.value,
        "checkout_stage": event.extra.get("checkout_stage"),
        "basket_items": event.extra.get("items"),
    }
