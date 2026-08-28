"""Recoverability scoring: how likely is this money to come back, and what is it worth.

Deterministic. No LLM. This is the layer that decides which events are worth
thinking hard about, and roughly four out of five events never need more than
this - the taxonomy plus a handful of customer signals settles them.

A note on calibration, because it matters for reading the results
-----------------------------------------------------------------
The priors this scorer starts from live in recoup/taxonomy.py, and the outcome
simulator in recoup/seed/world.py is grounded in the same domain reasoning. So
in the sandbox the scorer looks well-calibrated - but that reflects internal
consistency, not predictive validity. Against a real merchant's traffic the
priors would be wrong on day one.

That is what `fit_from_outcomes()` is for: once real outcomes exist, per-reason
recovery rates are estimated from them and shrunk toward the prior in proportion
to how little evidence there is. It is the honest path from "reasonable guess"
to "measured", and the report states which of the two produced any given number.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from recoup.db import Assessment, Cohort, Customer, Outcome, RevenueEvent
from recoup.detect.features import extract, liquidity_window
from recoup.taxonomy import Strategy, profile_for

SCORER_VERSION = "v1"

PRIOR_STRENGTH = 25.0
"""Pseudo-observations backing each taxonomy prior.

Shrinkage weight. With 25 pseudo-observations, a reason code needs on the order
of 25 real outcomes before the empirical rate meaningfully overrides the prior -
which stops a bucket with four events and three lucky recoveries from declaring
itself a 75% recovery opportunity.
"""


@dataclass(frozen=True)
class Score:
    recoverability: float
    expected_value_paise: int
    strategy: Strategy
    earliest_action_at: datetime
    features: dict
    basis: str
    """'prior' or 'fitted:<n>' - where the base rate came from. Surfaced in the UI."""


def _clamp(x: float, lo: float = 0.0, hi: float = 0.98) -> float:
    return max(lo, min(hi, x))


def score_event(
    event: RevenueEvent,
    customer: Customer,
    now: datetime,
    fitted_rates: dict[str, tuple[float, int]] | None = None,
) -> Score:
    """Estimate P(recovered | we act correctly) and the value of acting.

    `fitted_rates` maps reason_code -> (empirical_rate, n_observations). When
    absent, taxonomy priors are used unchanged.
    """
    profile = profile_for(event.reason_code)
    f = extract(event, customer, now)

    # --- base rate -------------------------------------------------------
    base = profile.base_recoverability
    basis = "prior"
    if fitted_rates and event.reason_code in fitted_rates:
        empirical, n = fitted_rates[event.reason_code]
        weight = n / (n + PRIOR_STRENGTH)
        base = weight * empirical + (1 - weight) * base
        basis = f"fitted:{n}"

    # A reason the taxonomy refuses to act on is worth nothing to act on,
    # whatever the customer history says. No adjustment can rescue it.
    if profile.strategy is Strategy.DO_NOT_RETRY:
        return Score(
            recoverability=0.0,
            expected_value_paise=0,
            strategy=profile.strategy,
            earliest_action_at=now,
            features=f,
            basis=basis,
        )

    p = base

    # --- customer signal -------------------------------------------------
    # Someone who has recovered from failures before is the single strongest
    # positive signal available, and it is a real one: past recovery behaviour
    # is observable, unlike intent.
    if f["prior_failure_count"] >= 3:
        p += 0.18 * (f["historical_recovery_rate"] - 0.35)

    if f["is_repeat_customer"]:
        p += 0.05
    if f["customer_tenure_days"] > 365:
        p += 0.03

    # Chronic failers convert worse regardless of reason.
    if f["prior_failure_count"] > 10 and f["historical_success_rate"] < 0.5:
        p -= 0.08

    # --- transaction signal ----------------------------------------------
    # Higher ticket, more deliberation, lower recovery - with a floor, because
    # the effect flattens out well before the tail of the distribution.
    amount_inr = f["amount_inr"]
    if amount_inr > 500:
        p -= min(0.14, 0.045 * (amount_inr / 5000))

    # A second failure on the same order is meaningfully worse than the first.
    if f["attempt_no"] > 1:
        p -= 0.10 * (f["attempt_no"] - 1)

    # Failing on a rail the customer does not normally use is often
    # instrument-specific rather than intent-driven, so switching has a real
    # chance of working.
    if not f["rail_is_preferred"] and profile.switch_to:
        p += 0.04

    # --- decay -----------------------------------------------------------
    # Recovery probability decays with age. Customer-present failures decay
    # fastest; receivables barely decay at all over a week.
    age = f["event_age_hours"]
    if profile.strategy in (Strategy.RETRY_NOW,):
        p *= max(0.30, 1.0 - 0.05 * age)
    elif profile.strategy is Strategy.PERSUADE:
        p *= max(0.45, 1.0 - 0.012 * age)
    else:
        p *= max(0.55, 1.0 - 0.006 * age)

    p = _clamp(p)

    # --- when it becomes worth acting ------------------------------------
    if profile.strategy is Strategy.RETRY_ON_LIQUIDITY:
        earliest = liquidity_window(event.occurred_at, now)
    else:
        earliest = event.occurred_at + timedelta(minutes=profile.retry_after_minutes)

    return Score(
        recoverability=round(p, 4),
        # From the feature dict, not the column. features.extract() coerces
        # nullable columns to their schema defaults at the boundary; reading the
        # raw attribute here reintroduced the crash that coercion exists to
        # prevent, two lines after the value had already been sanitised.
        expected_value_paise=int(p * f["amount_paise"]),
        strategy=profile.strategy,
        earliest_action_at=max(earliest, event.occurred_at),
        features=f,
        basis=basis,
    )


def fit_from_outcomes(session: Session) -> dict[str, tuple[float, int]]:
    """Estimate per-reason recovery rates from observed treated outcomes.

    Treatment arm only. The control arm measures what happens with no
    intervention, which is a different quantity - mixing the two would blend the
    organic rate into the intervened rate and quietly understate both.

    Returns reason_code -> (rate, n). Shrinkage toward the prior is applied at
    scoring time, not here, so the raw evidence stays inspectable.
    """
    rows = session.execute(
        select(RevenueEvent.reason_code, Outcome.recovered)
        .join(Outcome, Outcome.event_id == RevenueEvent.id)
        .where(RevenueEvent.cohort == Cohort.TREATMENT)
    ).all()

    tally: dict[str, list[int]] = {}
    for reason_code, recovered in rows:
        bucket = tally.setdefault(reason_code, [0, 0])
        bucket[0] += 1 if recovered else 0
        bucket[1] += 1

    return {code: (hits / n, n) for code, (hits, n) in tally.items() if n > 0}


def assess(
    session: Session,
    event: RevenueEvent,
    now: datetime,
    fitted_rates: dict[str, tuple[float, int]] | None = None,
) -> Assessment:
    """Score an event and persist the Assessment row."""
    customer = session.get(Customer, event.customer_id)
    if customer is None:
        raise ValueError(f"event {event.id} references unknown customer")

    s = score_event(event, customer, now, fitted_rates)

    assessment = Assessment(
        event_id=event.id,
        features=s.features,
        recoverability=s.recoverability,
        expected_value_paise=s.expected_value_paise,
        recommended_strategy=s.strategy.value,
        earliest_action_at=s.earliest_action_at,
        scorer_version=f"{SCORER_VERSION}/{s.basis}",
    )
    session.add(assessment)
    return assessment
