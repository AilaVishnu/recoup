"""The hidden world model. THIS IS THE SIMULATOR, NOT THE PRODUCT.

Read this file before believing any number Recoup reports.

Recoup has no access to a real merchant's post-failure customer behaviour, so
outcomes are simulated. This module holds the latent variables that decide
whether a given event recovers. Nothing in recoup/detect, recoup/agent,
recoup/policy or recoup/execute may import from here - the pipeline is blind to
these values by construction, and `tests/test_no_oracle_leak.py` enforces that.

What is honestly measured, and what is not
------------------------------------------
MEASURED FOR REAL - the decision quality of the pipeline. Whether Recoup picks
the right strategy for a failure reason, respects its bounds, refuses to retry
risk declines, times liquidity retries sensibly, and never exceeds its budget.
These are properties of the code and hold regardless of what this file says.

SIMULATED - the rupee figures. How much money comes back depends on the lift
parameters below, which are assumptions. They are drawn to be plausible and
conservative, but they are assumptions, and no amount of decimal places changes
that.

Because of this, recoup/eval/report.py runs a sensitivity sweep across a range
of lift assumptions and reports the range, not a single flattering point
estimate. A result that only holds at one parameter setting is not a result.

The one rule this file obeys
----------------------------
Lift is granted for the *correct* action, not for any action. Retrying a fraud
decline, discounting a bank outage, or hammering an insufficient-funds failure
five minutes later all earn zero lift or worse. A simulator that rewarded
activity would make Recoup look good for doing something stupid, which would
make the entire exercise worthless.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from recoup.taxonomy import Strategy, profile_for


@dataclass(frozen=True)
class LiftAssumptions:
    """How much a correctly-chosen action moves the needle.

    Expressed as absolute additions to recovery probability, capped at 0.95.
    Defaults are the conservative end of what published cart-recovery and
    payment-retry literature suggests; the sweep in recoup/eval/report.py runs
    from `pessimistic()` to `optimistic()`.
    """

    correct_retry: float = 0.18
    """Re-presenting a payment that failed for a transient/correctable reason."""

    correct_rail_switch: float = 0.14
    """Offering a working instrument when the original one cannot succeed."""

    correct_nudge: float = 0.06
    """A reminder alone, no money attached."""

    incentive_bonus: float = 0.09
    """Additional lift from a discount, on top of the nudge. Intent buckets only."""

    wrong_action_penalty: float = -0.03
    """Acting against the taxonomy. Mild, but negative - a badly-timed or
    irrelevant contact makes a customer marginally less likely to return, and a
    simulator that priced it at zero would let sloppy decisions look free."""

    @staticmethod
    def pessimistic() -> LiftAssumptions:
        return LiftAssumptions(
            correct_retry=0.09,
            correct_rail_switch=0.07,
            correct_nudge=0.02,
            incentive_bonus=0.03,
            wrong_action_penalty=-0.05,
        )

    @staticmethod
    def optimistic() -> LiftAssumptions:
        return LiftAssumptions(
            correct_retry=0.28,
            correct_rail_switch=0.22,
            correct_nudge=0.10,
            incentive_bonus=0.15,
            wrong_action_penalty=-0.01,
        )


# ---------------------------------------------------------------------------
# Organic recovery: what happens if Recoup does nothing at all.
# ---------------------------------------------------------------------------


def organic_recovery_probability(
    reason_code: str,
    amount_paise: int,
    prior_recovery_count: int,
    prior_failure_count: int,
    lifetime_value_paise: int,
    rng: random.Random,
) -> float:
    """P(customer resolves this themselves, with no intervention).

    This is the number that makes control groups necessary. A naive recovery
    system claims credit for all of it.
    """
    profile = profile_for(reason_code)

    # Transient technical failures self-heal at a high rate: the customer simply
    # tries again in an hour. This is the single biggest source of overclaiming
    # in recovery tooling - the outage ends, the payment succeeds, and the
    # recovery agent that happened to send an email takes the credit.
    base = {
        Strategy.RETRY_DELAYED: 0.42,
        Strategy.RETRY_NOW: 0.35,
        Strategy.RETRY_ON_LIQUIDITY: 0.22,
        Strategy.SWITCH_RAIL: 0.15,
        Strategy.PERSUADE: 0.07,
        Strategy.DO_NOT_RETRY: 0.02,
    }[profile.strategy]

    # Engaged customers come back on their own; the amount works against it.
    loyalty = min(0.20, 0.04 * prior_recovery_count + 0.00000002 * lifetime_value_paise)
    fatigue = min(0.12, 0.02 * prior_failure_count)
    price_drag = min(0.18, (amount_paise / 100_000) * 0.02)

    p = base + loyalty - fatigue - price_drag
    p *= rng.uniform(0.85, 1.15)  # per-customer idiosyncrasy
    return max(0.0, min(0.95, p))


# ---------------------------------------------------------------------------
# Treated recovery: what happens when Recoup acts.
# ---------------------------------------------------------------------------


def treated_recovery_probability(
    organic_p: float,
    reason_code: str,
    action_type: str,
    hours_waited: float,
    incentive_paise: int,
    amount_paise: int,
    assumptions: LiftAssumptions,
) -> float:
    """P(recovered | Recoup took `action_type`).

    Lift is earned by matching the taxonomy, not by acting.
    """
    profile = profile_for(reason_code)

    # Never-retry means never-retry. Acting here is punished, not merely wasted.
    if profile.strategy is Strategy.DO_NOT_RETRY:
        if action_type in ("no_action", "escalate_to_human"):
            return organic_p
        return max(0.0, organic_p + assumptions.wrong_action_penalty * 2)

    if action_type in ("no_action", "escalate_to_human"):
        return organic_p

    correct = _action_matches_strategy(action_type, profile.strategy)
    if not correct:
        return max(0.0, organic_p + assumptions.wrong_action_penalty)

    if profile.strategy in (Strategy.RETRY_NOW, Strategy.RETRY_DELAYED):
        lift = assumptions.correct_retry
    elif profile.strategy is Strategy.RETRY_ON_LIQUIDITY:
        lift = assumptions.correct_retry
    elif profile.strategy is Strategy.SWITCH_RAIL:
        lift = assumptions.correct_rail_switch
    else:  # PERSUADE
        lift = assumptions.correct_nudge

    lift *= _timing_multiplier(profile, hours_waited)

    # Incentives only move intent-driven failures, and only with diminishing
    # returns past a meaningful fraction of the order value.
    if incentive_paise > 0:
        if profile.incentive_eligible and amount_paise > 0:
            depth = incentive_paise / amount_paise
            lift += assumptions.incentive_bonus * min(1.0, depth / 0.10)
        else:
            # Paid a discount where the blocker was technical. No lift, and the
            # money is gone - which the metrics will show as negative net.
            pass

    return max(0.0, min(0.95, organic_p + lift))


def _action_matches_strategy(action_type: str, strategy: Strategy) -> bool:
    allowed = {
        Strategy.RETRY_NOW: {"retry_payment", "payment_link"},
        Strategy.RETRY_DELAYED: {"retry_payment", "payment_link"},
        Strategy.RETRY_ON_LIQUIDITY: {"retry_payment", "payment_link"},
        Strategy.SWITCH_RAIL: {"payment_link"},
        Strategy.PERSUADE: {"nudge", "nudge_with_incentive", "payment_link"},
    }
    return action_type in allowed.get(strategy, set())


def _timing_multiplier(profile, hours_waited: float) -> float:
    """Acting too early is the most common way to waste a recovery attempt.

    Retrying an insufficient-funds failure ten minutes later does not fail
    because the strategy was wrong - it fails because the balance has not
    changed. The taxonomy already encodes the right wait; this penalises
    ignoring it.
    """
    required = profile.retry_after_minutes / 60.0
    if required <= 0:
        # Customer-present cases decay fast - the window is the session.
        return 1.0 if hours_waited <= 1 else max(0.35, 1.0 - 0.1 * hours_waited)

    if hours_waited < required * 0.5:
        return 0.25  # far too early
    if hours_waited < required:
        return 0.6
    if hours_waited <= required * 6:
        return 1.0  # the sweet spot
    return max(0.4, 1.0 - 0.05 * (hours_waited - required * 6) / required)
