"""Measurement. The flattering number and the honest one, computed side by side.

Recovery tooling almost always reports *gross* recovery: of the payments that
failed, this fraction eventually succeeded. It is the easiest number to compute
and the least informative one available, because a large share of failed
payments recover with no help at all - the outage ends, the customer tries
again, and whoever sent an email in between claims the credit.

So every figure here comes in pairs:

    gross_recovery_rate         treatment arm recoveries / treatment arm events
    control_recovery_rate       the same thing where Recoup did nothing
    lift                        the difference, which is the only one that is a
                                result, reported with an interval

and every rupee figure separates money that came back from money that came back
*because of an action*.

Two attributions of incremental value, deliberately kept apart
-------------------------------------------------------------
1. `lift` (+ CI) is the arm difference. It is what a real deployment can
   compute, it is noisy, and its interval is usually wide at this sample size.
2. `incremental_recovered_paise` sums only Attribution.AGENT recoveries - the
   exact per-event counterfactual the frozen roll makes available. It has no
   sampling error at all, and it is available *only* because this is a
   simulation. Reading it as though it were measurable in production is the
   mistake this file exists to prevent, so the report labels it as such.

Intention-to-treat is the headline
----------------------------------
Arm rates are computed over every event in the arm, including the ones policy
denied and the ones the taxonomy refused to touch. That is what a merchant
experiences when they switch Recoup on: the refusals are part of the product,
not an inconvenience to be excluded. The per-protocol rate over acted-on events
only is kept as `treated_recovery_rate` for diagnosis and is never headlined -
conditioning on having acted selects for events that were easier to begin with.

All money stays integer paise. Rates are floats because they are not money.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from recoup.db import Cohort, Decision, DecisionSource, PolicyReview
from recoup.eval.resolve import Resolution, ResolvedOutcome
from recoup.taxonomy import Strategy, profile_for

Z_95 = 1.959963984540054
"""Two-sided normal quantile for a 95% interval."""

MIN_ARM_N = 30
"""Below this many events *per arm*, a segment is marked UNRELIABLE.

Not a significance threshold - a legibility one. At n=30 in each arm with rates
near 0.4, the 95% interval on the difference is about +/- 25 percentage points,
which is wider than any effect this system could plausibly produce. Printing a
point estimate there without a flag invites a reader to believe a number that
the data cannot support, and reason-code segments are small by construction:
payment_risk_check_failed is 1.5% of the mix.
"""


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmStats:
    n: int
    recovered: int
    rate: float | None
    """None, not 0.0, when the arm is empty. Zero recoveries out of zero events
    is not a 0% recovery rate, and a chart that plots it as one is lying."""
    recovered_paise: int
    value_at_risk_paise: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "recovered": self.recovered,
            "rate": self.rate,
            "recovered_paise": self.recovered_paise,
            "value_at_risk_paise": self.value_at_risk_paise,
        }


@dataclass(frozen=True)
class Lift:
    absolute: float | None
    """treatment_rate - control_rate, in probability units. Multiply by 100 for
    percentage points; it is a difference of rates, never a percentage change."""
    ci_low: float | None
    ci_high: float | None
    standard_error: float | None
    reliable: bool
    caveat: str

    @property
    def crosses_zero(self) -> bool:
        if self.ci_low is None or self.ci_high is None:
            return True
        return self.ci_low <= 0.0 <= self.ci_high

    def as_dict(self) -> dict[str, Any]:
        return {
            "absolute": self.absolute,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "standard_error": self.standard_error,
            "reliable": self.reliable,
            "crosses_zero": self.crosses_zero,
            "caveat": self.caveat,
        }


@dataclass(frozen=True)
class Segment:
    dimension: str
    key: str

    n: int
    value_at_risk_paise: int

    treatment: ArmStats
    control: ArmStats
    lift: Lift

    gross_recovery_rate: float | None
    """The flattering number: everything that came back in the treatment arm,
    cause unexamined. Carried at the same level as the honest one so a reader
    cannot see one without the other."""
    gross_recovered_paise: int
    control_recovery_rate: float | None

    incremental_recovered_paise: int
    incremental_n: int

    treated_n: int
    treated_recovery_rate: float | None

    harmed_n: int
    harmed_paise: int

    incentive_committed_paise: int
    incentive_realised_paise: int
    channel_cost_paise: int
    cannibalised_paise: int
    cannibalised_n: int

    cost_paise: int
    net_paise: int
    cost_per_incremental_rupee: float | None

    @property
    def label(self) -> str:
        return self.key if self.dimension != "overall" else "overall"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "key": self.key,
            "n": self.n,
            "value_at_risk_paise": self.value_at_risk_paise,
            "treatment": self.treatment.as_dict(),
            "control": self.control.as_dict(),
            "lift": self.lift.as_dict(),
            "gross_recovery_rate": self.gross_recovery_rate,
            "gross_recovered_paise": self.gross_recovered_paise,
            "control_recovery_rate": self.control_recovery_rate,
            "incremental_recovered_paise": self.incremental_recovered_paise,
            "incremental_n": self.incremental_n,
            "treated_n": self.treated_n,
            "treated_recovery_rate": self.treated_recovery_rate,
            "harmed_n": self.harmed_n,
            "harmed_paise": self.harmed_paise,
            "incentive_committed_paise": self.incentive_committed_paise,
            "incentive_realised_paise": self.incentive_realised_paise,
            "channel_cost_paise": self.channel_cost_paise,
            "cannibalised_paise": self.cannibalised_paise,
            "cannibalised_n": self.cannibalised_n,
            "cost_paise": self.cost_paise,
            "net_paise": self.net_paise,
            "cost_per_incremental_rupee": self.cost_per_incremental_rupee,
        }


@dataclass(frozen=True)
class Suppression:
    """At-risk value Recoup deliberately declined to chase.

    Reported, never netted away. A recovery system that quietly drops the events
    it cannot handle looks better than one that counts them, and is worse.
    """

    events: int
    value_paise: int
    by_reason: dict[str, dict[str, int]]
    untouched_events: int
    untouched_value_paise: int
    """Treatment-arm value no action reached, for any reason - taxonomy refusal,
    policy denial, or a pipeline that never got to it. The gap between this and
    `value_paise` is the part that was not a principled refusal."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "value_paise": self.value_paise,
            "by_reason": self.by_reason,
            "untouched_events": self.untouched_events,
            "untouched_value_paise": self.untouched_value_paise,
        }


@dataclass(frozen=True)
class LLMUsage:
    decisions: int
    llm_decisions: int
    share: float | None
    input_tokens: int
    output_tokens: int
    by_model: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "llm_decisions": self.llm_decisions,
            "share": self.share,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "by_model": self.by_model,
        }


@dataclass(frozen=True)
class EvalMetrics:
    seed: int
    assumptions: dict[str, float]
    n_events: int
    overall: Segment
    by_event_kind: list[Segment]
    by_reason_code: list[Segment]
    by_strategy: list[Segment]
    suppression: Suppression
    policy_denials: dict[str, int]
    llm: LLMUsage
    integrity: dict[str, int]
    unreliable_segments: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "assumptions": self.assumptions,
            "n_events": self.n_events,
            "overall": self.overall.as_dict(),
            "by_event_kind": [s.as_dict() for s in self.by_event_kind],
            "by_reason_code": [s.as_dict() for s in self.by_reason_code],
            "by_strategy": [s.as_dict() for s in self.by_strategy],
            "suppression": self.suppression.as_dict(),
            "policy_denials": self.policy_denials,
            "llm": self.llm.as_dict(),
            "integrity": self.integrity,
            "unreliable_segments": self.unreliable_segments,
        }


# ---------------------------------------------------------------------------
# The interval
# ---------------------------------------------------------------------------


def difference_ci(
    recovered_t: int, n_t: int, recovered_c: int, n_c: int, z: float = Z_95
) -> Lift:
    """95% interval on the difference of two independent proportions.

    Normal (Wald) approximation:

        se = sqrt( p_t(1-p_t)/n_t + p_c(1-p_c)/n_c )

    This is the standard estimator and it is weakest in exactly the conditions
    this dataset produces most: small arms, and rates near 0 or 1, where it
    understates the interval and can put a bound outside [-1, 1]. Nothing here
    corrects for that - no continuity correction, no Newcombe interval. Instead
    segments thinner than MIN_ARM_N per arm are marked unreliable and the report
    declines to draw a conclusion from them, which is the honest response to an
    approximation that has stopped applying rather than a tighter-looking number.

    The randomisation is over events, so the two arms are independent samples
    and this is the right family of estimator. Note what the frozen roll does
    and does not buy: it removes noise from the *per-event counterfactual*
    (Attribution.AGENT), not from this comparison, which is between two
    different sets of events and stays as noisy as its sample size.
    """
    if n_t == 0 or n_c == 0:
        return Lift(
            absolute=None,
            ci_low=None,
            ci_high=None,
            standard_error=None,
            reliable=False,
            caveat="one arm is empty - no comparison exists",
        )

    p_t = recovered_t / n_t
    p_c = recovered_c / n_c
    diff = p_t - p_c

    se = float(
        np.sqrt(p_t * (1.0 - p_t) / n_t + p_c * (1.0 - p_c) / n_c)
    )
    half = z * se

    reliable = n_t >= MIN_ARM_N and n_c >= MIN_ARM_N
    caveat = (
        ""
        if reliable
        else f"UNRELIABLE: {min(n_t, n_c)} events in the smaller arm, "
        f"below the {MIN_ARM_N} the normal approximation needs"
    )

    return Lift(
        absolute=diff,
        ci_low=diff - half,
        ci_high=diff + half,
        standard_error=se,
        reliable=reliable,
        caveat=caveat,
    )


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Columns:
    """One numpy column per quantity, so every segment is a boolean mask."""

    n: int
    amount: np.ndarray
    recovered: np.ndarray
    treatment: np.ndarray
    treated: np.ndarray
    agent: np.ndarray
    harmed: np.ndarray
    incentive_committed: np.ndarray
    incentive_realised: np.ndarray
    cannibalised: np.ndarray
    channel_cost: np.ndarray
    reason_code: list[str]
    event_kind: list[str]
    strategy: list[str]


def _columns(rows: Sequence[ResolvedOutcome]) -> _Columns:
    n = len(rows)
    ints = lambda vals: np.fromiter(vals, dtype=np.int64, count=n)  # noqa: E731
    bools = lambda vals: np.fromiter(vals, dtype=bool, count=n)  # noqa: E731

    return _Columns(
        n=n,
        amount=ints(o.amount_paise for o in rows),
        recovered=bools(o.recovered for o in rows),
        treatment=bools(o.cohort is Cohort.TREATMENT for o in rows),
        treated=bools(o.treated for o in rows),
        agent=bools(o.agent_caused for o in rows),
        harmed=bools(o.harmed for o in rows),
        incentive_committed=ints(o.incentive_committed_paise for o in rows),
        incentive_realised=ints(o.incentive_realised_paise for o in rows),
        cannibalised=ints(o.cannibalised_paise for o in rows),
        channel_cost=ints(o.channel_cost_paise for o in rows),
        reason_code=[o.reason_code for o in rows],
        event_kind=[o.event_kind for o in rows],
        strategy=[o.strategy for o in rows],
    )


def _arm(cols: _Columns, mask: np.ndarray) -> ArmStats:
    n = int(mask.sum())
    rec = int((mask & cols.recovered).sum())
    return ArmStats(
        n=n,
        recovered=rec,
        rate=(rec / n) if n else None,
        recovered_paise=int(cols.amount[mask & cols.recovered].sum()),
        value_at_risk_paise=int(cols.amount[mask].sum()),
    )


def _segment(dimension: str, key: str, cols: _Columns, mask: np.ndarray) -> Segment:
    treat_mask = mask & cols.treatment
    control_mask = mask & ~cols.treatment

    treatment = _arm(cols, treat_mask)
    control = _arm(cols, control_mask)
    lift = difference_ci(treatment.recovered, treatment.n, control.recovered, control.n)

    agent_mask = treat_mask & cols.agent
    incremental_paise = int(cols.amount[agent_mask].sum())

    harmed_mask = treat_mask & cols.harmed
    cannibal_mask = treat_mask & (cols.cannibalised > 0)

    incentive_realised = int(cols.incentive_realised[treat_mask].sum())
    channel_cost = int(cols.channel_cost[treat_mask].sum())

    # Cost is charged exactly once. Channel spend is gone the moment the message
    # goes out; a discount is only a cost when someone redeems it, which is why
    # realised and committed are tracked apart. Discounts redeemed by customers
    # who would have paid anyway are inside this figure - they are the
    # cannibalisation line below, not a separate charge.
    cost = incentive_realised + channel_cost

    # Per-protocol: acted-on events only. Kept for diagnosis, never headlined -
    # conditioning on having acted selects for events that were easier already.
    acted_mask = treat_mask & cols.treated
    acted_n = int(acted_mask.sum())
    acted_recovered = int((acted_mask & cols.recovered).sum())

    return Segment(
        dimension=dimension,
        key=key,
        n=int(mask.sum()),
        value_at_risk_paise=int(cols.amount[mask].sum()),
        treatment=treatment,
        control=control,
        lift=lift,
        gross_recovery_rate=treatment.rate,
        gross_recovered_paise=treatment.recovered_paise,
        control_recovery_rate=control.rate,
        incremental_recovered_paise=incremental_paise,
        incremental_n=int(agent_mask.sum()),
        treated_n=acted_n,
        treated_recovery_rate=(acted_recovered / acted_n) if acted_n else None,
        harmed_n=int(harmed_mask.sum()),
        harmed_paise=int(cols.amount[harmed_mask].sum()),
        incentive_committed_paise=int(cols.incentive_committed[treat_mask].sum()),
        incentive_realised_paise=incentive_realised,
        channel_cost_paise=channel_cost,
        cannibalised_paise=int(cols.cannibalised[cannibal_mask].sum()),
        cannibalised_n=int(cannibal_mask.sum()),
        cost_paise=cost,
        net_paise=incremental_paise - cost,
        cost_per_incremental_rupee=(
            cost / incremental_paise if incremental_paise > 0 else None
        ),
    )


def _by(dimension: str, values: list[str], cols: _Columns) -> list[Segment]:
    keys = sorted(set(values))
    arr = np.array(values, dtype=object)
    segments = [_segment(dimension, k, cols, arr == k) for k in keys]
    # Biggest exposure first: a reader scanning three rows should be looking at
    # the three that matter, not the three that sort alphabetically.
    return sorted(segments, key=lambda s: s.value_at_risk_paise, reverse=True)


def build_segments(rows: Sequence[ResolvedOutcome]) -> dict[str, Any]:
    """Every segment view, from resolved outcomes alone. No database, no I/O."""
    cols = _columns(rows)
    everything = np.ones(cols.n, dtype=bool)
    return {
        "overall": _segment("overall", "overall", cols, everything),
        "by_event_kind": _by("event_kind", cols.event_kind, cols),
        "by_reason_code": _by("reason_code", cols.reason_code, cols),
        "by_strategy": _by("strategy", cols.strategy, cols),
    }


# ---------------------------------------------------------------------------
# Refusals, denials, model spend
# ---------------------------------------------------------------------------


def suppression(rows: Sequence[ResolvedOutcome]) -> Suppression:
    """At-risk value the taxonomy refused on principle, plus what went untouched.

    Derived from the taxonomy rather than from event status, so it means the
    same thing whether or not the pipeline has run: these are the events Recoup
    was never going to act on - risk declines and codes it does not recognise.
    """
    by_reason: dict[str, dict[str, int]] = {}
    events = 0
    value = 0
    untouched_events = 0
    untouched_value = 0

    for o in rows:
        if o.cohort is not Cohort.TREATMENT:
            continue
        if not o.treated:
            untouched_events += 1
            untouched_value += o.amount_paise

        profile = profile_for(o.reason_code)
        refused = profile.strategy is Strategy.DO_NOT_RETRY or profile.code == "unknown"
        if refused and not o.treated:
            events += 1
            value += o.amount_paise
            bucket = by_reason.setdefault(o.reason_code, {"events": 0, "value_paise": 0})
            bucket["events"] += 1
            bucket["value_paise"] += o.amount_paise

    return Suppression(
        events=events,
        value_paise=value,
        by_reason=dict(sorted(by_reason.items(), key=lambda kv: -kv[1]["value_paise"])),
        untouched_events=untouched_events,
        untouched_value_paise=untouched_value,
    )


def policy_denials(session: Session) -> dict[str, int]:
    """Which bound stopped what, counted across every review ever run.

    `control_arm_suppression` will normally be the largest entry and equal the
    size of the holdout. That is the guardrail working, not a fault - it is the
    only direct evidence in the system that the control arm was actually held.
    """
    tally: Counter[str] = Counter()
    for (violations,) in session.execute(select(PolicyReview.violations)).all():
        tally.update(violations or [])
    return dict(tally.most_common())


def llm_usage(session: Session) -> LLMUsage:
    """How much of the decision volume actually needed a model.

    Recoup's claim is that the taxonomy settles most events and the LLM is spent
    on the ambiguous minority. That claim is only worth making if the share is
    reported, whatever it turns out to be.
    """
    rows = session.execute(
        select(
            Decision.source, Decision.model, Decision.input_tokens, Decision.output_tokens
        )
    ).all()

    total = len(rows)
    llm_rows = [r for r in rows if r[0] is DecisionSource.LLM]
    by_model: Counter[str] = Counter(r[1] for r in llm_rows if r[1])

    return LLMUsage(
        decisions=total,
        llm_decisions=len(llm_rows),
        share=(len(llm_rows) / total) if total else None,
        input_tokens=sum(r[2] or 0 for r in rows),
        output_tokens=sum(r[3] or 0 for r in rows),
        by_model=dict(by_model.most_common()),
    )


def compute(session: Session, resolution: Resolution, seed: int) -> EvalMetrics:
    """Everything the report prints, in one object with no formatting in it."""
    segments = build_segments(resolution.outcomes)
    all_segments = [
        segments["overall"],
        *segments["by_event_kind"],
        *segments["by_reason_code"],
        *segments["by_strategy"],
    ]

    return EvalMetrics(
        seed=seed,
        assumptions=asdict(resolution.assumptions),
        n_events=len(resolution.outcomes),
        overall=segments["overall"],
        by_event_kind=segments["by_event_kind"],
        by_reason_code=segments["by_reason_code"],
        by_strategy=segments["by_strategy"],
        suppression=suppression(resolution.outcomes),
        policy_denials=policy_denials(session),
        llm=llm_usage(session),
        integrity=resolution.integrity,
        unreliable_segments=[
            f"{s.dimension}:{s.key}" for s in all_segments if not s.lift.reliable
        ],
    )
