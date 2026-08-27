"""Outcome resolution: what happened to each event, and who is allowed to claim it.

This module is the grader, not the player. It is one of the few files permitted
to import recoup.seed.world and read data/oracle.json - the pipeline is
structurally blind to both (tests/test_no_oracle_leak.py) and this file lives on
the other side of that wall.

The frozen roll
---------------
Every event was assigned one uniform draw at generation time and that draw never
changes. Resolution *compares* against a probability, it does not sample:

    control    recovered <=> roll < organic_p
    treatment  recovered <=> roll < treated_p(organic_p, action, timing, incentive)

Because the roll is shared, the same event faces identical luck in both arms, so
the gap between the arms is the gap between the decisions rather than the gap
between two sets of coin flips. Re-drawing here - once, for a quick check, on a
branch - would swap a measured effect for sampling noise, and nothing in the
output would look wrong.

It also buys the one thing a real A/B test can never have: the per-event
counterfactual. A treated event that recovers with `roll >= organic_p` recovered
*only* because Recoup acted - the same event, the same luck, no action, is lost.
That is what Attribution.AGENT means here, and it is why the incremental rupee
figure is exact rather than estimated. Inside the simulation, and only inside
it: metrics.py reports the arm-difference estimate beside it precisely because
that second number is the one a real deployment could actually obtain.

What resolution refuses to credit
---------------------------------
- Actions that never reached the customer. Dry-run and failed sends earn
  nothing, so a demo run under RECOUP_DRY_RUN produces zero lift. That is the
  correct answer, not a bug - the alternative is a flag that inflates results.
- Actions on the control arm. Policy forbids them; if one appears anyway the
  event resolves as untouched and the contamination is reported at the top of
  the eval, because a holdout that was quietly acted on invalidates every
  number computed from it.
- Actions whose timing cannot be reconstructed. No credit, no blame, and the
  recovery is marked UNCLEAR so it can never reach the headline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from recoup.config import PROJECT_ROOT
from recoup.db import (
    ActionRun,
    ActionStatus,
    ActionType,
    Attribution,
    Cohort,
    Decision,
    EventStatus,
    Outcome,
    RevenueEvent,
)
from recoup.seed.generate import RECOVERY_WINDOW_DAYS
from recoup.seed.world import LiftAssumptions, treated_recovery_probability
from recoup.taxonomy import profile_for

ORACLE_PATH = PROJECT_ROOT / "data" / "oracle.json"

ACTING_TYPES = {
    ActionType.RETRY_PAYMENT,
    ActionType.PAYMENT_LINK,
    ActionType.NUDGE,
    ActionType.NUDGE_WITH_INCENTIVE,
}
"""Actions capable of changing an outcome.

NO_ACTION and ESCALATE_TO_HUMAN are recorded decisions, not interventions - the
world model returns the organic probability for both. Counting them as treated
would dilute the per-protocol rate with events nobody touched.
"""

RECOVERY_WINDOW_HOURS = RECOVERY_WINDOW_DAYS * 24


class OracleMismatch(RuntimeError):
    """The answer key does not describe the events being graded."""


def _now_naive() -> datetime:
    """Naive UTC, matching how the generator writes every timestamp.

    A tz-aware value in these columns is uncomparable with the naive occurred_at
    that comes back out of SQLite, and the failure surfaces far from here as a
    TypeError in the middle of a report.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def load_oracle(path: Path | None = None) -> dict[str, dict]:
    path = path or ORACLE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No ground truth at {path}. Run scripts/seed.py first - the eval "
            "grades against the rolls frozen at generation time and cannot "
            "invent them after the fact."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The unit of resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutedAction:
    """One intervention that actually reached the customer."""

    action_type: str
    executed_at: datetime
    incentive_paise: int = 0
    channel_cost_paise: int = 0


@dataclass(frozen=True)
class ResolvedOutcome:
    event_id: str
    cohort: Cohort
    event_kind: str
    reason_code: str
    strategy: str
    amount_paise: int

    treated: bool
    """An acting intervention reached this customer. False for the entire
    control arm, and for treatment events policy denied, suppressed, or never
    got to."""

    action_type: str | None
    hours_waited: float | None

    incentive_committed_paise: int
    channel_cost_paise: int

    organic_p: float
    treated_p: float
    roll: float

    recovered: bool
    recovered_paise: int
    """Full order value. The discount is subtracted exactly once, as cost, in
    metrics.py - netting it out here as well would hide a redeemed discount that
    went to a customer who was going to pay anyway."""

    attribution: Attribution
    hours_to_recovery: float | None
    note: str

    @property
    def incentive_realised_paise(self) -> int:
        """A discount costs the merchant only when the customer redeems it.

        Committed spend is the budget question, and the policy engine already
        enforces it at send time. Realised spend is the P&L question. They are
        different numbers and the report carries both.
        """
        return self.incentive_committed_paise if self.recovered else 0

    @property
    def agent_caused(self) -> bool:
        return self.recovered and self.attribution is Attribution.AGENT

    @property
    def cannibalised_paise(self) -> int:
        """Discount handed to a customer who would have paid without it."""
        if self.recovered and self.attribution is Attribution.ORGANIC:
            return self.incentive_committed_paise
        return 0

    @property
    def harmed(self) -> bool:
        """Acting destroyed a recovery that would have happened on its own.

        `roll < organic_p` says the customer was coming back; not recovering
        after an intervention means the intervention pushed the probability
        under their draw. Rare and small by construction, and reported anyway -
        a harness that can only find upside is not measuring anything.
        """
        return self.treated and not self.recovered and self.roll < self.organic_p


@dataclass
class Resolution:
    """Every outcome under one set of assumptions, plus everything that was wrong."""

    outcomes: list[ResolvedOutcome]
    assumptions: LiftAssumptions
    missing_from_oracle: list[str]
    contaminated_control: list[str]
    counters: dict[str, int]

    @property
    def integrity(self) -> dict[str, int]:
        return {
            "missing_from_oracle": len(self.missing_from_oracle),
            "contaminated_control": len(self.contaminated_control),
            **self.counters,
        }


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def resolve_one(
    *,
    event_id: str,
    cohort: Cohort,
    event_kind: str,
    reason_code: str,
    amount_paise: int,
    occurred_at: datetime,
    organic_p: float,
    roll: float,
    action: ExecutedAction | None,
    assumptions: LiftAssumptions,
) -> ResolvedOutcome:
    """Resolve a single event. Pure - no session, no clock, no randomness.

    Every claim the eval makes rests on this function, so it is kept small
    enough to check by reading it.
    """
    contaminated = cohort is Cohort.CONTROL and action is not None
    effective = None if cohort is Cohort.CONTROL else action

    unclear = False
    hours_waited: float | None = None
    treated_p = organic_p

    if effective is not None:
        hours_waited = (effective.executed_at - occurred_at).total_seconds() / 3600.0
        if hours_waited < 0:
            # An action stamped before the failure it responds to. The timing
            # multiplier is undefined for it, so it earns neither credit nor
            # blame and the event is graded as though untouched.
            hours_waited = None
            unclear = True
        else:
            treated_p = treated_recovery_probability(
                organic_p=organic_p,
                reason_code=reason_code,
                action_type=effective.action_type,
                hours_waited=hours_waited,
                incentive_paise=effective.incentive_paise,
                amount_paise=amount_paise,
                assumptions=assumptions,
            )

    treated = effective is not None and not unclear
    recovered = roll < treated_p

    if not recovered:
        # Nothing to attribute. ORGANIC is the schema default and no metric
        # reads attribution without filtering on `recovered` first.
        attribution = Attribution.ORGANIC
        note = (
            "recoverable without intervention, lost after it"
            if treated and roll < organic_p
            else "not recovered within the window"
        )
    elif unclear:
        attribution = Attribution.UNCLEAR
        note = "action timing unreconstructable - recovery cannot be claimed"
    elif roll < organic_p:
        attribution = Attribution.ORGANIC
        note = "would have recovered without intervention"
    else:
        attribution = Attribution.AGENT
        note = f"recovered only under the action taken ({effective.action_type})"

    if contaminated:
        note = f"control arm contaminated - action executed on a holdout event; {note}"

    return ResolvedOutcome(
        event_id=event_id,
        cohort=cohort,
        event_kind=event_kind,
        reason_code=reason_code,
        strategy=profile_for(reason_code).strategy.value,
        amount_paise=amount_paise,
        treated=treated,
        action_type=effective.action_type if effective is not None else None,
        hours_waited=hours_waited,
        incentive_committed_paise=effective.incentive_paise if effective else 0,
        channel_cost_paise=effective.channel_cost_paise if effective else 0,
        organic_p=organic_p,
        treated_p=treated_p,
        roll=roll,
        recovered=recovered,
        recovered_paise=amount_paise if recovered else 0,
        attribution=attribution,
        hours_to_recovery=(
            _hours_to_recovery(roll, hours_waited, attribution) if recovered else None
        ),
        note=note,
    )


def _hours_to_recovery(
    roll: float, hours_waited: float | None, attribution: Attribution
) -> float:
    """Illustrative timing, derived from the frozen roll so replays match.

    Feeds the dashboard timeline and nothing else. No headline metric reads it,
    because the simulator has no opinion worth reporting about *when* money
    comes back - only about whether it does.
    """
    if attribution is Attribution.AGENT and hours_waited is not None:
        return round(hours_waited + 0.5 + 12.0 * roll, 1)
    return round(RECOVERY_WINDOW_HOURS * (0.05 + 0.90 * roll), 1)


# ---------------------------------------------------------------------------
# Whole-dataset resolution
# ---------------------------------------------------------------------------


def executed_actions(session: Session) -> tuple[dict[str, ExecutedAction], dict[str, int]]:
    """The one intervention per event that could have moved the outcome.

    Recoup executes at most one action per event by design. A second delivered
    run is a bug rather than a strategy, so extras are counted and excluded
    instead of being averaged into something plausible-looking.
    """
    rows = session.execute(
        select(Decision.event_id, ActionRun)
        .join(ActionRun, ActionRun.decision_id == Decision.id)
        .order_by(ActionRun.executed_at)
    ).all()

    counters = {
        "dry_run_actions": 0,
        "failed_actions": 0,
        "pending_actions": 0,
        "inert_actions": 0,
        "extra_actions": 0,
    }
    chosen: dict[str, ExecutedAction] = {}

    for event_id, run in rows:
        if run.status is ActionStatus.FAILED:
            counters["failed_actions"] += 1
            continue

        if run.status is ActionStatus.SKIPPED_DRY_RUN:
            # Dispatched in simulation rather than over the wire - and graded
            # anyway, deliberately.
            #
            # The tempting rule is "a dry run cannot manufacture lift", but it
            # does not survive contact with the rest of the system. The outbox
            # never messages a real customer in ANY mode, so nudges would always
            # earn credit while Payment Links would earn it only when
            # RECOUP_DRY_RUN=false. Two actions, both simulated, graded
            # differently - and the resulting report would not be measuring
            # recovery strategy at all, only which executor happens to write an
            # outbox row.
            #
            # What actually decides an outcome here is recoup/seed/world.py, and
            # it has no opinion about whether an HTTP call left the machine. So
            # the honest line is not to exclude these, it is to grade them and
            # say plainly which mode produced the number - which the report does,
            # in its header, every time.
            counters["dry_run_actions"] += 1
        elif run.status is not ActionStatus.SENT:
            counters["pending_actions"] += 1
            continue
        if run.action_type not in ACTING_TYPES:
            counters["inert_actions"] += 1
            continue
        if event_id in chosen:
            counters["extra_actions"] += 1
            continue
        chosen[event_id] = ExecutedAction(
            action_type=run.action_type.value,
            executed_at=run.executed_at,
            incentive_paise=run.incentive_paise or 0,
            channel_cost_paise=run.channel_cost_paise or 0,
        )

    return chosen, counters


def resolve_all(
    session: Session,
    *,
    assumptions: LiftAssumptions | None = None,
    oracle: dict[str, dict] | None = None,
    persist: bool = True,
) -> Resolution:
    """Resolve every event in the database against the frozen ground truth.

    `persist=False` for the sensitivity sweep: only the default assumptions are
    written to the Outcome table, or the audit trail ends up holding whichever
    hypothetical happened to run last.
    """
    assumptions = assumptions or LiftAssumptions()
    oracle = load_oracle() if oracle is None else oracle

    events = list(session.scalars(select(RevenueEvent).order_by(RevenueEvent.occurred_at)))
    actions, counters = executed_actions(session)

    outcomes: list[ResolvedOutcome] = []
    missing: list[str] = []
    contaminated: list[str] = []
    stale: list[str] = []

    for event in events:
        truth = oracle.get(event.id)
        if truth is None:
            missing.append(event.id)
            continue
        # A stale answer key grades a different dataset, and every number
        # downstream of it is fiction. It stops the run rather than warning.
        if (
            truth.get("reason_code") != event.reason_code
            or int(truth.get("amount_paise", -1)) != event.amount_paise
        ):
            stale.append(event.id)
            continue

        action = actions.get(event.id)
        if event.cohort is Cohort.CONTROL and action is not None:
            contaminated.append(event.id)

        outcomes.append(
            resolve_one(
                event_id=event.id,
                cohort=event.cohort,
                event_kind=event.kind.value,
                reason_code=event.reason_code,
                amount_paise=event.amount_paise,
                occurred_at=event.occurred_at,
                organic_p=float(truth["organic_p"]),
                roll=float(truth["roll"]),
                action=action,
                assumptions=assumptions,
            )
        )

    if stale:
        raise OracleMismatch(
            f"{len(stale)} event(s) disagree with data/oracle.json on amount or "
            "reason code - the answer key belongs to a different dataset. Re-run "
            "scripts/seed.py so the events and the ground truth are generated "
            "together."
        )
    if events and len(missing) == len(events):
        raise OracleMismatch(
            "No event in the database appears in data/oracle.json. Nothing here "
            "can be graded; re-run scripts/seed.py."
        )

    if persist:
        _persist(session, outcomes)

    return Resolution(
        outcomes=outcomes,
        assumptions=assumptions,
        missing_from_oracle=missing,
        contaminated_control=contaminated,
        counters=counters,
    )


def _persist(session: Session, outcomes: list[ResolvedOutcome]) -> None:
    """Rebuild the Outcome table wholesale.

    Resolution is a pure function of (oracle, action log, assumptions), so it is
    regenerated rather than patched. Patching leaves rows from an older
    assumption set beside the current ones with no column recording which is
    which.

    Note what the persisted rows do *not* carry: the latent probabilities. The
    notes stay qualitative so the Outcome table never becomes a back-channel
    through which the pipeline could read the simulator it is meant to be blind
    to.
    """
    resolved_at = _now_naive()
    session.execute(delete(Outcome))

    for o in outcomes:
        session.add(
            Outcome(
                event_id=o.event_id,
                resolved_at=resolved_at,
                recovered=o.recovered,
                recovered_paise=o.recovered_paise,
                attribution=o.attribution,
                hours_to_recovery=o.hours_to_recovery,
                note=o.note,
            )
        )
        event = session.get(RevenueEvent, o.event_id)
        if event is not None:
            event.status = EventStatus.RECOVERED if o.recovered else EventStatus.LOST

    session.commit()
