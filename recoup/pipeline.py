"""The orchestrator: assess -> decide -> review -> execute, one event at a time.

This is the only place the five subsystems meet, and it is deliberately thin.
Each stage is already tested in isolation; what this file has to get right is
the *order*, and one thing the stages cannot decide for themselves - when
Recoup would actually have acted.

On simulated time
-----------------
Events in the dataset happened days ago, so "now" cannot be the wall clock: a
retry evaluated at today's timestamp would sail past every timing rule by sheer
elapsed time, and the liquidity-window logic - the most interesting timing
decision in the project - would never fire. Instead each event is processed at
the moment its action window opens, which is what a live scheduler would do.

That makes the run reproducible: the same seed produces the same timestamps,
so two runs of the report are comparable.

On deferral versus denial
-------------------------
If the window opens at 02:00 IST and the chosen action would message someone,
the pipeline defers to 08:00 rather than proposing a 2am send and letting the
policy engine deny it. Both outcomes protect the customer, but they mean
different things: a denial says the system tried to do something it should not
have, and a deferral says it scheduled around a constraint it understood.

The quiet-hours rule stays as the backstop. A scheduler that respects a bound
and a rule that enforces it are not redundant - the rule is what catches the
scheduler being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from recoup.agent.decide import decide
from recoup.db import (
    ActionRun,
    ActionType,
    Assessment,
    Cohort,
    Customer,
    Decision,
    DecisionSource,
    EventStatus,
    PolicyReview,
    PolicyVerdict,
    RevenueEvent,
)
from recoup.detect.features import IST_OFFSET, is_quiet_hours, to_ist
from recoup.detect.scorer import assess, fit_from_outcomes
from recoup.execute.actions import execute
from recoup.policy.rules import CONTACT_ACTIONS, Bounds, ReviewContext, review


@dataclass
class RunStats:
    """What a pipeline run did, for the terminal summary. Not a metrics source -
    recoup/eval owns anything that gets reported as a result."""

    events: int = 0
    assessed: int = 0
    executed: int = 0
    suppressed_by_policy: int = 0
    suppressed_for_holdout: int = 0
    escalated: int = 0
    deferred_for_quiet_hours: int = 0
    llm_decisions: int = 0
    rules_decisions: int = 0
    incentive_paise: int = 0
    channel_cost_paise: int = 0
    denials_by_rule: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def next_business_hour(dt: datetime) -> datetime:
    """Push a timestamp forward to 08:00 IST if it lands in quiet hours.

    Returns UTC, like everything stored. Quiet hours run 21:00-08:00 IST, so a
    22:00 timestamp defers to 08:00 the following morning and a 03:00 one to
    08:00 the same day.
    """
    if not is_quiet_hours(dt):
        return dt
    ist = to_ist(dt)
    target_date = ist.date() if ist.hour < 8 else (ist + timedelta(days=1)).date()
    return datetime.combine(target_date, time(hour=8)) - IST_OFFSET


def _attempts_so_far(session: Session, event_id: str) -> int:
    """Executed actions already made against this event.

    Counted from ActionRun rather than a column on the event, so the number
    cannot drift from what actually happened.
    """
    return (
        session.scalar(
            select(func.count())
            .select_from(ActionRun)
            .join(Decision, Decision.id == ActionRun.decision_id)
            .where(Decision.event_id == event_id)
        )
        or 0
    )


def process_event(
    session: Session,
    event: RevenueEvent,
    customer: Customer,
    stats: RunStats,
    fitted_rates: dict[str, tuple[float, int]] | None = None,
    bounds: Bounds | None = None,
) -> None:
    """Run one event through the whole pipeline."""
    bounds = bounds or Bounds()

    # --- 1. assess ---------------------------------------------------------
    assessment: Assessment = assess(
        session, event, now=event.occurred_at, fitted_rates=fitted_rates
    )
    session.flush()
    event.status = EventStatus.ASSESSED
    stats.assessed += 1

    # --- 2. decide, at the moment the window opens -------------------------
    decision_time = max(event.occurred_at, assessment.earliest_action_at)
    decision: Decision = decide(session, event, assessment, customer, now=decision_time)
    session.flush()
    event.status = EventStatus.DECIDED

    if decision.source is DecisionSource.LLM:
        stats.llm_decisions += 1
    else:
        stats.rules_decisions += 1

    # --- 3. schedule around quiet hours ------------------------------------
    action_time = decision_time
    if decision.action_type in CONTACT_ACTIONS:
        action_time = next_business_hour(decision_time)
        if action_time != decision_time:
            stats.deferred_for_quiet_hours += 1
            # Recorded on the decision so the dashboard can show the deferral.
            # A run where nothing was ever deferred would mean the scheduler is
            # not doing its job, and that should be visible rather than assumed.
            decision.params = {
                **(decision.params or {}),
                "deferred_from": decision_time.isoformat(),
                "deferred_to": action_time.isoformat(),
                "deferred_reason": "quiet hours 21:00-08:00 IST",
            }

    # --- 4. review ---------------------------------------------------------
    ctx = ReviewContext(
        event_id=event.id,
        customer_id=event.customer_id,
        cohort=event.cohort,
        reason_code=event.reason_code,
        amount_paise=event.amount_paise,
        expected_value_paise=assessment.expected_value_paise,
        recoverability=assessment.recoverability,
        earliest_action_at=assessment.earliest_action_at,
        attempts_so_far=_attempts_so_far(session, event.id),
        action_type=decision.action_type,
        incentive_paise=int((decision.params or {}).get("incentive_paise", 0)),
        now=action_time,
        bounds=bounds,
    )
    result = review(ctx, session)

    session.add(
        PolicyReview(
            decision_id=decision.id,
            verdict=result.verdict,
            checks=[c.as_dict() for c in result.checks],
            violations=result.violations,
        )
    )
    session.flush()

    # --- 5. act on the verdict ---------------------------------------------
    if result.verdict is PolicyVerdict.ALLOW:
        try:
            run = execute(session, decision, result, event, customer, now=action_time)
        except Exception as exc:  # noqa: BLE001 - one bad event must not end the run
            stats.errors.append(f"{event.id}: {type(exc).__name__}: {exc}")
            return
        stats.executed += 1
        stats.incentive_paise += run.incentive_paise
        stats.channel_cost_paise += run.channel_cost_paise
        return

    for name in result.violations:
        stats.denials_by_rule[name] = stats.denials_by_rule.get(name, 0) + 1

    if result.verdict is PolicyVerdict.ESCALATE:
        event.status = EventStatus.AWAITING_APPROVAL
        stats.escalated += 1
        return

    event.status = EventStatus.SUPPRESSED
    # A held-out event and a policy-denied one are both suppressed, but only one
    # of them says anything about the bounds. Kept apart so the run summary does
    # not read as though the policy engine denied a third of all traffic.
    if event.cohort is Cohort.CONTROL and "control_arm_suppression" in result.violations:
        stats.suppressed_for_holdout += 1
    else:
        stats.suppressed_by_policy += 1


def run(
    session: Session,
    *,
    limit: int | None = None,
    recalibrate: bool = False,
    bounds: Bounds | None = None,
) -> RunStats:
    """Process every open event.

    `recalibrate=True` fits per-reason recovery rates from outcomes already in
    the database before scoring. Off by default because on a fresh run there is
    nothing to fit and the priors are all there is.
    """
    stats = RunStats()

    fitted = fit_from_outcomes(session) if recalibrate else None

    stmt = (
        select(RevenueEvent)
        .where(RevenueEvent.status == EventStatus.OPEN)
        .order_by(RevenueEvent.occurred_at)
    )
    if limit:
        stmt = stmt.limit(limit)

    events = list(session.scalars(stmt))
    stats.events = len(events)

    for event in events:
        customer = session.get(Customer, event.customer_id)
        if customer is None:
            stats.errors.append(f"{event.id}: unknown customer {event.customer_id}")
            continue
        process_event(session, event, customer, stats, fitted, bounds)
        # Committed per event on purpose. A crash halfway through a 600-event run
        # leaves a coherent partial audit trail rather than losing all of it, and
        # the remaining events are still OPEN so a rerun picks up exactly where
        # this one stopped.
        session.commit()

    return stats
