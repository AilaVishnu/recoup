"""The dashboard's read model: every query the UI runs, in one place.

Read-only by construction. Nothing under recoup/api writes to the database or
calls Razorpay - the dashboard is a window onto the audit trail, and a window
that could also move the furniture would be a worse demo and a worse product.

Two rules this module holds to.

**No N+1.** The events table renders fifty rows, each needing its customer,
assessment, latest decision, that decision's policy review, its action run and
the outcome. Fetched naively that is three hundred round trips per page load.
Everything here is either an aggregate the database computes, or an eager load
with an explicit strategy.

**Arithmetic in integers.** Incremental recovery is a ratio of two sums applied
to a third; written as ``a * b // c`` it never touches a float, so the headline
number on the overview page is exactly reproducible from the rows beneath it.
"""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import func, inspect as sa_inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload, selectinload

from recoup.config import PROJECT_ROOT
from recoup.db import (
    ActionRun,
    ActionStatus,
    Assessment,
    Attribution,
    Cohort,
    ContactLog,
    Decision,
    DecisionSource,
    EventStatus,
    Outcome,
    PolicyReview,
    PolicyVerdict,
    RevenueEvent,
    SpendLog,
)
from recoup.policy import rules as policy_rules
from recoup.taxonomy import FailureProfile, profile_for

PER_PAGE = 50
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def schema_ready(session: Session) -> bool:
    """True when the tables exist at all.

    A fresh clone has no database until scripts/seed.py runs. Asking the
    inspector is cheaper and far more legible than letting every route catch an
    OperationalError and guess at what it meant.
    """
    try:
        return sa_inspect(session.get_bind()).has_table(RevenueEvent.__tablename__)
    except SQLAlchemyError:
        return False


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmStats:
    cohort: Cohort
    events: int = 0
    value_paise: int = 0
    recovered_events: int = 0
    recovered_paise: int = 0

    @property
    def recovery_rate(self) -> float:
        """Value-weighted, not event-weighted.

        A merchant does not care what share of tickets came back, it cares what
        share of rupees came back, and the two diverge sharply when the amount
        distribution has a tail this long.
        """
        return self.recovered_paise / self.value_paise if self.value_paise else 0.0

    @property
    def event_rate(self) -> float:
        return self.recovered_events / self.events if self.events else 0.0


@dataclass(frozen=True)
class RuleStats:
    name: str
    evaluated: int = 0
    failed: int = 0
    verdict: str = PolicyVerdict.DENY.value
    exposure_paise: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed / self.evaluated if self.evaluated else 0.0


@dataclass(frozen=True)
class Overview:
    events: int
    value_at_risk_paise: int
    treatment: ArmStats
    control: ArmStats
    counterfactual_paise: int
    incremental_paise: int
    cost_paise: int
    net_paise: int
    by_status: list[tuple[EventStatus, int, int]]
    by_source: list[tuple[DecisionSource, int, int, int]]
    by_verdict: list[tuple[PolicyVerdict, int]]
    by_attribution: list[tuple[Attribution, int, int]]
    by_run_status: list[tuple[ActionStatus, int, int]]
    binding_rules: list[RuleStats]

    @property
    def gross_paise(self) -> int:
        """What a vendor with no holdout would put on the slide."""
        return self.treatment.recovered_paise

    @property
    def decisions(self) -> int:
        return sum(n for _, n, _, _ in self.by_source)

    @property
    def llm_decisions(self) -> int:
        return sum(n for src, n, _, _ in self.by_source if src is DecisionSource.LLM)

    @property
    def llm_share(self) -> float:
        return self.llm_decisions / self.decisions if self.decisions else 0.0

    @property
    def tokens(self) -> int:
        return sum(i + o for _, _, i, o in self.by_source)

    @property
    def denials(self) -> list[RuleStats]:
        return [r for r in self.binding_rules if r.failed]

    @property
    def has_outcomes(self) -> bool:
        """Whether any recovery figure on this page is measured rather than nil.

        Seeded-but-not-run is the normal state of a fresh clone, and printing
        Rs 0 of incremental recovery for it would read as a result rather than
        as an absence.
        """
        return bool(self.treatment.recovered_events or self.control.recovered_events)


def overview(session: Session) -> Overview:
    totals = {
        cohort: (n, v)
        for cohort, n, v in session.execute(
            select(
                RevenueEvent.cohort,
                func.count(),
                func.coalesce(func.sum(RevenueEvent.amount_paise), 0),
            ).group_by(RevenueEvent.cohort)
        ).all()
    }
    recovered = {
        cohort: (n, v)
        for cohort, n, v in session.execute(
            select(
                RevenueEvent.cohort,
                func.count(),
                func.coalesce(func.sum(Outcome.recovered_paise), 0),
            )
            .join(Outcome, Outcome.event_id == RevenueEvent.id)
            .where(Outcome.recovered.is_(True))
            .group_by(RevenueEvent.cohort)
        ).all()
    }

    arms = {
        cohort: ArmStats(
            cohort=cohort,
            events=totals.get(cohort, (0, 0))[0],
            value_paise=totals.get(cohort, (0, 0))[1],
            recovered_events=recovered.get(cohort, (0, 0))[0],
            recovered_paise=recovered.get(cohort, (0, 0))[1],
        )
        for cohort in Cohort
    }
    treatment, control = arms[Cohort.TREATMENT], arms[Cohort.CONTROL]

    # The counterfactual: what the treated population would have recovered had
    # nobody touched it, priced at the holdout's own value-weighted rate. This
    # is the number that turns a gross figure into an honest one, and it is
    # integer throughout - a ratio of two sums applied to a third.
    counterfactual = (
        control.recovered_paise * treatment.value_paise // control.value_paise
        if control.value_paise
        else 0
    )

    by_run_status = session.execute(
        select(
            ActionRun.status,
            func.count(),
            func.coalesce(
                func.sum(ActionRun.incentive_paise + ActionRun.channel_cost_paise), 0
            ),
        ).group_by(ActionRun.status)
    ).all()

    # A send that failed cost nothing. A dry run records what it would have
    # spent and is counted, because a demo that hides its own cost is the exact
    # dishonesty this project exists to argue against.
    cost = sum(v for status, _, v in by_run_status if status is not ActionStatus.FAILED)
    incremental = treatment.recovered_paise - counterfactual

    return Overview(
        events=sum(n for n, _ in totals.values()),
        value_at_risk_paise=sum(v for _, v in totals.values()),
        treatment=treatment,
        control=control,
        counterfactual_paise=counterfactual,
        incremental_paise=incremental,
        cost_paise=cost,
        net_paise=incremental - cost,
        by_status=session.execute(
            select(
                RevenueEvent.status,
                func.count(),
                func.coalesce(func.sum(RevenueEvent.amount_paise), 0),
            ).group_by(RevenueEvent.status)
        ).all(),
        by_source=session.execute(
            select(
                Decision.source,
                func.count(),
                func.coalesce(func.sum(Decision.input_tokens), 0),
                func.coalesce(func.sum(Decision.output_tokens), 0),
            ).group_by(Decision.source)
        ).all(),
        by_verdict=session.execute(
            select(PolicyReview.verdict, func.count()).group_by(PolicyReview.verdict)
        ).all(),
        by_attribution=session.execute(
            select(
                Outcome.attribution,
                func.count(),
                func.coalesce(func.sum(Outcome.recovered_paise), 0),
            )
            .join(RevenueEvent, RevenueEvent.id == Outcome.event_id)
            .where(Outcome.recovered.is_(True), RevenueEvent.cohort == Cohort.TREATMENT)
            .group_by(Outcome.attribution)
        ).all(),
        by_run_status=by_run_status,
        binding_rules=rule_stats(session),
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def rule_order(session: Session) -> list[str]:
    """The canonical rule sequence, preferring names the engine actually recorded.

    A rule's function name and the name it writes into the review are allowed to
    differ - `_rule_control_arm_is_never_executed` records itself as
    `control_arm_suppression` - and the recorded name is what the audit trail is
    keyed on. The fallback to function names only matters on an empty database,
    where this page is listing rules that have never run.
    """
    latest = session.scalars(
        select(PolicyReview).order_by(PolicyReview.id.desc()).limit(1)
    ).first()
    if latest and latest.checks:
        return [c.get("name", "?") for c in latest.checks]
    return [fn.__name__.removeprefix("_rule_") for fn in policy_rules.RULES]


def rule_stats(session: Session) -> list[RuleStats]:
    """How often each bound was evaluated, how often it bound, and on what value.

    Exposure is per-rule, not a partition: when three rules object to the same
    action, that event's amount appears under all three. Summing the column
    would double-count. It exists to answer "which bound is doing the work",
    not "how much did policy block in total".
    """
    reviews = session.scalars(
        select(PolicyReview).options(
            joinedload(PolicyReview.decision).joinedload(Decision.event)
        )
    ).all()

    evaluated: dict[str, int] = {}
    failed: dict[str, int] = {}
    verdicts: dict[str, str] = {}
    exposure: dict[str, int] = {}
    examples: dict[str, list[str]] = {}

    for review in reviews:
        event = review.decision.event if review.decision else None
        amount = event.amount_paise if event else 0
        for check in review.checks or []:
            name = check.get("name", "?")
            evaluated[name] = evaluated.get(name, 0) + 1
            if check.get("passed", True):
                continue
            failed[name] = failed.get(name, 0) + 1
            verdicts[name] = check.get("verdict", PolicyVerdict.DENY.value)
            exposure[name] = exposure.get(name, 0) + amount
            detail = check.get("detail", "")
            seen = examples.setdefault(name, [])
            if detail and detail not in seen and len(seen) < 3:
                seen.append(detail)

    order = rule_order(session)
    order += [name for name in evaluated if name not in order]
    return [
        RuleStats(
            name=name,
            evaluated=evaluated.get(name, 0),
            failed=failed.get(name, 0),
            verdict=verdicts.get(name, PolicyVerdict.DENY.value),
            exposure_paise=exposure.get(name, 0),
            examples=examples.get(name, []),
        )
        for name in order
    ]


@dataclass(frozen=True)
class BoundRow:
    name: str
    value: Any
    why: str


def _bound_docs() -> dict[str, str]:
    """Pull each Bounds field's justification out of the source of rules.py.

    Attribute docstrings are discarded at runtime, so they are read back from
    the AST. The point is that this page cannot drift from the policy engine: it
    renders whatever the dataclass currently holds, alongside whatever reason
    the author currently gives for it, with no second copy to fall behind.
    """
    try:
        tree = ast.parse(inspect.getsource(policy_rules))
    except (OSError, TypeError, SyntaxError):
        return {}

    docs: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "Bounds":
            continue
        pending: str | None = None
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                pending = stmt.target.id
            elif (
                pending
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                docs[pending] = " ".join(stmt.value.value.split())
                pending = None
            else:
                pending = None
    return docs


def bounds_table(session: Session | None = None) -> list[BoundRow]:
    """The limits that actually governed the last run, not the ones in the source.

    Reading DEFAULT_BOUNDS here made the page truthful only by coincidence. A
    reviewer put it plainly: with a loosened override in play, /policy reported a
    Rs 25,000 autonomy limit for a run that had none. Overrides are clamped now,
    so the two agree again - but agreeing by construction and agreeing by luck
    are different properties, and only one of them survives the next change.

    Every PolicyReview snapshots the bounds in force, so the page reads them from
    the run itself and falls back to the defaults only when there is no run to
    describe.
    """
    docs = _bound_docs()
    defaults = policy_rules.DEFAULT_BOUNDS

    applied: dict[str, Any] = {}
    if session is not None:
        row = session.scalars(
            select(PolicyReview).order_by(PolicyReview.id.desc()).limit(1)
        ).first()
        if row and row.bounds:
            applied = row.bounds

    return [
        BoundRow(
            name=f.name,
            value=applied.get(f.name, getattr(defaults, f.name)),
            why=docs.get(f.name, ""),
        )
        for f in fields(defaults)
    ]


# ---------------------------------------------------------------------------
# The events table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Filters:
    cohort: Cohort | None = None
    status: EventStatus | None = None
    reason: str | None = None
    verdict: PolicyVerdict | None = None

    @property
    def active(self) -> bool:
        return any((self.cohort, self.status, self.reason, self.verdict))


@dataclass(frozen=True)
class EventRow:
    event: RevenueEvent
    assessment: Assessment | None
    decision: Decision | None
    review: PolicyReview | None
    run: ActionRun | None
    outcome: Outcome | None

    @property
    def profile(self) -> FailureProfile:
        return profile_for(self.event.reason_code)


def _latest_decision(event: RevenueEvent) -> Decision | None:
    """The decision the event is currently living under.

    Decisions are append-only, so the highest id is the operative one. Earlier
    rows are superseded proposals and stay visible on the detail page.
    """
    return max(event.decisions, key=lambda d: d.id, default=None)


def _criteria(f: Filters) -> list:
    where = []
    if f.cohort:
        where.append(RevenueEvent.cohort == f.cohort)
    if f.status:
        where.append(RevenueEvent.status == f.status)
    if f.reason:
        where.append(RevenueEvent.reason_code == f.reason)
    if f.verdict:
        # EXISTS rather than a join: an event carrying two decisions would
        # otherwise appear twice in the table and corrupt the page count.
        where.append(
            select(PolicyReview.id)
            .join(Decision, Decision.id == PolicyReview.decision_id)
            .where(
                Decision.event_id == RevenueEvent.id,
                PolicyReview.verdict == f.verdict,
            )
            .exists()
        )
    return where


def list_events(
    session: Session, f: Filters, page: int = 1, per_page: int = PER_PAGE
) -> tuple[list[EventRow], int, int]:
    """One page of events, ordered by what they are worth acting on.

    Expected value descending rather than recency: the first screen of this
    table should be the money. An unassessed event sorts last, not first.

    Returns the rows, the unpaginated total, and the page actually served - a
    page number past the end lands on the last page rather than on an empty
    table that looks like a broken filter.
    """
    where = _criteria(f)
    total = session.scalar(
        select(func.count()).select_from(RevenueEvent).where(*where)
    ) or 0
    page = min(max(page, 1), max(1, -(-total // per_page)))

    stmt = (
        select(RevenueEvent)
        .outerjoin(Assessment, Assessment.event_id == RevenueEvent.id)
        .where(*where)
        .options(
            joinedload(RevenueEvent.customer),
            joinedload(RevenueEvent.assessment),
            joinedload(RevenueEvent.outcome),
            selectinload(RevenueEvent.decisions).joinedload(Decision.review),
            selectinload(RevenueEvent.decisions).joinedload(Decision.run),
        )
        .order_by(
            func.coalesce(Assessment.expected_value_paise, -1).desc(),
            RevenueEvent.occurred_at.desc(),
        )
        .offset((page - 1) * per_page)
        .limit(per_page)
    )

    rows: list[EventRow] = []
    for event in session.scalars(stmt).unique().all():
        decision = _latest_decision(event)
        rows.append(
            EventRow(
                event=event,
                assessment=event.assessment,
                decision=decision,
                review=decision.review if decision else None,
                run=decision.run if decision else None,
                outcome=event.outcome,
            )
        )
    return rows, total, page


def reason_codes(session: Session) -> list[str]:
    """Reason codes present in the data, for the filter dropdown.

    Read from the events rather than from the taxonomy: a filter offering
    sixteen options where the dataset holds fourteen sends the viewer to two
    guaranteed-empty pages.
    """
    return sorted(
        code
        for code in session.scalars(select(RevenueEvent.reason_code).distinct()).all()
        if code
    )


# ---------------------------------------------------------------------------
# One event, end to end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One decision and everything that happened to it."""

    decision: Decision
    review: PolicyReview | None
    run: ActionRun | None

    @property
    def checks(self) -> list[dict[str, Any]]:
        return list(self.review.checks or []) if self.review else []

    @property
    def cleared(self) -> int:
        return sum(1 for c in self.checks if c.get("passed"))


@dataclass(frozen=True)
class Replay:
    event: RevenueEvent
    profile: FailureProfile
    assessment: Assessment | None
    stages: list[Stage]
    outcome: Outcome | None
    contacts: list[ContactLog]
    spends: list[SpendLog]

    @property
    def is_control(self) -> bool:
        return self.event.cohort is Cohort.CONTROL

    @property
    def basis(self) -> str:
        """'prior' or 'fitted:<n>' - which base rate produced the score.

        The scorer stamps it onto scorer_version as 'v1/<basis>' so a score read
        back months later still states where its number came from.
        """
        version = self.assessment.scorer_version if self.assessment else ""
        return version.partition("/")[2] or "prior"


def replay(session: Session, event_id: str) -> Replay | None:
    event = (
        session.scalars(
            select(RevenueEvent)
            .where(RevenueEvent.id == event_id)
            .options(
                joinedload(RevenueEvent.customer),
                joinedload(RevenueEvent.assessment),
                joinedload(RevenueEvent.outcome),
                selectinload(RevenueEvent.decisions).joinedload(Decision.review),
                selectinload(RevenueEvent.decisions).joinedload(Decision.run),
            )
        )
        .unique()
        .first()
    )
    if event is None:
        return None

    return Replay(
        event=event,
        profile=profile_for(event.reason_code),
        assessment=event.assessment,
        stages=[
            Stage(decision=d, review=d.review, run=d.run)
            for d in sorted(event.decisions, key=lambda d: d.id)
        ],
        outcome=event.outcome,
        contacts=list(
            session.scalars(
                select(ContactLog)
                .where(ContactLog.event_id == event_id)
                .order_by(ContactLog.occurred_at)
            ).all()
        ),
        spends=list(
            session.scalars(
                select(SpendLog)
                .where(SpendLog.event_id == event_id)
                .order_by(SpendLog.occurred_at)
            ).all()
        ),
    )


# ---------------------------------------------------------------------------
# The eval report
# ---------------------------------------------------------------------------

HEADLINE_KEYS: dict[str, tuple[str, ...]] = {
    "incremental": ("incremental_recovered_paise", "incremental_paise", "incremental"),
    "gross": ("gross_recovered_paise", "gross_paise", "gross_recovery_paise"),
    "cost": ("cost_paise", "total_cost_paise", "spend_paise"),
    "net": ("net_paise", "net_recovered_paise", "net_recovery_paise"),
    "cannibalised": (
        "cannibalised_paise",
        "cannibalisation_paise",
        "counterfactual_paise",
        "organic_paise",
    ),
}


def _flatten(obj: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _flatten(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj[:8]):
            yield from _flatten(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


@dataclass(frozen=True)
class SweepRow:
    """One point of the sensitivity sweep, normalised."""

    label: str
    incremental_paise: int | None = None
    cannibalised_paise: int | None = None
    cost_paise: int | None = None
    net_paise: int | None = None
    lift: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    @property
    def crosses_zero(self) -> bool:
        if self.ci_low is None or self.ci_high is None:
            return False
        return self.ci_low <= 0 <= self.ci_high


def _num(source: Any, *names: str) -> Any:
    if not isinstance(source, dict):
        return None
    for name in names:
        value = source.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


@dataclass(frozen=True)
class Report:
    """The last eval run, read defensively.

    recoup/eval owns this file's shape; the dashboard does not. Keys are matched
    by name at any depth rather than by fixed path, and anything unrecognised is
    still rendered verbatim underneath. A rename in the grader should make this
    panel less specific - never blank, and never a stack trace mid-pitch.
    """

    path: Path
    modified_at: datetime
    data: dict[str, Any]

    @property
    def provenance(self) -> str:
        for key in ("provenance", "note", "units"):
            value = self.data.get(key)
            if isinstance(value, str):
                return value
        return ""

    def headline(self) -> dict[str, int]:
        found: dict[str, int] = {}
        flat = list(_flatten(self.data))
        for label, aliases in HEADLINE_KEYS.items():
            for key, value in flat:
                if key.rsplit(".", 1)[-1] in aliases and isinstance(value, int):
                    found[label] = value
                    break
        return found

    def sweep(self) -> list[SweepRow]:
        """The sensitivity sweep, if the harness ran one.

        Given its own panel rather than left in the verbatim dump because it is
        the report's most important claim: a lift figure that only survives at
        one setting of the lift assumptions is not a finding, and the range is
        what a reader is entitled to see.
        """
        for key in ("sweep", "sensitivity", "range"):
            block = self.data.get(key)
            if isinstance(block, dict):
                block = block.get("points") or block.get("rows")
            if not isinstance(block, list):
                continue
            points = [p for p in block if isinstance(p, dict)]
            if not points:
                continue
            return [
                SweepRow(
                    label=str(p.get("label") or p.get("name") or f"point {i + 1}"),
                    incremental_paise=_num(p, *HEADLINE_KEYS["incremental"]),
                    cannibalised_paise=_num(p, *HEADLINE_KEYS["cannibalised"]),
                    cost_paise=_num(p, *HEADLINE_KEYS["cost"]),
                    net_paise=_num(p, *HEADLINE_KEYS["net"]),
                    lift=_num(p.get("lift"), "absolute", "value") or _num(p, "lift"),
                    ci_low=_num(p.get("lift"), "ci_low"),
                    ci_high=_num(p.get("lift"), "ci_high"),
                )
                for i, p in enumerate(points)
            ]
        return []

    def rows(self, limit: int = 80) -> list[tuple[str, Any]]:
        """Everything else, verbatim.

        List contents are dropped: the per-segment breakdowns are long, and the
        one list worth showing already has a panel of its own above.
        """
        return [(k, v) for k, v in _flatten(self.data) if "[" not in k][:limit]


def latest_report(reports_dir: Path | None = None) -> Report | None:
    reports_dir = reports_dir or REPORTS_DIR
    try:
        candidates = list(reports_dir.glob("eval_*.json"))
    except OSError:
        return None
    if not candidates:
        return None

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return Report(
        path=newest,
        # Naive UTC, like every other timestamp in this system. The renderer
        # adds the IST offset itself, so handing it a local-time mtime would
        # print every report five and a half hours into the future.
        modified_at=datetime.fromtimestamp(newest.stat().st_mtime, timezone.utc).replace(
            tzinfo=None
        ),
        data=data,
    )
