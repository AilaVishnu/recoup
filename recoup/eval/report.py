"""The report. Incremental first, gross beside it, and the assumptions in plain sight.

Ordering is an argument. A report that opens with gross recovery and mentions
the holdout in a footnote has already told the reader what to believe, so this
one opens with the incremental number and puts the gross figure immediately
next to it, labelled as what a system that skipped the holdout would have
claimed. The gap between the two panels is the entire thesis of the project.

The sensitivity sweep exists for the same reason. Every rupee figure here is
downstream of the lift parameters in recoup/seed/world.py, which are
assumptions. Reporting a single point estimate from an assumption is how a
simulation turns into a claim. So the eval re-resolves every outcome under
pessimistic, default and optimistic settings and reports the range - and says so
when the sign of the effect is not stable across it.

On rounding: headline and per-segment figures are printed coarsely, in lakhs,
because they are simulator output and printing "Rs 1,24,538" implies a
measurement that was never made. The economics table carries whole rupees
because its column has to add up, and data/reports/eval_<seed>.json carries
exact paise for anything downstream.

Output is ASCII only. This runs on a Windows console as often as not, and a
report that renders "Rs 1.2L" as mojibake in front of a reviewer has failed at
the only job it has.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from recoup.config import PROJECT_ROOT
from recoup.eval.metrics import EvalMetrics, Lift, Segment, build_segments, compute
from recoup.eval.resolve import Resolution, load_oracle, resolve_all
from recoup.seed.world import LiftAssumptions

REPORT_DIR = PROJECT_ROOT / "data" / "reports"


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    label: str
    assumptions: dict[str, float]
    lift: Lift
    incremental_recovered_paise: int
    incremental_n: int
    cannibalised_paise: int
    cost_paise: int
    net_paise: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "assumptions": self.assumptions,
            "lift": self.lift.as_dict(),
            "incremental_recovered_paise": self.incremental_recovered_paise,
            "incremental_n": self.incremental_n,
            "cannibalised_paise": self.cannibalised_paise,
            "cost_paise": self.cost_paise,
            "net_paise": self.net_paise,
        }


@dataclass(frozen=True)
class Sweep:
    points: list[SweepPoint]

    def _span(self, attr: str) -> tuple[float | int | None, float | int | None]:
        vals = [getattr(p, attr) for p in self.points]
        vals = [v for v in vals if v is not None]
        return (min(vals), max(vals)) if vals else (None, None)

    @property
    def lift_span(self) -> tuple[float | None, float | None]:
        vals = [p.lift.absolute for p in self.points if p.lift.absolute is not None]
        return (min(vals), max(vals)) if vals else (None, None)

    @property
    def incremental_span(self) -> tuple[int | None, int | None]:
        return self._span("incremental_recovered_paise")

    @property
    def net_span(self) -> tuple[int | None, int | None]:
        return self._span("net_paise")

    @property
    def sign_is_stable(self) -> bool:
        """False when the assumption range alone can flip the conclusion.

        The one thing a sweep is for. If pessimistic says the agent lost money
        and optimistic says it made money, there is no result here, only a
        parameter choice.
        """
        lo, hi = self.net_span
        if lo is None or hi is None:
            return False
        return (lo >= 0 and hi >= 0) or (lo <= 0 and hi <= 0)

    def as_dict(self) -> dict[str, Any]:
        lo_l, hi_l = self.lift_span
        lo_i, hi_i = self.incremental_span
        lo_n, hi_n = self.net_span
        return {
            "points": [p.as_dict() for p in self.points],
            "lift_span": [lo_l, hi_l],
            "incremental_recovered_paise_span": [lo_i, hi_i],
            "net_paise_span": [lo_n, hi_n],
            "sign_is_stable": self.sign_is_stable,
        }


def run_sensitivity_sweep(session, *, oracle: dict[str, dict] | None = None) -> Sweep:
    """Re-resolve every outcome under pessimistic, default and optimistic lift.

    Nothing is persisted: only the default resolution belongs in the Outcome
    table, or the audit trail ends up recording a hypothetical.

    The events, the actions and the frozen rolls are identical across all three
    runs - the only thing that changes is how much a correct action is assumed
    to be worth. So the spread is pure parameter sensitivity, with sampling
    noise held fixed, which is what makes it readable as a range.
    """
    oracle = load_oracle() if oracle is None else oracle
    settings: list[tuple[str, LiftAssumptions]] = [
        ("pessimistic", LiftAssumptions.pessimistic()),
        ("default", LiftAssumptions()),
        ("optimistic", LiftAssumptions.optimistic()),
    ]

    points: list[SweepPoint] = []
    for label, assumptions in settings:
        resolution = resolve_all(
            session, assumptions=assumptions, oracle=oracle, persist=False
        )
        overall: Segment = build_segments(resolution.outcomes)["overall"]
        points.append(
            SweepPoint(
                label=label,
                assumptions=asdict(assumptions),
                lift=overall.lift,
                incremental_recovered_paise=overall.incremental_recovered_paise,
                incremental_n=overall.incremental_n,
                cannibalised_paise=overall.cannibalised_paise,
                cost_paise=overall.cost_paise,
                net_paise=overall.net_paise,
            )
        )
    return Sweep(points=points)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def rs(paise: int | None) -> str:
    """Whole rupees. For table columns, which have to add up."""
    if paise is None:
        return "-"
    return f"Rs {paise / 100:,.0f}"


def rs_coarse(paise: int | None) -> str:
    """Lakhs and crores, for headline figures.

    Deliberately imprecise. These are simulator output, and a headline printed
    to the rupee invites a reader to treat it as a measurement.
    """
    if paise is None:
        return "-"
    rupees = paise / 100
    sign = "-" if rupees < 0 else ""
    r = abs(rupees)
    if r >= 1_00_00_000:
        return f"{sign}Rs {r / 1_00_00_000:.2f}Cr"
    if r >= 1_00_000:
        return f"{sign}Rs {r / 1_00_000:.1f}L"
    if r >= 1_000:
        return f"{sign}Rs {r / 1_000:.1f}k"
    return f"{sign}Rs {r:.0f}"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def pp(x: float | None) -> str:
    """Percentage points. A difference of rates is never a percentage change."""
    return "n/a" if x is None else f"{x * 100:+.1f}pp"


def _ci(lift: Lift) -> str:
    if lift.ci_low is None or lift.ci_high is None:
        return "no interval"
    return f"95% CI {lift.ci_low * 100:+.1f} .. {lift.ci_high * 100:+.1f}pp"


def _lift_cell(lift: Lift) -> str:
    """Point estimate and half-interval in one narrow column.

    The full bounds are in the JSON. Here the half-width is the useful part: a
    reader scanning sixteen rows needs to see which estimates are swamped by
    their own error bar, and `+7.4 +/-19.0` says that at a glance.
    """
    if lift.absolute is None:
        return "n/a"
    if lift.ci_low is None or lift.ci_high is None:
        return f"{lift.absolute * 100:+.1f}"
    half = (lift.ci_high - lift.ci_low) / 2 * 100
    return f"{lift.absolute * 100:+.1f} +/-{half:.1f}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _integrity_banner(metrics: EvalMetrics) -> Panel | None:
    """Anything that makes the numbers below untrustworthy, before the numbers.

    A quiet warning under a headline is worse than no warning - the reader has
    already taken the number.
    """
    i = metrics.integrity
    alarms: list[str] = []
    if i.get("contaminated_control"):
        alarms.append(
            f"[bold red]{i['contaminated_control']} control-arm event(s) were acted on.[/] "
            "The holdout is no longer a holdout and every comparison below is void. "
            "This should be impossible - see the control_arm_suppression bound."
        )
    if i.get("missing_from_oracle"):
        alarms.append(
            f"{i['missing_from_oracle']} event(s) have no ground-truth row and were "
            "excluded. Re-seed so events and oracle are generated together."
        )
    if i.get("extra_actions"):
        alarms.append(
            f"{i['extra_actions']} event(s) received more than one delivered action. "
            "Recoup executes one action per event; the extras are excluded, so the "
            "cost figures below are a lower bound."
        )
    if not alarms:
        return None
    return Panel(
        "\n".join(alarms), title="[bold red]integrity[/]", border_style="red"
    )


def _no_actions_notice(metrics: EvalMetrics) -> Panel | None:
    if metrics.overall.treated_n > 0:
        return None
    i = metrics.integrity
    skipped = i.get("dry_run_actions", 0)
    detail = (
        f"{skipped} action(s) were dispatched in dry-run mode. They are graded, "
        "but no Razorpay call left the machine - see the mode line in the header."
        if skipped
        else "No action has been executed against any event yet."
    )
    return Panel(
        f"{detail}\n"
        "Every treatment event therefore resolves exactly as a control event would, "
        "so the arm difference below is sampling noise around zero and is [bold]the "
        "correct output[/] for this state, not a failure.",
        title="[bold yellow]nothing was executed[/]",
        border_style="yellow",
    )


def _headline(metrics: EvalMetrics) -> Columns:
    o = metrics.overall
    lift = o.lift
    tone = "green" if (lift.absolute or 0) > 0 else "red"

    incremental = (
        f"[bold {tone}]{pp(lift.absolute)}[/] incremental recovery rate\n"
        f"[dim]{_ci(lift)}[/]\n\n"
        f"[bold]{rs_coarse(o.incremental_recovered_paise)}[/] recovered that would "
        f"not have been\n"
        f"[dim]{o.incremental_n} of {o.treatment.n} treatment events, each one a case "
        f"where the same event with the same luck and no action was lost[/]"
    )
    if not lift.reliable:
        incremental += f"\n\n[yellow]{lift.caveat}[/]"
    elif lift.crosses_zero:
        incremental += (
            "\n\n[yellow]the interval includes zero - this sample cannot rule out "
            "no effect[/]"
        )

    gross = (
        f"[bold yellow]{pct(o.gross_recovery_rate)}[/] gross recovery rate\n"
        f"[dim]control arm: {pct(o.control_recovery_rate)} with no help at all[/]\n\n"
        f"[bold]{rs_coarse(o.gross_recovered_paise)}[/] came back in the treatment arm\n"
        f"[dim]the number a system without a holdout would report as its own - "
        f"most of it would have arrived anyway[/]"
    )

    return Columns(
        [
            Panel(
                incremental,
                title="[bold]incremental - what Recoup caused[/]",
                border_style=tone,
                width=56,
            ),
            Panel(
                gross,
                title="[bold]gross - what a careless system claims[/]",
                border_style="yellow",
                width=56,
            ),
        ]
    )


def _sweep_panel(sweep: Sweep) -> Panel:
    lo_l, hi_l = sweep.lift_span
    lo_i, hi_i = sweep.incremental_span
    lo_n, hi_n = sweep.net_span

    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("assumption", style="cyan")
    table.add_column("lift", justify="right")
    table.add_column("incremental", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("net", justify="right")
    for p in sweep.points:
        style = "bold" if p.label == "default" else "dim"
        table.add_row(
            p.label,
            f"[{style}]{pp(p.lift.absolute)}[/]",
            f"[{style}]{rs(p.incremental_recovered_paise)}[/]",
            f"[{style}]{rs(p.cost_paise)}[/]",
            f"[{style}]{rs(p.net_paise)}[/]",
        )

    verdict = (
        "[green]The sign holds across the whole assumption range.[/]"
        if sweep.sign_is_stable
        else "[bold red]The sign flips inside the assumption range - there is no "
        "result here, only a parameter choice.[/]"
    )
    span = (
        f"lift [bold]{pp(lo_l)} to {pp(hi_l)}[/]   "
        f"incremental [bold]{rs_coarse(lo_i)} to {rs_coarse(hi_i)}[/]   "
        f"net [bold]{rs_coarse(lo_n)} to {rs_coarse(hi_n)}[/]"
    )

    return Panel(
        Group(span, "", table, "", verdict),
        title="[bold]sensitivity - the range is the result, not the middle row[/]",
        border_style="blue",
    )


def _economics_table(metrics: EvalMetrics) -> Table:
    o = metrics.overall
    t = Table(
        title="Economics of the treatment arm",
        title_style="bold",
        box=None,
        pad_edge=False,
    )
    t.add_column("", style="cyan", no_wrap=True)
    t.add_column("", justify="right")
    t.add_column("", style="dim")

    t.add_row("gross recovered", rs(o.gross_recovered_paise), "everything that came back")
    t.add_row(
        "agent-attributed",
        f"[bold]{rs(o.incremental_recovered_paise)}[/]",
        "the only part Recoup may claim",
    )
    t.add_section()
    t.add_row(
        "incentive committed",
        rs(o.incentive_committed_paise),
        "discount promised, counted against the daily budget at send time",
    )
    t.add_row(
        "incentive redeemed",
        rs(o.incentive_realised_paise),
        "a discount only costs the merchant when the customer uses it",
    )
    t.add_row(
        "[bold]  of which cannibalised[/]",
        f"[bold red]{rs(o.cannibalised_paise)}[/]",
        f"{o.cannibalised_n} customer(s) paid to do what they were going to do anyway",
    )
    t.add_row("channel cost", rs(o.channel_cost_paise), "messages sent, redeemed or not")
    t.add_row(
        "recoveries destroyed",
        f"[red]{rs(o.harmed_paise)}[/]",
        f"{o.harmed_n} event(s) that would have recovered untouched",
    )
    t.add_section()
    t.add_row(
        "[bold]net[/]",
        f"[bold]{rs(o.net_paise)}[/]",
        "incremental, less spend, less recoveries the action destroyed",
    )
    # Printed in paise, not rupees-per-rupee. The ratio is tiny by design -
    # recovery here costs a few paise of messaging per rupee returned - and at
    # three decimal places it rendered as "0.000" directly beside a caption
    # insisting the value was "undefined, not zero". A row whose own explainer
    # contradicts the number it labels teaches a reader to distrust the rest.
    if o.cost_per_incremental_rupee is None:
        cost_ratio, cost_note = (
            "undefined",
            "undefined, not zero, when nothing incremental was recovered",
        )
    else:
        cost_ratio = f"{o.cost_per_incremental_rupee * 100:.2f}p"
        cost_note = "paise spent per rupee of incremental recovery"
    t.add_row("cost per incremental rupee", cost_ratio, cost_note)
    return t


def _segment_table(title: str, segments: Sequence[Segment]) -> Table:
    """One row per segment, narrow enough to survive an 80-column console.

    Gross and control sit in one cell on purpose: they are only meaningful as a
    pair, and a reader who sees them apart will read the first one alone.
    """
    t = Table(title=title, title_style="bold", header_style="dim", box=None, pad_edge=False)
    # Truncated here rather than by the renderer: rich elides with a unicode
    # ellipsis, which is the one character that would break the ASCII rule.
    t.add_column("segment", style="cyan", no_wrap=True, overflow="crop")
    t.add_column("T/C", justify="right")
    t.add_column("at risk", justify="right")
    t.add_column("gross/ctrl", justify="right")
    t.add_column("lift pp", justify="right")
    t.add_column("incremental", justify="right")

    for s in segments:
        unreliable = not s.lift.reliable
        key = s.key if len(s.key) <= 22 else f"{s.key[:20]}.."
        name = f"{key}{' !' if unreliable else ''}"
        lift_cell = _lift_cell(s.lift)
        if unreliable:
            lift_cell = f"[dim]{lift_cell}[/]"
        elif (s.lift.absolute or 0) > 0 and not s.lift.crosses_zero:
            lift_cell = f"[green]{lift_cell}[/]"
        t.add_row(
            f"[dim]{name}[/]" if unreliable else name,
            f"{s.treatment.n}/{s.control.n}",
            rs_coarse(s.value_at_risk_paise),
            f"[yellow]{pct(s.gross_recovery_rate)}[/] / {pct(s.control_recovery_rate)}",
            lift_cell,
            rs_coarse(s.incremental_recovered_paise),
        )
    t.caption = (
        "[dim]lift is the arm difference in percentage points, +/- half its 95% "
        "interval. ! = fewer than 30 events in one arm, where the normal "
        "approximation has stopped applying - read those rows as counts, not "
        "results.[/]"
    )
    return t


def _refusals_panel(metrics: EvalMetrics) -> Panel:
    s = metrics.suppression
    lines = [
        f"[bold]{rs(s.value_paise)}[/] across {s.events} event(s) was refused on "
        "principle - risk declines and reason codes the taxonomy does not "
        "recognise. Re-presenting these earns chargebacks and a worse decline "
        "rate, so the value is reported rather than quietly dropped.",
    ]
    for reason, stats in s.by_reason.items():
        lines.append(
            f"  [cyan]{reason}[/]  {stats['events']} event(s)  {rs(stats['value_paise'])}"
        )
    lines.append(
        f"\n[dim]{rs(s.untouched_value_paise)} of treatment-arm value went untouched "
        f"in total ({s.untouched_events} events) - principled refusals plus policy "
        f"denials plus anything the pipeline did not reach.[/]"
    )
    return Panel("\n".join(lines), title="[bold]refused on purpose[/]", border_style="magenta")


def _denials_table(metrics: EvalMetrics) -> Table:
    t = Table(
        title="Policy denials by bound",
        title_style="bold",
        box=None,
        header_style="dim",
    )
    t.add_column("bound", style="cyan")
    t.add_column("times it fired", justify="right")
    if not metrics.policy_denials:
        t.add_row("[dim]no reviews recorded[/]", "-")
        return t
    for name, count in metrics.policy_denials.items():
        t.add_row(name, str(count))
    t.caption = (
        "[dim]control_arm_suppression firing once per holdout event is the guardrail "
        "working - it is the only direct evidence the holdout was actually held.[/]"
    )
    return t


def _footer(metrics: EvalMetrics) -> Panel:
    # Count every interval the report actually renders, not just the reason-code
    # ones. The warning said "15 segments are compared here" while 25 confidence
    # intervals were printed above it - understating the family-wise error rate
    # in the very sentence being honest about it.
    n_intervals = (
        len(metrics.by_reason_code)
        + len(metrics.by_strategy)
        + len(metrics.by_event_kind)
        + 1  # overall
    )
    llm = metrics.llm
    llm_line = (
        f"{llm.llm_decisions} of {llm.decisions} decisions used the model "
        f"({pct(llm.share)}), {llm.input_tokens:,} in / {llm.output_tokens:,} out tokens"
        if llm.decisions
        else "no decisions recorded"
    )
    return Panel(
        "[bold]Real[/] - every decision, bound and refusal above. Which strategy was "
        "chosen for which failure reason, which bounds fired, what was never touched. "
        "Properties of the code, true regardless of any simulation parameter.\n\n"
        "[bold]Simulated[/] - every rupee. Outcomes come from recoup/seed/world.py, "
        "whose lift parameters are assumptions. The sensitivity range is the honest "
        "form of the headline; the point estimate is not.\n\n"
        "[bold]Not corrected for[/] - "
        f"{n_intervals} confidence intervals are printed above - across reason "
        "codes, strategies, event kinds and overall. At 95% confidence roughly "
        "one in twenty looks significant by chance, so this family would be "
        f"expected to throw up about {max(1, round(n_intervals * 0.05))} spurious "
        "result(s); no multiple-comparison correction is applied, and no "
        "per-segment interval should be read as a discovery.\n\n"
        f"[bold]Model use[/] - {llm_line}.",
        title="[bold]what is measured, and what is assumed[/]",
        border_style="dim",
    )


def render(
    metrics: EvalMetrics, sweep: Sweep | None = None, console: Console | None = None
) -> None:
    console = console or Console()
    o = metrics.overall

    console.print()
    console.rule(
        f"[bold]Recoup evaluation[/]  seed {metrics.seed}  |  {metrics.n_events} events "
        f"|  {rs_coarse(o.value_at_risk_paise)} at risk"
    )

    # Execution mode belongs in the header, not a footnote. Dry-run actions are
    # graded (see recoup/eval/resolve.py on why), so a reader who misses this
    # line would take a simulated dispatch for a live one.
    #
    # Counted from what could reach Razorpay, never from ActionStatus alone. An
    # outbox nudge is SENT in every mode, so inferring "sent to Razorpay" from
    # that status made an earlier version of this line report 168 live gateway
    # calls during a run in which none were made. A header that overstates what
    # the system did is worse than no header.
    i = metrics.integrity
    sent = i.get("gateway_sent", 0)
    withheld = i.get("gateway_withheld", 0)
    outbox = i.get("outbox_only_actions", 0)

    if sent and withheld:
        mode = (
            f"[yellow]mixed[/] - {sent} gateway call(s) made, {withheld} withheld "
            "in dry run"
        )
    elif withheld:
        mode = (
            f"[yellow]dry run[/] - {withheld} gateway call(s) withheld, "
            "nothing left the machine"
        )
    elif sent:
        mode = f"[green]live test mode[/] - {sent} gateway call(s) to Razorpay"
    else:
        mode = "[dim]no gateway actions executed[/]"

    console.print(f"  {mode}")
    if outbox:
        console.print(
            f"  [dim]{outbox} outbox message(s) - simulated in every mode, never "
            "delivered to a real customer[/]"
        )
    console.print(
        "  [dim]outcomes are simulated in every mode - see recoup/seed/world.py[/]"
    )
    console.print()

    banner = _integrity_banner(metrics)
    if banner is not None:
        console.print(banner)
        console.print()

    notice = _no_actions_notice(metrics)
    if notice is not None:
        console.print(notice)
        console.print()

    console.print(_headline(metrics))
    console.print()

    if sweep is not None:
        console.print(_sweep_panel(sweep))
        console.print()

    console.print(_economics_table(metrics))
    console.print()
    console.print(_segment_table("By strategy", metrics.by_strategy))
    console.print()
    console.print(_segment_table("By reason code", metrics.by_reason_code))
    console.print()
    console.print(_segment_table("By event kind", metrics.by_event_kind))
    console.print()
    console.print(_refusals_panel(metrics))
    console.print()
    console.print(_denials_table(metrics))
    console.print()
    console.print(_footer(metrics))
    console.print()


# ---------------------------------------------------------------------------
# Machine-readable
# ---------------------------------------------------------------------------


def write_json(
    metrics: EvalMetrics, sweep: Sweep | None = None, path: Path | None = None
) -> Path:
    """Exact paise, no formatting, for anything downstream of this report."""
    path = path or REPORT_DIR / f"eval_{metrics.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "units": "all money in integer paise; rates in [0,1]; lift in rate difference",
        "provenance": (
            "Outcomes are simulated from recoup/seed/world.py against rolls frozen at "
            "generation time. Decision quality is real; rupee figures are assumption "
            "output and are only meaningful as the range in `sweep`."
        ),
        "metrics": metrics.as_dict(),
        "sweep": sweep.as_dict() if sweep is not None else None,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run(
    session,
    *,
    seed: int,
    sweep: bool = True,
    console: Console | None = None,
) -> tuple[EvalMetrics, Sweep | None, Path]:
    """Resolve, measure, print, write. The whole harness in one call."""
    oracle = load_oracle()
    resolution: Resolution = resolve_all(session, oracle=oracle, persist=True)
    metrics = compute(session, resolution, seed=seed)
    sweep_result = run_sensitivity_sweep(session, oracle=oracle) if sweep else None

    render(metrics, sweep_result, console=console)
    path = write_json(metrics, sweep_result)
    return metrics, sweep_result, path
