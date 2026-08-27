"""Run every open revenue event through the full pipeline.

    python scripts/run_pipeline.py                # all open events
    python scripts/run_pipeline.py --limit 50     # a sample, for a quick look
    python scripts/run_pipeline.py --recalibrate  # fit rates from prior outcomes first

Nothing here reports results. It says what the system *did*; whether any of it
worked is recoup/eval's job, and keeping those two apart is deliberate - a
runner that also graded itself would be marking its own homework.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from recoup.config import get_settings
from recoup.db import get_session
from recoup.pipeline import run

console = Console()


def rupees(paise: int) -> str:
    return f"Rs {paise / 100:,.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Recoup recovery pipeline.")
    ap.add_argument("--limit", type=int, default=None, help="process at most N events")
    ap.add_argument(
        "--recalibrate",
        action="store_true",
        help="fit per-reason rates from existing outcomes before scoring",
    )
    args = ap.parse_args()

    settings = get_settings()
    mode = (
        "[green]DRY RUN[/] - intended calls are logged, not made"
        if settings.dry_run
        else "[yellow]LIVE TEST MODE[/] - real Payment Links will be created"
    )
    console.print(f"\n[bold]Recoup pipeline[/]  ({mode})\n")

    session = get_session()
    stats = run(session, limit=args.limit, recalibrate=args.recalibrate)
    session.close()

    if stats.events == 0:
        console.print(
            "[yellow]No open events.[/] Reseed with: python scripts/seed.py\n"
        )
        return 0

    decided = stats.llm_decisions + stats.rules_decisions
    llm_share = stats.llm_decisions / decided if decided else 0.0

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("events", f"[bold]{stats.events}[/] processed")
    t.add_row(
        "decided by",
        f"{stats.rules_decisions} rules / [cyan]{stats.llm_decisions} model[/] "
        f"({llm_share:.0%} escalated)",
    )
    t.add_row("executed", f"[green]{stats.executed}[/]")
    t.add_row("escalated to human", str(stats.escalated))
    t.add_row("held out (control)", f"[magenta]{stats.suppressed_for_holdout}[/]")
    t.add_row("denied by policy", f"[red]{stats.suppressed_by_policy}[/]")
    t.add_row("deferred past quiet hours", str(stats.deferred_for_quiet_hours))
    t.add_row("incentive spend", rupees(stats.incentive_paise))
    t.add_row("channel cost", rupees(stats.channel_cost_paise))
    console.print(t)

    if stats.denials_by_rule:
        console.print("\n[bold]Which bounds actually bound[/]")
        dt = Table(box=None, padding=(0, 2))
        dt.add_column("rule", style="cyan")
        dt.add_column("blocked", justify="right")
        for name, count in sorted(
            stats.denials_by_rule.items(), key=lambda kv: -kv[1]
        ):
            dt.add_row(name, str(count))
        console.print(dt)
        console.print(
            "\n[dim]A rule that never fires is either unnecessary or untested. "
            "This table is how you tell which.[/]"
        )

    if stats.errors:
        console.print(f"\n[red]{len(stats.errors)} event(s) errored[/]")
        for err in stats.errors[:10]:
            console.print(f"  {err}")
        if len(stats.errors) > 10:
            console.print(f"  ... and {len(stats.errors) - 10} more")

    console.print("\n[dim]Now grade it:  python scripts/run_eval.py[/]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
