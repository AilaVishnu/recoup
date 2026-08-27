"""Build the synthetic merchant dataset.

    python scripts/seed.py --events 600 --customers 220
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from recoup.db import Cohort, RevenueEvent, get_session
from recoup.seed.generate import generate

console = Console()


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed the Recoup synthetic merchant.")
    ap.add_argument("--events", type=int, default=600)
    ap.add_argument("--customers", type=int, default=220)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--keep",
        action="store_true",
        help="Append to the existing database instead of dropping it.",
    )
    args = ap.parse_args()

    summary = generate(
        n_customers=args.customers,
        n_events=args.events,
        seed=args.seed,
        drop=not args.keep,
    )

    console.print()
    console.print("[bold green]Synthetic merchant built.[/]")
    console.print(
        f"  {summary['customers']} customers, {summary['events']} revenue-at-risk events"
    )
    console.print(
        f"  [bold]Rs {summary['value_at_risk_inr']:,.0f}[/] at risk  "
        f"([cyan]{summary['treatment']}[/] treatment / "
        f"[magenta]{summary['control']}[/] control)"
    )
    console.print(f"  ground truth -> {summary['oracle_path']}")
    console.print()

    session = get_session()
    events = session.query(RevenueEvent).all()

    by_reason = Counter(e.reason_code for e in events)
    value_by_reason: Counter[str] = Counter()
    for e in events:
        value_by_reason[e.reason_code] += e.amount_paise

    table = Table(title="Revenue at risk by failure reason", title_style="bold")
    table.add_column("reason", style="cyan", no_wrap=True)
    table.add_column("events", justify="right")
    table.add_column("share", justify="right")
    table.add_column("value at risk", justify="right", style="yellow")

    total_value = sum(value_by_reason.values())
    for reason, count in by_reason.most_common():
        table.add_row(
            reason,
            str(count),
            f"{count / len(events):.1%}",
            f"Rs {value_by_reason[reason] / 100:,.0f}",
        )
    table.add_section()
    table.add_row(
        "[bold]total[/]",
        f"[bold]{len(events)}[/]",
        "",
        f"[bold]Rs {total_value / 100:,.0f}[/]",
    )
    console.print(table)

    control_value = sum(
        e.amount_paise for e in events if e.cohort is Cohort.CONTROL
    )
    console.print(
        f"\n[dim]Holdout carries Rs {control_value / 100:,.0f} of the at-risk value. "
        f"Recoup will not touch it - that is the price of knowing whether any of "
        f"this works.[/]\n"
    )
    session.close()


if __name__ == "__main__":
    main()
