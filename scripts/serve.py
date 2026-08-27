"""Run the dashboard.

    python scripts/serve.py

Read-only, local, and deliberately boring to start: on a five-minute clock the
demo cannot depend on anything that might need a second attempt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from rich.console import Console
from sqlalchemy import func, select

from recoup.api.format import rupees
from recoup.api.read import latest_report, schema_ready
from recoup.db import Cohort, RevenueEvent, get_session

console = Console()


def banner(host: str, port: int) -> None:
    session = get_session()
    try:
        if not schema_ready(session):
            console.print(
                "\n[bold yellow]No database yet.[/] The dashboard will start and "
                "tell you the same thing.\n  Build the merchant first: "
                "[bold]python scripts/seed.py[/]\n"
            )
        else:
            events = session.scalar(select(func.count()).select_from(RevenueEvent)) or 0
            at_risk = session.scalar(
                select(func.coalesce(func.sum(RevenueEvent.amount_paise), 0))
            ) or 0
            held = session.scalar(
                select(func.coalesce(func.sum(RevenueEvent.amount_paise), 0)).where(
                    RevenueEvent.cohort == Cohort.CONTROL
                )
            ) or 0
            # Indian grouping here too. The dashboard argues that Rs 1,172,450
            # is a tell; printing it in the terminal on the way to the browser
            # would be the same tell with a smaller audience.
            console.print(
                f"\n[bold]{events}[/] events  |  "
                f"[bold]{rupees(at_risk)}[/] at risk  |  "
                f"[magenta]{rupees(held)}[/] held out"
            )
            report = latest_report()
            console.print(
                f"  eval report: {report.path.name if report else '[dim]none yet - python scripts/run_eval.py[/]'}"
            )
    finally:
        session.close()

    console.print(f"\n[bold green]Recoup[/] -> [bold]http://{host}:{port}[/]")
    console.print("  [dim]/[/]              value at risk, incremental vs gross, what policy blocked")
    console.print("  [dim]/events[/]        every event, sorted by what it is worth acting on")
    console.print("  [dim]/events/<id>[/]   the whole chain of reasoning for one event")
    console.print("  [dim]/policy[/]        the bounds, and which of them actually bind")
    console.print("\n[dim]ctrl-c to stop[/]\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the Recoup dashboard.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--reload",
        action="store_true",
        help="Reload on source changes. Off by default - a reloader restarting "
        "mid-sentence is not something to discover on camera.",
    )
    args = ap.parse_args()

    banner(args.host, args.port)
    uvicorn.run(
        "recoup.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
