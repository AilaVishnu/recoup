"""Grade the run: resolve every outcome, measure incremental lift, print the report.

    python scripts/run_eval.py
    python scripts/run_eval.py --no-sweep        # skip the assumption range
    python scripts/run_eval.py --seed 7          # label the report for a dataset

The eval itself draws no random numbers. Every roll was frozen when the dataset
was generated, so two runs over the same database produce byte-identical output;
`--seed` names which dataset is being graded and titles the JSON report, it does
not seed anything here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from recoup.config import get_settings
from recoup.db import get_session
from recoup.eval.report import run
from recoup.eval.resolve import OracleMismatch

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a Recoup run.")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Dataset seed, used to name data/reports/eval_<seed>.json.",
    )
    ap.add_argument(
        "--sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-resolve under pessimistic/default/optimistic lift and report the range.",
    )
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else get_settings().seed
    session = get_session()
    try:
        _, _, path = run(session, seed=seed, sweep=args.sweep)
    except FileNotFoundError as exc:
        console.print(f"[bold red]{exc}[/]")
        return 1
    except OracleMismatch as exc:
        # The grader refusing to grade is the correct outcome here: a mismatched
        # answer key produces numbers that look fine and mean nothing.
        console.print(f"[bold red]Refusing to report:[/] {exc}")
        return 1
    finally:
        session.close()

    console.print(f"[dim]machine-readable report -> {path}[/]")
    if not args.sweep:
        console.print(
            "[yellow]Sweep skipped.[/] The point estimate above holds at one "
            "assumption setting, which is not a result. Re-run with --sweep before "
            "quoting any rupee figure."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
