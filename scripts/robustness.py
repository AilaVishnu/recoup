"""Does the result survive a different dataset?

    python scripts/robustness.py --seeds 1 7 42 99 2024

Every figure this project reports comes from seed 42. That is reproducible,
which is not the same as robust: a single synthetic dataset can carry a result
that exists only in its own particular draw of customers, amounts and rolls.

So this re-runs the whole thing - seed, pipeline, resolve, measure - on several
independent datasets and reports the spread. It is the natural companion to the
sensitivity sweep in recoup/eval/report.py: that one varies the *assumptions*
with the data held fixed, this one varies the *data* with the assumptions held
fixed. A result needs to survive both.

What it deliberately does not do
--------------------------------
It runs with the decision model disabled, even when a key is configured. Two
reasons, and the second is the real one. A paced model run takes about fifteen
minutes per seed, so five seeds would be over an hour. More importantly, the
question here is whether the *system's* effect is stable across datasets, and
letting a non-deterministic component vary alongside the data would confound the
two - a lift that moved between seeds would leave you unable to say whether the
data or the model moved it. The model's contribution is measured separately in
ARCHITECTURE.md section 15, on one dataset, against the same taxonomy.

Each seed gets its own database and its own oracle file, so a run never touches
the working data/recoup.db or the report the dashboard is reading.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()
WORKDIR = Path(__file__).resolve().parent.parent / "data" / "robustness"


def run_one(seed: int, events: int) -> dict | None:
    """Seed, run and grade one dataset in its own database. Returns a row dict."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    db = WORKDIR / f"seed_{seed}.db"
    oracle = WORKDIR / f"oracle_{seed}.json"
    if db.exists():
        db.unlink()

    # Point every layer at this seed's files before anything imports settings.
    os.environ["RECOUP_DB_URL"] = f"sqlite:///{db.as_posix()}"
    os.environ["RECOUP_SEED"] = str(seed)
    os.environ["RECOUP_DRY_RUN"] = "true"
    # Disabling the model here is what keeps this a test of the data rather than
    # of the data and the model together.
    os.environ["RECOUP_LLM_PROVIDER"] = ""
    os.environ["RECOUP_LLM_API_KEY"] = ""
    os.environ["ANTHROPIC_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""

    # Re-import per seed: config, the engine and the oracle path are all cached
    # at module scope, and a stale cache would silently grade seed N against
    # seed N-1's answer key - which would look like a stable result.
    for mod in [m for m in list(sys.modules) if m.startswith("recoup")]:
        del sys.modules[mod]

    import recoup.db as db_mod
    import recoup.eval.resolve as resolve_mod
    from recoup.config import get_settings
    from recoup.eval.metrics import build_segments
    from recoup.pipeline import run as run_pipeline
    from recoup.seed import generate as gen

    get_settings.cache_clear()
    gen.ORACLE_PATH = oracle
    resolve_mod.ORACLE_PATH = oracle

    gen.generate(n_customers=220, n_events=events, seed=seed, drop=True)

    session = db_mod.get_session()
    stats = run_pipeline(session)
    resolution = resolve_mod.resolve_all(
        session, oracle=resolve_mod.load_oracle(oracle), persist=True
    )
    overall = build_segments(resolution.outcomes)["overall"]
    session.close()

    return {
        "seed": seed,
        "events": stats.events,
        "executed": stats.executed,
        "lift": overall.lift.absolute,
        "ci_low": overall.lift.ci_low,
        "ci_high": overall.lift.ci_high,
        "gross_rate": overall.gross_recovery_rate,
        "control_rate": overall.control_recovery_rate,
        "incremental_paise": overall.incremental_recovered_paise,
        "net_paise": overall.net_paise,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-run Recoup across several datasets.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 42, 99, 2024])
    ap.add_argument("--events", type=int, default=600)
    args = ap.parse_args()

    console.print(
        f"\n[bold]Recoup - robustness across {len(args.seeds)} datasets[/]  "
        f"({args.events} events each, model disabled)\n"
    )

    rows = []
    for seed in args.seeds:
        console.print(f"  seed {seed} ...", end="")
        try:
            row = run_one(seed, args.events)
        except Exception as exc:  # noqa: BLE001
            console.print(f" [red]failed: {type(exc).__name__}: {exc}[/]")
            continue
        rows.append(row)
        console.print(f" lift {row['lift'] * 100:+.1f}pp")

    if not rows:
        console.print("\n[red]No seed completed.[/]\n")
        return 1

    t = Table(title="Per dataset", title_style="bold", box=None, padding=(0, 2))
    for col, just in (
        ("seed", "right"), ("lift", "right"), ("95% CI", "right"),
        ("gross", "right"), ("control", "right"), ("incremental", "right"),
    ):
        t.add_column(col, justify=just)
    for r in rows:
        t.add_row(
            str(r["seed"]),
            f"{r['lift'] * 100:+.1f}pp",
            f"{r['ci_low'] * 100:+.1f} .. {r['ci_high'] * 100:+.1f}",
            f"{r['gross_rate'] * 100:.1f}%",
            f"{r['control_rate'] * 100:.1f}%",
            f"Rs {r['incremental_paise'] / 100:,.0f}",
        )
    console.print(t)

    lifts = [r["lift"] * 100 for r in rows]
    positive = sum(1 for x in lifts if x > 0)
    console.print()
    console.print(f"  lift across datasets   {min(lifts):+.1f}pp to {max(lifts):+.1f}pp")
    console.print(f"  median                 {statistics.median(lifts):+.1f}pp")
    if len(lifts) > 1:
        console.print(f"  spread (sd)            {statistics.stdev(lifts):.1f}pp")
    console.print(f"  positive point estimate {positive} of {len(lifts)} datasets")

    # A positive point estimate on every dataset and an interval that excludes
    # zero are different claims, and quoting only the first would be the exact
    # overclaiming this project exists to avoid. Most of these intervals contain
    # zero: the direction is consistent, the magnitude is not established.
    excludes = sum(
        1
        for r in rows
        if r["ci_low"] is not None and r["ci_high"] is not None and r["ci_low"] > 0
    )
    console.print(f"  interval excludes zero  {excludes} of {len(lifts)} datasets")

    if positive == len(lifts):
        console.print(
            f"\n  [green]The direction holds on every dataset.[/] The magnitude "
            f"moves by roughly {max(lifts) / max(min(lifts), 0.1):.0f}x across "
            f"them, and only {excludes} of {len(lifts)} intervals exclude zero - "
            "so this is evidence of a consistent sign, not of a reliable "
            "effect size. Report it that way."
        )
    else:
        console.print(
            f"\n  [yellow]The direction does not hold everywhere[/] - "
            f"{len(lifts) - positive} dataset(s) came out negative. That belongs "
            "in the README beside the headline, not in a footnote."
        )
    console.print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
