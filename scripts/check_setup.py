"""Verify the local environment before running anything that costs money.

    python scripts/check_setup.py

Every check here is read-only. Nothing is created, charged, or sent. Run it
after cloning, after rotating keys, and before recording a demo - a Payment Link
that fails on camera because a key expired is a bad five minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from recoup.config import PROJECT_ROOT, get_settings

console = Console()

OK = "[green]OK  [/]"
WARN = "[yellow]WARN[/]"
FAIL = "[red]FAIL[/]"


def mask(value: str, keep: int = 8) -> str:
    """Never print a secret in full - not to a terminal that may be screen-shared."""
    if not value:
        return "(empty)"
    if len(value) <= keep:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * (len(value) - keep)} ({len(value)} chars)"


def check_env_file() -> bool:
    path = PROJECT_ROOT / ".env"
    if path.exists():
        console.print(f"{OK} .env found at {path}")
        return True
    console.print(f"{FAIL} no .env - copy .env.example and fill it in")
    return False


def check_razorpay(settings) -> bool:
    if not settings.razorpay_configured:
        console.print(f"{WARN} Razorpay keys absent - executors will run in dry-run only")
        return False

    console.print(f"  key id     {settings.razorpay_key_id}")
    console.print(f"  key secret {mask(settings.razorpay_key_secret)}")

    try:
        import razorpay
    except ImportError:
        console.print(f"{FAIL} razorpay SDK not installed - pip install -r requirements.txt")
        return False

    client = razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))
    try:
        # Read-only. Lists at most one payment; creates nothing.
        result = client.payment.all({"count": 1})
    except Exception as exc:  # noqa: BLE001 - the SDK raises several unrelated types
        message = str(exc)
        if "authentication" in message.lower() or "401" in message:
            console.print(f"{FAIL} Razorpay rejected the credentials.")
            console.print(
                "       The secret is shown only once at generation. If you did not "
                "copy it, regenerate the key pair and update BOTH halves in .env."
            )
        else:
            console.print(f"{FAIL} Razorpay call failed: {message}")
        return False

    count = result.get("count", 0)
    console.print(f"{OK} Razorpay test mode authenticated ({count} existing payment(s) visible)")
    if count == 0:
        console.print(
            "       [dim]A fresh test account with no payments yet - expected, and fine. "
            "Recoup creates its own.[/]"
        )
    return True


def check_anthropic(settings) -> bool:
    if not settings.anthropic_configured:
        console.print(
            f"{WARN} ANTHROPIC_API_KEY empty - the agent falls back to the rules engine."
        )
        console.print(
            "       [dim]The pipeline still runs end to end. You lose the ambiguous "
            "decisions, which are the interesting ones.[/]"
        )
        return False
    console.print(f"{OK} Anthropic key present ({mask(settings.anthropic_api_key, 10)})")
    return True


def check_database() -> bool:
    from sqlalchemy import inspect

    from recoup.db import get_engine

    try:
        tables = inspect(get_engine()).get_table_names()
    except Exception as exc:  # noqa: BLE001
        console.print(f"{FAIL} database unreachable: {exc}")
        return False

    if not tables:
        console.print(f"{WARN} database empty - run: python scripts/seed.py")
        return False

    from recoup.db import RevenueEvent, get_session

    session = get_session()
    n = session.query(RevenueEvent).count()
    session.close()
    console.print(f"{OK} database ready ({len(tables)} tables, {n} events)")
    return True


def main() -> int:
    console.print("\n[bold]Recoup - environment check[/]\n")

    check_env_file()
    settings = get_settings()

    console.print("\n[bold]Razorpay[/]")
    razorpay_ok = check_razorpay(settings)

    console.print("\n[bold]Anthropic[/]")
    check_anthropic(settings)

    console.print("\n[bold]Data[/]")
    check_database()

    console.print(f"\n[bold]Mode[/]\n  RECOUP_DRY_RUN={settings.dry_run}")
    if settings.dry_run:
        console.print(
            "  [dim]Executors will log intended calls instead of making them. "
            "Flip to false in .env for real Payment Links.[/]"
        )
    elif razorpay_ok:
        console.print(
            "  [yellow]LIVE TEST-MODE CALLS ENABLED[/] - a pipeline run will create "
            "real Payment Links in your test account."
        )

    console.print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
