"""Prove the Razorpay path works against the real API, once, safely.

    python scripts/verify_live.py

Unlike scripts/check_setup.py, which is strictly read-only, this CREATES one
Payment Link in your test account and then reads it back. That is the point:
every other check in this project verifies that Recoup *would* call Razorpay
correctly, and a payload assembled from SDK source and documentation is not the
same claim as a 200 from the live service.

It creates exactly one link, for Rs 499, and never runs the pipeline. A full
live run would create several hundred.

The assertion that matters
--------------------------
Recoup's seeded customers carry fabricated email addresses and phone numbers.
Razorpay will happily send a Payment Link to whatever contact details it is
given, on its own schedule, outside every bound this project enforces - quiet
hours, the weekly contact cap, the cost ledger. So every link Recoup creates
sets notify.email, notify.sms and reminder_enable to false, and this script
reads them back off the created object rather than trusting that the request
carried them.

That check failing is not a cosmetic problem. It means a synthetic dataset has
started emailing real inboxes at addresses that happen to exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from recoup.config import get_settings
from recoup.db import Customer, get_session

console = Console()
OK = "[green]OK  [/]"
FAIL = "[red]FAIL[/]"


def main() -> int:
    settings = get_settings()

    if not settings.razorpay_configured:
        console.print(f"\n{FAIL} No Razorpay keys. Nothing to verify.\n")
        return 1

    # Forced on regardless of RECOUP_DRY_RUN. The whole purpose of this script is
    # to leave the machine, so honouring the dry-run flag here would make it
    # silently verify nothing - the failure mode where a green check means the
    # code never ran.
    import os

    os.environ["RECOUP_DRY_RUN"] = "false"
    get_settings.cache_clear()

    from recoup.execute import razorpay_client as rc

    if rc.is_dry_run():
        console.print(f"\n{FAIL} Still in dry run after forcing it off - aborting.\n")
        return 1

    console.print("\n[bold]Recoup - live Razorpay verification[/]")
    console.print(f"  key id  {get_settings().razorpay_key_id}")
    console.print("  [dim]creating one Rs 499 Payment Link in TEST mode[/]\n")

    session = get_session()
    customer = session.query(Customer).first()
    if customer is None:
        console.print(f"{FAIL} No customers. Run scripts/seed.py first.\n")
        return 1

    try:
        link = rc.create_payment_link(
            amount_paise=499_00,
            customer=customer,
            description="Recoup live-path verification",
            notes={"recoup_verification": "true"},
            reference_id="recoup_live_check",
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"{FAIL} create_payment_link failed: {exc}\n")
        session.close()
        return 1

    console.print(f"{OK} Payment Link created")
    console.print(f"     id        {link.get('id')}")
    console.print(f"     short_url  {link.get('short_url')}")

    fetched = rc.fetch_payment_link(link["id"])
    console.print(f"{OK} round-trip fetch, status={fetched.get('status')}")

    notify = fetched.get("notify") or {}
    reminders = fetched.get("reminder_enable")
    silent = not any(notify.values()) and not reminders

    if silent:
        console.print(f"{OK} Razorpay will not contact the customer")
        console.print(
            f"     notify={notify}  reminder_enable={reminders}"
        )
        console.print(
            "     [dim]Required, not incidental: seeded contact details are "
            "fabricated, and Razorpay messages on its own schedule - outside "
            "quiet hours, the weekly cap and the cost ledger.[/]"
        )
    else:
        console.print(f"{FAIL} Razorpay WILL contact the customer")
        console.print(f"     notify={notify}  reminder_enable={reminders}")
        console.print(
            "     [red]Stop. A synthetic dataset is about to message real "
            "inboxes at addresses that happen to exist.[/]"
        )

    console.print(
        f"\n[dim]Visible under Payment Links in the Razorpay dashboard "
        f"(Test Mode). It is never paid, so it expires on its own.[/]\n"
    )
    session.close()
    return 0 if silent else 1


if __name__ == "__main__":
    raise SystemExit(main())
