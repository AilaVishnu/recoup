"""Presentation primitives: money, clock, identity.

Two of these are load-bearing rather than cosmetic.

**Indian digit grouping.** ``Rs 11,72,450`` and ``Rs 1,172,450`` are the same
number, and only one of them reads as money to the people this is built for.
Western grouping on an Indian payments dashboard is a small, immediate tell that
the thing was assembled for a different audience.

**Integers the whole way to the screen.** Every rupee printed here is integer
division of the integer paise in the database. No float ever holds money on its
way out, so the page cannot disagree with the audit trail in the third decimal
place - which is the entire reason the schema stores paise in the first place.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from recoup.detect.features import to_ist

DASH = "\u2014"
"""What an absent number renders as. Never a zero - a stage that has not run yet
and a stage that ran and produced nothing are different facts, and a dashboard
that prints Rs 0 for both is lying about the first one."""


def _to_rupees(paise: int) -> int:
    """Paise to whole rupees, half-up, sign preserved."""
    sign = -1 if paise < 0 else 1
    return sign * ((abs(int(paise)) + 50) // 100)


def group_indian(n: int) -> str:
    """12345678 -> '1,23,45,678'. Last three digits, then pairs."""
    sign = "-" if n < 0 else ""
    digits = str(abs(int(n)))
    if len(digits) <= 3:
        return sign + digits

    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return sign + ",".join([*groups, tail])


def rupees(paise: int | None) -> str:
    if paise is None:
        return DASH
    return f"Rs {group_indian(_to_rupees(paise))}"


def compact_rupees(paise: int | None) -> str:
    """Lakh/crore shorthand for headline figures. Integer arithmetic only.

    The decimal is assembled from a quotient and a remainder rather than a
    division, so this stays inside the no-floats-for-money rule.
    """
    if paise is None:
        return DASH
    r = _to_rupees(paise)
    sign, a = ("-" if r < 0 else ""), abs(r)
    if a >= 1_00_00_000:
        hundredths = (a + 50_000) // 1_00_000
        return f"{sign}Rs {hundredths // 100}.{hundredths % 100:02d}Cr"
    if a >= 1_00_000:
        tenths = (a + 5_000) // 10_000
        return f"{sign}Rs {tenths // 10}.{tenths % 10}L"
    return rupees(paise)


def pct(x: float | None, dp: int = 1) -> str:
    if x is None:
        return DASH
    return f"{x * 100:.{dp}f}%"


def ist(dt: datetime | None) -> str:
    """Stored naive-UTC rendered in the only timezone this merchant cares about."""
    if dt is None:
        return DASH
    return to_ist(dt).strftime("%d %b %Y, %H:%M") + " IST"


def ist_short(dt: datetime | None) -> str:
    if dt is None:
        return DASH
    return to_ist(dt).strftime("%d %b %H:%M")


def hours(x: float | None) -> str:
    if x is None:
        return DASH
    if x < 48:
        return f"{x:.1f}h"
    return f"{x / 24:.1f}d"


def sid(value: str | None, keep: int = 14) -> str:
    """Trim an identifier for table cells. Full value stays in the row's link."""
    if not value:
        return DASH
    return value if len(value) <= keep else value[:keep] + "\u2026"


def pretty(value: Any) -> str:
    """JSON blobs (action params, gateway responses) as something readable."""
    if value in (None, {}, []):
        return DASH
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def humanise(key: str) -> str:
    return key.replace("_", " ")


FILTERS = {
    "rupees": rupees,
    "crupees": compact_rupees,
    "pct": pct,
    "ist": ist,
    "ist_short": ist_short,
    "hours": hours,
    "sid": sid,
    "pretty": pretty,
    "humanise": humanise,
    "group_indian": group_indian,
}
