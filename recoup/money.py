"""Rupee formatting, in one place.

Money is stored as integer paise everywhere in this project. Rendering it is the
one operation that happens in three unrelated layers - policy check details, the
terminal report, the dashboard - and each had grown its own f-string.

That is how the Indian digit grouping broke on exactly the two screens the pitch
video spends longest on: the dashboard formatted `1,23,456` correctly while the
policy engine's own check details, rendered verbatim on the same page, printed
`123,456`. Nobody notices a formatter drifting until two of them are side by
side in front of an audience that groups digits the other way.

So: one function, imported by everything that prints a rupee.
"""

from __future__ import annotations


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
    """Whole rupees with Indian grouping, e.g. 'Rs 1,23,456'.

    Integer division rather than `/100`, so a large paise value cannot pick up a
    float rounding artefact on its way to a display string. Sub-rupee amounts
    round to the nearest rupee; the exact paise stay in the database and in
    data/reports/*.json for anything that needs to add up.
    """
    if paise is None:
        return "-"
    return f"Rs {group_indian(round(paise / 100))}"
