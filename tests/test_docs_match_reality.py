"""The documentation makes checkable claims. This checks them.

README.md, ARCHITECTURE.md and PITCH.md quote counts - tests, reason codes,
bounds, tables - and a reader has no way to tell a current number from one that
was true four commits ago. These had already drifted: two documents claimed 252
tests and a third claimed 245, while the suite held 256, and the two figures
disagreed with each other inside the same file.

That is a small error with a disproportionate cost here. This project's entire
argument is that its numbers can be trusted, and a reviewer who finds one stale
count has no reason to believe the ones they cannot check as easily.

So the counts are asserted rather than maintained. Adding a test or a taxonomy
entry now fails this file until the documents are updated, which is the point:
the failure is the reminder nobody would otherwise get.

Deliberately narrow. Only claims with a single unambiguous source of truth in
the code belong here - a test that tried to verify prose would fail on rewording
and teach everyone to ignore it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = ("README.md", "ARCHITECTURE.md", "PITCH.md")


def doc_texts() -> dict[str, str]:
    out = {}
    for name in DOCS:
        p = PROJECT_ROOT / name
        if p.exists():
            out[name] = p.read_text(encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Test count
# ---------------------------------------------------------------------------


def test_quoted_test_counts_match_the_suite(request):
    """Every "N tests" in the docs must be the number pytest actually collected.

    `testscollected` is the whole suite even when a subset is being run, so this
    stays honest under `pytest -k`. It is skipped when someone runs a single
    file, because the count is then genuinely unknown rather than wrong.
    """
    collected = request.session.testscollected
    if collected < 50:
        pytest.skip(f"partial run ({collected} collected) - cannot judge the claim")

    # Only suite-scale claims. ARCHITECTURE.md legitimately says "an adversarial
    # review wrote 27 tests against the policy engine" - a true statement about
    # one file, which this test cannot distinguish from a stale total except by
    # magnitude. Flagging it would have meant editing a correct document to
    # satisfy a careless assertion, which is the wrong direction to resolve a
    # disagreement between a test and the truth.
    suite_scale = 100

    wrong: list[str] = []
    for name, text in doc_texts().items():
        for match in re.finditer(r"(\d{2,4})\s+tests\b", text):
            claimed = int(match.group(1))
            if claimed >= suite_scale and claimed != collected:
                line = text[: match.start()].count("\n") + 1
                wrong.append(f"{name}:{line} claims {claimed}, suite has {collected}")

    assert not wrong, (
        "documentation quotes a stale test count:\n  " + "\n  ".join(wrong)
    )


# ---------------------------------------------------------------------------
# Domain counts
# ---------------------------------------------------------------------------


def test_the_reason_code_count_is_right():
    from recoup.taxonomy import all_codes

    n = len(all_codes())
    wrong = [
        f"{name}: says {m.group(1)} reason codes, taxonomy has {n}"
        for name, text in doc_texts().items()
        for m in re.finditer(r"(\d+|[Ss]ixteen)\s+(?:failure\s+)?reason codes", text)
        if (16 if m.group(1).lower() == "sixteen" else int(m.group(1))) != n
    ]
    assert not wrong, "\n  ".join(wrong)


def test_the_bounds_and_checks_counts_are_right():
    """Thirteen bounds, fourteen checks. The extra one is an input-validation gate.

    Both numbers appear in the docs and in the dashboard chrome, and they are
    one apart, which is exactly the kind of pair that drifts into agreeing with
    each other and disagreeing with the code.
    """
    from recoup.policy.rules import RULES

    checks = len(RULES)
    bounds = checks - 1

    wrong: list[str] = []
    for name, text in doc_texts().items():
        for m in re.finditer(r"(\d+|thirteen|fourteen)\s+bounds\b", text, re.I):
            word = m.group(1).lower()
            claimed = {"thirteen": 13, "fourteen": 14}.get(word) or int(word)
            if claimed != bounds:
                wrong.append(f"{name}: says {claimed} bounds, RULES implies {bounds}")
        for m in re.finditer(r"(\d+|thirteen|fourteen)\s+checks\b", text, re.I):
            word = m.group(1).lower()
            claimed = {"thirteen": 13, "fourteen": 14}.get(word) or int(word)
            if claimed != checks:
                wrong.append(f"{name}: says {claimed} checks, RULES has {checks}")

    assert not wrong, "\n  ".join(sorted(set(wrong)))


def test_the_schema_table_count_is_right():
    from recoup.db import Base

    n = len(Base.metadata.tables)
    wrong = [
        f"{name}: says {m.group(1)} tables, schema has {n}"
        for name, text in doc_texts().items()
        for m in re.finditer(r"(\d+)\s+tables\b", text)
        if int(m.group(1)) != n
    ]
    assert not wrong, "\n  ".join(wrong)


def test_the_holdout_fraction_is_right():
    """The 30% holdout is quoted everywhere and is the basis of every result."""
    from recoup.seed.generate import CONTROL_FRACTION

    pct = round(CONTROL_FRACTION * 100)
    wrong = [
        f"{name}: says {m.group(1)}% holdout, generator uses {pct}%"
        for name, text in doc_texts().items()
        for m in re.finditer(r"(\d+)%\s+of events are (?:randomly )?assigned", text)
        if int(m.group(1)) != pct
    ]
    assert not wrong, "\n  ".join(wrong)
