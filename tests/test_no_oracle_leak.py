"""Structural guarantee: the pipeline cannot see the answers.

Recoup's whole claim is that its decisions are made from observable signals. If
any pipeline module could reach the latent recovery propensities in
recoup/seed/world.py or data/oracle.json, every metric downstream would be
worthless and the leak would be invisible in the output - the numbers would just
quietly be too good.

So it is enforced here rather than trusted. These tests read the source of every
pipeline module and fail on any path to the oracle, direct or transitive.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PACKAGES = ("detect", "agent", "policy", "execute", "api")

FORBIDDEN_MODULES = {"recoup.seed.world", "recoup.seed.generate", "recoup.seed"}
FORBIDDEN_NAMES = {
    "organic_recovery_probability",
    "treated_recovery_probability",
    "LiftAssumptions",
    "ORACLE_PATH",
}


def pipeline_files() -> list[Path]:
    files: list[Path] = []
    for pkg in PIPELINE_PACKAGES:
        files.extend((PROJECT_ROOT / "recoup" / pkg).rglob("*.py"))
    return [f for f in files if f.name != "__init__.py"]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


@pytest.mark.parametrize("path", pipeline_files(), ids=lambda p: p.name)
def test_pipeline_module_does_not_import_the_oracle(path: Path) -> None:
    imports = imported_modules(path)
    leaked = {m for m in imports if any(m.startswith(f) for f in FORBIDDEN_MODULES)}
    assert not leaked, (
        f"{path.relative_to(PROJECT_ROOT)} imports the simulator: {sorted(leaked)}. "
        "Pipeline code must decide from observable signals only."
    )


@pytest.mark.parametrize("path", pipeline_files(), ids=lambda p: p.name)
def test_pipeline_module_does_not_name_oracle_symbols(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    hits = [n for n in FORBIDDEN_NAMES if n in source]
    assert not hits, (
        f"{path.relative_to(PROJECT_ROOT)} references oracle symbols {hits}."
    )


@pytest.mark.parametrize("path", pipeline_files(), ids=lambda p: p.name)
def test_pipeline_module_does_not_read_the_oracle_file(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert "oracle.json" not in source, (
        f"{path.relative_to(PROJECT_ROOT)} reads the ground-truth file directly."
    )


def test_the_guard_would_actually_catch_a_leak(tmp_path: Path) -> None:
    """A test that never fails proves nothing - verify the detector detects."""
    leaky = tmp_path / "leaky.py"
    leaky.write_text(
        "from recoup.seed.world import organic_recovery_probability\n",
        encoding="utf-8",
    )
    imports = imported_modules(leaky)
    assert any(m.startswith("recoup.seed") for m in imports)
