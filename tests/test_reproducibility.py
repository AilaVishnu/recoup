"""The same seed must produce the same dataset. This is load-bearing.

recoup/pipeline.py states that "the same seed produces the same timestamps, so
two runs of the report are comparable", the README tells a reviewer the headline
is reproducible from a clean checkout, and scripts/robustness.py compares five
datasets on the assumption that the only thing differing between them is the
seed. All three claims rest on this file passing.

They were all false for a while, and invisibly so. The generator anchored event
timestamps to datetime.now(), so the hour of day each event landed on moved with
when the seeder happened to run - and hour of day decides quiet-hours deferral,
which decides which actions fire, which decides the outcome. Two runs of seed 42
a few hours apart produced different headline numbers while every document
claimed they could not.

Nothing failed when that was true. The numbers simply moved, and the only way to
notice was to run the same seed twice and compare, which nothing did.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from recoup.seed.generate import SIMULATION_EPOCH


def _events(session):
    from recoup.db import RevenueEvent

    return sorted(
        (
            (e.id, e.occurred_at.isoformat(), e.reason_code, e.amount_paise, e.cohort.value)
            for e in session.query(RevenueEvent).all()
        )
    )


def _generate_into(tmp_path, monkeypatch, seed: int, tag: str):
    """Seed one dataset into its own database and return its events + oracle."""
    import json
    import sys

    db = tmp_path / f"{tag}.db"
    oracle = tmp_path / f"{tag}.json"
    monkeypatch.setenv("RECOUP_DB_URL", f"sqlite:///{db.as_posix()}")

    for mod in [m for m in list(sys.modules) if m.startswith("recoup")]:
        del sys.modules[mod]

    import recoup.db as db_mod
    from recoup.config import get_settings
    from recoup.seed import generate as gen

    get_settings.cache_clear()
    gen.ORACLE_PATH = oracle
    gen.generate(n_customers=25, n_events=60, seed=seed, drop=True)

    session = db_mod.get_session()
    events = _events(session)
    session.close()
    return events, json.loads(oracle.read_text(encoding="utf-8"))


def test_the_epoch_is_fixed_rather_than_the_wall_clock():
    """The single line that made everything else reproducible.

    Asserted directly, because the failure it prevents is silent: a wall-clock
    anchor produces a perfectly valid dataset every time, just not the same one.
    """
    assert isinstance(SIMULATION_EPOCH, datetime)
    assert SIMULATION_EPOCH.tzinfo is None, "naive UTC, like every stored timestamp"
    assert SIMULATION_EPOCH < datetime(2030, 1, 1), "a fixed instant, not a moving one"


def test_the_same_seed_produces_the_same_events(tmp_path, monkeypatch):
    a_events, a_oracle = _generate_into(tmp_path, monkeypatch, 42, "a")
    b_events, b_oracle = _generate_into(tmp_path, monkeypatch, 42, "b")

    assert a_events == b_events, (
        "two runs of seed 42 produced different events - every reproducibility "
        "claim in the README and in pipeline.py depends on this"
    )
    assert a_oracle == b_oracle, "the answer key moved between runs of the same seed"


def test_the_timestamps_specifically_are_stable(tmp_path, monkeypatch):
    """Timestamps get their own test because they were the thing that broke.

    Event ids and amounts come straight from the seeded RNG and were always
    stable; only the timestamps were anchored to the clock. A test comparing
    whole rows would have passed on ids alone if the tuple ever stopped carrying
    occurred_at, so the hours are checked on their own.
    """
    a_events, _ = _generate_into(tmp_path, monkeypatch, 7, "ts_a")
    b_events, _ = _generate_into(tmp_path, monkeypatch, 7, "ts_b")

    a_hours = [datetime.fromisoformat(e[1]).hour for e in a_events]
    b_hours = [datetime.fromisoformat(e[1]).hour for e in b_events]
    assert a_hours == b_hours, (
        "hour-of-day drifted between runs. That decides quiet-hours deferral, "
        "which decides which actions fire, which decides the outcome."
    )


def test_different_seeds_still_produce_different_data(tmp_path, monkeypatch):
    """A fixed epoch must not have collapsed the datasets into one.

    Without this, the previous tests would also pass if generate() had started
    ignoring its seed entirely - and scripts/robustness.py would then be
    comparing five copies of the same data and calling the agreement robustness.
    """
    a_events, _ = _generate_into(tmp_path, monkeypatch, 1, "s1")
    b_events, _ = _generate_into(tmp_path, monkeypatch, 2, "s2")
    assert a_events != b_events, "two different seeds produced identical datasets"
