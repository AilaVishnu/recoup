"""Synthetic merchant generator.

Produces a customer base and a stream of revenue-at-risk events with a reason-code
mix drawn to resemble Indian online checkout traffic. Deterministic given a seed.

The latent recovery propensities live in data/oracle.json - written here, read
only by recoup/eval. The pipeline never sees them; see recoup/seed/world.py.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone

from faker import Faker

from recoup.config import PROJECT_ROOT
from recoup.db import Cohort, Customer, EventKind, RevenueEvent, get_session, init_db
from recoup.seed.world import organic_recovery_probability

ORACLE_PATH = PROJECT_ROOT / "data" / "oracle.json"

# Reason-code mix. Roughly mirrors published Indian e-commerce decline
# distributions: liquidity and intent dominate, hard technical failures are the
# long tail, and fraud declines are rare but must be represented because
# mishandling them is the most expensive mistake available.
FAILURE_MIX: list[tuple[str, float]] = [
    ("insufficient_funds", 0.18),
    ("payment_cancelled", 0.16),
    ("incorrect_otp", 0.13),
    ("authentication_failed", 0.10),
    ("gateway_technical_error", 0.08),
    ("card_declined", 0.08),
    ("issuer_down", 0.06),
    ("payment_timeout", 0.06),
    ("invalid_cvv", 0.05),
    ("card_expired", 0.04),
    ("payment_limit_exceeded", 0.03),
    ("fraud_suspected", 0.015),
    ("international_transaction_not_allowed", 0.010),
    ("transaction_not_permitted", 0.005),
]

KIND_MIX: list[tuple[EventKind, float]] = [
    (EventKind.PAYMENT_FAILED, 0.60),
    (EventKind.CHECKOUT_ABANDONED, 0.30),
    (EventKind.INVOICE_OVERDUE, 0.10),
]

RAIL_MIX: list[tuple[str, float]] = [
    ("upi", 0.46),
    ("card", 0.34),
    ("netbanking", 0.12),
    ("wallet", 0.08),
]

CONTROL_FRACTION = 0.30
"""Share of events held out from execution.

30% is larger than a mature system would run, and deliberately so: this dataset
is a few hundred events, and a 10% holdout would leave the incremental-lift
estimate too noisy to say anything. The cost of a wide holdout is forgone
recovery; the cost of a narrow one is not knowing whether you recovered
anything. At this sample size the second is worse.
"""

RECOVERY_WINDOW_DAYS = 7

SIMULATION_EPOCH = datetime(2026, 8, 1, 12, 0)
"""The instant the synthetic merchant's clock is anchored to. Fixed, not now().

This used to read datetime.now(), which quietly broke the reproducibility the
rest of the project claims. Event timestamps are placed relative to the anchor,
so with a wall-clock anchor the *hour of day* every event lands on moves with
when you happen to run the seeder - and hour of day decides quiet-hours
deferral, which decides which actions fire, which decides the outcome. Two runs
of seed 42 hours apart produced different headline numbers, while
recoup/pipeline.py claimed in its own docstring that "the same seed produces the
same timestamps, so two runs of the report are comparable".

A fixed epoch makes that claim true. The consequence is that generated events
are always dated around this date rather than around today, which is the right
trade: a demo dataset that reads "13 Aug 2026" and reproduces exactly is worth
more than one that reads "yesterday" and does not.

Everything downstream stays relative to the event's own timestamp, so nothing
here depends on the epoch being close to the present.
"""



def _weighted(rng: random.Random, choices: list[tuple]) -> object:
    population = [c[0] for c in choices]
    weights = [c[1] for c in choices]
    return rng.choices(population, weights=weights, k=1)[0]


def _amount_paise(rng: random.Random, kind: EventKind) -> int:
    """Log-normal-ish order values. B2B receivables are an order larger."""
    if kind is EventKind.INVOICE_OVERDUE:
        rupees = rng.lognormvariate(10.2, 0.9)  # ~Rs 27k median
        return int(min(max(rupees, 5_000), 500_000) * 100)
    rupees = rng.lognormvariate(7.1, 1.0)  # ~Rs 1.2k median, long tail
    return int(min(max(rupees, 99), 60_000) * 100)


def generate(
    n_customers: int = 220,
    n_events: int = 600,
    seed: int | None = None,
    drop: bool = True,
) -> dict:
    """Build a synthetic merchant. Returns a summary dict."""
    from recoup.config import get_settings

    seed = seed if seed is not None else get_settings().seed
    rng = random.Random(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    init_db(drop=drop)
    session = get_session()
    now = SIMULATION_EPOCH

    # ---- Customers -------------------------------------------------------
    customers: list[Customer] = []
    for _ in range(n_customers):
        successes = rng.randint(0, 40)
        failures = rng.randint(0, max(1, successes // 2 + 2))
        recoveries = rng.randint(0, max(1, failures))
        cust = Customer(
            id=f"cust_{uuid.UUID(int=rng.getrandbits(128)).hex[:14]}",
            name=fake.name(),
            email=fake.email(),
            contact=f"+91{rng.randint(6000000000, 9999999999)}",
            created_at=now - timedelta(days=rng.randint(1, 900)),
            prior_success_count=successes,
            prior_failure_count=failures,
            prior_recovery_count=recoveries,
            lifetime_value_paise=int(successes * rng.lognormvariate(7.0, 0.8)) * 100,
            preferred_rail=_weighted(rng, RAIL_MIX),
        )
        customers.append(cust)
    session.add_all(customers)
    session.flush()

    # ---- Events ----------------------------------------------------------
    oracle: dict[str, dict] = {}
    events: list[RevenueEvent] = []

    for _ in range(n_events):
        cust = rng.choice(customers)
        kind: EventKind = _weighted(rng, KIND_MIX)

        if kind is EventKind.PAYMENT_FAILED:
            reason = _weighted(rng, FAILURE_MIX)
            rail = cust.preferred_rail if rng.random() < 0.7 else _weighted(rng, RAIL_MIX)
        elif kind is EventKind.CHECKOUT_ABANDONED:
            reason = "checkout_abandoned"
            rail = _weighted(rng, RAIL_MIX)
        else:
            reason = "invoice_overdue"
            rail = "netbanking"

        amount = _amount_paise(rng, kind)

        # Events land across the last 30 days, but nothing inside the recovery
        # window - every event in the dataset has had time to resolve, so the
        # eval never has to guess about unfinished business.
        occurred = now - timedelta(
            days=rng.uniform(RECOVERY_WINDOW_DAYS, 30),
            hours=rng.uniform(0, 24),
        )

        eid = f"evt_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]}"
        cohort = Cohort.CONTROL if rng.random() < CONTROL_FRACTION else Cohort.TREATMENT

        events.append(
            RevenueEvent(
                id=eid,
                kind=kind,
                customer_id=cust.id,
                amount_paise=amount,
                currency="INR",
                occurred_at=occurred,
                reason_code=reason,
                rail=rail,
                cohort=cohort,
                attempt_no=1 if rng.random() < 0.82 else 2,
                extra={
                    "checkout_stage": (
                        "payment" if kind is EventKind.PAYMENT_FAILED else "cart"
                    ),
                    "items": rng.randint(1, 6),
                    "is_repeat_customer": cust.prior_success_count > 3,
                },
            )
        )

        oracle[eid] = {
            "organic_p": round(
                organic_recovery_probability(
                    reason_code=reason,
                    amount_paise=amount,
                    prior_recovery_count=cust.prior_recovery_count,
                    prior_failure_count=cust.prior_failure_count,
                    lifetime_value_paise=cust.lifetime_value_paise,
                    rng=rng,
                ),
                4,
            ),
            # Drawn now and frozen, so that treated and control arms face the same
            # luck. Resolving an outcome is a comparison against this number, not
            # a fresh coin flip - otherwise the measured lift would be dominated
            # by sampling noise rather than by the decisions under test.
            "roll": round(rng.random(), 6),
            "reason_code": reason,
            "amount_paise": amount,
        }

    session.add_all(events)
    session.commit()

    ORACLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ORACLE_PATH.write_text(json.dumps(oracle, indent=1), encoding="utf-8")

    treated = sum(1 for e in events if e.cohort is Cohort.TREATMENT)
    at_risk = sum(e.amount_paise for e in events)
    summary = {
        "seed": seed,
        "customers": len(customers),
        "events": len(events),
        "treatment": treated,
        "control": len(events) - treated,
        "value_at_risk_paise": at_risk,
        "value_at_risk_inr": round(at_risk / 100, 2),
        "oracle_path": str(ORACLE_PATH),
    }
    session.close()
    return summary
