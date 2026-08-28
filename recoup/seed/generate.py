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
from recoup.taxonomy import codes_for_rail

ORACLE_PATH = PROJECT_ROOT / "data" / "oracle.json"

# Reason-code prevalence, as weights rather than a flat distribution.
#
# Rail-scoped, because the earlier version was not: it drew a reason and a rail
# independently, so 46% of the traffic was UPI and a quarter of failed UPI
# payments came back as card_expired or incorrect_cvv - failures that cannot
# happen on that rail. A Razorpay reviewer would notice that in the seed data
# before reading a line of the recovery logic.
#
# Weights are relative prevalence across all rails; the generator filters to the
# codes possible on the chosen rail and renormalises, so UPI traffic draws UPI
# failures and cards draw card failures without maintaining a table per rail.
#
# The shape reflects Indian checkout: liquidity and intent dominate, credential
# slips (OTP and PIN) are the next tier, infrastructure faults are a steady
# background, and risk declines are rare but must appear because mishandling
# them is the most expensive mistake available.
FAILURE_WEIGHTS: dict[str, float] = {
    # liquidity and intent
    "insufficient_funds": 0.155,
    "payment_cancelled": 0.140,
    # credential slips - customer present, cheapest to recover
    "incorrect_otp": 0.080,
    "incorrect_pin": 0.075,
    "incorrect_cvv": 0.035,
    "otp_expired": 0.030,
    "authentication_failed": 0.070,
    # UPI-specific friction, the single largest rail in India
    "payment_collect_request_expired": 0.055,
    "invalid_vpa": 0.030,
    "vpa_resolution_failed": 0.020,
    "transaction_frequency_limit_exceeded": 0.018,
    "transaction_on_vpa_restricted": 0.010,
    "psp_not_available": 0.028,
    "upi_app_technical_error": 0.030,
    "pin_attempts_exceeded": 0.012,
    # limits
    "transaction_daily_limit_exceeded": 0.040,
    "transaction_limit_exceeded": 0.025,
    "credit_limit_exceeded": 0.018,
    # infrastructure
    "gateway_technical_error": 0.045,
    "bank_not_available": 0.038,
    "issuer_technical_error": 0.025,
    "payment_declined_due_to_high_traffic": 0.020,
    "payment_timed_out": 0.035,
    # instrument unusable
    "card_declined": 0.045,
    "card_expired": 0.028,
    "debit_instrument_blocked": 0.012,
    "international_transaction_not_allowed": 0.010,
    "otp_attempts_exceeded": 0.014,
    # never retried - rare, and the most expensive to get wrong
    "payment_risk_check_failed": 0.014,
    "payment_amount_tampered": 0.003,
    "compliance_violation": 0.002,
}

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



def _failure_for_rail(rng: random.Random, rail: str) -> str:
    """Draw a failure reason that is physically possible on `rail`.

    Filtering and renormalising per rail is what keeps the dataset honest. Drawing
    reason and rail independently produced UPI payments failing with card_expired,
    which is not a rare edge case in the data - it was roughly a quarter of failed
    UPI events, and UPI is 46% of the traffic.
    """
    allowed = codes_for_rail(rail)
    weights = [(c, FAILURE_WEIGHTS[c]) for c in allowed if c in FAILURE_WEIGHTS]
    if not weights:
        # No modelled failure for this rail. Better to fall back to a rail-agnostic
        # reason than to invent one the taxonomy would reject.
        return "payment_cancelled"
    return _weighted(rng, weights)


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
            # Rail first, then a reason that can actually occur on it.
            rail = cust.preferred_rail if rng.random() < 0.7 else _weighted(rng, RAIL_MIX)
            reason = _failure_for_rail(rng, rail)
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
