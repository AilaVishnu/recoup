"""The eval is the claim. It gets tested harder than anything it measures.

Every test here names a specific way this project could report a number that
flatters it - crediting a recovery that would have happened anyway, letting a
dry run manufacture lift, quietly moving the baseline, or printing a point
estimate from thirty events as though it were a finding - and asserts that the
harness does not.

The frozen roll is the load-bearing idea and gets three tests of its own: the
control arm must be identical across repeated runs, and identical across
pessimistic and optimistic assumptions, because a baseline that drifts with a
parameter makes every lift figure downstream meaningless.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta

import pytest
from rich.console import Console
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from recoup.db import (
    ActionRun,
    ActionStatus,
    ActionType,
    Attribution,
    Base,
    Cohort,
    Customer,
    Decision,
    DecisionSource,
    EventKind,
    Outcome,
    RevenueEvent,
)
from recoup.eval.metrics import MIN_ARM_N, build_segments, compute, difference_ci
from recoup.eval.report import render, run_sensitivity_sweep, write_json
from recoup.eval.resolve import (
    ExecutedAction,
    OracleMismatch,
    resolve_all,
    resolve_one,
)
from recoup.seed.world import LiftAssumptions

OCCURRED = datetime(2026, 3, 1, 9, 0)
CUSTOMER_ID = "cust_eval_test"

# incorrect_otp: RETRY_NOW, no required wait, never incentive-eligible. With the
# default assumptions a correct retry inside the session window adds 0.18, so an
# organic 0.30 becomes a treated 0.48 and the interesting rolls sit between them.
ORGANIC = 0.30
TREATED = 0.48


@pytest.fixture
def session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, future=True)()
    s.add(
        Customer(
            id=CUSTOMER_ID,
            name="Test Customer",
            email="t@example.com",
            contact="+919000000000",
            created_at=OCCURRED - timedelta(days=400),
        )
    )
    s.commit()
    yield s
    s.close()


def retry(hours: float = 0.5, **kwargs) -> ExecutedAction:
    return ExecutedAction(
        action_type=ActionType.RETRY_PAYMENT.value,
        executed_at=OCCURRED + timedelta(hours=hours),
        **kwargs,
    )


def resolve(
    *,
    roll: float,
    cohort: Cohort = Cohort.TREATMENT,
    action: ExecutedAction | None = None,
    reason_code: str = "incorrect_otp",
    organic_p: float = ORGANIC,
    amount_paise: int = 1_000_00,
    assumptions: LiftAssumptions | None = None,
):
    return resolve_one(
        event_id="evt_test",
        cohort=cohort,
        event_kind=EventKind.PAYMENT_FAILED.value,
        reason_code=reason_code,
        amount_paise=amount_paise,
        occurred_at=OCCURRED,
        organic_p=organic_p,
        roll=roll,
        action=action,
        assumptions=assumptions or LiftAssumptions(),
    )


def seed_event(
    session,
    oracle: dict,
    *,
    eid: str,
    roll: float,
    organic_p: float = ORGANIC,
    cohort: Cohort = Cohort.TREATMENT,
    reason_code: str = "incorrect_otp",
    amount_paise: int = 1_000_00,
    kind: EventKind = EventKind.PAYMENT_FAILED,
    action_type: ActionType | None = None,
    status: ActionStatus = ActionStatus.SENT,
    incentive_paise: int = 0,
    channel_cost_paise: int = 0,
    hours: float = 0.5,
) -> None:
    session.add(
        RevenueEvent(
            id=eid,
            kind=kind,
            customer_id=CUSTOMER_ID,
            amount_paise=amount_paise,
            occurred_at=OCCURRED,
            reason_code=reason_code,
            rail="card",
            cohort=cohort,
            extra={},
        )
    )
    oracle[eid] = {
        "organic_p": organic_p,
        "roll": roll,
        "reason_code": reason_code,
        "amount_paise": amount_paise,
    }
    if action_type is not None:
        decision = Decision(
            event_id=eid,
            source=DecisionSource.RULES,
            action_type=action_type,
            params={},
            rationale="test",
        )
        session.add(decision)
        session.flush()
        session.add(
            ActionRun(
                decision_id=decision.id,
                executed_at=OCCURRED + timedelta(hours=hours),
                action_type=action_type,
                status=status,
                incentive_paise=incentive_paise,
                channel_cost_paise=channel_cost_paise,
            )
        )
    session.commit()


# --- attribution: the line between "we recovered it" and "it came back" -----


def test_recovery_only_possible_under_the_action_is_credited_to_the_agent():
    """roll sits above organic and below treated: no action, no recovery."""
    o = resolve(roll=0.40, action=retry())
    assert o.recovered
    assert o.attribution is Attribution.AGENT
    assert o.agent_caused
    # The counterfactual is checkable directly - same event, same luck, no action.
    assert not resolve(roll=0.40, action=None).recovered


def test_recovery_that_would_have_happened_anyway_is_not_credited():
    """The single most common overclaim in recovery tooling."""
    o = resolve(roll=0.20, action=retry())
    assert o.recovered
    assert o.attribution is Attribution.ORGANIC
    assert not o.agent_caused
    # It recovers with no action too, which is the whole point.
    assert resolve(roll=0.20, action=None).recovered


def test_a_lost_event_is_never_credited_to_the_agent():
    o = resolve(roll=0.60, action=retry())
    assert not o.recovered
    assert o.recovered_paise == 0
    assert not o.agent_caused


def test_the_control_arm_can_never_be_credited_to_the_agent():
    """Even handed an executed action, a holdout event resolves as untouched."""
    for roll in (0.10, 0.20, 0.40, 0.60, 0.90):
        plain = resolve(roll=roll, cohort=Cohort.CONTROL)
        assert plain.attribution is not Attribution.AGENT
        assert not plain.treated

        contaminated = resolve(roll=roll, cohort=Cohort.CONTROL, action=retry())
        assert contaminated.attribution is not Attribution.AGENT
        assert contaminated.recovered == (roll < ORGANIC)
        assert "contaminated" in contaminated.note


def test_an_action_that_cannot_be_dated_earns_neither_credit_nor_blame():
    """Fail closed: an unclaimable recovery is better than a fabricated one."""
    before = ExecutedAction(
        action_type=ActionType.RETRY_PAYMENT.value,
        executed_at=OCCURRED - timedelta(hours=2),
    )
    o = resolve(roll=0.20, action=before)
    assert o.recovered
    assert o.attribution is Attribution.UNCLEAR
    assert not o.agent_caused
    assert not o.treated


def test_acting_can_destroy_a_recovery_and_that_is_reported():
    """Discounting a technical failure earns the wrong-action penalty.

    bank_not_available would have recovered on its own; a nudge is not a retry, so the
    world model pushes the probability below the roll and the event is lost.
    """
    nudge = ExecutedAction(
        action_type=ActionType.NUDGE.value,
        executed_at=OCCURRED + timedelta(hours=3),
    )
    o = resolve(roll=0.29, organic_p=0.30, reason_code="bank_not_available", action=nudge)
    assert not o.recovered
    assert o.harmed
    assert resolve(roll=0.29, organic_p=0.30, reason_code="bank_not_available").recovered


# --- the frozen roll -------------------------------------------------------


def test_resolution_is_a_comparison_not_a_draw(session):
    """Two runs, same answers. Nothing here samples."""
    oracle: dict = {}
    for i in range(40):
        seed_event(
            session,
            oracle,
            eid=f"evt_{i}",
            roll=i / 40,
            cohort=Cohort.CONTROL if i % 3 == 0 else Cohort.TREATMENT,
        )

    def fingerprint(resolution):
        return [(o.event_id, o.recovered, o.attribution) for o in resolution.outcomes]

    first = resolve_all(session, oracle=oracle, persist=False)
    second = resolve_all(session, oracle=oracle, persist=False)
    assert fingerprint(first) == fingerprint(second)


def test_the_baseline_does_not_move_when_the_assumptions_do(session):
    """A control arm that drifts with a lift parameter is not a baseline.

    The sensitivity sweep only means something if the thing being compared
    against holds still across it.
    """
    oracle: dict = {}
    for i in range(40):
        seed_event(session, oracle, eid=f"evt_{i}", roll=i / 40, cohort=Cohort.CONTROL)

    outcomes = {}
    for label, assumptions in (
        ("pessimistic", LiftAssumptions.pessimistic()),
        ("default", LiftAssumptions()),
        ("optimistic", LiftAssumptions.optimistic()),
    ):
        resolution = resolve_all(
            session, assumptions=assumptions, oracle=oracle, persist=False
        )
        outcomes[label] = [(o.event_id, o.recovered) for o in resolution.outcomes]

    assert outcomes["pessimistic"] == outcomes["default"] == outcomes["optimistic"]


# --- what does not count as an intervention --------------------------------


def test_dry_run_is_not_a_switch_that_moves_the_numbers(session):
    """RECOUP_DRY_RUN must not change a result. It is a transport flag, not a dial.

    The earlier version of this rule excluded dry-run actions from grading, on
    the reasoning that a simulated dispatch has not earned any credit. It sounds
    right and it was not, for a reason the flag itself exposes: the outbox never
    messages a real customer in ANY mode, so nudges were graded always while
    Payment Links were graded only with the flag off. Turning dry-run off would
    therefore have moved every number in the report - which is precisely what
    this test's name forbids.

    Grading both identically is the stronger guarantee. What produces an outcome
    is recoup/seed/world.py, which has no opinion about whether an HTTP request
    left the machine, so the flag genuinely cannot reach the result. The mode is
    disclosed in the report header instead of being smuggled into the maths.
    """
    oracle: dict = {}
    # Identical events, identical luck, identical action. Only the transport differs.
    seed_event(
        session,
        oracle,
        eid="evt_dry",
        roll=0.40,
        action_type=ActionType.RETRY_PAYMENT,
        status=ActionStatus.SKIPPED_DRY_RUN,
    )
    seed_event(
        session,
        oracle,
        eid="evt_live",
        roll=0.40,
        action_type=ActionType.RETRY_PAYMENT,
        status=ActionStatus.SENT,
    )

    resolution = resolve_all(session, oracle=oracle, persist=False)
    by_id = {o.event_id: o for o in resolution.outcomes}
    dry, live = by_id["evt_dry"], by_id["evt_live"]

    assert dry.treated == live.treated
    assert dry.recovered == live.recovered
    assert dry.attribution == live.attribution
    assert dry.recovered_paise == live.recovered_paise

    # Still counted, so the header can say which mode produced the number.
    assert resolution.integrity["dry_run_actions"] == 1


def test_a_dry_run_action_is_graded_not_discarded(session):
    """The specific regression: 190 of 384 real actions once vanished from the eval.

    A full pipeline run in dry mode dispatched 134 retries and 56 Payment Links
    that the grader dropped on the floor, while 172 nudges counted. Every retry
    strategy consequently reported zero incremental recovery, and the report read
    as though only persuasion worked.
    """
    oracle: dict = {}
    seed_event(
        session,
        oracle,
        eid="evt_dry_graded",
        roll=0.40,
        action_type=ActionType.RETRY_PAYMENT,
        status=ActionStatus.SKIPPED_DRY_RUN,
    )
    resolution = resolve_all(session, oracle=oracle, persist=False)
    (outcome,) = resolution.outcomes

    assert outcome.treated, "a dry-run action must still be graded as an action"


def test_a_failed_send_earns_nothing(session):
    oracle: dict = {}
    seed_event(
        session,
        oracle,
        eid="evt_failed",
        roll=0.40,
        action_type=ActionType.RETRY_PAYMENT,
        status=ActionStatus.FAILED,
    )
    resolution = resolve_all(session, oracle=oracle, persist=False)
    assert not resolution.outcomes[0].treated
    assert resolution.integrity["failed_actions"] == 1


def test_escalating_to_a_human_is_not_an_intervention(session):
    """A recorded decision to do nothing is not a treatment."""
    oracle: dict = {}
    seed_event(
        session,
        oracle,
        eid="evt_escalated",
        roll=0.40,
        action_type=ActionType.ESCALATE_TO_HUMAN,
    )
    resolution = resolve_all(session, oracle=oracle, persist=False)
    assert not resolution.outcomes[0].treated
    assert resolution.integrity["inert_actions"] == 1


# --- cannibalisation: the false-positive cost ------------------------------


def _cannibalising_event(session, oracle, eid: str, roll: float) -> None:
    """A discount on an intent failure, landing on a customer who would pay anyway.

    payment_cancelled is the one bucket the taxonomy lets money into, which is
    exactly why it is the one bucket where money can be wasted.
    """
    seed_event(
        session,
        oracle,
        eid=eid,
        roll=roll,
        organic_p=0.20,
        reason_code="payment_cancelled",
        amount_paise=1_000_00,
        action_type=ActionType.NUDGE_WITH_INCENTIVE,
        incentive_paise=100_00,
        channel_cost_paise=50,
        hours=5.0,
    )


def test_discounts_paid_to_customers_who_would_have_paid_anyway_are_counted(session):
    oracle: dict = {}
    _cannibalising_event(session, oracle, "evt_cannibal", roll=0.10)

    resolution = resolve_all(session, oracle=oracle, persist=False)
    (outcome,) = resolution.outcomes

    assert outcome.recovered
    assert outcome.attribution is Attribution.ORGANIC
    assert outcome.cannibalised_paise == 100_00

    overall = build_segments(resolution.outcomes)["overall"]
    assert overall.cannibalised_paise == 100_00
    assert overall.cannibalised_n == 1
    assert overall.incremental_recovered_paise == 0
    # Pure loss: a redeemed discount plus the message, against nothing gained.
    assert overall.cost_paise == 100_00 + 50
    assert overall.net_paise == -(100_00 + 50)


def test_an_unredeemed_discount_costs_nothing_but_the_message(session):
    """Committed spend is the budget question; redeemed spend is the P&L one."""
    oracle: dict = {}
    _cannibalising_event(session, oracle, "evt_unredeemed", roll=0.95)

    resolution = resolve_all(session, oracle=oracle, persist=False)
    overall = build_segments(resolution.outcomes)["overall"]

    assert not resolution.outcomes[0].recovered
    assert overall.incentive_committed_paise == 100_00
    assert overall.incentive_realised_paise == 0
    assert overall.cannibalised_paise == 0
    assert overall.cost_paise == 50


# --- inference that knows its own limits -----------------------------------


def test_thin_segments_are_flagged_rather_than_quietly_reported():
    thin = difference_ci(recovered_t=5, n_t=10, recovered_c=4, n_c=10)
    assert not thin.reliable
    assert "UNRELIABLE" in thin.caveat

    thick = difference_ci(recovered_t=20, n_t=60, recovered_c=15, n_c=60)
    assert thick.reliable
    assert thick.caveat == ""


def test_an_empty_arm_yields_no_comparison_rather_than_a_zero():
    """Zero recoveries out of zero events is not a 0% recovery rate."""
    lift = difference_ci(recovered_t=3, n_t=10, recovered_c=0, n_c=0)
    assert lift.absolute is None
    assert lift.ci_low is None
    assert not lift.reliable


def test_the_interval_is_wide_enough_to_be_honest_at_the_threshold():
    """At the reliability floor the interval is still about +/-25 points.

    The flag is not a licence to believe the rows that clear it.
    """
    lift = difference_ci(
        recovered_t=int(0.4 * MIN_ARM_N),
        n_t=MIN_ARM_N,
        recovered_c=int(0.4 * MIN_ARM_N),
        n_c=MIN_ARM_N,
    )
    assert lift.reliable
    assert (lift.ci_high - lift.ci_low) > 0.4


def test_small_reason_code_segments_reach_the_unreliable_list(session):
    oracle: dict = {}
    for i in range(20):
        seed_event(
            session,
            oracle,
            eid=f"evt_otp_{i}",
            roll=i / 20,
            reason_code="incorrect_otp",
            cohort=Cohort.CONTROL if i % 2 else Cohort.TREATMENT,
        )
    resolution = resolve_all(session, oracle=oracle, persist=False)
    metrics = compute(session, resolution, seed=1)

    assert "reason_code:incorrect_otp" in metrics.unreliable_segments
    assert "overall:overall" in metrics.unreliable_segments


# --- gross vs incremental --------------------------------------------------


def test_gross_recovery_exceeds_incremental_recovery(session):
    """The gap between the two is the number this project exists to show."""
    oracle: dict = {}
    for i in range(30):
        seed_event(
            session,
            oracle,
            eid=f"evt_t_{i}",
            roll=i / 30,
            action_type=ActionType.RETRY_PAYMENT,
        )
    for i in range(30):
        seed_event(session, oracle, eid=f"evt_c_{i}", roll=i / 30, cohort=Cohort.CONTROL)

    overall = build_segments(resolve_all(session, oracle=oracle, persist=False).outcomes)[
        "overall"
    ]

    assert overall.gross_recovered_paise > overall.incremental_recovered_paise > 0
    assert overall.gross_recovery_rate > overall.control_recovery_rate
    assert overall.lift.absolute == pytest.approx(
        overall.gross_recovery_rate - overall.control_recovery_rate
    )


# --- refusing to grade the wrong thing -------------------------------------


def test_a_stale_answer_key_stops_the_run(session):
    """Grading against the wrong oracle produces numbers that look fine."""
    oracle: dict = {}
    seed_event(session, oracle, eid="evt_stale", roll=0.4)
    oracle["evt_stale"]["amount_paise"] = 99_99_999

    with pytest.raises(OracleMismatch):
        resolve_all(session, oracle=oracle, persist=False)


def test_an_empty_oracle_stops_the_run(session):
    oracle: dict = {}
    seed_event(session, oracle, eid="evt_orphan", roll=0.4)

    with pytest.raises(OracleMismatch):
        resolve_all(session, oracle={}, persist=False)


def test_contaminated_control_events_are_named_not_swallowed(session):
    oracle: dict = {}
    seed_event(
        session,
        oracle,
        eid="evt_leaky",
        roll=0.40,
        cohort=Cohort.CONTROL,
        action_type=ActionType.RETRY_PAYMENT,
    )
    resolution = resolve_all(session, oracle=oracle, persist=False)
    assert resolution.contaminated_control == ["evt_leaky"]
    assert resolution.integrity["contaminated_control"] == 1


# --- persistence and the audit trail ---------------------------------------


def test_persisted_outcomes_do_not_leak_the_simulator(session):
    """The Outcome table must not become a back-channel to the world model.

    Anything written here is readable by the pipeline, which is supposed to be
    blind to the latent probabilities.
    """
    oracle: dict = {}
    seed_event(
        session, oracle, eid="evt_persist", roll=0.40, action_type=ActionType.RETRY_PAYMENT
    )
    resolve_all(session, oracle=oracle, persist=True)

    row = session.query(Outcome).one()
    assert row.recovered
    assert row.attribution is Attribution.AGENT
    assert "0.3" not in row.note and "0.4" not in row.note


def test_re_resolving_replaces_outcomes_rather_than_accumulating(session):
    oracle: dict = {}
    seed_event(session, oracle, eid="evt_once", roll=0.40)
    resolve_all(session, oracle=oracle, persist=True)
    resolve_all(session, oracle=oracle, persist=True)

    assert session.query(Outcome).count() == 1


# --- the report ------------------------------------------------------------


def test_the_sweep_reports_a_range_and_notices_an_unstable_sign(session):
    oracle: dict = {}
    for i in range(30):
        seed_event(
            session,
            oracle,
            eid=f"evt_s_{i}",
            roll=i / 30,
            action_type=ActionType.RETRY_PAYMENT,
        )
    for i in range(30):
        seed_event(session, oracle, eid=f"evt_sc_{i}", roll=i / 30, cohort=Cohort.CONTROL)

    sweep = run_sensitivity_sweep(session, oracle=oracle)
    assert [p.label for p in sweep.points] == ["pessimistic", "default", "optimistic"]

    lo, hi = sweep.lift_span
    assert lo <= hi
    # Optimistic assumptions must never look worse than pessimistic ones; if they
    # do, the sweep is wired backwards and the range means nothing.
    assert (
        sweep.points[0].incremental_recovered_paise
        <= sweep.points[2].incremental_recovered_paise
    )
    assert isinstance(sweep.sign_is_stable, bool)


def test_the_report_renders_with_empty_arms_and_undefined_ratios(session, tmp_path):
    """Every None path in one pass: no control arm, no cost, nothing incremental.

    These are the cells that crash a formatter, and they turn up in the real
    dataset - payment_risk_check_failed lands entirely in one arm.
    """
    oracle: dict = {}
    seed_event(session, oracle, eid="evt_fraud", roll=0.9, reason_code="payment_risk_check_failed")
    seed_event(session, oracle, eid="evt_kind", roll=0.1, kind=EventKind.INVOICE_OVERDUE,
               reason_code="invoice_overdue")

    resolution = resolve_all(session, oracle=oracle, persist=False)
    metrics = compute(session, resolution, seed=7)
    assert metrics.overall.cost_per_incremental_rupee is None

    console = Console(file=io.StringIO(), width=100, no_color=True)
    render(metrics, run_sensitivity_sweep(session, oracle=oracle), console=console)
    out = console.file.getvalue()
    assert "gross" in out

    # Rich picks its own box glyphs and downgrades them to +--+ on a console that
    # cannot render them. Everything this project writes has to be ASCII already,
    # so strip the box-drawing block and nothing exotic should remain - a stray
    # ellipsis or rupee sign becomes mojibake in front of a reviewer.
    content = "".join(ch for ch in out if not 0x2500 <= ord(ch) <= 0x257F)
    assert content.isascii(), sorted({c for c in content if not c.isascii()})

    path = write_json(metrics, None, path=tmp_path / "eval_7.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"]["overall"]["incremental_recovered_paise"] == 0
    assert payload["units"].startswith("all money in integer paise")


def test_the_report_names_the_holdout_as_untouched_value(session):
    oracle: dict = {}
    seed_event(session, oracle, eid="evt_refused", roll=0.9, reason_code="payment_risk_check_failed")
    metrics = compute(
        session, resolve_all(session, oracle=oracle, persist=False), seed=3
    )

    assert metrics.suppression.events == 1
    assert metrics.suppression.value_paise == 1_000_00
    assert "payment_risk_check_failed" in metrics.suppression.by_reason
