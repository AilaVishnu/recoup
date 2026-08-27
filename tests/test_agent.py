"""The decision layer, tested where it can be wrong.

Three things this file is protecting:

1. **The split is real.** The claim "we did not call a model 600 times when a
   lookup table answers most of it" is only worth making if the routing rule
   actually sends the settled cases to the table. Every deterministic reason code
   is checked, and so is the property that routing ignores the cohort - a control
   event routed differently from a treatment event would corrupt the holdout.
2. **The model path is optional.** The whole pipeline runs with no API key.
   These tests pass in CI with none present, and they would fail if anything in
   the agent grew a hard dependency on a live endpoint.
3. **Bad model output does not become a Decision row.** Structured output
   guarantees shape, not sense; validation is what stops the audit trail filling
   with proposals that were never coherent.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from recoup.agent import brain, prompts, rules_engine
from recoup.agent.decide import decide
from recoup.agent.router import should_escalate_to_model
from recoup.db import (
    ActionType,
    Assessment,
    Base,
    Cohort,
    Customer,
    DecisionSource,
    EventKind,
    RevenueEvent,
)
from recoup.policy.rules import Bounds
from recoup.taxonomy import PROFILES, Strategy, all_codes, profile_for

NOW = datetime(2026, 3, 3, 8, 30)  # 14:00 IST on the 3rd
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def session():
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, future=True)()
    yield s
    s.close()


def make_customer(**overrides) -> Customer:
    base = dict(
        id="cust_test",
        name="Test Customer",
        email="test@example.com",
        contact="+919000000000",
        created_at=NOW - timedelta(days=400),
        prior_success_count=12,
        prior_failure_count=4,
        prior_recovery_count=1,
        lifetime_value_paise=40_000_00,
        preferred_rail="upi",
    )
    base.update(overrides)
    return Customer(**base)


def make_event(**overrides) -> RevenueEvent:
    base = dict(
        id="evt_test",
        kind=EventKind.PAYMENT_FAILED,
        customer_id="cust_test",
        amount_paise=1_500_00,
        currency="INR",
        occurred_at=NOW - timedelta(hours=6),
        reason_code="incorrect_otp",
        rail="card",
        cohort=Cohort.TREATMENT,
        attempt_no=1,
        extra={"checkout_stage": "payment", "items": 2},
    )
    base.update(overrides)
    return RevenueEvent(**base)


def make_assessment(event: RevenueEvent, **overrides) -> Assessment:
    profile = profile_for(event.reason_code)
    base = dict(
        event_id=event.id,
        features={
            "historical_recovery_rate": 0.25,
            "prior_failure_count": 4,
            "prior_success_count": 12,
            "customer_tenure_days": 400,
            "lifetime_value_inr": 40_000.0,
            "preferred_rail": "upi",
            "rail_is_preferred": False,
            "event_age_hours": 6.0,
            "occurred_hour_ist": 14,
            "occurred_day_of_month": 3,
            "checkout_stage": "payment",
            "basket_items": 2,
        },
        recoverability=0.4,
        expected_value_paise=int(0.4 * event.amount_paise),
        recommended_strategy=profile.strategy.value,
        earliest_action_at=NOW,
        scorer_version="v1/prior",
    )
    base.update(overrides)
    return Assessment(**base)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_settled_reason_codes_never_reach_the_model():
    """The load-bearing claim. A lookup answers these; a model would only be slower."""
    for code in ("incorrect_otp", "invalid_cvv", "issuer_down", "gateway_technical_error"):
        event = make_event(reason_code=code)
        use_llm, why = should_escalate_to_model(event, make_assessment(event))
        assert not use_llm, f"{code} was routed to the model: {why}"
        assert code in why


def test_fraud_declines_are_not_worth_a_model_call():
    event = make_event(reason_code="fraud_suspected")
    use_llm, why = should_escalate_to_model(
        event, make_assessment(event, recoverability=0.0, expected_value_paise=0)
    )
    assert not use_llm
    assert "do-not-retry" in why


def test_unknown_reason_codes_are_not_worth_a_model_call():
    """Guessing at an unclassified failure is the taxonomy's job to refuse, not the model's."""
    event = make_event(reason_code="some_new_code_from_2027")
    use_llm, _ = should_escalate_to_model(event, make_assessment(event))
    assert not use_llm


def test_incentive_decisions_reach_the_model_when_the_money_is_worth_deliberating():
    event = make_event(reason_code="payment_cancelled", amount_paise=8_000_00)
    use_llm, why = should_escalate_to_model(event, make_assessment(event))
    assert use_llm
    assert "incentive depth" in why


def test_small_incentive_eligible_events_stay_on_the_rules_path():
    """A 15% cap on Rs 400 leaves nothing to deliberate about."""
    event = make_event(reason_code="payment_cancelled", amount_paise=400_00)
    use_llm, _ = should_escalate_to_model(event, make_assessment(event))
    assert not use_llm


def test_high_value_below_the_review_line_reaches_the_model():
    event = make_event(reason_code="insufficient_funds", amount_paise=20_000_00)
    assessment = make_assessment(event, recoverability=0.5, expected_value_paise=10_000_00)
    use_llm, why = should_escalate_to_model(event, assessment)
    assert use_llm
    assert "high stakes" in why


def test_events_below_the_action_floor_are_not_worth_thinking_about():
    event = make_event(reason_code="payment_cancelled", amount_paise=60_000_00)
    assessment = make_assessment(event, recoverability=0.001, expected_value_paise=10_00)
    use_llm, why = should_escalate_to_model(event, assessment)
    assert not use_llm
    assert "action floor" in why


def test_a_hard_decline_against_a_strong_recovery_history_reaches_the_model():
    event = make_event(reason_code="card_declined")
    assessment = make_assessment(
        event, features={**make_assessment(event).features, "historical_recovery_rate": 0.8}
    )
    use_llm, why = should_escalate_to_model(event, assessment)
    assert use_llm
    assert "conflicting signals" in why


def test_a_second_attempt_reaches_the_model():
    """The taxonomy's answer has already been tried. Repeating it is not a decision."""
    event = make_event(reason_code="issuer_down", attempt_no=2)
    use_llm, why = should_escalate_to_model(event, make_assessment(event))
    assert use_llm
    assert "repeat attempt" in why


def test_routing_ignores_the_cohort():
    """A control event decided differently is a control event that measures nothing."""
    treatment = make_event(reason_code="payment_cancelled", amount_paise=8_000_00)
    control = make_event(
        reason_code="payment_cancelled", amount_paise=8_000_00, cohort=Cohort.CONTROL
    )
    assert should_escalate_to_model(treatment, make_assessment(treatment)) == (
        should_escalate_to_model(control, make_assessment(control))
    )


# ---------------------------------------------------------------------------
# Rules engine
# ---------------------------------------------------------------------------

EXPECTED_ACTION = {
    Strategy.RETRY_NOW: ActionType.RETRY_PAYMENT,
    Strategy.RETRY_DELAYED: ActionType.RETRY_PAYMENT,
    Strategy.RETRY_ON_LIQUIDITY: ActionType.RETRY_PAYMENT,
    Strategy.SWITCH_RAIL: ActionType.PAYMENT_LINK,
    Strategy.PERSUADE: ActionType.NUDGE,
    Strategy.DO_NOT_RETRY: ActionType.NO_ACTION,
}


@pytest.mark.parametrize("code", all_codes())
def test_every_reason_code_maps_to_the_action_its_strategy_implies(code):
    event = make_event(reason_code=code)
    action, params, rationale = rules_engine.propose_from_rules(
        event, make_assessment(event), make_customer()
    )
    profile = profile_for(code)
    assert action is EXPECTED_ACTION[profile.strategy]
    assert code in rationale, "the rationale must name the reason code it came from"
    assert params["incentive_paise"] == 0


def test_every_strategy_is_covered_by_the_mapping():
    """A strategy added to the taxonomy with no action mapping must not slip through."""
    assert set(EXPECTED_ACTION) == set(Strategy)
    exercised = {profile_for(c).strategy for c in all_codes()}
    assert exercised == set(Strategy), f"untested strategies: {set(Strategy) - exercised}"


def test_the_rules_path_never_proposes_spending_money():
    """The taxonomy says a discount is permitted; it cannot say one is warranted."""
    for code in all_codes():
        event = make_event(reason_code=code, amount_paise=50_000_00)
        action, params, _ = rules_engine.propose_from_rules(
            event, make_assessment(event), make_customer()
        )
        assert params["incentive_paise"] == 0
        assert action is not ActionType.NUDGE_WITH_INCENTIVE


def test_switch_rail_offers_a_rail_the_taxonomy_endorses():
    event = make_event(reason_code="card_expired", rail="card")
    _, params, _ = rules_engine.propose_from_rules(
        event, make_assessment(event), make_customer(preferred_rail="netbanking")
    )
    assert params["rail"] in {r.value for r in profile_for("card_expired").switch_to}
    assert params["rail"] != "card", "never offer back the instrument that just failed"


def test_retries_stay_on_the_rail_that_failed():
    event = make_event(reason_code="issuer_down", rail="netbanking")
    action, params, _ = rules_engine.propose_from_rules(
        event, make_assessment(event), make_customer()
    )
    assert action is ActionType.RETRY_PAYMENT
    assert params["rail"] == "netbanking"


def test_unclassified_failures_go_to_a_human_not_to_silence():
    """NO_ACTION would absorb a stale taxonomy invisibly. Coverage has to stay countable."""
    event = make_event(reason_code="some_new_code_from_2027")
    action, _, rationale = rules_engine.propose_from_rules(
        event, make_assessment(event), make_customer()
    )
    assert action is ActionType.ESCALATE_TO_HUMAN
    assert "taxonomy" in rationale


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_every_policy_bound_is_described_to_the_model():
    """A bound the policy engine enforces and the prompt omits is a guaranteed denial."""
    rendered = prompts.render_bounds(Bounds())
    assert rendered.count("\n") + 1 == len(Bounds.__dataclass_fields__)


def test_the_prompt_tracks_the_real_bounds_rather_than_a_copy():
    tightened = prompts.system_prompt(Bounds(max_incentive_fraction=0.05))
    assert "5% of order value" in tightened
    assert "15% of order value" not in tightened


def test_the_brief_never_leaks_the_cohort_or_the_customer_identity():
    event = make_event(cohort=Cohort.CONTROL)
    brief = prompts.event_brief(event, make_assessment(event)).lower()
    for forbidden in ("control", "treatment", "cohort", "test customer", "@example.com"):
        assert forbidden not in brief, f"the brief leaked '{forbidden}'"


def test_the_brief_states_whether_an_incentive_is_permitted():
    technical = make_event(reason_code="issuer_down")
    intent = make_event(reason_code="payment_cancelled")
    assert "NOT PERMITTED" in prompts.event_brief(technical, make_assessment(technical))
    assert "PERMITTED for this reason" in prompts.event_brief(intent, make_assessment(intent))


# ---------------------------------------------------------------------------
# Brain: degradation
# ---------------------------------------------------------------------------


@pytest.fixture
def no_api_key(monkeypatch):
    """Force the keyless path regardless of the developer's environment."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        brain.config, "get_settings", lambda: SimpleNamespace(anthropic_api_key="")
    )


def test_brain_falls_back_to_rules_with_no_api_key(no_api_key):
    event = make_event(reason_code="payment_cancelled", amount_paise=8_000_00)
    action, params, rationale, usage = brain.propose(
        event, make_assessment(event), make_customer()
    )
    assert usage.fell_back is True
    assert usage.model is None
    assert usage.input_tokens == 0
    assert action is ActionType.NUDGE
    assert params["incentive_paise"] == 0
    assert "fell back to rules" in rationale
    assert "ANTHROPIC_API_KEY" in rationale


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")


def _raising_client(exc: Exception):
    def create(**kwargs):
        raise exc

    return SimpleNamespace(messages=SimpleNamespace(create=create))


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    "exc, expected",
    [
        (
            anthropic.RateLimitError(
                "429", response=httpx2.Response(429, request=_request()), body=None
            ),
            "rate-limited",
        ),
        (
            anthropic.APIStatusError(
                "500", response=httpx2.Response(500, request=_request()), body=None
            ),
            "provider error",
        ),
        (
            anthropic.APIStatusError(
                "400", response=httpx2.Response(400, request=_request()), body=None
            ),
            "rejected our request",
        ),
        (anthropic.APIConnectionError(request=_request()), "could not reach"),
    ],
    ids=["rate_limit", "server_error", "bad_request", "connection"],
)
def test_every_api_failure_degrades_to_rules(monkeypatch, fake_key, exc, expected):
    """Distinct causes, one behaviour: the event still gets a decision."""
    monkeypatch.setattr(brain, "_client", lambda: _raising_client(exc))
    event = make_event(reason_code="issuer_down")
    action, _, rationale, usage = brain.propose(
        event, make_assessment(event), make_customer()
    )
    assert usage.fell_back is True
    assert expected in rationale
    assert action is ActionType.RETRY_PAYMENT


# ---------------------------------------------------------------------------
# Brain: validating what the model said
# ---------------------------------------------------------------------------


def fake_response(payload, stop_reason: str = "end_turn"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
    )


def good_payload(**overrides):
    base = {
        "action_type": "nudge",
        "incentive_paise": 0,
        "rail": None,
        "delay_hours": 0,
        "rationale": "payment_cancelled with a strong history; a plain reminder is enough.",
    }
    base.update(overrides)
    return base


def test_a_well_formed_proposal_survives_validation():
    event = make_event(reason_code="payment_cancelled")
    result = brain._validate(fake_response(good_payload()), event)
    assert result is not None
    action, params, rationale = result
    assert action is ActionType.NUDGE
    assert params == {"rail": None, "incentive_paise": 0, "delay_hours": 0.0}
    assert rationale.startswith("payment_cancelled")


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        json.dumps(["a", "list"]),
        json.dumps(good_payload(action_type="send_carrier_pigeon")),
        json.dumps(good_payload(action_type=None)),
        json.dumps(good_payload(rationale="")),
        json.dumps(good_payload(incentive_paise="two hundred")),
    ],
    ids=["prose", "not_an_object", "invented_action", "null_action", "no_rationale", "text_money"],
)
def test_malformed_model_output_is_rejected(payload):
    event = make_event(reason_code="payment_cancelled")
    assert brain._validate(fake_response(payload), event) is None


def test_a_retry_on_a_do_not_retry_reason_is_rejected_outright():
    """Policy would deny it anyway; letting it through only pollutes the trail."""
    event = make_event(reason_code="fraud_suspected")
    payload = good_payload(action_type="retry_payment")
    assert brain._validate(fake_response(payload), event) is None


def test_a_discount_on_a_technical_failure_is_stripped_not_rejected():
    """The nudge is still worth sending. Only the margin burn is removed."""
    event = make_event(reason_code="issuer_down")
    payload = good_payload(action_type="nudge_with_incentive", incentive_paise=200_00)
    action, params, rationale = brain._validate(fake_response(payload), event)
    assert action is ActionType.NUDGE
    assert params["incentive_paise"] == 0
    assert "discount removed" in rationale


def test_negative_incentives_are_clamped():
    event = make_event(reason_code="payment_cancelled")
    payload = good_payload(action_type="nudge_with_incentive", incentive_paise=-500_00)
    action, params, _ = brain._validate(fake_response(payload), event)
    assert params["incentive_paise"] == 0
    assert action is ActionType.NUDGE, "an incentive action with no incentive is a nudge"


def test_an_unrecognised_rail_is_dropped():
    event = make_event(reason_code="card_expired")
    payload = good_payload(action_type="payment_link", rail="cheque")
    _, params, _ = brain._validate(fake_response(payload), event)
    assert params["rail"] is None


def test_absurd_delays_are_clamped_to_the_outcome_window():
    event = make_event(reason_code="payment_cancelled")
    payload = good_payload(delay_hours=10_000)
    _, params, _ = brain._validate(fake_response(payload), event)
    assert params["delay_hours"] == brain.MAX_DELAY_HOURS


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_an_incomplete_response_degrades_to_rules(monkeypatch, fake_key, stop_reason):
    response = fake_response(good_payload(), stop_reason=stop_reason)
    monkeypatch.setattr(
        brain,
        "_client",
        lambda: SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response)),
    )
    event = make_event(reason_code="payment_cancelled", amount_paise=8_000_00)
    _, _, rationale, usage = brain.propose(event, make_assessment(event), make_customer())
    assert usage.fell_back is True
    assert usage.input_tokens == 1200, "tokens spent on an unusable answer still cost money"
    assert "fell back to rules" in rationale


# ---------------------------------------------------------------------------
# decide(): the row that gets written
# ---------------------------------------------------------------------------


def test_a_rules_decision_records_its_source_and_costs_nothing(session):
    customer = make_customer()
    event = make_event(reason_code="incorrect_otp")
    session.add_all([customer, event])
    session.flush()

    decision = decide(session, event, make_assessment(event), customer, NOW)

    assert decision.source is DecisionSource.RULES
    assert decision.action_type is ActionType.RETRY_PAYMENT
    assert decision.model is None
    assert decision.input_tokens == 0 and decision.output_tokens == 0
    assert decision.created_at == NOW
    assert decision.id is not None, "PolicyReview needs this id, so decide() must flush"


def test_a_fallback_is_recorded_as_rules_not_as_the_model(session, no_api_key):
    """Counting fallbacks as LLM would inflate the exact number the project argues from."""
    customer = make_customer()
    event = make_event(reason_code="payment_cancelled", amount_paise=8_000_00)
    session.add_all([customer, event])
    session.flush()

    decision = decide(session, event, make_assessment(event), customer, NOW)

    assert decision.source is DecisionSource.RULES
    assert decision.params["routed_to"] == "model"
    assert "incentive depth" in decision.params["routing_reason"]


def test_every_decision_records_why_it_took_the_path_it_took(session):
    customer = make_customer()
    event = make_event(reason_code="issuer_down")
    session.add_all([customer, event])
    session.flush()

    decision = decide(session, event, make_assessment(event), customer, NOW)

    assert decision.params["routed_to"] == "rules"
    assert "issuer_down" in decision.params["routing_reason"]
    assert set(decision.params) >= {"rail", "incentive_paise", "delay_hours"}


def test_decide_leaves_the_event_status_to_the_stage_that_can_see_the_verdict(session):
    customer = make_customer()
    event = make_event()
    session.add_all([customer, event])
    session.flush()
    before = event.status

    decide(session, event, make_assessment(event), customer, NOW)

    assert event.status is before


# ---------------------------------------------------------------------------
# The structural guarantee, restated at the package level
# ---------------------------------------------------------------------------


def test_the_agent_cannot_see_the_simulator():
    """Duplicated from tests/test_no_oracle_leak.py on purpose - the agent is the
    module a future contributor is most tempted to 'help' with ground truth."""
    for path in (PROJECT_ROOT / "recoup" / "agent").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.startswith("recoup.seed") for n in names), (
                f"{path.name} imports the simulator"
            )


def test_the_taxonomy_still_permits_exactly_two_ways_to_spend_money():
    """If this changes, the router's incentive trigger is measuring something else."""
    spenders = {c for c, p in PROFILES.items() if p.incentive_eligible}
    assert spenders == {"payment_cancelled"}
    assert profile_for("checkout_abandoned").incentive_eligible
