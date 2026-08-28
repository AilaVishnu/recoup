"""The decision model is a swappable component. This is what makes that true.

Recoup claims the model is a part, not the architecture - that the taxonomy,
the bounds and the measurement are the product, and the thing proposing actions
can be replaced without any of them noticing. A claim like that is worth exactly
as much as the test that holds it up.

So: the same event, decided by two different providers, must produce proposals
that are validated the same way and reviewed against the same bounds. What must
never differ is what Recoup is allowed to do about the answer.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from recoup.agent import brain, providers


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_unknown_providers_fall_back_rather_than_crash(monkeypatch):
    """A typo in .env must not take the pipeline down.

    Falling back to the default costs a wrong-provider call that fails and
    degrades to rules. Raising would end a 600-event run over a spelling
    mistake in a config file.
    """
    monkeypatch.setattr(
        providers.config,
        "get_settings",
        lambda: SimpleNamespace(llm_provider="oepnai"),
    )
    assert providers.active_provider() == "anthropic"


@pytest.mark.parametrize("name", ["openai", "OpenAI", "  openai  "])
def test_provider_names_are_read_leniently(monkeypatch, name):
    """Env files are typed by hand and casing is not a meaningful signal."""
    monkeypatch.setattr(
        providers.config, "get_settings", lambda: SimpleNamespace(llm_provider=name)
    )
    assert providers.active_provider() == "openai"


def test_every_adapter_is_reachable_by_name():
    """A provider in the table with no adapter is a KeyError at the worst moment."""
    for name in providers._ADAPTERS:
        assert callable(providers._ADAPTERS[name])
        assert name in providers._KEY_ENV


def test_the_missing_key_note_names_the_variable_the_user_must_set(monkeypatch):
    """"No API key" is useless advice when two are possible."""
    monkeypatch.setattr(providers, "active_provider", lambda: "openai")
    assert "OPENAI_API_KEY" in providers.missing_key_note()
    monkeypatch.setattr(providers, "active_provider", lambda: "anthropic")
    assert "ANTHROPIC_API_KEY" in providers.missing_key_note()


@pytest.mark.parametrize(
    "base_url", ["http://localhost:11434/v1", "http://127.0.0.1:1234/v1"]
)
def test_a_local_endpoint_needs_no_key(monkeypatch, base_url):
    """Ollama and LM Studio authenticate nothing. Demanding a key locks them out."""
    monkeypatch.setattr(
        providers.config,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="openai", llm_api_key="", llm_base_url=base_url
        ),
    )
    assert providers.key_available() is True


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.groq.com/openai/v1",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "https://openrouter.ai/api/v1",
    ],
)
def test_a_remote_endpoint_without_a_key_is_not_reachable(monkeypatch, base_url):
    """The carve-out is for localhost, not for every configured base_url.

    Treating any base_url as key-free reported "decision model reachable" for a
    Groq endpoint with an empty key - a false OK, which is worse than a failure
    because it sends someone looking for the problem somewhere it is not.
    """
    monkeypatch.setattr(
        providers.config,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="openai", llm_api_key="", llm_base_url=base_url
        ),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert providers.key_available() is False


# ---------------------------------------------------------------------------
# The claim itself
# ---------------------------------------------------------------------------

PAYLOAD = {
    "action_type": "nudge",
    "incentive_paise": 0,
    "rail": None,
    "delay_hours": 0,
    "rationale": "strong history, a reminder is enough",
}


def test_two_providers_produce_the_same_validated_proposal(monkeypatch):
    """The swappability claim, tested rather than asserted.

    Same event, same payload, two providers, identical outcome. If this ever
    fails, the model has stopped being a component and become part of the
    architecture.
    """
    from tests.test_agent import make_assessment, make_customer, make_event

    event = make_event(reason_code="payment_cancelled", amount_paise=8_000_00)
    assessment, customer = make_assessment(event), make_customer()

    results = []
    for name in ("anthropic", "openai"):
        monkeypatch.setattr(providers, "active_provider", lambda n=name: n)
        monkeypatch.setattr(providers, "key_available", lambda: True)
        monkeypatch.setattr(
            providers,
            "call",
            lambda **kw: providers.ModelReply(
                text=json.dumps(PAYLOAD), model=f"{name}-model", input_tokens=10, output_tokens=5
            ),
        )
        action, params, rationale, usage = brain.propose(event, assessment, customer)
        results.append((action, params, usage.fell_back))

    assert results[0] == results[1], (
        "the same proposal decided by two providers diverged - the model is "
        "supposed to be interchangeable below the validator"
    )
    assert not results[0][2], "neither should have fallen back"


def test_a_fenced_reply_is_still_accepted(monkeypatch):
    """Endpoints without json_schema support fence their JSON constantly.

    Rejecting a fenced body would make exactly the free-tier providers look
    unreliable, when the JSON inside is fine. Everything after the unwrap still
    has to pass validation unchanged.
    """
    from tests.test_agent import make_event

    event = make_event(reason_code="payment_cancelled", amount_paise=8_000_00)
    fenced = "```json\n" + json.dumps(PAYLOAD) + "\n```"

    assert brain._validate(fenced, event) is not None
    assert brain._validate(json.dumps(PAYLOAD), event) == brain._validate(fenced, event)


def test_a_fenced_reply_that_is_not_json_is_still_rejected(monkeypatch):
    """Unwrapping the fence must not become tolerance for garbage inside it."""
    from tests.test_agent import make_event

    event = make_event(reason_code="payment_cancelled")
    assert brain._validate("```json\nnot json at all\n```", event) is None


def test_response_format_negotiation_starts_strict_and_degrades():
    """Strictest first, plain text last - never the other way round."""
    formats = providers._response_formats({"type": "object"})
    assert formats[0]["type"] == "json_schema"
    assert formats[1]["type"] == "json_object"
    assert formats[-1] is None, "the last resort must be an unconstrained request"


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------


def test_pacing_is_off_by_default(monkeypatch):
    """An unset RPM must not introduce latency into a paid-tier run."""
    monkeypatch.setattr(
        providers.config,
        "get_settings",
        lambda: SimpleNamespace(llm_requests_per_minute=0),
    )
    slept: list[float] = []
    monkeypatch.setattr(providers.time, "sleep", lambda s: slept.append(s))
    providers._throttle()
    providers._throttle()
    assert slept == []


def test_pacing_waits_between_calls_when_configured(monkeypatch):
    """The second call inside the interval must wait, not fire.

    Without this, a free-tier run answers a handful of events and rate-limits the
    rest - and the reported LLM share becomes a measure of the rate limiter
    rather than of the routing rule.
    """
    monkeypatch.setattr(
        providers.config,
        "get_settings",
        lambda: SimpleNamespace(llm_requests_per_minute=10),  # 6s apart
    )
    clock = {"t": 1000.0}
    slept: list[float] = []
    monkeypatch.setattr(providers.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(providers.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(providers, "_last_call_at", 0.0)

    providers._throttle()          # first call, nothing to wait for
    assert slept == []

    providers._throttle()          # immediately after: must wait ~6s
    assert slept and abs(slept[0] - 6.0) < 0.01


def test_pacing_does_not_wait_once_the_interval_has_passed(monkeypatch):
    """Pacing is a floor on spacing, not a fixed delay per call."""
    monkeypatch.setattr(
        providers.config,
        "get_settings",
        lambda: SimpleNamespace(llm_requests_per_minute=10),
    )
    clock = {"t": 1000.0}
    slept: list[float] = []
    monkeypatch.setattr(providers.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(providers.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(providers, "_last_call_at", 0.0)

    providers._throttle()
    clock["t"] += 30.0             # well past the 6s interval
    providers._throttle()
    assert slept == []
