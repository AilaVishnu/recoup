"""Model providers. The decision layer is swappable; the policy engine is not.

Recoup asks a model for one structured proposal per ambiguous event. Nothing
about that requires a particular vendor, and pinning the project to one would
misrepresent where its value sits: the taxonomy, the bounds and the measurement
are the product, and the model is a component with an interface.

So the provider is configuration. Every adapter here returns the same
`ModelReply`, `brain.py` validates it identically, and `policy/rules.py` never
learns which one answered - a proposal from a frontier model and a proposal from
a free-tier model pass through exactly the same thirteen bounds.

Two adapters cover almost everything:

  anthropic  - the Anthropic SDK, native.
  openai     - any OpenAI-compatible /chat/completions endpoint.

The second is doing more work than its name suggests. OpenAI, Groq, Google
Gemini, OpenRouter, Together, DeepSeek, Mistral and a local Ollama all expose
OpenAI-compatible endpoints, so pointing `RECOUP_LLM_BASE_URL` at one of them is
the whole integration. See .env.example for the base URLs.

On structured output
--------------------
Compatibility is uneven in exactly the place it matters. Full `json_schema`
response formats are supported by OpenAI and Gemini, partially by Groq, and not
at all by some smaller endpoints - and an unsupported value is usually a 400
rather than a graceful degradation. So the adapter negotiates downward:
json_schema, then json_object, then plain text with the schema in the prompt.

That is not defensive padding. `brain._validate` rejects anything malformed
whatever the transport promised, so the negotiation only decides how often the
model gets it right the first time, never whether a bad proposal can slip
through. The floor is the validator, not the response format.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from recoup import config


class ProviderError(RuntimeError):
    """A call failed in a way the caller should treat as "no proposal".

    Deliberately one type. brain.py falls back to the rules engine for every
    provider failure, so distinguishing a 429 from a DNS failure in the type
    system would buy nothing - the reason travels in `.note` for the audit
    trail, which is where it is actually read.
    """

    def __init__(self, note: str) -> None:
        super().__init__(note)
        self.note = note


@dataclass(frozen=True)
class ModelReply:
    """One provider's answer, normalised."""

    text: str | None
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False
    refused: bool = False


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic(max_retries=2)


def _call_anthropic(
    system: str, user: str, schema: dict[str, Any], max_tokens: int
) -> ModelReply:
    import anthropic

    model = config.get_settings().llm_model or config.DECISION_MODEL
    try:
        response = _anthropic_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={
                "effort": config.DECISION_EFFORT,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
    except anthropic.RateLimitError:
        # No sleep-and-retry. Recoup decides events in batches, so a 429 means
        # the whole batch is about to hit the same wall - waiting turns one slow
        # event into a slow run, and the taxonomy answer is available instantly.
        raise ProviderError("model rate-limited") from None
    except anthropic.APIStatusError as exc:
        blame = "model provider error" if exc.status_code >= 500 else "rejected our request"
        raise ProviderError(f"{blame} (HTTP {exc.status_code})") from None
    except anthropic.APIConnectionError:
        raise ProviderError("could not reach the model") from None

    text = next((b.text for b in response.content if b.type == "text"), None)
    return ModelReply(
        text=text,
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        truncated=response.stop_reason == "max_tokens",
        refused=response.stop_reason == "refusal",
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    settings = config.get_settings()
    return OpenAI(
        api_key=settings.llm_api_key or "not-needed",
        base_url=settings.llm_base_url or None,
        max_retries=2,
    )


def _response_formats(schema: dict[str, Any]) -> list[dict[str, Any] | None]:
    """Response formats to try, strictest first.

    `strict` is omitted rather than set False: some compatible endpoints reject
    the key outright, and the validator is the real guarantee regardless.
    """
    return [
        {
            "type": "json_schema",
            "json_schema": {"name": "recovery_decision", "schema": schema},
        },
        {"type": "json_object"},
        None,
    ]


def _call_openai_compatible(
    system: str, user: str, schema: dict[str, Any], max_tokens: int
) -> ModelReply:
    import openai

    settings = config.get_settings()
    model = settings.llm_model or "gpt-4o-mini"
    client = _openai_client()

    # A plain-text or json_object request has no schema attached, so the schema
    # is restated in the user turn. Harmless when the response format already
    # carries it, and the difference between a usable answer and a fallback when
    # it does not.
    user_with_schema = (
        f"{user}\n\nReply with JSON only, matching this schema exactly:\n"
        f"{json.dumps(schema, indent=1)}"
    )

    last: ProviderError | None = None
    for response_format in _response_formats(schema):
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user if response_format and
                    response_format.get("type") == "json_schema" else user_with_schema,
                },
            ],
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            response = client.chat.completions.create(**payload)
        except openai.RateLimitError:
            raise ProviderError("model rate-limited") from None
        except openai.APIStatusError as exc:
            if exc.status_code == 400 and response_format is not None:
                # Almost always "this endpoint does not support that response
                # format". Step down and try again rather than falling back to
                # rules over a transport detail.
                last = ProviderError("rejected our request (HTTP 400)")
                continue
            blame = (
                "model provider error" if exc.status_code >= 500 else "rejected our request"
            )
            raise ProviderError(f"{blame} (HTTP {exc.status_code})") from None
        except openai.APIConnectionError:
            raise ProviderError("could not reach the model") from None

        choice = response.choices[0]
        usage = response.usage
        return ModelReply(
            text=choice.message.content,
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            truncated=choice.finish_reason == "length",
            refused=getattr(choice.message, "refusal", None) is not None,
        )

    raise last or ProviderError("no usable response format")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_ADAPTERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai_compatible,
}

_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def active_provider() -> str:
    name = (config.get_settings().llm_provider or "anthropic").strip().lower()
    return name if name in _ADAPTERS else "anthropic"


def key_available() -> bool:
    """True if the active provider has a key, bridging .env into os.environ.

    Both SDKs read os.environ in their zero-argument constructors while settings
    come from .env via pydantic-settings. Without the bridge the project would
    use the model from a shell export and silently fall back to rules from a
    .env file - a difference only ever noticed during a demo.
    """
    settings = config.get_settings()
    env_name = _KEY_ENV[active_provider()]

    if os.environ.get(env_name):
        return True

    key = settings.llm_api_key
    if key:
        os.environ[env_name] = key
        return True

    # A local endpoint (Ollama, LM Studio) needs no key at all.
    return bool(settings.llm_base_url) and active_provider() == "openai"


def missing_key_note() -> str:
    return (
        f"no {_KEY_ENV[active_provider()]} configured, so the taxonomy decided this one"
    )


def call(system: str, user: str, schema: dict[str, Any], max_tokens: int) -> ModelReply:
    """Ask the configured provider for one structured proposal."""
    return _ADAPTERS[active_provider()](system, user, schema, max_tokens)
