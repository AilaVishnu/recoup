"""The model call. One event in, one validated proposal out - or the rules path.

This is the expensive path, taken for roughly one event in five (see
recoup/agent/router.py). Three things it is built around:

**It degrades to rules, never to nothing.** No API key, a rate limit, a network
that is down, a refusal, a truncated response, a proposal that fails validation -
every one of them falls through to recoup/agent/rules_engine.py and records the
Decision with source=RULES and a rationale that says why. The pipeline runs
end-to-end on a laptop with no ANTHROPIC_API_KEY set, produces a full audit
trail, and the eval still works. A recovery agent that stops recovering because
a model endpoint is busy is not a recovery agent.

**It validates before it returns.** Structured output guarantees the shape, not
the sense. The model can still name an incentive on a reason code the taxonomy
forbids one on, or a rail nobody accepts. The policy engine is the real gate and
would catch those - but handing it proposals that are already known-wrong fills
the audit trail with denials that teach a reviewer nothing about the system's
judgement. Malformed proposals are dropped here; borderline ones are passed
through and left for policy to rule on.

**It reports what it cost.** Token counts go onto the Decision row, so "we did
not call a model 600 times" is a number in the database rather than a claim in a
README.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from recoup import config
from recoup.agent import prompts, providers, rules_engine
from recoup.agent.rules_engine import Params
from recoup.db import ActionType, Assessment, Customer, RevenueEvent
from recoup.taxonomy import Rail, Strategy, profile_for

MAX_TOKENS = 4000
"""Enough for adaptive thinking plus a five-field JSON object with a short
rationale. A truncated response is not a partial answer - structured output that
hits the cap is invalid JSON - so the ceiling is set well clear of the need."""

MAX_DELAY_HOURS = 168.0
"""A week. The outcome window closes at seven days, so a proposal to wait longer
than that is a proposal to do nothing while pretending otherwise."""

MAX_RATIONALE_CHARS = 900
"""The rationale is displayed in the audit UI next to the policy checks. Longer
than this is an essay nobody reads, and the useful part is always at the front."""

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [a.value for a in ActionType],
            "description": "The single recovery action to take on this event.",
        },
        "incentive_paise": {
            "type": "integer",
            "description": (
                "Discount to attach, in integer paise. 0 when no discount is "
                "proposed. Must be 0 unless the brief says the reason code is "
                "incentive-eligible."
            ),
        },
        "rail": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": (
                "Payment rail to use: card, upi, netbanking, wallet or emi. For a "
                "retry this is the rail that failed; for a payment link it is the "
                "replacement. null when the action does not involve a rail."
            ),
        },
        "delay_hours": {
            "type": "number",
            "description": (
                "Additional hours to wait beyond the taxonomy's own timing floor "
                "('Not before' in the brief). 0 to act as soon as that window opens."
            ),
        },
        "rationale": {
            "type": "string",
            "description": (
                "Two or three sentences naming the reason code and the specific "
                "signal that drove this call. Written for a merchant reading the "
                "audit trail."
            ),
        },
    },
    "required": ["action_type", "incentive_paise", "rail", "delay_hours", "rationale"],
    "additionalProperties": False,
}

_RAIL_VALUES = {r.value for r in Rail}


@dataclass(frozen=True)
class Usage:
    """What the call cost, and whether it happened at all."""

    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    fell_back: bool = False
    """True when the rules engine produced this proposal instead of the model.
    recoup/agent/decide.py reads it to set DecisionSource correctly - a fallback
    recorded as source=LLM would overstate how much of the system is a model."""

    note: str = ""
    """Why it fell back. Appended to the rationale so the audit trail explains
    itself without anyone having to correlate log lines."""


def propose(
    event: RevenueEvent, assessment: Assessment, customer: Customer
) -> tuple[ActionType, Params, str, Usage]:
    """Ask the model for one action. Returns (action_type, params, rationale, usage).

    Never raises on an API failure. Every path returns a usable proposal; the
    Usage object says whether it came from the model.
    """
    if not providers.key_available():
        return _fallback(event, assessment, customer, providers.missing_key_note())

    try:
        reply = providers.call(
            system=prompts.system_prompt(),
            user=prompts.event_brief(event, assessment),
            schema=DECISION_SCHEMA,
            max_tokens=MAX_TOKENS,
        )
    except providers.ProviderError as exc:
        # Every provider failure lands here and falls back to the taxonomy.
        # Distinguishing a 429 from a DNS failure would change nothing about
        # what Recoup does next; the reason travels into the audit trail, which
        # is where it is actually read.
        return _fallback(event, assessment, customer, exc.note)

    usage = Usage(
        model=reply.model,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
    )

    if reply.truncated:
        return _fallback(
            event, assessment, customer, "model response truncated before it was valid", usage
        )
    if reply.refused:
        return _fallback(event, assessment, customer, "model declined to answer", usage)

    parsed = _validate(reply.text, event)
    if parsed is None:
        return _fallback(
            event, assessment, customer, "model output failed validation", usage
        )

    action_type, params, rationale = parsed
    return action_type, params, rationale, usage


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    """Unwrap ```json ... ``` if a model wrapped its answer in one.

    Not cosmetic tolerance. Endpoints without json_schema support are exactly
    the ones that fence their output, so refusing a fenced body would make the
    free-tier providers look worse than they are - the JSON inside is usually
    perfectly good, and everything after this still has to pass validation
    unchanged.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.split("\n", 1)[-1]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[: -len("```")]
    return body.strip()


def _validate(
    text: str | None, event: RevenueEvent
) -> tuple[ActionType, Params, str] | None:
    """Turn a model response into a proposal, or None if it is not usable.

    Rejection and repair are different things and the line between them is
    whether the mistake is about *shape* or about *degree*. An action type that
    is not an action type cannot be repaired without inventing intent, so it is
    rejected. An incentive on a reason code that forbids one is a known-wrong
    detail on an otherwise sensible proposal, so it is stripped and the rest
    kept - policy still gets to rule on what is left.
    """
    if not text:
        return None
    try:
        raw = json.loads(_strip_code_fence(text))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None

    try:
        action_type = ActionType(raw.get("action_type"))
    except ValueError:
        return None

    profile = profile_for(event.reason_code)

    # The taxonomy's hard stops are not the model's to overrule. Reaching here
    # means the router misfired or the prompt was ignored; either way, a proposal
    # that contradicts a do-not-retry is not evidence about anything except that
    # something is wrong, and it does not belong in the trace as a real decision.
    if profile.strategy is Strategy.DO_NOT_RETRY and action_type not in (
        ActionType.NO_ACTION,
        ActionType.ESCALATE_TO_HUMAN,
    ):
        return None

    incentive = raw.get("incentive_paise", 0)
    if isinstance(incentive, bool) or not isinstance(incentive, (int, float)):
        return None
    incentive = max(0, int(incentive))

    notes: list[str] = []
    if incentive and not profile.incentive_eligible:
        # Policy would deny this outright and the whole event would be lost. The
        # rest of the proposal is usually fine, so drop the discount and let the
        # nudge through.
        incentive = 0
        notes.append(
            f"[discount removed: {event.reason_code} is a technical failure and is "
            "not incentive-eligible]"
        )
    if action_type is ActionType.NUDGE_WITH_INCENTIVE and incentive == 0:
        action_type = ActionType.NUDGE
    if incentive and action_type is not ActionType.NUDGE_WITH_INCENTIVE:
        # A discount can only be delivered by the action that carries a message.
        incentive = 0
        notes.append(f"[discount removed: {action_type.value} cannot carry one]")

    rail = raw.get("rail")
    if isinstance(rail, str):
        rail = rail.strip().lower()
        if rail not in _RAIL_VALUES:
            rail = None
    else:
        rail = None

    delay = raw.get("delay_hours", 0)
    if isinstance(delay, bool) or not isinstance(delay, (int, float)):
        delay = 0.0
    delay = min(MAX_DELAY_HOURS, max(0.0, float(delay)))

    rationale = raw.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return None
    rationale = rationale.strip()[:MAX_RATIONALE_CHARS]
    if notes:
        rationale = f"{rationale} {' '.join(notes)}"

    params: Params = {
        "rail": rail,
        "incentive_paise": incentive,
        "delay_hours": delay,
    }
    return action_type, params, rationale


def _fallback(
    event: RevenueEvent,
    assessment: Assessment,
    customer: Customer,
    why: str,
    usage: Usage | None = None,
) -> tuple[ActionType, Params, str, Usage]:
    """Degrade to the taxonomy and say so, in the rationale itself.

    The tokens already spent are kept on the Usage object even when the answer
    is thrown away - a model call that produced nothing usable still cost money,
    and a cost report that hides those is a cost report that flatters itself.
    """
    action_type, params, rationale = rules_engine.propose_from_rules(
        event, assessment, customer
    )
    usage = usage or Usage()
    return (
        action_type,
        params,
        f"[fell back to rules: {why}] {rationale}",
        Usage(
            model=usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            fell_back=True,
            note=why,
        ),
    )
