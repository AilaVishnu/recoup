"""The deterministic proposal path: taxonomy in, action out.

Four out of five events are decided here, with no model involved. That is not a
cost-saving compromise - it is the correct answer. `card_expired` means the card
cannot be charged; there is exactly one sensible action and no amount of
reasoning produces a second one. Sending that to a model would add latency, cost
and non-determinism to a lookup.

Two properties this file holds to:

**It never proposes spending money.** The taxonomy says whether a discount is
*permitted*; it cannot say whether one is *warranted*, because that depends on
the customer and the order rather than on the failure reason. So the rules path
always proposes the free version of the action, and any decision to spend
margin goes through the model (recoup/agent/router.py) and then through the
policy engine. A deterministic path that quietly discounted 40% of events would
be the single most expensive bug in the system.

**It never escalates on value.** Above the autonomy limit the policy engine
returns ESCALATE and the action queues for a human - that is its job, and
duplicating the threshold here would give the project two copies of a number
that must agree. The rules engine proposes the action it believes in and lets
policy decide who is allowed to authorise it.
"""

from __future__ import annotations

from typing import Any

from recoup.db import ActionType, Assessment, Customer, RevenueEvent
from recoup.taxonomy import FailureProfile, Strategy, profile_for

Params = dict[str, Any]

_RETRY_STRATEGIES = {
    Strategy.RETRY_NOW,
    Strategy.RETRY_DELAYED,
    Strategy.RETRY_ON_LIQUIDITY,
}


def _params(rail: str | None = None, incentive_paise: int = 0, delay_hours: float = 0.0) -> Params:
    """One params shape for every proposal, rules-authored or model-authored.

    The executor and the audit UI read a single set of keys regardless of which
    path produced the decision, which is what makes the two comparable in the
    trace. `delay_hours` is additional wait *beyond* Assessment.earliest_action_at
    - the rules path never asks for any, because the taxonomy's timing floor is
    already encoded there.
    """
    return {"rail": rail, "incentive_paise": incentive_paise, "delay_hours": delay_hours}


def _switch_rail(profile: FailureProfile, event: RevenueEvent, customer: Customer) -> str | None:
    """Pick the rail to offer instead of the one that failed.

    Best-first from the taxonomy, except that a rail the customer already uses
    beats a marginally better one they have never touched - a UPI-native
    customer offered netbanking is a customer who does not complete. The failed
    rail is excluded even if it is their favourite; it just demonstrated that it
    does not work.
    """
    options = [r.value for r in profile.switch_to if r.value != event.rail]
    if not options:
        return None
    if customer.preferred_rail in options:
        return customer.preferred_rail
    return options[0]


def propose_from_rules(
    event: RevenueEvent, assessment: Assessment, customer: Customer
) -> tuple[ActionType, Params, str]:
    """Map a failure reason to the one action its strategy implies.

    Returns (action_type, params, rationale). The rationale names the reason
    code and says why the strategy follows from it - it is shown to the merchant
    in the audit trail, so "the rules engine decided" is not an acceptable
    answer.
    """
    profile = profile_for(event.reason_code)

    # Unknown reason codes are not a strategy decision - they are a coverage
    # failure. NO_ACTION would silently absorb them; escalating puts the event
    # in front of a human and keeps the taxonomy's blind spots countable.
    if profile.code == "unknown":
        return (
            ActionType.ESCALATE_TO_HUMAN,
            _params(),
            f"'{event.reason_code}' is not in the taxonomy. Recoup will not guess at "
            f"a recovery strategy for a failure it cannot classify - Rs "
            f"{event.amount_paise / 100:,.0f} is queued for a human and counted "
            "against coverage rather than quietly dropped.",
        )

    if profile.strategy is Strategy.DO_NOT_RETRY:
        return (
            ActionType.NO_ACTION,
            _params(),
            f"{event.reason_code} ({profile.label}) is do-not-retry. {profile.note} "
            f"Rs {event.amount_paise / 100:,.0f} is reported as suppressed value, "
            "not as a recovery opportunity.",
        )

    if profile.strategy in _RETRY_STRATEGIES:
        return (
            ActionType.RETRY_PAYMENT,
            _params(rail=event.rail),
            _retry_rationale(profile, event, assessment),
        )

    if profile.strategy is Strategy.SWITCH_RAIL:
        rail = _switch_rail(profile, event, customer)
        if rail is None:
            # A switch strategy with nowhere to switch to. Fail closed rather
            # than re-presenting the instrument that just refused.
            return (
                ActionType.ESCALATE_TO_HUMAN,
                _params(),
                f"{event.reason_code} requires a different payment rail, but the "
                f"taxonomy offers no alternative to {event.rail} for this failure. "
                "Retrying the same instrument is guaranteed to fail, so this needs "
                "a human rather than an automated attempt.",
            )
        return (
            ActionType.PAYMENT_LINK,
            _params(rail=rail),
            f"{event.reason_code} ({profile.label}) means the instrument itself "
            f"cannot complete this payment, so retrying {event.rail} would burn one "
            f"of {profile.max_attempts} permitted attempt(s) on a certain failure. "
            f"A Payment Link on {rail} routes around it. {profile.note}".strip(),
        )

    if profile.strategy is Strategy.PERSUADE:
        return (
            ActionType.NUDGE,
            _params(),
            f"{event.reason_code} ({profile.label}) is an intent failure, not a "
            f"technical one - nothing broke, so there is nothing to retry. A nudge "
            f"after {_hours(profile.retry_after_minutes)} is the whole intervention. "
            "No incentive is proposed here: whether this customer is worth paying "
            "to convert is a margin judgement the taxonomy does not make.",
        )

    # Unreachable while Strategy has six members and all six are handled above.
    # Kept as a fail-closed floor: a strategy added to the taxonomy without a
    # mapping here must not silently become "do nothing and say nothing".
    raise ValueError(
        f"no action mapping for strategy {profile.strategy!r} "
        f"(reason code {event.reason_code}) - recoup/agent/rules_engine.py is "
        "behind recoup/taxonomy.py"
    )


def _retry_rationale(
    profile: FailureProfile, event: RevenueEvent, assessment: Assessment
) -> str:
    """Why a retry, and why *this* retry - the timing is most of the argument."""
    head = f"{event.reason_code} ({profile.label}) is recoverable by re-presenting "
    head += "the same payment on the same rail"

    if profile.strategy is Strategy.RETRY_NOW:
        why = (
            " immediately - the customer is likely still in session, which is the "
            "highest-yield and cheapest recovery available."
        )
    elif profile.strategy is Strategy.RETRY_DELAYED:
        why = (
            f" once the fault clears. The taxonomy holds this for "
            f"{_hours(profile.retry_after_minutes)}; retrying sooner just reproduces "
            "the same failure and spends an attempt doing it."
        )
    else:
        why = (
            " when the money is likely to be there. Intent is intact, the balance "
            "is not, so the retry is timed to the next salary window rather than to "
            f"the next few minutes - scheduled for "
            f"{assessment.earliest_action_at:%d %b %H:%M} UTC."
        )

    return (
        head
        + why
        + f" No incentive: {profile.source}-side technical failures do not get cheaper "
        "with a discount."
    )


def _hours(minutes: int) -> str:
    if minutes <= 0:
        return "no delay"
    if minutes < 60:
        return f"{minutes} minutes"
    if minutes < 60 * 24:
        return f"{minutes // 60}h"
    return f"{minutes // (60 * 24)}d"
