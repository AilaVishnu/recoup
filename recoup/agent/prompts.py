"""What the model is told, and what it is deliberately not told.

Two rules govern everything in this file.

**The bounds are rendered from recoup/policy/rules.py, never retyped.** A prompt
that lists "never discount more than 15%" as prose is a second copy of a number
that already exists in code, and the two will disagree the first time anyone
tunes the real one. `render_bounds()` walks the Bounds dataclass field by field
and raises if a field has no description - so a bound added to the policy engine
and not explained to the model fails loudly at import rather than silently
producing proposals that get denied.

**The model is told what it is for, not just what it may not do.** Its proposal
is scored by a policy engine it cannot reach around, and a denied proposal is
not a near miss - the event gets one decision, so an out-of-bounds suggestion is
a wasted event and a lost recovery. The prompt says so plainly, because a model
optimising for "maximal" and a model optimising for "allowed" behave differently
and only one of them is useful here.

The per-event brief carries observable signals only. No cohort - a control event
must be reasoned about exactly as a treatment event is or the holdout measures
nothing - no outcome, and no customer name, email or phone number, because none
of them are inputs to choosing a recovery action.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable

from recoup.db import ActionType, Assessment, RevenueEvent
from recoup.policy.rules import DEFAULT_BOUNDS, Bounds
from recoup.taxonomy import profile_for


def _rs(paise: int) -> str:
    return f"Rs {paise / 100:,.0f}"


# ---------------------------------------------------------------------------
# Bounds, rendered from the policy engine so the two cannot drift apart.
# ---------------------------------------------------------------------------

_BOUND_TEXT: dict[str, Callable[[Any], str]] = {
    "max_contacts_per_customer_per_week": lambda v: (
        f"At most {v} customer contacts in any 7 days, counted across all of that "
        "customer's events, not per event. Silent retries are not contact."
    ),
    "quiet_hours_enforced": lambda v: (
        "No customer contact between 21:00 and 08:00 IST."
        if v
        else "Quiet hours are not enforced in this configuration."
    ),
    "max_incentive_fraction": lambda v: f"A discount may never exceed {v:.0%} of order value.",
    "max_incentive_paise": lambda v: (
        f"...and never more than {_rs(v)} in absolute terms, however large the order."
    ),
    "daily_incentive_budget_paise": lambda v: (
        f"{_rs(v)} of discount per day across every event. Earlier events today may "
        "already have consumed it."
    ),
    "human_approval_above_paise": lambda v: (
        f"Orders of {_rs(v)} or more are routed to a human for approval whatever you "
        "propose. Propose the action you believe in; do not water it down to stay "
        "under the line."
    ),
    "min_expected_value_paise": lambda v: (
        f"Below {_rs(v)} of expected recovery, no action is taken at all - the channel "
        "cost and the customer's attention are worth more than the upside."
    ),
    "min_incremental_ev_ratio": lambda v: (
        f"An incentive must buy at least {v:.1f}x its own cost in expected incremental "
        "recovery. Break-even is not a reason to spend money. The estimate is "
        "deliberately conservative: it credits a discount only for the recovery "
        "headroom left over after the probability the customer converts anyway."
    ),
    "max_attempts_override": lambda v: (
        "Per-reason attempt ceilings come from the failure taxonomy."
        if v is None
        else f"Attempts are globally capped at {v}, tighter than the taxonomy."
    ),
}


def render_bounds(bounds: Bounds = DEFAULT_BOUNDS) -> str:
    """Render every field of Bounds as prose the model can plan against.

    Walks the dataclass rather than a hand-written list. Adding a bound to
    recoup/policy/rules.py without describing it here raises immediately, which
    is the point: the failure mode this guards against is a policy engine that
    silently enforces a rule the model was never told about.
    """
    lines = []
    for f in fields(bounds):
        try:
            render = _BOUND_TEXT[f.name]
        except KeyError:  # pragma: no cover - guard against silent drift
            raise RuntimeError(
                f"Bounds.{f.name} exists in recoup/policy/rules.py but has no "
                "description in recoup/agent/prompts.py. The model would be judged "
                "against a rule it was never given."
            ) from None
        lines.append(f"- {render(getattr(bounds, f.name))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_ACTION_MENU = {
    ActionType.RETRY_PAYMENT: (
        "re-present the same payment on the same rail. Silent - the customer sees "
        "nothing unless it succeeds."
    ),
    ActionType.PAYMENT_LINK: (
        "send a Razorpay Payment Link on a different rail. Counts as customer contact."
    ),
    ActionType.NUDGE: (
        "a message reminding the customer to complete the payment, at full price. "
        "Counts as customer contact."
    ),
    ActionType.NUDGE_WITH_INCENTIVE: (
        "the same message with a discount attached. Counts as customer contact and "
        "spends real margin."
    ),
    ActionType.ESCALATE_TO_HUMAN: (
        "hand the event to a person. The right answer when the situation is outside "
        "what the taxonomy covers."
    ),
    ActionType.NO_ACTION: (
        "do nothing and report the value as unrecoverable. A legitimate answer, not a "
        "failure to decide."
    ),
}


def system_prompt(bounds: Bounds = DEFAULT_BOUNDS) -> str:
    """The standing instructions. Stable across events, so it caches cleanly."""
    menu = "\n".join(f"- {a.value}: {desc}" for a, desc in _ACTION_MENU.items())
    return f"""\
You are the decision layer of Recoup, a revenue-recovery agent for an Indian
merchant on Razorpay. A payment has failed, a checkout was abandoned, or an
invoice has gone past due, and you choose the single recovery action for that
one event. There is no second decision: whatever you propose is what happens to
this event, or nothing happens to it.

This runs against Razorpay TEST MODE. Payment Links, retries and messages are
created against test credentials, so no customer is really charged - but treat
every decision as though they were, because the evaluation measures what would
have been recovered and what it would have cost.

All money is integer paise. Rs 1 = 100 paise. Never emit a fractional amount.

Most events never reach you. A failure taxonomy settles roughly four in five
deterministically - an expired card gets a new rail, a fraud decline gets
nothing, an issuer outage gets a two-hour wait. You are called for the minority
where the deterministic answer is genuinely underdetermined: an intent failure
where a discount might or might not be warranted, a high-value event nobody will
review, a customer whose history contradicts the reason code, or a second
attempt where the obvious action has already failed once. Assume the easy call
has already been made and something about this event is not easy.

THE ACTIONS AVAILABLE
{menu}

THE BOUNDS YOU ARE OPERATING INSIDE
{render_bounds(bounds)}

These are not guidelines. A policy engine evaluates your proposal after you make
it, against exactly these rules, and denies anything outside them. It runs in
code you cannot influence, and it does not negotiate.

A denied proposal is not a near miss - it is a lost event. The event does not
come back for a second decision, so a discount two hundred rupees over the cap
does not get trimmed to the cap; it gets rejected, and that customer is never
contacted at all. Aim to be allowed, not to be maximal. If you are unsure
whether a deeper discount would clear the incremental-value hurdle, propose the
shallower one that certainly will, or propose no discount.

NEVER DISCOUNT A TECHNICAL FAILURE
A discount can only move a failure whose cause was intent. When a bank was down,
a card had expired, an OTP was mistyped or a gateway timed out, the customer
already wanted to pay - the money did not arrive for a mechanical reason a coupon
does not touch. Discounting those is pure margin burn: you pay a customer to do
what they were going to do anyway, and you do it at scale. The brief tells you
whether the taxonomy considers this failure incentive-eligible. If it says no,
the answer is no, regardless of how valuable the order is or how sympathetic the
customer looks.

HOW TO DECIDE
Read the taxonomy's strategy as a strong prior - it encodes the mechanics of why
this payment died. Depart from it only when something specific in this event
contradicts it, and say what that something was. Prefer the cheapest action that
could plausibly work: a silent retry costs nothing and annoys nobody, a message
costs attention, a discount costs margin permanently.

Your rationale is written into the merchant's audit trail and read by a human
reviewing what the agent did. Two or three sentences, concrete about this event.
Cite the reason code and the specific signal that drove the call. Do not restate
the bounds back - the reviewer can see them - and do not hedge across two
actions; you are choosing one.\
"""


# ---------------------------------------------------------------------------
# Per-event brief
# ---------------------------------------------------------------------------


def _row(label: str, value: str) -> str:
    return f"  {label:<18}{value}"


def _age(hours: float) -> str:
    """Hours are the right unit for a fresh failure and useless for a stale one."""
    if hours < 48:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.0f}d ago"


def _wait(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} minutes"
    if minutes < 60 * 24:
        return f"{minutes // 60}h"
    return f"{minutes / (60 * 24):.0f}d"


def event_brief(event: RevenueEvent, assessment: Assessment) -> str:
    """Render one event as a readable brief.

    Deliberately not a JSON dump of Assessment.features. The features dict is the
    reproducibility record - forty keys, several of them internal - and pasting
    it in would bury the four signals that matter under thirty-six that do not.

    Customer signals come from the frozen feature dict rather than from the live
    Customer row on purpose: the brief the model saw is then reconstructible from
    the Assessment alone, months later, even after the customer's counters have
    moved on.
    """
    profile = profile_for(event.reason_code)
    f = assessment.features or {}
    lines: list[str] = []

    lines.append(f"EVENT {event.id}  ({event.kind.value})")
    lines.append("")

    lines.append(_row("Order", f"{_rs(event.amount_paise)} on {event.rail}"))
    lines.append(
        _row(
            "Attempt",
            f"{event.attempt_no} of at most {profile.max_attempts} permitted for this reason",
        )
    )
    lines.append(
        _row(
            "Failed",
            f"{_age(float(f.get('event_age_hours', 0) or 0))}, at "
            f"{f.get('occurred_hour_ist', 0):02d}:00 IST on the "
            f"{f.get('occurred_day_of_month', 0)}th of the month",
        )
    )
    lines.append("")

    lines.append(_row("Reason code", f"{event.reason_code} - {profile.label}"))
    lines.append(_row("", f"source={profile.source}  step={profile.step}"))
    lines.append(_row("Taxonomy strategy", profile.strategy.value))
    lines.append(
        _row(
            "Incentive",
            "PERMITTED for this reason - intent, not mechanics, is the blocker"
            if profile.incentive_eligible
            else "NOT PERMITTED for this reason - the cause is technical",
        )
    )
    if profile.switch_to:
        lines.append(
            _row("Rails to consider", ", ".join(r.value for r in profile.switch_to))
        )
    if profile.retry_after_minutes:
        lines.append(
            _row(
                "Taxonomy timing",
                f"hold {_wait(profile.retry_after_minutes)} before acting",
            )
        )
    if profile.note:
        lines.append(_row("Taxonomy note", profile.note))
    lines.append("")

    lines.append(
        _row(
            "Scored",
            f"recoverability {assessment.recoverability:.0%}, expected value "
            f"{_rs(assessment.expected_value_paise)} ({assessment.scorer_version})",
        )
    )
    lines.append(
        _row("Not before", f"{assessment.earliest_action_at:%Y-%m-%d %H:%M} UTC")
    )
    lines.append("")

    successes = f.get("prior_success_count", 0)
    failures = f.get("prior_failure_count", 0)
    lines.append(
        _row("Customer", f"{successes} successful payments, {failures} failures")
    )
    if failures:
        lines.append(
            _row(
                "",
                f"recovered from {f.get('historical_recovery_rate', 0):.0%} of those "
                "on their own",
            )
        )
    lines.append(
        _row(
            "",
            f"{f.get('customer_tenure_days', 0)}d tenure, lifetime value "
            f"{_rs(int(f.get('lifetime_value_inr', 0) * 100))}, prefers "
            f"{f.get('preferred_rail', 'unknown')}",
        )
    )
    if not f.get("rail_is_preferred", True):
        lines.append(
            _row("", f"this payment used {event.rail}, which is not their usual rail")
        )

    stage = f.get("checkout_stage")
    items = f.get("basket_items")
    if stage or items:
        lines.append("")
        lines.append(_row("Basket", f"{items} item(s), checkout stage '{stage}'"))

    lines.append("")
    lines.append(
        "Decide the single recovery action for this event, and say why in terms a "
        "merchant reviewing the audit trail would accept."
    )
    return "\n".join(lines)
