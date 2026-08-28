"""When is a decision worth a model call, and when is it a lookup?

Recoup does not call an LLM 600 times to answer a question a 16-row table
already answers. Most failure reasons are *determined*: an expired card cannot
be retried on the same card, a fraud decline must never be re-presented, an
issuer outage wants two hours and nothing else. There is no judgement in those -
only lookup - and dressing a lookup up as reasoning costs latency, money, and
reproducibility while adding nothing.

So the model is reserved for the cases where the deterministic path is genuinely
underdetermined. Four of them:

  1. **Incentive-eligible and large enough to matter.** Two reason codes in the
     taxonomy permit spending money. Whether to spend it, and how deep, is a
     margin decision the table cannot make - it depends on the customer, the
     order, and how much of the recovery would have happened anyway.
  2. **High expected value that nobody else will check.** Above the autonomy
     limit a human reviews the action anyway, so model judgement is redundant.
     Below the action floor nothing happens at all. The band in between is where
     an expensive mistake ships unreviewed - that is where the marginal decision
     quality is worth paying for.
  3. **Conflicting signals.** A customer who usually recovers on their own,
     hitting a hard decline - an issuer refusal or an unusable instrument - that
     the taxonomy scores as near-hopeless. The prior and the evidence disagree,
     and a table cannot arbitrate that. Restricted to the switch-rail family on
     purpose: an intent failure and a stubborn customer are not in conflict,
     they are the same fact twice.
  4. **The obvious action already failed.** A second attempt on the same order
     means the taxonomy's first recommendation has been tried and did not work.
     Repeating it is the definition of not thinking.

Everything else routes to recoup/agent/rules_engine.py.

Measured share on the seeded 600-event dataset: **26.8% routed to the model**
(161 events), the other 439 settled by the taxonomy alone. The split by trigger:

    68.5%  taxonomy settles it            411 events   no model call
    12.5%  high stakes, unreviewed         75 events   -> model
     6.5%  incentive depth                 39 events   -> model
     4.7%  repeat attempt                  28 events   -> model
     4.0%  below the action floor          24 events   no model call
     3.2%  conflicting signals             19 events   -> model
     0.7%  do-not-retry                     4 events   no model call

These are counted from Decision rows, not predicted. An earlier version of this
docstring claimed 20.7% from a run whose triggers were computed against
different expected values - the attempt cap could not fire then, so more events
cleared the action floor and fewer reached the high-stakes band. A share quoted
in a comment drifts silently from the share the code produces; re-derive it with
`python -m recoup.agent.router` after any change that moves scores.
     0.7%  do-not-retry                     4 events   no model call

Rerun with `python -m recoup.agent.router` to recount against the current
database. The number is printed rather than asserted in a test on purpose: it is
a property of the reason-code mix, not a contract, and a dataset that shifts it
deserves a human reading the breakdown rather than a red build.

The routing decision reads only observable event and assessment fields. It never
looks at the cohort: a control event must be routed, reasoned about, and decided
exactly as a treatment event is, or the holdout stops being a counterfactual for
anything.
"""

from __future__ import annotations

from recoup.db import Assessment, RevenueEvent
from recoup.policy.rules import DEFAULT_BOUNDS
from recoup.taxonomy import Strategy, profile_for

INCENTIVE_DELIBERATION_FLOOR_PAISE = 30_00_00
"""Below Rs 3,000 an incentive is not a decision worth deliberating.

Policy caps a discount at 15% of order value, so on a Rs 900 basket the entire
decision space is "nudge, or nudge with up to Rs 135 off" - a model adds nothing
to a choice that narrow. Rs 3,000 is where the cap first clears Rs 450, which is
roughly where the money on the table exceeds the cost of thinking about it.
"""

HIGH_STAKES_EV_PAISE = 25_00_00
"""Expected recovery above which a wrong action is expensive rather than routine.

Deliberately one tenth of the human-approval threshold: the events that reach
the model unreviewed are the ones where Rs 2,500 of expected value rides on the
call, but not so much value that a person is going to read it anyway.
"""

CONFLICT_PRIOR_CEILING = 0.45
"""A reason code the taxonomy scores at or below this is a hard decline - the
generic issuer refusals and the instrument-unusable family."""

CONFLICT_HISTORY_FLOOR = 0.55
"""...and a customer who has recovered from more than half of their past
failures is evidence pointing the other way. Both at once is a real conflict."""

CONFLICT_MIN_FAILURES = 3
"""Below three prior failures a "recovery rate" is one or two coin flips.
Requiring three keeps the conflict trigger from firing on noise."""

_RETRY_STRATEGIES = {
    Strategy.RETRY_NOW,
    Strategy.RETRY_DELAYED,
    Strategy.RETRY_ON_LIQUIDITY,
}


def should_escalate_to_model(
    event: RevenueEvent, assessment: Assessment
) -> tuple[bool, str]:
    """Decide whether this event needs a model. Returns (use_llm, why).

    The reason string is stored on the Decision row, so the split is auditable
    per event rather than being a claim in a README: every decision records
    which path chose it and on what grounds.
    """
    profile = profile_for(event.reason_code)
    features = assessment.features or {}
    ev = assessment.expected_value_paise

    # --- floors: cases where no amount of reasoning changes the answer -----

    if profile.strategy is Strategy.DO_NOT_RETRY:
        # Includes the unknown-reason fallback. Both are absolute, and policy
        # would deny anything else regardless of how well the model argued it.
        return False, (
            f"do-not-retry: {event.reason_code} is suppressed by the taxonomy, "
            "and no argument the model could make would change that"
        )

    if ev < DEFAULT_BOUNDS.min_expected_value_paise:
        return False, (
            f"below action floor: expected value Rs {ev / 100:,.0f} against the Rs "
            f"{DEFAULT_BOUNDS.min_expected_value_paise / 100:,.0f} action floor - "
            "policy will refuse to act whatever is proposed"
        )

    # --- triggers ---------------------------------------------------------

    if (
        profile.incentive_eligible
        and event.amount_paise >= INCENTIVE_DELIBERATION_FLOOR_PAISE
    ):
        return True, (
            f"incentive depth: {event.reason_code} permits a discount on Rs "
            f"{event.amount_paise / 100:,.0f} - whether to spend margin here, and "
            "how much, is a judgement the taxonomy deliberately does not make"
        )

    if (
        ev >= HIGH_STAKES_EV_PAISE
        and event.amount_paise < DEFAULT_BOUNDS.human_approval_above_paise
    ):
        return True, (
            f"high stakes: Rs {ev / 100:,.0f} of expected recovery, below the Rs "
            f"{DEFAULT_BOUNDS.human_approval_above_paise / 100:,.0f} review line - "
            "expensive to get wrong and nobody else will check it"
        )

    recovery_rate = float(features.get("historical_recovery_rate", 0.0) or 0.0)
    prior_failures = int(features.get("prior_failure_count", 0) or 0)
    if (
        profile.strategy is Strategy.SWITCH_RAIL
        and profile.base_recoverability <= CONFLICT_PRIOR_CEILING
        and recovery_rate >= CONFLICT_HISTORY_FLOOR
        and prior_failures >= CONFLICT_MIN_FAILURES
    ):
        return True, (
            f"conflicting signals: {event.reason_code} scores "
            f"{profile.base_recoverability:.0%} at the reason level, but this "
            f"customer has recovered from {recovery_rate:.0%} of "
            f"{prior_failures} prior failures"
        )

    if event.attempt_no > 1 and profile.strategy in _RETRY_STRATEGIES:
        return True, (
            f"repeat attempt: attempt {event.attempt_no}, so the taxonomy's answer for "
            f"{event.reason_code} has already been tried once and failed"
        )

    return False, (
        f"taxonomy: {event.reason_code} settles to {profile.strategy.value}, "
        f"{'incentive permitted' if profile.incentive_eligible else 'no incentive'}, "
        f"max {profile.max_attempts} attempt(s)"
    )


def _report() -> None:
    """Recount the routing split against the seeded database.

    Kept here rather than in a test: the target share is a design intention, not
    a contract. A test that pinned it would fail every time the reason-code mix
    changed, which is exactly when a human should be *reading* the number rather
    than being told it broke.
    """
    from collections import Counter
    from datetime import datetime, timezone

    from recoup.db import Customer, get_session
    from recoup.detect.scorer import score_event

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    session = get_session()
    events = session.query(RevenueEvent).all()

    routed = 0
    why: Counter[str] = Counter()
    for event in events:
        customer = session.get(Customer, event.customer_id)
        score = score_event(event, customer, now)
        assessment = Assessment(
            event_id=event.id,
            features=score.features,
            recoverability=score.recoverability,
            expected_value_paise=score.expected_value_paise,
            recommended_strategy=score.strategy.value,
            earliest_action_at=score.earliest_action_at,
        )
        use_llm, reason = should_escalate_to_model(event, assessment)
        routed += int(use_llm)
        why[reason.split(":", 1)[0]] += 1

    total = len(events) or 1
    print(f"{routed}/{total} routed to the model ({routed / total:.1%})")
    for trigger, n in why.most_common():
        print(f"  {n:>4}  {n / total:>6.1%}  {trigger}")
    session.close()


if __name__ == "__main__":  # pragma: no cover - operator tool
    _report()
