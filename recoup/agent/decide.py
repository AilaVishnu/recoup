"""One event, one Decision row. The stage boundary the rest of the pipeline joins on.

Deliberately narrow. This function routes, proposes, and writes - it does not
run the policy engine and it does not execute anything. Keeping proposal and
authorisation in separate modules is the whole architecture: a Decision row is
what the agent *wanted*, a PolicyReview row is what it was *allowed*, and the
gap between them is the most interesting column in the audit trail. Merging the
two would make that gap unobservable and turn "the model proposes, policy
disposes" into an unfalsifiable slogan.

It also does not advance RevenueEvent.status. A decision alone does not tell you
whether the event ends up ACTED, SUPPRESSED or AWAITING_APPROVAL - only the
policy verdict does - so the status pointer is moved by the stage that can see
it.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from recoup.agent import brain, rules_engine
from recoup.agent.router import should_escalate_to_model
from recoup.db import Assessment, Customer, Decision, DecisionSource, RevenueEvent


def decide(
    session: Session,
    event: RevenueEvent,
    assessment: Assessment,
    customer: Customer,
    now: datetime,
) -> Decision:
    """Choose an action for one event and persist the proposal.

    `now` is injected rather than read from the clock so that a replay over
    historical events reproduces the timestamps it originally wrote.
    """
    use_llm, routing_reason = should_escalate_to_model(event, assessment)

    if use_llm:
        action_type, params, rationale, usage = brain.propose(event, assessment, customer)
        # A model call that fell back is a rules decision that cost tokens. It is
        # recorded as RULES, because DecisionSource answers "what produced this
        # action", and counting fallbacks as LLM would inflate exactly the number
        # this project uses to argue the model is used sparingly.
        source = DecisionSource.RULES if usage.fell_back else DecisionSource.LLM
        model = usage.model
        input_tokens, output_tokens = usage.input_tokens, usage.output_tokens
    else:
        action_type, params, rationale = rules_engine.propose_from_rules(
            event, assessment, customer
        )
        source = DecisionSource.RULES
        model = None
        input_tokens = output_tokens = 0

    # The routing decision travels with the proposal. Without it the audit trail
    # can say a decision came from rules but not whether that was because the
    # taxonomy settled it or because a model call failed - which are very
    # different facts about a run.
    params = {
        **params,
        "routed_to": "model" if use_llm else "rules",
        "routing_reason": routing_reason,
    }

    decision = Decision(
        event_id=event.id,
        created_at=now,
        source=source,
        action_type=action_type,
        params=params,
        rationale=rationale,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    session.add(decision)
    # Flushed, not committed: PolicyReview needs decision.id, and the caller owns
    # the transaction boundary for the whole event.
    session.flush()
    return decision
