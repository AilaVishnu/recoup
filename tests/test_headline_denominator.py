"""The headline is intention-to-treat. Nothing enforced that until this file.

A mutation audit swapped the denominator in the arm rate from every event in the
arm to only the events Recoup acted on. The published headline moved from +7.6pp
with an interval straddling zero to +10.9pp with an interval clearing it - the
difference between "cannot rule out no effect" and "demonstrated" - and all 280
tests still passed.

That is the most dangerous shape a defect can take here. It requires no bug: the
per-protocol rate is a real quantity, `treated_recovery_rate` reports it
deliberately, and a reader who saw only the larger number would have no way to
tell which denominator produced it. metrics.py argues the choice at length in its
own docstring, and an argument in a docstring is not a constraint.

So the two rates are pinned apart, and the headline is pinned to the conservative
one. Conditioning on having acted selects for events that were easier to begin
with - the ones policy cleared, the taxonomy was willing to touch, and the
scheduler reached in time - which is exactly the selection an intention-to-treat
denominator exists to refuse.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recoup.db import ActionType, Attribution, Cohort, EventKind
from recoup.eval.metrics import build_segments
from recoup.eval.resolve import ResolvedOutcome

OCCURRED = datetime(2026, 3, 1, 9, 0)


def outcome(
    *,
    eid: str,
    cohort: Cohort = Cohort.TREATMENT,
    treated: bool,
    recovered: bool,
    attribution: Attribution = Attribution.ORGANIC,
    amount_paise: int = 1_000_00,
) -> ResolvedOutcome:
    return ResolvedOutcome(
        event_id=eid,
        cohort=cohort,
        event_kind=EventKind.PAYMENT_FAILED.value,
        reason_code="incorrect_otp",
        strategy="retry_now",
        amount_paise=amount_paise,
        treated=treated,
        action_type=ActionType.RETRY_PAYMENT.value if treated else None,
        hours_waited=0.5 if treated else None,
        incentive_committed_paise=0,
        channel_cost_paise=0,
        organic_p=0.3,
        treated_p=0.48,
        roll=0.4,
        recovered=recovered,
        recovered_paise=amount_paise if recovered else 0,
        attribution=attribution if recovered else Attribution.ORGANIC,
        hours_to_recovery=2.0 if recovered else None,
        note="",
    )


def a_population():
    """A treatment arm where acting was selective, which is the realistic case.

    Ten treated events, six of them acted on and doing well; four untouched -
    policy denied them, or the taxonomy refused - and doing badly. Any denominator
    that drops the four reports a materially better product than the merchant has.
    """
    rows = []
    # Acted on: 4 of 6 recover.
    for i in range(6):
        rows.append(
            outcome(
                eid=f"t_acted_{i}",
                treated=True,
                recovered=i < 4,
                attribution=Attribution.AGENT if i < 4 else Attribution.ORGANIC,
            )
        )
    # In the arm, never acted on: 0 of 4 recover.
    for i in range(4):
        rows.append(outcome(eid=f"t_untouched_{i}", treated=False, recovered=False))
    # Control arm: 3 of 10.
    for i in range(10):
        rows.append(
            outcome(eid=f"c_{i}", cohort=Cohort.CONTROL, treated=False, recovered=i < 3)
        )
    return rows


def test_the_arm_rate_counts_every_event_in_the_arm():
    """4 of 10, not 4 of 6. The refusals are part of the product."""
    overall = build_segments(a_population())["overall"]

    assert overall.treatment.n == 10, "the arm is every event assigned to it"
    assert overall.gross_recovery_rate == pytest.approx(0.4), (
        "gross recovery must be 4/10 (intention-to-treat), not 4/6 (per-protocol)"
    )


def test_the_per_protocol_rate_is_reported_but_is_not_the_headline():
    """Both numbers exist. Only one of them is allowed to be the result."""
    overall = build_segments(a_population())["overall"]

    assert overall.treated_n == 6, "per-protocol denominator is acted-on events"
    assert overall.treated_recovery_rate == pytest.approx(4 / 6)
    assert overall.gross_recovery_rate < overall.treated_recovery_rate, (
        "if these ever coincide this test proves nothing - the fixture must keep "
        "a population where policy refused to act on part of the arm"
    )


def test_lift_is_computed_from_the_intention_to_treat_rate():
    """The number the project publishes, pinned to the conservative denominator.

    ITT: 0.40 - 0.30 = +10pp. Per-protocol: 0.667 - 0.30 = +36.7pp. A reader has
    no way to tell those apart from the figure alone, which is why the choice
    cannot live only in a docstring.
    """
    overall = build_segments(a_population())["overall"]

    assert overall.lift.absolute == pytest.approx(0.4 - 0.3), (
        "lift moved off the intention-to-treat denominator"
    )
    per_protocol_lift = (4 / 6) - 0.3
    assert overall.lift.absolute != pytest.approx(per_protocol_lift)


def test_an_arm_of_untouched_events_does_not_report_a_perfect_rate():
    """The degenerate case the per-protocol denominator produces.

    If nothing in the arm was acted on, per-protocol has no denominator at all
    and a naive implementation divides by zero or reports 0/0 as 100%. ITT gives
    the honest answer: the arm exists, nothing recovered, the rate is zero.
    """
    rows = [outcome(eid=f"t_{i}", treated=False, recovered=False) for i in range(8)]
    rows += [
        outcome(eid=f"c_{i}", cohort=Cohort.CONTROL, treated=False, recovered=i < 2)
        for i in range(8)
    ]
    overall = build_segments(rows)["overall"]

    assert overall.treated_n == 0
    assert overall.gross_recovery_rate == pytest.approx(0.0)
    assert overall.lift.absolute == pytest.approx(-0.25), (
        "an arm Recoup never touched should show the control arm ahead, not a "
        "vacuous 100% from an empty per-protocol denominator"
    )


def test_net_subtracts_recoveries_the_action_destroyed():
    """The report prints harm as a cost line, so net must actually subtract it.

    It did not. `net` was incremental minus spend, while "recoveries destroyed"
    sat directly above it in the cost block captioned as part of "all cost". That
    reads correctly only while harm is zero, which is how a figure like this
    survives review - right by luck, wrong by construction, and overstating the
    moment an action starts costing recoveries.

    The two sets are disjoint. An AGENT attribution is a recovery that happened
    only because Recoup acted; a harmed event is a recovery that failed to happen
    only because Recoup acted. Subtracting the full amount double-counts nothing.
    """
    rows = []
    # Two events Recoup caused to recover.
    for i in range(2):
        rows.append(
            outcome(
                eid=f"t_gain_{i}",
                treated=True,
                recovered=True,
                attribution=Attribution.AGENT,
                amount_paise=10_000_00,
            )
        )
    # One the action destroyed: it would have recovered untouched.
    harmed = outcome(eid="t_harm", treated=True, recovered=False, amount_paise=4_000_00)
    harmed = type(harmed)(**{**harmed.__dict__, "roll": 0.1, "organic_p": 0.9})
    rows.append(harmed)
    rows += [
        outcome(eid=f"c_{i}", cohort=Cohort.CONTROL, treated=False, recovered=i < 1)
        for i in range(4)
    ]

    overall = build_segments(rows)["overall"]

    assert overall.harmed_n == 1, "fixture must actually produce a harmed event"
    assert overall.harmed_paise == 4_000_00
    assert overall.net_paise == (
        overall.incremental_recovered_paise
        - overall.cost_paise
        - overall.harmed_paise
    ), "net must subtract destroyed recoveries, not merely print them"
