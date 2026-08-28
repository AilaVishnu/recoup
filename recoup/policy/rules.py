"""The bounds. Every proposed action passes through here before anything happens.

Design principle: **the model proposes, the policy engine disposes.**

Recoup's agent can suggest whatever it likes. It cannot message a customer at
3am, cannot discount a bank outage, cannot retry a fraud decline, cannot spend
past the daily budget, and cannot move on a high-value account without a human.
Those are not instructions in a prompt - they are code that runs after the model
has spoken, and the model has no way to reach around them.

That matters because prompt-level guardrails fail silently and invisibly. A
model that has been told "never exceed a 15% discount" will mostly obey, and the
one time it does not, nothing catches it. A rule that runs afterward catches it
every time, and writes down that it caught it.

Two properties every rule here holds to:

1. **Fail closed.** A rule that cannot evaluate - missing data, unknown reason
   code, arithmetic it cannot complete - denies. Recovery is worth money;
   uncontrolled action costs more.

2. **Record the passes too.** The PolicyReview row stores every check that ran,
   not just the ones that failed. "This action was permitted because these
   thirteen bounds were checked and cleared" is the claim the audit trail exists
   to support, and it cannot be made from a list of failures alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from recoup.db import (
    ActionType,
    Cohort,
    ContactLog,
    PolicyVerdict,
    SpendLog,
)
from recoup.detect.features import is_quiet_hours
from recoup.money import rupees
from recoup.taxonomy import Strategy, profile_for


# ---------------------------------------------------------------------------
# The bounds themselves, in one reviewable block.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bounds:
    """Every limit Recoup operates under. Deliberately one small, readable object.

    A reviewer should be able to read this dataclass and know exactly what the
    system is permitted to do, without reading any other file.
    """

    max_contacts_per_customer_per_week: int = 3
    """Message fatigue is the cost customers actually feel. Three touches in
    seven days across all events - not per event, which is the loophole that
    turns a reasonable cap into a spam cannon for anyone with several failures."""

    max_incentive_fraction: float = 0.15
    """A discount may never exceed 15% of the order value."""

    max_incentive_paise: int = 2_000_00
    """...and never more than Rs 2,000 in absolute terms, whatever the order size.
    A percentage cap alone is unbounded on a large enough order."""

    daily_incentive_budget_paise: int = 25_000_00
    """Rs 25,000 of discount per day, across everything. The blast radius of a
    scoring bug is one day's budget."""

    human_approval_above_paise: int = 25_000_00
    """Above Rs 25,000 of exposure, a person decides. Not because the agent is
    likely wrong, but because the cost of it being wrong stops being routine."""

    min_expected_value_paise: int = 50_00
    """Below Rs 50 of expected recovery, acting is not worth the channel cost or
    the customer's attention."""

    min_incremental_ev_ratio: float = 2.0
    """An incentive must buy at least twice its own cost in expected incremental
    recovery. Not 1.0 - break-even is not a reason to spend money, and the
    estimate has error bars the ratio needs to absorb."""

    max_attempts_override: int | None = None
    """Set to tighten the per-reason attempt ceilings globally. None = trust the
    taxonomy."""

    def clamped_to_default(self) -> "Bounds":
        """Return these bounds, field-wise no looser than the defaults.

        `bounds` is a caller-supplied argument, and until this existed it was a
        switch that turned the policy engine off: passing
        Bounds(human_approval_above_paise=10**14) put a Rs 5,00,000 event through
        to execution with no human - twenty times the autonomy limit - while the
        stored audit row recorded high_value_needs_human as *passed*, "within
        autonomy limit", because a check reports its verdict and never the
        threshold behind it.

        Clamping rather than raising is deliberate. A raise would be louder, and
        would also end the run for every remaining event over one bad call from
        one caller; and the safe fallback here is unambiguous, because a bound
        that is too strict costs a little recovery while one that is too loose
        costs money and customer trust. Tightening still works, so a cautious
        caller keeps its say.

        The safe direction differs per field, which is the whole reason this is
        explicit rather than a min() over a loop: a smaller contact cap is
        stricter, a larger expected-value floor is stricter.
        """
        d = DEFAULT_BOUNDS
        return replace(
            self,
            # Smaller permits less.
            max_contacts_per_customer_per_week=min(
                self.max_contacts_per_customer_per_week,
                d.max_contacts_per_customer_per_week,
            ),
            max_incentive_fraction=min(
                self.max_incentive_fraction, d.max_incentive_fraction
            ),
            max_incentive_paise=min(self.max_incentive_paise, d.max_incentive_paise),
            daily_incentive_budget_paise=min(
                self.daily_incentive_budget_paise, d.daily_incentive_budget_paise
            ),
            human_approval_above_paise=min(
                self.human_approval_above_paise, d.human_approval_above_paise
            ),
            # Larger permits less.
            min_expected_value_paise=max(
                self.min_expected_value_paise, d.min_expected_value_paise
            ),
            min_incremental_ev_ratio=max(
                self.min_incremental_ev_ratio, d.min_incremental_ev_ratio
            ),
            # None means "no override", which is looser than any ceiling.
            max_attempts_override=(
                d.max_attempts_override
                if self.max_attempts_override is None
                else min(
                    self.max_attempts_override,
                    d.max_attempts_override
                    if d.max_attempts_override is not None
                    else self.max_attempts_override,
                )
            ),
        )


DEFAULT_BOUNDS = Bounds()


# ---------------------------------------------------------------------------
# Check plumbing
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    verdict: PolicyVerdict
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "verdict": self.verdict.value,
            "detail": self.detail,
        }


def _naive_utc(value: datetime, field_name: str) -> datetime:
    """Normalise to naive UTC, the one representation the schema stores.

    Not pedantry. db.utcnow() returns an *aware* datetime while every DateTime
    column holds naive values, so an aware `now` reaching _rule_timing_floor
    raises "can't compare offset-naive and offset-aware datetimes" - and a raise
    inside review() used to mean no PolicyReview row at all. Converting at the
    boundary makes the mismatch impossible instead of merely unlikely.
    """
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


@dataclass
class ReviewContext:
    """Everything the rules need, gathered once - and validated before they run.

    Validation lives here so no individual rule has to be defensive about types.
    A rule that has to guard its own inputs will eventually forget, and the
    forgetting is silent.
    """

    event_id: str
    customer_id: str
    cohort: Cohort
    reason_code: str
    amount_paise: int
    expected_value_paise: int
    recoverability: float
    earliest_action_at: datetime
    attempts_so_far: int
    action_type: ActionType
    incentive_paise: int
    now: datetime
    bounds: Bounds = DEFAULT_BOUNDS

    validation_errors: list[str] = field(default_factory=list)
    """Problems found while normalising. Denied by rule, never raised.

    Raising here would be the obvious move and is wrong: construction happens
    before any row is written, so a raise leaves the event with no PolicyReview
    at all - no denial, no record, just a traceback. An unevaluable event is
    exactly the case where the audit trail earns its keep.
    """

    def __post_init__(self) -> None:
        """Normalise to something every rule can evaluate. Never raise.

        Bad values are replaced with inert ones and recorded, so the full set of
        thirteen checks still runs and still produces a reviewable record, with
        _rule_context_is_evaluable supplying the deny.
        """
        self.bounds = self.bounds.clamped_to_default()

        # Coerce enums rather than trusting the caller's spelling. These are
        # str-enums, so a plain string compares equal but is not identical - and
        # one rule in this file used to test identity, which silently classified
        # a held-out event as treatment.
        try:
            self.cohort = Cohort(self.cohort)
        except (ValueError, KeyError):
            self.validation_errors.append(f"unrecognised cohort {self.cohort!r}")
        try:
            self.action_type = ActionType(self.action_type)
        except (ValueError, KeyError):
            self.validation_errors.append(f"unrecognised action_type {self.action_type!r}")

        for name in (
            "amount_paise",
            "expected_value_paise",
            "attempts_so_far",
            "incentive_paise",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self.validation_errors.append(f"{name} is not numeric ({value!r})")
                setattr(self, name, 0)
            else:
                setattr(self, name, int(value))

        if isinstance(self.recoverability, bool) or not isinstance(
            self.recoverability, (int, float)
        ):
            self.validation_errors.append(
                f"recoverability is not numeric ({self.recoverability!r})"
            )
            self.recoverability = 0.0

        for name in ("now", "earliest_action_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime):
                self.validation_errors.append(f"{name} is not a datetime ({value!r})")
                setattr(self, name, datetime.min)
            elif value.tzinfo is not None:
                # db.utcnow() returns an aware datetime while every DateTime
                # column stores naive, so an aware value here used to raise
                # "can't compare offset-naive and offset-aware datetimes" out of
                # the timing rule. Converting at the boundary makes the mismatch
                # impossible rather than merely unlikely.
                setattr(
                    self, name, value.astimezone(timezone.utc).replace(tzinfo=None)
                )


@dataclass
class Review:
    """A verdict, and the provenance that says what it was a verdict *about*.

    The provenance fields are not bookkeeping. Without them a Review is an
    unauthenticated capability token: a genuine ALLOW computed for a harmless
    nudge on one event will authorise a fraud retry on another, a message to a
    held-out customer, or a send at 3am for an approval granted at noon -
    demonstrated for each case in tests/test_policy_bypass.py. execute() can
    only refuse those if the Review says which event, customer and instant it
    was issued for.

    `bounds` is snapshotted for a different reason: the checks record their
    verdicts but not the thresholds behind them, so an audit row could say
    "within autonomy limit" without recording what the limit was.
    """

    verdict: PolicyVerdict
    checks: list[Check] = field(default_factory=list)

    event_id: str | None = None
    customer_id: str | None = None
    action_type: ActionType | None = None
    reviewed_at: datetime | None = None
    bounds: dict[str, Any] = field(default_factory=dict)

    @property
    def violations(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    @property
    def allowed(self) -> bool:
        return self.verdict is PolicyVerdict.ALLOW


CONTACT_ACTIONS = {
    ActionType.NUDGE,
    ActionType.NUDGE_WITH_INCENTIVE,
    ActionType.PAYMENT_LINK,
}
"""Actions the customer perceives. Silent retries are not contact and are not
counted against the fatigue cap - conflating them would make the system refuse
to retry a bank outage because it had sent two emails last week."""


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _rule_never_retry_risk_declines(ctx: ReviewContext, session: Session) -> Check:
    profile = profile_for(ctx.reason_code)
    if profile.strategy is Strategy.DO_NOT_RETRY:
        ok = ctx.action_type in (ActionType.NO_ACTION, ActionType.ESCALATE_TO_HUMAN)
        return Check(
            "never_retry_risk_declines",
            ok,
            PolicyVerdict.ALLOW if ok else PolicyVerdict.DENY,
            f"{ctx.reason_code} is do-not-retry; proposed {ctx.action_type.value}",
        )
    return Check(
        "never_retry_risk_declines", True, PolicyVerdict.ALLOW, "not a do-not-retry reason"
    )


def _rule_context_is_evaluable(ctx: ReviewContext, session: Session) -> Check:
    """Deny anything the context could not be normalised into.

    The inert values substituted in __post_init__ keep the other twelve rules
    evaluable, which is what makes a complete record possible - but they are not
    the caller's data, so nothing may be approved on the strength of them.
    """
    if ctx.validation_errors:
        return Check(
            "context_is_evaluable",
            False,
            PolicyVerdict.DENY,
            "; ".join(ctx.validation_errors) + " - failing closed",
        )
    return Check("context_is_evaluable", True, PolicyVerdict.ALLOW, "inputs well-formed")


def _rule_unknown_reason_fails_closed(ctx: ReviewContext, session: Session) -> Check:
    profile = profile_for(ctx.reason_code)
    if profile.code == "unknown" and ctx.action_type not in (
        ActionType.NO_ACTION,
        ActionType.ESCALATE_TO_HUMAN,
    ):
        return Check(
            "unknown_reason_fails_closed",
            False,
            PolicyVerdict.DENY,
            f"reason '{ctx.reason_code}' is not in the taxonomy - refusing to guess",
        )
    return Check(
        "unknown_reason_fails_closed", True, PolicyVerdict.ALLOW, "reason recognised"
    )


def _rule_attempt_cap(ctx: ReviewContext, session: Session) -> Check:
    profile = profile_for(ctx.reason_code)
    ceiling = profile.max_attempts
    if ctx.bounds.max_attempts_override is not None:
        ceiling = min(ceiling, ctx.bounds.max_attempts_override)

    if ctx.action_type in (ActionType.NO_ACTION, ActionType.ESCALATE_TO_HUMAN):
        return Check("attempt_cap", True, PolicyVerdict.ALLOW, "non-acting proposal")

    ok = ctx.attempts_so_far < ceiling
    return Check(
        "attempt_cap",
        ok,
        PolicyVerdict.ALLOW if ok else PolicyVerdict.DENY,
        f"{ctx.attempts_so_far} prior attempt(s), ceiling {ceiling} for {ctx.reason_code}",
    )


def _rule_timing_floor(ctx: ReviewContext, session: Session) -> Check:
    if ctx.action_type in (ActionType.NO_ACTION, ActionType.ESCALATE_TO_HUMAN):
        return Check("timing_floor", True, PolicyVerdict.ALLOW, "non-acting proposal")

    ok = ctx.now >= ctx.earliest_action_at
    wait = (ctx.earliest_action_at - ctx.now).total_seconds() / 3600
    return Check(
        "timing_floor",
        ok,
        PolicyVerdict.ALLOW if ok else PolicyVerdict.DENY,
        "action window open"
        if ok
        else f"too early by {wait:.1f}h - acting now wastes the attempt",
    )


def _rule_quiet_hours(ctx: ReviewContext, session: Session) -> Check:
    if ctx.action_type not in CONTACT_ACTIONS:
        return Check("quiet_hours", True, PolicyVerdict.ALLOW, "no customer contact")

    quiet = is_quiet_hours(ctx.now)
    return Check(
        "quiet_hours",
        not quiet,
        PolicyVerdict.ALLOW if not quiet else PolicyVerdict.DENY,
        "outside quiet hours" if not quiet else "21:00-08:00 IST - defer to morning",
    )


def _rule_contact_frequency(ctx: ReviewContext, session: Session) -> Check:
    if ctx.action_type not in CONTACT_ACTIONS:
        return Check("contact_frequency", True, PolicyVerdict.ALLOW, "no customer contact")

    since = ctx.now - timedelta(days=7)
    recent = session.scalar(
        select(func.count())
        .select_from(ContactLog)
        .where(ContactLog.customer_id == ctx.customer_id, ContactLog.occurred_at >= since)
    ) or 0

    cap = ctx.bounds.max_contacts_per_customer_per_week
    ok = recent < cap
    return Check(
        "contact_frequency",
        ok,
        PolicyVerdict.ALLOW if ok else PolicyVerdict.DENY,
        f"{recent} contact(s) in the last 7 days, cap {cap}",
    )


def _rule_incentive_eligibility(ctx: ReviewContext, session: Session) -> Check:
    if ctx.incentive_paise <= 0:
        return Check("incentive_eligibility", True, PolicyVerdict.ALLOW, "no incentive")

    profile = profile_for(ctx.reason_code)
    ok = profile.incentive_eligible
    return Check(
        "incentive_eligibility",
        ok,
        PolicyVerdict.ALLOW if ok else PolicyVerdict.DENY,
        "intent-driven failure - discount can move it"
        if ok
        else f"{ctx.reason_code} is a technical failure; a discount buys nothing",
    )


def _rule_incentive_depth(ctx: ReviewContext, session: Session) -> Check:
    if ctx.incentive_paise <= 0:
        return Check("incentive_depth", True, PolicyVerdict.ALLOW, "no incentive")

    if ctx.amount_paise <= 0:
        return Check(
            "incentive_depth", False, PolicyVerdict.DENY, "order value unknown - fail closed"
        )

    frac = ctx.incentive_paise / ctx.amount_paise
    over_frac = frac > ctx.bounds.max_incentive_fraction
    over_abs = ctx.incentive_paise > ctx.bounds.max_incentive_paise
    ok = not (over_frac or over_abs)

    reasons = []
    if over_frac:
        reasons.append(f"{frac:.1%} > {ctx.bounds.max_incentive_fraction:.0%} cap")
    if over_abs:
        reasons.append(
            f"{rupees(ctx.incentive_paise)} > {rupees(ctx.bounds.max_incentive_paise)} absolute cap"
        )

    return Check(
        "incentive_depth",
        ok,
        PolicyVerdict.ALLOW if ok else PolicyVerdict.DENY,
        f"{rupees(ctx.incentive_paise)} ({frac:.1%})"
        if ok
        else "; ".join(reasons),
    )


def _rule_incentive_ev_positive(ctx: ReviewContext, session: Session) -> Check:
    """A discount must buy more than it costs - by a margin, not at break-even.

    The incremental gain is estimated as the incentive's share of remaining
    upside, which is intentionally conservative: it never credits a discount for
    recovery that would have happened anyway.
    """
    if ctx.incentive_paise <= 0:
        return Check("incentive_ev_positive", True, PolicyVerdict.ALLOW, "no incentive")

    headroom = max(0.0, 1.0 - ctx.recoverability)
    est_incremental_paise = int(headroom * 0.35 * ctx.amount_paise)
    ratio = est_incremental_paise / ctx.incentive_paise if ctx.incentive_paise else 0.0
    ok = ratio >= ctx.bounds.min_incremental_ev_ratio

    return Check(
        "incentive_ev_positive",
        ok,
        PolicyVerdict.ALLOW if ok else PolicyVerdict.DENY,
        f"est. incremental {rupees(est_incremental_paise)} vs cost "
        f"{rupees(ctx.incentive_paise)} (ratio {ratio:.1f}x, "
        f"need {ctx.bounds.min_incremental_ev_ratio:.1f}x)",
    )


def _rule_daily_budget(ctx: ReviewContext, session: Session) -> Check:
    if ctx.incentive_paise <= 0:
        return Check("daily_budget", True, PolicyVerdict.ALLOW, "no spend")

    day_start = ctx.now.replace(hour=0, minute=0, second=0, microsecond=0)
    spent = session.scalar(
        select(func.coalesce(func.sum(SpendLog.amount_paise), 0)).where(
            SpendLog.occurred_at >= day_start
        )
    ) or 0

    budget = ctx.bounds.daily_incentive_budget_paise
    ok = spent + ctx.incentive_paise <= budget
    return Check(
        "daily_budget",
        ok,
        PolicyVerdict.ALLOW if ok else PolicyVerdict.DENY,
        f"{rupees(spent)} spent today + {rupees(ctx.incentive_paise)} "
        f"vs {rupees(budget)} budget",
    )


def _rule_minimum_expected_value(ctx: ReviewContext, session: Session) -> Check:
    if ctx.action_type in (ActionType.NO_ACTION, ActionType.ESCALATE_TO_HUMAN):
        return Check("minimum_expected_value", True, PolicyVerdict.ALLOW, "non-acting")

    ok = ctx.expected_value_paise >= ctx.bounds.min_expected_value_paise
    return Check(
        "minimum_expected_value",
        ok,
        PolicyVerdict.ALLOW if ok else PolicyVerdict.DENY,
        f"EV {rupees(ctx.expected_value_paise)} vs floor "
        f"{rupees(ctx.bounds.min_expected_value_paise)}",
    )


def _rule_high_value_needs_human(ctx: ReviewContext, session: Session) -> Check:
    """Not a denial - an escalation. The action may well be right."""
    if ctx.action_type in (ActionType.NO_ACTION, ActionType.ESCALATE_TO_HUMAN):
        return Check("high_value_needs_human", True, PolicyVerdict.ALLOW, "non-acting")

    if ctx.amount_paise >= ctx.bounds.human_approval_above_paise:
        return Check(
            "high_value_needs_human",
            False,
            PolicyVerdict.ESCALATE,
            f"{rupees(ctx.amount_paise)} exceeds the "
            f"{rupees(ctx.bounds.human_approval_above_paise)} autonomy limit",
        )
    return Check("high_value_needs_human", True, PolicyVerdict.ALLOW, "within autonomy limit")


def _rule_control_arm_is_never_executed(ctx: ReviewContext, session: Session) -> Check:
    """The holdout is only worth holding if it is genuinely held.

    Last check to run and the one that must never be relaxed for a demo. A
    control arm that gets acted on "just this once" is not a control arm, and
    every number computed from it afterwards is fiction.
    """
    # Equality, never identity. Cohort is a str-enum, so `'control' is
    # Cohort.CONTROL` is False while `==` is True - and this was the only rule in
    # the file testing identity, which meant a cohort arriving as a plain string
    # from a replay or a webhook was waved through as "treatment arm". A
    # contaminated holdout cannot be detected after the fact; every number
    # computed from it is quietly wrong forever.
    #
    # Unrecognised cohorts deny rather than defaulting to treatment. ReviewContext
    # now coerces this field, so reaching the else branch means something is very
    # wrong, and guessing is the wrong response to that.
    if ctx.cohort == Cohort.TREATMENT:
        return Check(
            "control_arm_suppression", True, PolicyVerdict.ALLOW, "treatment arm"
        )
    if ctx.cohort != Cohort.CONTROL:
        return Check(
            "control_arm_suppression",
            False,
            PolicyVerdict.DENY,
            f"unrecognised cohort {ctx.cohort!r} - refusing to assume treatment",
        )
    if ctx.action_type in (ActionType.NO_ACTION, ActionType.ESCALATE_TO_HUMAN):
        return Check("control_arm_suppression", True, PolicyVerdict.ALLOW, "non-acting")
    return Check(
        "control_arm_suppression",
        False,
        PolicyVerdict.DENY,
        "control arm - decision recorded, execution suppressed",
    )


RULES: list[Callable[[ReviewContext, Session], Check]] = [
    _rule_context_is_evaluable,
    _rule_unknown_reason_fails_closed,
    _rule_never_retry_risk_declines,
    _rule_attempt_cap,
    _rule_timing_floor,
    _rule_quiet_hours,
    _rule_contact_frequency,
    _rule_incentive_eligibility,
    _rule_incentive_depth,
    _rule_incentive_ev_positive,
    _rule_daily_budget,
    _rule_minimum_expected_value,
    _rule_high_value_needs_human,
    _rule_control_arm_is_never_executed,
]


def review(ctx: ReviewContext, session: Session) -> Review:
    """Run every rule. All of them, always - no short-circuiting.

    Stopping at the first failure would be faster and would make the audit trail
    useless: "denied by quiet_hours" hides that the action also blew the budget
    and exceeded the attempt cap. Recoup runs the full set so the record shows
    everything that was wrong, not merely the first thing noticed.
    """
    checks: list[Check] = []
    for rule in RULES:
        try:
            checks.append(rule(ctx, session))
        except Exception as exc:  # noqa: BLE001 - the point is to catch everything
            # Fail closed, and say so in the audit trail.
            #
            # This module's docstring has always promised that a rule which
            # cannot evaluate denies. Until this existed the promise was false:
            # five different malformed inputs raised TypeError out of review()
            # instead, and because the raise happened before any row was
            # written, the event ended up with no PolicyReview at all - no
            # denial, no record, just a stack trace. An unevaluable event is
            # exactly the case where the audit trail matters most.
            checks.append(
                Check(
                    rule.__name__.removeprefix("_rule_"),
                    False,
                    PolicyVerdict.DENY,
                    f"rule raised {type(exc).__name__}: {exc} - failing closed",
                )
            )

    if any(c.verdict is PolicyVerdict.DENY and not c.passed for c in checks):
        verdict = PolicyVerdict.DENY
    elif any(c.verdict is PolicyVerdict.ESCALATE and not c.passed for c in checks):
        verdict = PolicyVerdict.ESCALATE
    else:
        verdict = PolicyVerdict.ALLOW

    return Review(
        verdict=verdict,
        checks=checks,
        # Provenance, so execute() can verify this Review was issued for the
        # thing it is about to act on rather than trusting whatever it was
        # handed. See the Review docstring.
        event_id=ctx.event_id,
        customer_id=ctx.customer_id,
        action_type=ctx.action_type,
        reviewed_at=ctx.now,
        bounds=asdict(ctx.bounds),
    )
