"""Persistence layer and the audit trail.

The schema is shaped by one requirement above all others: **every rupee Recoup
claims to have recovered must be traceable back through the exact decision that
recovered it.** A single wide "events" table with a status column would have been
less code, but it would make the central claim of the project unverifiable.

So one row per stage, immutable once written:

    RevenueEvent   what went wrong, and how much money is at stake
      -> Assessment    what the deterministic scorer computed, and from what
      -> Decision      what was proposed, by whom (rules or LLM), and why
      -> PolicyReview  which bounds were checked, and what the verdict was
      -> ActionRun     what actually executed against Razorpay, and what it cost
      -> Outcome       whether the money came back, and whether we may claim it

Nothing is updated in place except the event's own status pointer. A replay of
any event is a join, not a reconstruction.

All money is stored as integer paise. Never floats - a rupee is not a float, and
a recovery report that disagrees with itself in the third decimal place is a
report nobody trusts.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from recoup.config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventKind(str, enum.Enum):
    PAYMENT_FAILED = "payment_failed"
    CHECKOUT_ABANDONED = "checkout_abandoned"
    INVOICE_OVERDUE = "invoice_overdue"


class Cohort(str, enum.Enum):
    """Randomised assignment, fixed at event creation and never revisited.

    TREATMENT events are acted on. CONTROL events are scored and decided exactly
    as treatment events are - the full decision is recorded - but execution is
    suppressed at the last step. That asymmetry is what makes the eval honest:
    we know precisely what Recoup *would* have done to each control event, so
    the comparison is against a counterfactual we can actually inspect rather
    than an untouched population we merely hope is comparable.
    """

    TREATMENT = "treatment"
    CONTROL = "control"


class EventStatus(str, enum.Enum):
    OPEN = "open"
    ASSESSED = "assessed"
    DECIDED = "decided"
    SUPPRESSED = "suppressed"
    """Policy or taxonomy forbade any action. Value is reported, not discarded."""
    AWAITING_APPROVAL = "awaiting_approval"
    """Exceeded an autonomy bound and is queued for a human."""
    ACTED = "acted"
    RECOVERED = "recovered"
    LOST = "lost"


class PolicyVerdict(str, enum.Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"
    """Defensible, but above the autonomy threshold. A human decides."""


class DecisionSource(str, enum.Enum):
    RULES = "rules"
    """Resolved from the taxonomy alone. No LLM call was made or needed."""
    LLM = "llm"
    SUPPRESSED = "suppressed"


class ActionType(str, enum.Enum):
    RETRY_PAYMENT = "retry_payment"
    PAYMENT_LINK = "payment_link"
    NUDGE = "nudge"
    NUDGE_WITH_INCENTIVE = "nudge_with_incentive"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    NO_ACTION = "no_action"


class ActionStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    SKIPPED_DRY_RUN = "skipped_dry_run"


class Attribution(str, enum.Enum):
    """Why we believe a recovery happened. Central to not overclaiming."""

    AGENT = "agent"
    """Recovered through a link or retry Recoup itself created. Directly traceable."""
    ORGANIC = "organic"
    """Customer came back on their own. Recoup takes no credit."""
    UNCLEAR = "unclear"
    """Recovered after an action, but not through it. Counted separately and
    never folded into the headline number - see recoup/eval/metrics.py."""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(200))
    contact: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Behavioural history. The scorer leans on these heavily - a customer who has
    # recovered twice before is a very different prospect from a first-timer.
    prior_success_count: Mapped[int] = mapped_column(Integer, default=0)
    prior_failure_count: Mapped[int] = mapped_column(Integer, default=0)
    prior_recovery_count: Mapped[int] = mapped_column(Integer, default=0)
    lifetime_value_paise: Mapped[int] = mapped_column(Integer, default=0)
    preferred_rail: Mapped[str] = mapped_column(String(20), default="card")

    razorpay_customer_id: Mapped[str | None] = mapped_column(String(40), nullable=True)

    events: Mapped[list[RevenueEvent]] = relationship(back_populates="customer")


class RevenueEvent(Base):
    """One unit of revenue at risk."""

    __tablename__ = "revenue_events"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[EventKind] = mapped_column(SAEnum(EventKind))
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)

    amount_paise: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)

    reason_code: Mapped[str] = mapped_column(String(60), index=True)
    """Key into recoup.taxonomy.PROFILES."""
    rail: Mapped[str] = mapped_column(String(20), default="card")

    razorpay_order_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    razorpay_payment_id: Mapped[str | None] = mapped_column(String(40), nullable=True)

    cohort: Mapped[Cohort] = mapped_column(SAEnum(Cohort), index=True)
    status: Mapped[EventStatus] = mapped_column(
        SAEnum(EventStatus), default=EventStatus.OPEN, index=True
    )

    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    """Which attempt on this order this failure represents."""

    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    customer: Mapped[Customer] = relationship(back_populates="events")
    assessment: Mapped[Assessment | None] = relationship(
        back_populates="event", uselist=False
    )
    decisions: Mapped[list[Decision]] = relationship(back_populates="event")
    outcome: Mapped[Outcome | None] = relationship(back_populates="event", uselist=False)


class Assessment(Base):
    """Deterministic scoring output. No LLM involved at this stage."""

    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("revenue_events.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    """Every input the score was computed from, verbatim. Makes the score reproducible."""

    recoverability: Mapped[float] = mapped_column(Float)
    """P(recovered | intervention), in [0, 1]."""

    expected_value_paise: Mapped[int] = mapped_column(Integer)
    """recoverability * amount, before any action cost is subtracted."""

    recommended_strategy: Mapped[str] = mapped_column(String(40))
    earliest_action_at: Mapped[datetime] = mapped_column(DateTime)
    """Before this instant, acting is worse than waiting. Enforced by policy."""

    scorer_version: Mapped[str] = mapped_column(String(20), default="v1")

    event: Mapped[RevenueEvent] = relationship(back_populates="assessment")


class Decision(Base):
    """A proposed action, before policy has had its say."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("revenue_events.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    source: Mapped[DecisionSource] = mapped_column(SAEnum(DecisionSource))
    action_type: Mapped[ActionType] = mapped_column(SAEnum(ActionType))
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    rationale: Mapped[str] = mapped_column(Text, default="")
    """Why this action. Written by the rules engine or the model; shown in the UI."""

    model: Mapped[str | None] = mapped_column(String(60), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    event: Mapped[RevenueEvent] = relationship(back_populates="decisions")
    review: Mapped[PolicyReview | None] = relationship(
        back_populates="decision", uselist=False
    )
    run: Mapped[ActionRun | None] = relationship(back_populates="decision", uselist=False)


class PolicyReview(Base):
    """The bounds check. Runs on every decision, LLM-authored or not."""

    __tablename__ = "policy_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[int] = mapped_column(ForeignKey("decisions.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    verdict: Mapped[PolicyVerdict] = mapped_column(SAEnum(PolicyVerdict))

    checks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    """Every rule evaluated, with its outcome - passes included.

    Recording the passes matters as much as the failures: "the agent was allowed
    to do this because these eleven bounds were checked and cleared" is the claim
    the audit trail has to support.
    """

    violations: Mapped[list[str]] = mapped_column(JSON, default=list)

    decision: Mapped[Decision] = relationship(back_populates="review")


class ActionRun(Base):
    """What actually happened when the approved action was executed."""

    __tablename__ = "action_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[int] = mapped_column(ForeignKey("decisions.id"), index=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    action_type: Mapped[ActionType] = mapped_column(SAEnum(ActionType))
    status: Mapped[ActionStatus] = mapped_column(SAEnum(ActionStatus))

    razorpay_ref: Mapped[str | None] = mapped_column(String(60), nullable=True)
    """Payment Link id, order id, or whatever the executor created. The thread
    that ties a later recovery back to this specific action."""

    incentive_paise: Mapped[int] = mapped_column(Integer, default=0)
    """Discount actually granted. Counts against the daily budget and is
    subtracted from gross recovery in every reported figure."""

    channel_cost_paise: Mapped[int] = mapped_column(Integer, default=0)
    """Messaging cost. Small per unit, not small in aggregate."""

    response: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    decision: Mapped[Decision] = relationship(back_populates="run")


class Outcome(Base):
    """Ground truth, resolved after the recovery window closes."""

    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("revenue_events.id"), index=True)
    resolved_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    recovered: Mapped[bool] = mapped_column(Boolean, default=False)
    recovered_paise: Mapped[int] = mapped_column(Integer, default=0)
    attribution: Mapped[Attribution] = mapped_column(
        SAEnum(Attribution), default=Attribution.ORGANIC
    )
    hours_to_recovery: Mapped[float | None] = mapped_column(Float, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")

    event: Mapped[RevenueEvent] = relationship(back_populates="outcome")


class ContactLog(Base):
    """Append-only record of every customer touch, for enforcing contact caps.

    Separate from ActionRun on purpose: the cap must be enforceable by a cheap
    indexed count over a time window, and it must survive changes to how actions
    are modelled. Fatigue limits are the one bound whose failure the customer
    feels directly.
    """

    __tablename__ = "contact_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(String(40), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    action_type: Mapped[ActionType] = mapped_column(SAEnum(ActionType))
    event_id: Mapped[str] = mapped_column(String(40))


class SpendLog(Base):
    """Append-only incentive spend, for enforcing the daily budget cap."""

    __tablename__ = "spend_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    amount_paise: Mapped[int] = mapped_column(Integer)
    event_id: Mapped[str] = mapped_column(String(40))


# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------

_engine = None
_Session = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        url = settings.db_url
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
            # Resolve the relative path against the project root rather than the
            # caller's cwd, so scripts/ and the API server share one database.
            from recoup.config import PROJECT_ROOT

            rel = url.removeprefix("sqlite:///")
            target = PROJECT_ROOT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{target}"
        _engine = create_engine(url, future=True)
    return _engine


def get_session():
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine(), future=True, expire_on_commit=False)
    return _Session()


def init_db(drop: bool = False) -> None:
    engine = get_engine()
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
