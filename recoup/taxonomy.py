"""Payment-failure taxonomy: why a payment died, and what that implies for recovery.

This is the domain core of Recoup. Every downstream component - the scorer, the
agent, the policy engine - reads from this table rather than guessing.

The key insight the whole project rests on: *a failed payment is not one thing.*
An issuer outage and a fraud decline both surface as "payment failed", but one
should be retried in two hours untouched and the other must never be retried at
all. Treating them alike is how naive recovery systems burn money and, worse,
push declined-for-risk transactions back through the rails.

Reason codes and the source/step vocabulary mirror Razorpay's payment error
schema (error_source / error_step / error_reason).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Rail(str, Enum):
    """A payment instrument family. Recovery often means switching rails."""

    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMI = "emi"


class Strategy(str, Enum):
    """The recovery archetype implied by a failure reason."""

    RETRY_NOW = "retry_now"
    """Customer is still at the keyboard. Retry the same rail immediately."""

    RETRY_DELAYED = "retry_delayed"
    """A transient system fault. Wait for it to clear, then retry untouched."""

    RETRY_ON_LIQUIDITY = "retry_on_liquidity"
    """Customer lacked funds. Time the retry to when money is likely present."""

    SWITCH_RAIL = "switch_rail"
    """This instrument cannot work. Offer a different one via a Payment Link."""

    PERSUADE = "persuade"
    """Nothing was broken - the customer chose not to pay. An intent problem."""

    DO_NOT_RETRY = "do_not_retry"
    """Retrying is harmful, forbidden, or futile. Suppress and report."""


@dataclass(frozen=True)
class FailureProfile:
    """Everything Recoup knows about one failure reason code."""

    code: str
    label: str

    source: str
    """Razorpay error_source: customer | bank | issuer | gateway | network | business | internal."""

    step: str
    """Razorpay error_step: payment_initiation | payment_authentication | payment_authorization | payment_response."""

    strategy: Strategy

    base_recoverability: float
    """P(recovered | this strategy applied), from a cold start.

    These are priors, not measured truth. They seed the scorer on day one and are
    superseded by observed rates once the eval harness has enough events - see
    recoup/detect/scorer.py. Every number here is defensible as an order of
    magnitude, none as a decimal point, and the report says so.
    """

    retry_after_minutes: int = 0
    """Minimum wait before a retry is meaningful. 0 = the customer is still present."""

    max_attempts: int = 2
    """Hard ceiling on automated retries for this reason. Beyond this, stop."""

    incentive_eligible: bool = False
    """Whether a discount could plausibly change the outcome.

    False for every technical failure. Discounting a bank outage is pure margin
    burn - the customer already wanted to pay, so you paid them to do what they
    were going to do anyway. Incentives only make sense where intent is the
    blocker.
    """

    switch_to: tuple[Rail, ...] = field(default_factory=tuple)
    """Rails worth offering instead, best-first."""

    note: str = ""
    """Why this profile is what it is. Surfaced in the audit trail."""


# ---------------------------------------------------------------------------
# The table.
# ---------------------------------------------------------------------------

PROFILES: dict[str, FailureProfile] = {
    # ---- Transient infrastructure: high recovery, zero incentive, just wait ----
    "issuer_down": FailureProfile(
        code="issuer_down",
        label="Issuing bank unavailable",
        source="issuer",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        base_recoverability=0.82,
        retry_after_minutes=120,
        max_attempts=3,
        incentive_eligible=False,
        switch_to=(Rail.UPI, Rail.NETBANKING),
        note="Nothing is wrong with the customer or the card. Wait out the outage.",
    ),
    "gateway_technical_error": FailureProfile(
        code="gateway_technical_error",
        label="Gateway technical error",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        base_recoverability=0.78,
        retry_after_minutes=30,
        max_attempts=3,
        incentive_eligible=False,
        note="Transient. Retry before assuming customer-side intent.",
    ),
    "payment_timeout": FailureProfile(
        code="payment_timeout",
        label="Payment timed out",
        source="network",
        step="payment_response",
        strategy=Strategy.RETRY_NOW,
        base_recoverability=0.71,
        retry_after_minutes=0,
        max_attempts=2,
        incentive_eligible=False,
        note="Customer is likely still in-session. Re-present immediately.",
    ),
    # ---- Customer present, fixable friction: retry now, same rail ----
    "incorrect_otp": FailureProfile(
        code="incorrect_otp",
        label="Incorrect OTP entered",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.RETRY_NOW,
        base_recoverability=0.74,
        retry_after_minutes=0,
        max_attempts=2,
        incentive_eligible=False,
        note="Fat-finger. Highest-yield, lowest-cost recovery there is.",
    ),
    "invalid_cvv": FailureProfile(
        code="invalid_cvv",
        label="Invalid CVV",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.RETRY_NOW,
        base_recoverability=0.69,
        retry_after_minutes=0,
        max_attempts=2,
        incentive_eligible=False,
        note="Correctable in-session, like an OTP slip. No money needs to be spent.",
    ),
    "authentication_failed": FailureProfile(
        code="authentication_failed",
        label="3DS authentication failed",
        source="issuer",
        step="payment_authentication",
        strategy=Strategy.SWITCH_RAIL,
        base_recoverability=0.48,
        retry_after_minutes=5,
        max_attempts=2,
        incentive_eligible=False,
        switch_to=(Rail.UPI, Rail.NETBANKING),
        note="Repeated 3DS failure rarely self-resolves. Offer a rail without 3DS.",
    ),
    # ---- Liquidity: recoverable, but only if the timing is right ----
    "insufficient_funds": FailureProfile(
        code="insufficient_funds",
        label="Insufficient funds",
        source="customer",
        step="payment_authorization",
        strategy=Strategy.RETRY_ON_LIQUIDITY,
        base_recoverability=0.55,
        retry_after_minutes=60 * 24,
        max_attempts=3,
        incentive_eligible=False,
        switch_to=(Rail.UPI, Rail.EMI),
        note=(
            "Intent is intact, the balance is not. Retrying in five minutes fails "
            "again and burns an attempt; timing the retry to a salary window is the "
            "entire game. See recoup/detect/features.py:liquidity_window."
        ),
    ),
    "payment_limit_exceeded": FailureProfile(
        code="payment_limit_exceeded",
        label="Per-transaction limit exceeded",
        source="issuer",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        base_recoverability=0.52,
        retry_after_minutes=0,
        max_attempts=2,
        incentive_eligible=False,
        switch_to=(Rail.NETBANKING, Rail.UPI, Rail.EMI),
        note="The rail has a ceiling, the customer does not. Route around it.",
    ),
    # ---- Instrument unusable: must switch rails ----
    "card_expired": FailureProfile(
        code="card_expired",
        label="Card expired",
        source="issuer",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        base_recoverability=0.44,
        retry_after_minutes=0,
        max_attempts=1,
        incentive_eligible=False,
        switch_to=(Rail.UPI, Rail.NETBANKING),
        note="Retrying the same card is guaranteed to fail. One attempt, new rail.",
    ),
    "international_transaction_not_allowed": FailureProfile(
        code="international_transaction_not_allowed",
        label="International transactions blocked on card",
        source="issuer",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        base_recoverability=0.41,
        retry_after_minutes=0,
        max_attempts=1,
        incentive_eligible=False,
        switch_to=(Rail.UPI, Rail.NETBANKING),
        note="A card setting the customer cannot change mid-checkout.",
    ),
    "transaction_not_permitted": FailureProfile(
        code="transaction_not_permitted",
        label="Transaction type not permitted on instrument",
        source="issuer",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        base_recoverability=0.38,
        retry_after_minutes=0,
        max_attempts=1,
        incentive_eligible=False,
        switch_to=(Rail.UPI, Rail.NETBANKING),
    ),
    # ---- Intent problem: the only place an incentive is defensible ----
    "payment_cancelled": FailureProfile(
        code="payment_cancelled",
        label="Customer cancelled the payment",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.PERSUADE,
        base_recoverability=0.21,
        retry_after_minutes=60 * 4,
        max_attempts=1,
        incentive_eligible=True,
        note=(
            "Nothing broke. The customer looked at the price and walked. This is the "
            "one bucket where spending money can change the answer - and the one "
            "bucket where it is easiest to spend it on people who would have "
            "converted anyway. Hence the control group."
        ),
    ),
    # ---- Never retry ----
    "fraud_suspected": FailureProfile(
        code="fraud_suspected",
        label="Declined for suspected fraud",
        source="issuer",
        step="payment_authorization",
        strategy=Strategy.DO_NOT_RETRY,
        base_recoverability=0.0,
        retry_after_minutes=0,
        max_attempts=0,
        incentive_eligible=False,
        note=(
            "Hard stop. Re-presenting a risk decline is how a merchant earns a "
            "higher decline rate and a chargeback problem. Recoup suppresses these "
            "and reports the suppressed value rather than silently dropping it."
        ),
    ),
    "card_declined": FailureProfile(
        code="card_declined",
        label="Declined by issuer, no reason given",
        source="issuer",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        base_recoverability=0.29,
        retry_after_minutes=60,
        max_attempts=1,
        incentive_eligible=False,
        switch_to=(Rail.UPI,),
        note=(
            "A generic decline hides both soft and hard causes. Treated "
            "conservatively: one attempt, different rail, no incentive."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Non-payment-failure event kinds. Same interface, different origin.
# ---------------------------------------------------------------------------

CHECKOUT_ABANDONED = FailureProfile(
    code="checkout_abandoned",
    label="Checkout abandoned before payment",
    source="customer",
    step="payment_initiation",
    strategy=Strategy.PERSUADE,
    base_recoverability=0.13,
    retry_after_minutes=60,
    max_attempts=2,
    incentive_eligible=True,
    note="No payment was ever attempted, so there is nothing to retry - only intent to move.",
)

INVOICE_OVERDUE = FailureProfile(
    code="invoice_overdue",
    label="Receivable past due date",
    source="business",
    step="payment_initiation",
    strategy=Strategy.PERSUADE,
    base_recoverability=0.62,
    retry_after_minutes=60 * 24,
    max_attempts=4,
    incentive_eligible=False,
    note=(
        "B2B receivables are usually collected, not converted - the question is "
        "when, not whether. An escalation ladder beats discounting."
    ),
)

_EXTRA = {p.code: p for p in (CHECKOUT_ABANDONED, INVOICE_OVERDUE)}

UNKNOWN = FailureProfile(
    code="unknown",
    label="Unclassified failure",
    source="internal",
    step="payment_authorization",
    strategy=Strategy.DO_NOT_RETRY,
    base_recoverability=0.0,
    retry_after_minutes=0,
    max_attempts=0,
    incentive_eligible=False,
    note=(
        "Fail closed. An unrecognised reason code means the taxonomy is stale; the "
        "safe default is to touch nothing and surface it for a human. Volume here "
        "is reported as a coverage metric, not swept under the rug."
    ),
)


def profile_for(code: str | None) -> FailureProfile:
    """Look up a failure profile, falling back to a deliberately inert default."""
    if not code:
        return UNKNOWN
    key = code.strip().lower()
    return PROFILES.get(key) or _EXTRA.get(key) or UNKNOWN


def all_codes() -> list[str]:
    """Every reason code the taxonomy recognises."""
    return sorted([*PROFILES, *_EXTRA])
