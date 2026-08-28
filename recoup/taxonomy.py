"""Payment-failure taxonomy: why a payment died, and what that implies for recovery.

This is the domain core of Recoup. Every downstream component - the scorer, the
agent, the policy engine - reads from this table rather than guessing.

The key insight the whole project rests on: *a failed payment is not one thing.*
A bank outage and a risk decline both surface as "payment failed", but one
should be retried in two hours untouched and the other must never be retried at
all. Treating them alike is how naive recovery systems burn money and, worse,
push declined-for-risk transactions back through the rails.

Every reason code here is a real Razorpay `error_reason`
--------------------------------------------------------
Taken from https://razorpay.com/docs/errors/payments/list/. That sentence is
load-bearing, because an earlier version of this file was not: it invented
`invalid_cvv` (Razorpay emits `incorrect_cvv`), `payment_timeout` (`payment_timed_out`),
`issuer_down`, `fraud_suspected`, `payment_limit_exceeded` and
`transaction_not_permitted`. Near-miss spellings are the worst kind of wrong -
they read as correct to anyone who has not checked, and as careless to anyone who
has.

Codes are also rail-scoped. A UPI collect request cannot fail with `card_expired`,
and a card cannot fail with `invalid_vpa`; the generator uses `rails` to keep the
synthetic data physically possible, which the earlier version did not - it dealt
card failures to UPI payments for 46% of traffic.

Coverage is deliberately partial. Razorpay documents ~112 reason codes and this
table carries the ones that (a) occur often enough to matter and (b) imply a
*different* recovery action. Everything else falls through to UNKNOWN and is
escalated rather than guessed at, and `python -m recoup.taxonomy` prints the
coverage so the gap is a number rather than an impression.
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


ALL_RAILS: tuple[Rail, ...] = (Rail.CARD, Rail.UPI, Rail.NETBANKING, Rail.WALLET, Rail.EMI)


class Strategy(str, Enum):
    """The recovery archetype implied by a failure reason."""

    RETRY_NOW = "retry_now"
    """Customer is still at the keyboard. Re-present the same rail immediately."""

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
    """Everything Recoup knows about one Razorpay error_reason."""

    code: str
    """The exact `error_reason` string Razorpay emits. Not a paraphrase."""

    label: str

    source: str
    """Razorpay error_source: customer | business | bank | gateway | razorpay."""

    step: str
    """Razorpay error_step: payment_initiation | payment_authentication |
    payment_authorization | payment_response."""

    strategy: Strategy

    rails: tuple[Rail, ...]
    """Rails on which this failure is physically possible.

    A UPI payment cannot fail with card_expired. Without this the generator
    produced impossible events - and a Razorpay reviewer reading the seeded data
    would spot it before reading a line of the recovery logic.
    """

    base_recoverability: float
    """P(recovered | this strategy applied), from a cold start.

    Priors, not measured truth. They seed the scorer on day one and are
    superseded by observed rates once the eval harness has enough events - see
    recoup/detect/scorer.py. Defensible as an order of magnitude, not as a
    decimal point, and the report says so.
    """

    retry_after_minutes: int = 0
    """Minimum wait before a retry is meaningful. 0 = the customer is still present."""

    max_attempts: int = 2
    """Hard ceiling on automated retries for this reason."""

    incentive_eligible: bool = False
    """Whether a discount could plausibly change the outcome.

    False for every technical failure. Discounting a bank outage is pure margin
    burn - the customer already wanted to pay, so you paid them to do what they
    were going to do anyway. Incentives only make sense where intent is the
    blocker.
    """

    switch_to: tuple[Rail, ...] = field(default_factory=tuple)
    """Rails worth offering instead, best-first."""

    customer_present: bool = False
    """True when the customer is still in the checkout session.

    Separated from the strategy because it decides whether a *silent* retry is
    even possible. In India it usually is not: RBI's additional-factor rules mean
    a card re-presentment needs the cardholder, and a UPI re-presentment is a
    collect request that lights up their phone. See the note on RETRY_NOW below.
    """

    note: str = ""
    """Why this profile is what it is. Surfaced in the audit trail."""


CARD_RAILS = (Rail.CARD, Rail.EMI)
UPI_RAILS = (Rail.UPI,)
BANK_RAILS = (Rail.NETBANKING,)


# ---------------------------------------------------------------------------
# The table. Every `code` is a documented Razorpay error_reason.
# ---------------------------------------------------------------------------

PROFILES: dict[str, FailureProfile] = {
    # ---- Transient infrastructure: high recovery, zero incentive, just wait ----
    "bank_not_available": FailureProfile(
        code="bank_not_available",
        label="Bank unavailable (downtime)",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        rails=(Rail.NETBANKING, Rail.CARD, Rail.UPI, Rail.EMI),
        base_recoverability=0.82,
        retry_after_minutes=120,
        max_attempts=3,
        switch_to=(Rail.UPI, Rail.CARD),
        note="Nothing is wrong with the customer or the instrument. Wait out the outage.",
    ),
    "issuer_technical_error": FailureProfile(
        code="issuer_technical_error",
        label="Technical error at the card issuer",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        rails=CARD_RAILS,
        base_recoverability=0.79,
        retry_after_minutes=90,
        max_attempts=3,
        switch_to=(Rail.UPI, Rail.NETBANKING),
        note="Issuer-side fault. Transient, and not the customer's doing.",
    ),
    "gateway_technical_error": FailureProfile(
        code="gateway_technical_error",
        label="Gateway technical error",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        rails=ALL_RAILS,
        base_recoverability=0.78,
        retry_after_minutes=30,
        max_attempts=3,
        note="Transient. Retry before assuming customer-side intent.",
    ),
    "upi_app_technical_error": FailureProfile(
        code="upi_app_technical_error",
        label="Technical error at the customer's UPI app",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        rails=UPI_RAILS,
        base_recoverability=0.74,
        retry_after_minutes=45,
        max_attempts=3,
        switch_to=(Rail.CARD, Rail.NETBANKING),
        note="The PSP app failed, not the payer. A second collect usually lands.",
    ),
    "psp_not_available": FailureProfile(
        code="psp_not_available",
        label="UPI PSP unavailable",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        rails=UPI_RAILS,
        base_recoverability=0.76,
        retry_after_minutes=60,
        max_attempts=3,
        switch_to=(Rail.CARD, Rail.NETBANKING),
        note="PSP downtime. UPI outages are usually short and total.",
    ),
    "payment_declined_due_to_high_traffic": FailureProfile(
        code="payment_declined_due_to_high_traffic",
        label="Declined - gateway congestion",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        rails=ALL_RAILS,
        base_recoverability=0.80,
        retry_after_minutes=20,
        max_attempts=3,
        note="Load shedding. The most reliably recoverable failure there is.",
    ),
    # ---- Customer present and correctable ----
    "incorrect_otp": FailureProfile(
        code="incorrect_otp",
        label="Incorrect OTP entered",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.RETRY_NOW,
        rails=(Rail.CARD, Rail.NETBANKING, Rail.EMI),
        base_recoverability=0.71,
        max_attempts=2,
        customer_present=True,
        note=(
            "Mistyped, and only fixable while the customer is still there. Recoup "
            "cannot re-enter an OTP on their behalf, so the recovery is a fresh "
            "authenticated attempt - which is why the action is a link, not a "
            "silent re-presentment."
        ),
    ),
    "incorrect_cvv": FailureProfile(
        code="incorrect_cvv",
        label="Incorrect CVV",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.RETRY_NOW,
        rails=CARD_RAILS,
        base_recoverability=0.68,
        max_attempts=2,
        customer_present=True,
        note="Correctable in-session, like an OTP slip. No money needs to be spent.",
    ),
    "incorrect_pin": FailureProfile(
        code="incorrect_pin",
        label="Incorrect UPI PIN",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.RETRY_NOW,
        rails=UPI_RAILS,
        base_recoverability=0.70,
        max_attempts=2,
        customer_present=True,
        note="The UPI equivalent of a mistyped OTP, and about as recoverable.",
    ),
    "otp_expired": FailureProfile(
        code="otp_expired",
        label="OTP expired",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.RETRY_NOW,
        rails=(Rail.CARD, Rail.NETBANKING, Rail.EMI),
        base_recoverability=0.66,
        max_attempts=2,
        customer_present=True,
        note="They were there and got distracted. A fresh attempt usually completes.",
    ),
    "payment_timed_out": FailureProfile(
        code="payment_timed_out",
        label="Customer did not complete in time",
        source="gateway",
        step="payment_response",
        strategy=Strategy.RETRY_NOW,
        rails=ALL_RAILS,
        base_recoverability=0.64,
        max_attempts=2,
        customer_present=True,
        note="Session lapsed rather than failed. Re-present promptly.",
    ),
    "payment_collect_request_expired": FailureProfile(
        code="payment_collect_request_expired",
        label="UPI collect request expired",
        source="gateway",
        step="payment_authentication",
        strategy=Strategy.RETRY_NOW,
        rails=UPI_RAILS,
        base_recoverability=0.58,
        retry_after_minutes=15,
        max_attempts=2,
        note=(
            "The collect notification timed out unanswered. A second collect is "
            "cheap and often works - but it rings the customer's phone, so it is "
            "customer contact and counts against the fatigue cap."
        ),
    ),
    "otp_attempts_exceeded": FailureProfile(
        code="otp_attempts_exceeded",
        label="OTP attempts exhausted",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.SWITCH_RAIL,
        rails=(Rail.CARD, Rail.NETBANKING, Rail.EMI),
        base_recoverability=0.42,
        retry_after_minutes=30,
        max_attempts=1,
        switch_to=(Rail.UPI,),
        note="Issuer has locked the attempt window. Another OTP will not help.",
    ),
    "pin_attempts_exceeded": FailureProfile(
        code="pin_attempts_exceeded",
        label="UPI PIN attempts exhausted",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.SWITCH_RAIL,
        rails=UPI_RAILS,
        base_recoverability=0.40,
        retry_after_minutes=60,
        max_attempts=1,
        switch_to=(Rail.CARD, Rail.NETBANKING),
        note="NPCI locks the PIN after repeated failures. Offer another rail.",
    ),
    # ---- Liquidity: recoverable, but only with the right timing ----
    "insufficient_funds": FailureProfile(
        code="insufficient_funds",
        label="Insufficient funds",
        source="customer",
        step="payment_authorization",
        strategy=Strategy.RETRY_ON_LIQUIDITY,
        rails=ALL_RAILS,
        base_recoverability=0.55,
        retry_after_minutes=60 * 24,
        max_attempts=3,
        switch_to=(Rail.UPI, Rail.EMI),
        note=(
            "Intent is intact, the balance is not. Retrying in five minutes fails "
            "again and burns an attempt; timing the retry to a salary window is "
            "most of the value. See recoup/detect/features.py:liquidity_window."
        ),
    ),
    "transaction_limit_exceeded": FailureProfile(
        code="transaction_limit_exceeded",
        label="Per-transaction limit exceeded",
        source="customer",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        rails=(Rail.CARD, Rail.UPI, Rail.EMI),
        base_recoverability=0.52,
        max_attempts=2,
        switch_to=(Rail.NETBANKING, Rail.CARD),
        note="The rail has a ceiling, the customer does not. Route around it.",
    ),
    "transaction_daily_limit_exceeded": FailureProfile(
        code="transaction_daily_limit_exceeded",
        label="Daily limit exhausted",
        source="customer",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        rails=(Rail.UPI, Rail.CARD, Rail.NETBANKING),
        base_recoverability=0.61,
        retry_after_minutes=60 * 14,
        max_attempts=2,
        switch_to=(Rail.CARD,),
        note=(
            "Resets at midnight. UPI's per-day cap is hit constantly in India and "
            "the fix is simply tomorrow - one of the few failures where waiting "
            "beats every other action."
        ),
    ),
    "transaction_frequency_limit_exceeded": FailureProfile(
        code="transaction_frequency_limit_exceeded",
        label="NPCI frequency limit exhausted",
        source="customer",
        step="payment_authorization",
        strategy=Strategy.RETRY_DELAYED,
        rails=UPI_RAILS,
        base_recoverability=0.57,
        retry_after_minutes=60 * 4,
        max_attempts=2,
        switch_to=(Rail.CARD, Rail.NETBANKING),
        note="NPCI caps UPI attempts per window. Waiting clears it; retrying does not.",
    ),
    "credit_limit_exceeded": FailureProfile(
        code="credit_limit_exceeded",
        label="Credit limit exceeded",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        rails=(Rail.CARD, Rail.EMI),
        base_recoverability=0.45,
        max_attempts=1,
        switch_to=(Rail.UPI, Rail.NETBANKING),
        note="The card is maxed. A debit rail is the only thing that will work.",
    ),
    # ---- Instrument unusable: must switch rails ----
    "card_expired": FailureProfile(
        code="card_expired",
        label="Card expired",
        source="customer",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        rails=CARD_RAILS,
        base_recoverability=0.44,
        max_attempts=1,
        switch_to=(Rail.UPI, Rail.NETBANKING),
        note="Retrying the same card is guaranteed to fail. One attempt, new rail.",
    ),
    "debit_instrument_blocked": FailureProfile(
        code="debit_instrument_blocked",
        label="Card blocked by issuer or cardholder",
        source="customer",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        rails=CARD_RAILS,
        base_recoverability=0.36,
        max_attempts=1,
        switch_to=(Rail.UPI, Rail.NETBANKING),
        note="Nothing about this instrument will start working. Offer another.",
    ),
    "international_transaction_not_allowed": FailureProfile(
        code="international_transaction_not_allowed",
        label="International transactions blocked on card",
        source="customer",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        rails=CARD_RAILS,
        base_recoverability=0.41,
        max_attempts=1,
        switch_to=(Rail.UPI, Rail.NETBANKING),
        note=(
            "A card setting the customer cannot change mid-checkout. Razorpay "
            "classes the source as customer, but nothing they do in this session "
            "fixes it - so this is a rail switch, not a retry."
        ),
    ),
    "invalid_vpa": FailureProfile(
        code="invalid_vpa",
        label="Incorrect UPI VPA",
        source="customer",
        step="payment_initiation",
        strategy=Strategy.RETRY_NOW,
        rails=UPI_RAILS,
        base_recoverability=0.62,
        max_attempts=2,
        customer_present=True,
        switch_to=(Rail.UPI,),
        note="Mistyped handle. Re-presenting a UPI intent lets them pick the app instead.",
    ),
    "vpa_resolution_failed": FailureProfile(
        code="vpa_resolution_failed",
        label="VPA could not be resolved",
        source="gateway",
        step="payment_initiation",
        strategy=Strategy.RETRY_DELAYED,
        rails=UPI_RAILS,
        base_recoverability=0.54,
        retry_after_minutes=30,
        max_attempts=2,
        switch_to=(Rail.CARD, Rail.NETBANKING),
        note="UPI network could not validate the handle. Often transient.",
    ),
    "transaction_on_vpa_restricted": FailureProfile(
        code="transaction_on_vpa_restricted",
        label="VPA blocked by the PSP",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        rails=UPI_RAILS,
        base_recoverability=0.33,
        max_attempts=1,
        switch_to=(Rail.CARD, Rail.NETBANKING),
        note="The PSP has restricted this handle. Another UPI attempt will not clear it.",
    ),
    "card_declined": FailureProfile(
        code="card_declined",
        label="Declined by issuer, no reason given",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.SWITCH_RAIL,
        rails=CARD_RAILS,
        base_recoverability=0.29,
        retry_after_minutes=60,
        max_attempts=1,
        switch_to=(Rail.UPI,),
        note=(
            "A generic decline hides both soft and hard causes. Treated "
            "conservatively: one attempt, different rail, no incentive."
        ),
    ),
    "authentication_failed": FailureProfile(
        code="authentication_failed",
        label="Authentication failed or was cancelled",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.RETRY_NOW,
        rails=(Rail.CARD, Rail.NETBANKING, Rail.EMI),
        base_recoverability=0.52,
        retry_after_minutes=5,
        max_attempts=2,
        customer_present=True,
        switch_to=(Rail.UPI,),
        note=(
            "Razorpay defines this as a wrong OTP *or* the customer abandoning "
            "authentication - so it is predominantly a correctable in-session "
            "failure, not an instrument problem. An earlier version routed it "
            "straight to a rail switch, which gave up on the majority case."
        ),
    ),
    # ---- Intent problem: the only place an incentive is defensible ----
    "payment_cancelled": FailureProfile(
        code="payment_cancelled",
        label="Customer cancelled the payment",
        source="customer",
        step="payment_authentication",
        strategy=Strategy.PERSUADE,
        rails=ALL_RAILS,
        base_recoverability=0.21,
        retry_after_minutes=60 * 4,
        max_attempts=1,
        incentive_eligible=True,
        note=(
            "Nothing broke. The customer looked at the price and walked. This is "
            "the one bucket where spending money can change the answer - and the "
            "one where it is easiest to spend it on people who would have "
            "converted anyway. Hence the control group."
        ),
    ),
    # ---- Never retry ----
    "payment_risk_check_failed": FailureProfile(
        code="payment_risk_check_failed",
        label="Declined by risk checks",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.DO_NOT_RETRY,
        rails=ALL_RAILS,
        base_recoverability=0.0,
        max_attempts=0,
        note=(
            "Hard stop. Re-presenting a risk decline is how a merchant earns a "
            "higher decline rate and a chargeback problem. Recoup suppresses "
            "these and reports the suppressed value rather than dropping it."
        ),
    ),
    "compliance_violation": FailureProfile(
        code="compliance_violation",
        label="Payment violates a compliance requirement",
        source="business",
        step="payment_initiation",
        strategy=Strategy.DO_NOT_RETRY,
        rails=ALL_RAILS,
        base_recoverability=0.0,
        max_attempts=0,
        note="Retrying a compliance block is a merchant problem, not a recovery one.",
    ),
    "payment_amount_tampered": FailureProfile(
        code="payment_amount_tampered",
        label="Payment amount was tampered with",
        source="gateway",
        step="payment_authorization",
        strategy=Strategy.DO_NOT_RETRY,
        rails=ALL_RAILS,
        base_recoverability=0.0,
        max_attempts=0,
        note="Integrity failure. Nothing here should be re-presented automatically.",
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
    rails=ALL_RAILS,
    base_recoverability=0.13,
    retry_after_minutes=60,
    max_attempts=2,
    incentive_eligible=True,
    note="No payment was attempted, so there is nothing to retry - only intent to move.",
)

INVOICE_OVERDUE = FailureProfile(
    code="invoice_overdue",
    label="Receivable past due date",
    source="business",
    step="payment_initiation",
    strategy=Strategy.PERSUADE,
    rails=(Rail.NETBANKING, Rail.UPI),
    base_recoverability=0.62,
    retry_after_minutes=60 * 24,
    max_attempts=4,
    note=(
        "B2B receivables are usually collected, not converted - the question is "
        "when, not whether. An escalation ladder beats discounting."
    ),
)

_EXTRA = {p.code: p for p in (CHECKOUT_ABANDONED, INVOICE_OVERDUE)}

UNKNOWN = FailureProfile(
    code="unknown",
    label="Unclassified failure",
    source="razorpay",
    step="payment_authorization",
    strategy=Strategy.DO_NOT_RETRY,
    rails=ALL_RAILS,
    base_recoverability=0.0,
    max_attempts=0,
    note=(
        "Fail closed. Razorpay documents around 112 reason codes and this table "
        "carries the ones that imply a distinct action; anything else means the "
        "taxonomy has not seen it, and the safe response is to touch nothing and "
        "surface it. Volume here is reported as a coverage metric, not swept up."
    ),
)

DOCUMENTED_RAZORPAY_REASON_COUNT = 112
"""Approximate size of Razorpay's published error_reason list.

Carried as a number so coverage() can state the gap rather than implying there
isn't one. Recoup classifies the common, action-distinct subset; the rest fail
closed by design, and that is a choice worth being explicit about.
"""


def profile_for(code: str | None) -> FailureProfile:
    """Look up a failure profile, falling back to a deliberately inert default."""
    if not code:
        return UNKNOWN
    key = code.strip().lower()
    return PROFILES.get(key) or _EXTRA.get(key) or UNKNOWN


def all_codes() -> list[str]:
    """Every reason code the taxonomy recognises."""
    return sorted([*PROFILES, *_EXTRA])


def codes_for_rail(rail: str | Rail) -> list[str]:
    """Reason codes that can physically occur on `rail`.

    Used by the generator so a UPI payment never fails with `card_expired`.
    """
    target = Rail(rail) if not isinstance(rail, Rail) else rail
    return sorted(c for c, p in PROFILES.items() if target in p.rails)


def coverage() -> dict[str, object]:
    """How much of Razorpay's error vocabulary this table actually classifies."""
    return {
        "classified": len(PROFILES),
        "documented_by_razorpay": DOCUMENTED_RAZORPAY_REASON_COUNT,
        "share": round(len(PROFILES) / DOCUMENTED_RAZORPAY_REASON_COUNT, 3),
        "by_rail": {r.value: len(codes_for_rail(r)) for r in ALL_RAILS},
        "do_not_retry": sorted(
            c for c, p in PROFILES.items() if p.strategy is Strategy.DO_NOT_RETRY
        ),
        "incentive_eligible": sorted(
            c for c, p in {**PROFILES, **_EXTRA}.items() if p.incentive_eligible
        ),
    }


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import json

    print(json.dumps(coverage(), indent=2))
