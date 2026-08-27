"""Thin wrapper over the Razorpay SDK. Three guarantees, none of them optional.

**1. The pipeline runs end to end with no credentials.** Every call has a
dry-run path that returns a deterministic, unmistakably fake payload without
touching the network. A reviewer cloning this repo can seed, decide, execute and
evaluate before they have found their test keys - and the fake ids are stable
across runs, so a replay of the same event produces the same reference and the
eval's attribution join stays intact.

**2. No raw SDK exception reaches the executor.** `razorpay` raises three
exception types carrying only a description string, from a dozen call sites.
Letting those propagate would put SDK internals in ActionRun.error and force
every caller to import `razorpay.errors`. Everything comes back as
RecoupExecutionError with a structured payload instead.

**3. A 4xx is never retried.** Razorpay rejected the request for a reason -
malformed amount, cancelled link, expired order - and re-sending it unchanged
gets the same rejection at best. At worst the rejection was on the response leg
of a request that did land, and retrying it is how a recovery agent double-charges
a customer it was supposed to be helping. Only 5xx and gateway faults, where the
request demonstrably did not complete, are retried.

Reason codes and entity shapes follow Razorpay's Orders and Payment Links APIs.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import razorpay
import requests
from razorpay.errors import (
    BadRequestError,
    GatewayError,
    ServerError,
    SignatureVerificationError,
)

from recoup.config import get_settings
from recoup.db import Customer

MAX_RETRIES = 2
"""Two retries, then stop. A third attempt against a gateway that has already
failed twice is not resilience - it is a queue forming behind an outage, and the
taxonomy already says to come back in two hours."""

RETRY_BACKOFF_SECONDS = (0.5, 2.0)
"""Backoff before retry 1 and retry 2. Short, because the executor runs inside a
per-event loop and a stalled event blocks the ones behind it."""

LINK_MIN_LIFETIME = timedelta(minutes=16)
"""Razorpay rejects an expire_by less than 15 minutes out; 16 leaves room for
clock skew between this process and the API."""

LINK_MAX_LIFETIME = timedelta(days=180)
"""API ceiling is six months. A recovery link that outlives the recovery window
is a discount with no expiry date attached to it."""

DEFAULT_LINK_LIFETIME = timedelta(days=3)
"""Long enough to survive a weekend, short enough that the offer still reads as
an offer. Recovery intent decays fast - see the decay terms in detect/scorer.py."""

_client: razorpay.Client | None = None
_client_key: str | None = None


class RecoupExecutionError(RuntimeError):
    """One exception type for every way a Razorpay call can fail.

    `payload` is reconstructed rather than passed through: the SDK collapses
    Razorpay's JSON error body down to `error.description` before raising, so
    the code has to be recovered from which exception class came back. What
    survives is still enough to answer the only two questions the audit trail
    asks - what did the gateway refuse, and were we allowed to try again.
    """

    def __init__(
        self,
        operation: str,
        message: str,
        *,
        code: str,
        retryable: bool,
        attempts: int = 1,
    ) -> None:
        super().__init__(f"{operation}: [{code}] {message}")
        self.operation = operation
        self.code = code
        self.retryable = retryable
        self.attempts = attempts
        self.payload: dict[str, Any] = {
            "operation": operation,
            "error": {"code": code, "description": message},
            "retryable": retryable,
            "attempts": attempts,
        }


def get_client() -> razorpay.Client | None:
    """The memoised SDK client, or None when no test credentials are configured.

    Memoised on the key id rather than unconditionally, so a test that swaps
    settings does not keep talking to a client built from the previous ones.
    """
    global _client, _client_key

    settings = get_settings()
    if not settings.razorpay_configured:
        return None

    if _client is None or _client_key != settings.razorpay_key_id:
        _client = razorpay.Client(
            auth=(settings.razorpay_key_id, settings.razorpay_key_secret)
        )
        _client.set_app_details({"title": "Recoup", "version": "1.0"})
        _client_key = settings.razorpay_key_id

    return _client


def is_dry_run() -> bool:
    """True when nothing may touch the network - by configuration or by absence.

    Missing keys are treated identically to an explicit dry run. The alternative
    is an executor that raises halfway through a seeded dataset because someone
    has not filled in a .env, which teaches them nothing about the system.
    """
    settings = get_settings()
    return settings.dry_run or not settings.razorpay_configured


# ---------------------------------------------------------------------------
# Dry-run payloads
# ---------------------------------------------------------------------------


def _fake_id(prefix: str, *parts: object) -> str:
    """A stable pseudo-id derived from the request, not from a counter or a clock.

    Determinism is the point: two runs of the same seeded dataset produce the
    same references, so `ActionRun.razorpay_ref` still joins to the same event on
    a replay and the eval is reproducible.
    """
    material = "|".join(str(p) for p in parts).encode("utf-8")
    return f"{prefix}{hashlib.blake2s(material, digest_size=7).hexdigest()}"


def _dry_short_url(link_id: str) -> str:
    """A URL that is structurally incapable of resolving.

    `.invalid` is reserved by RFC 2606 and has no DNS record anywhere, so a fake
    link that leaks into a log, a screenshot or a test fixture cannot be mistaken
    for a live one or accidentally opened.
    """
    return f"https://recoup.invalid/dry-run/{link_id}"


def _dry_marker(entity: str) -> dict[str, Any]:
    return {
        "_dry_run": True,
        "_note": f"simulated {entity} - no Razorpay call was made",
    }


# ---------------------------------------------------------------------------
# Call plumbing
# ---------------------------------------------------------------------------


def _require_client() -> razorpay.Client:
    """Unreachable in practice - is_dry_run() already covers an unconfigured
    account - but the live path must not depend on that invariant holding."""
    client = get_client()
    if client is None:
        raise RecoupExecutionError(
            "get_client",
            "no Razorpay credentials configured",
            code="NOT_CONFIGURED",
            retryable=False,
        )
    return client


def _invoke(operation: str, fn: Callable[[], Any]) -> dict[str, Any]:
    """Run one SDK call, translating failures and retrying only the safe ones.

    Anything outside the known failure set is deliberately left to propagate. A
    TypeError while building the request body is a bug in Recoup, not a refusal
    by the gateway, and dressing it up as a gateway error would send it to the
    retry path and then bury it in ActionRun.error.
    """
    last: RecoupExecutionError | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn()

        except BadRequestError as exc:
            # Terminal by construction. See the module docstring: this is the
            # branch that stops Recoup double-charging anyone.
            raise RecoupExecutionError(
                operation,
                str(exc) or "request rejected by Razorpay",
                code="BAD_REQUEST_ERROR",
                retryable=False,
                attempts=attempt + 1,
            ) from exc

        except SignatureVerificationError as exc:
            raise RecoupExecutionError(
                operation,
                str(exc) or "signature verification failed",
                code="SIGNATURE_VERIFICATION_ERROR",
                retryable=False,
                attempts=attempt + 1,
            ) from exc

        except (GatewayError, ServerError) as exc:
            code = (
                "GATEWAY_ERROR" if isinstance(exc, GatewayError) else "SERVER_ERROR"
            )
            last = RecoupExecutionError(
                operation,
                str(exc) or "Razorpay is unavailable",
                code=code,
                retryable=True,
                attempts=attempt + 1,
            )

        except requests.RequestException as exc:
            # Never reached the API, so nothing was created and a retry is safe.
            last = RecoupExecutionError(
                operation,
                f"{type(exc).__name__}: {exc}",
                code="TRANSPORT_ERROR",
                retryable=True,
                attempts=attempt + 1,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    assert last is not None  # only reachable via a retryable branch
    raise last


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def create_order(
    amount_paise: int,
    receipt: str,
    notes: dict[str, str] | None = None,
    currency: str = "INR",
) -> dict[str, Any]:
    """Create an order for a silent re-presentment of a failed payment.

    `receipt` carries Recoup's own event/decision reference. Razorpay treats it
    as an opaque merchant string, which makes it the cheapest place to keep the
    thread back to the decision that caused this order to exist.
    """
    if amount_paise <= 0:
        raise ValueError("order amount must be positive paise")

    payload = {
        "amount": amount_paise,
        "currency": currency,
        "receipt": receipt[:40],  # API caps receipt at 40 characters
        "payment_capture": 1,
        "notes": {k: str(v) for k, v in (notes or {}).items()},
    }

    if is_dry_run():
        order_id = _fake_id("order_dry_", receipt, amount_paise)
        return {
            "id": order_id,
            "entity": "order",
            "amount": amount_paise,
            "amount_paid": 0,
            "amount_due": amount_paise,
            "currency": currency,
            "receipt": payload["receipt"],
            "status": "created",
            "attempts": 0,
            "notes": payload["notes"],
            **_dry_marker("order"),
        }

    client = _require_client()
    return _invoke("create_order", lambda: client.order.create(data=payload))


def create_payment_link(
    amount_paise: int,
    customer: Customer,
    description: str,
    expire_by: datetime | None = None,
    notes: dict[str, str] | None = None,
    reference_id: str | None = None,
) -> dict[str, Any]:
    """Create a Payment Link the customer can pay on any rail.

    Notifications are switched off on purpose. Razorpay will happily SMS and
    email the link itself, but Recoup owns customer contact: every touch has to
    pass the quiet-hours and fatigue bounds in policy/rules.py and land in the
    outbox with its cost attached. A link that notifies on its own bypasses all
    of that, and against a seeded dataset it would fire thousands of messages at
    addresses that belong to real people by coincidence.
    """
    if amount_paise <= 0:
        raise ValueError("payment link amount must be positive paise")

    now = datetime.now(timezone.utc)
    target = expire_by or (now + DEFAULT_LINK_LIFETIME)
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    target = min(max(target, now + LINK_MIN_LIFETIME), now + LINK_MAX_LIFETIME)

    payload = {
        "amount": amount_paise,
        "currency": "INR",
        "accept_partial": False,
        "description": description[:2048],
        "customer": {
            "name": customer.name,
            "email": customer.email,
            "contact": customer.contact,
        },
        "notify": {"sms": False, "email": False},
        "reminder_enable": False,
        "expire_by": int(target.timestamp()),
        "notes": {k: str(v) for k, v in (notes or {}).items()},
    }
    if reference_id:
        payload["reference_id"] = reference_id[:40]

    if is_dry_run():
        link_id = _fake_id(
            "plink_dry_", reference_id or description, customer.id, amount_paise
        )
        return {
            "id": link_id,
            "entity": "payment_link",
            "amount": amount_paise,
            "amount_paid": 0,
            "currency": "INR",
            "status": "created",
            "description": payload["description"],
            "short_url": _dry_short_url(link_id),
            "reference_id": payload.get("reference_id"),
            "expire_by": payload["expire_by"],
            "customer": payload["customer"],
            "notes": payload["notes"],
            **_dry_marker("payment link"),
        }

    client = _require_client()
    return _invoke(
        "create_payment_link", lambda: client.payment_link.create(data=payload)
    )


def fetch_payment_link(link_id: str) -> dict[str, Any]:
    """Read a link back - the settlement check that turns an action into an Outcome."""
    if is_dry_run():
        return {
            "id": link_id,
            "entity": "payment_link",
            "status": "created",
            "amount_paid": 0,
            "short_url": _dry_short_url(link_id),
            **_dry_marker("payment link fetch"),
        }

    client = _require_client()
    return _invoke("fetch_payment_link", lambda: client.payment_link.fetch(link_id))


def fetch_payment(payment_id: str) -> dict[str, Any]:
    if is_dry_run():
        return {
            "id": payment_id,
            "entity": "payment",
            "status": "created",
            "amount": 0,
            **_dry_marker("payment fetch"),
        }

    client = _require_client()
    return _invoke("fetch_payment", lambda: client.payment.fetch(payment_id))
