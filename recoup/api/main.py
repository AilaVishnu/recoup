"""The dashboard. Five read-only routes over the audit trail.

There is exactly one thing this application has to make effortless: pick any
event and see the whole chain of reasoning that led to what happened to it -
what failed, what the scorer thought it was worth, what was proposed and by
whom, which of the thirteen bounds were checked, what executed, and whether the
money came back. Everything else here is navigation to that page.

No writes. No Razorpay calls. No LLM. If this process died mid-demo the pipeline
would not notice, which is the correct relationship between a system and its
window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from recoup.api import read
from recoup.api.format import FILTERS
from recoup.config import get_settings
from recoup.db import Cohort, EventStatus, PolicyVerdict, get_session
from recoup.policy.rules import RULES
from recoup.taxonomy import profile_for

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="Recoup",
    description="Audit trail for a bounded revenue-recovery agent.",
    docs_url=None,
    redoc_url=None,
)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
templates.env.filters.update(FILTERS)
templates.env.globals.update(
    profile_for=profile_for,
    rule_count=len(RULES),
    cohorts=list(Cohort),
    statuses=list(EventStatus),
    verdicts=list(PolicyVerdict),
)


def db() -> Iterator[Session]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _enum(cls, raw: str | None):
    """Query params are user input. An unrecognised filter is dropped, not fatal."""
    if not raw:
        return None
    try:
        return cls(raw)
    except ValueError:
        return None


def _base(session: Session, active: str) -> dict:
    return {
        "active": active,
        "ready": read.schema_ready(session),
        "dry_run": get_settings().dry_run,
    }


# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def overview(request: Request, session: Session = Depends(db)) -> HTMLResponse:
    ctx = _base(session, "overview")
    ctx["ov"] = read.overview(session) if ctx["ready"] else None
    ctx["report"] = read.latest_report()
    return templates.TemplateResponse(request, "overview.html", ctx)


@app.get("/events", response_class=HTMLResponse)
def events(
    request: Request,
    session: Session = Depends(db),
    cohort: str | None = None,
    status: str | None = None,
    reason: str | None = None,
    verdict: str | None = None,
    page: int = Query(1, ge=1),
) -> HTMLResponse:
    ctx = _base(session, "events")
    filters = read.Filters(
        cohort=_enum(Cohort, cohort),
        status=_enum(EventStatus, status),
        reason=reason or None,
        verdict=_enum(PolicyVerdict, verdict),
    )

    if ctx["ready"]:
        rows, total, page = read.list_events(session, filters, page=page)
        ctx["reason_codes"] = read.reason_codes(session)
    else:
        rows, total, ctx["reason_codes"] = [], 0, []

    pages = max(1, -(-total // read.PER_PAGE))
    query = {
        k: v
        for k, v in (
            ("cohort", cohort),
            ("status", status),
            ("reason", reason),
            ("verdict", verdict),
        )
        if v
    }
    ctx.update(
        rows=rows,
        total=total,
        filters=filters,
        page=page,
        pages=pages,
        prev_url=f"/events?{urlencode({**query, 'page': page - 1})}" if page > 1 else None,
        next_url=f"/events?{urlencode({**query, 'page': page + 1})}" if page < pages else None,
    )
    return templates.TemplateResponse(request, "events.html", ctx)


@app.get("/events/{event_id}", response_class=HTMLResponse)
def event_detail(
    request: Request, event_id: str, session: Session = Depends(db)
) -> HTMLResponse:
    ctx = _base(session, "events")
    replay = read.replay(session, event_id) if ctx["ready"] else None
    if replay is None:
        ctx["event_id"] = event_id
        return templates.TemplateResponse(request, "not_found.html", ctx, status_code=404)

    ctx["r"] = replay
    return templates.TemplateResponse(request, "event.html", ctx)


@app.get("/policy", response_class=HTMLResponse)
def policy(request: Request, session: Session = Depends(db)) -> HTMLResponse:
    ctx = _base(session, "policy")
    ctx["bounds"] = read.bounds_table()
    ctx["rules"] = read.rule_stats(session) if ctx["ready"] else []
    return templates.TemplateResponse(request, "policy.html", ctx)


@app.get("/healthz")
def healthz(session: Session = Depends(db)) -> JSONResponse:
    return JSONResponse({"ok": True, "database": read.schema_ready(session)})
