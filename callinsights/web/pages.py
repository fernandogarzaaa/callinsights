"""Server-rendered HTML pages (dashboard, call intake, call detail, settings)."""

from __future__ import annotations

import math
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import func, select

from callinsights import __version__, models, scoring
from callinsights.services import calls as svc
from callinsights.web.calls import audio_response
from callinsights.web.deps import DbDep, LLMDep, OptionalUserDep, SettingsDep, TranscriberDep

router = APIRouter(include_in_schema=False)
PAGE_SIZE = 25


def _render(request: Request, name: str, ctx: dict[str, Any], status_code: int = 200) -> Response:
    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request, name, ctx, status_code=status_code
    )


def _login() -> Response:
    return RedirectResponse("/login", status_code=303)


@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: DbDep,
    user: OptionalUserDep,
    agent: str = "",
    outcome: str = "",
    q: str = "",
    days: Annotated[int | None, Query(ge=1, le=3650)] = None,
    page: Annotated[int, Query(ge=1, le=100_000)] = 1,
) -> Response:
    if user is None:
        return _login()
    filters = svc.CallFilters(
        agent=agent.strip(), outcome=outcome.strip(), q=q.strip()[:200], since=svc.since_days(days)
    )
    calls, total = svc.list_calls(db, filters, PAGE_SIZE, (page - 1) * PAGE_SIZE)
    stats = svc.analytics_summary(db, filters)
    query = {
        k: v for k, v in {"agent": agent, "outcome": outcome, "q": q, "days": days}.items() if v
    }
    return _render(
        request,
        "dashboard.html",
        {
            "user": user,
            "active": "dashboard",
            "calls": calls,
            "total": total,
            "page": page,
            "pages": max(1, math.ceil(total / PAGE_SIZE)),
            "stats": stats,
            "agents": svc.agents(db),
            "filters": filters,
            "days": days,
            "export_query": urlencode(query),
            "chart_data": {"outcomes": stats["outcomes"], "trend": stats["trend"]},
        },
    )


@router.get("/calls/new", response_class=HTMLResponse)
def call_new(request: Request, user: OptionalUserDep, transcriber: TranscriberDep) -> Response:
    if user is None:
        return _login()
    return _render(
        request,
        "call_new.html",
        {"user": user, "active": "new", "asr_enabled": transcriber is not None},
    )


@router.get("/calls/{call_id}", response_class=HTMLResponse)
def call_detail(
    call_id: int,
    request: Request,
    db: DbDep,
    user: OptionalUserDep,
    settings: SettingsDep,
    transcriber: TranscriberDep,
) -> Response:
    if user is None:
        return _login()
    call = db.get(models.Call, call_id)
    if call is None:
        return _render(
            request,
            "error.html",
            {"user": user, "status_code": 404, "detail": "Call not found."},
            status_code=404,
        )
    turns = scoring.parse_turns(call.transcript)
    has_audio = svc.resolve_audio_path(settings, call.audio_path) is not None
    return _render(
        request,
        "call_detail.html",
        {
            "user": user,
            "active": "",
            "call": call,
            "turns": turns,
            "analytics": scoring.score_call(turns) if turns else None,
            "highlights": call.highlights,
            "coaching": call.coaching,
            "audio_url": f"/api/calls/{call.id}/audio" if has_audio else None,
            "asr_enabled": transcriber is not None,
            "can_delete": user.is_admin or call.created_by_id == user.id,
        },
    )


@router.get("/audio/{call_id}")
def legacy_audio(call_id: int, db: DbDep, user: OptionalUserDep, settings: SettingsDep) -> Response:
    """Beta-era audio URL, kept for backwards compatibility."""
    if user is None:
        return Response(status_code=401)
    call = db.get(models.Call, call_id)
    if call is None:
        return Response(status_code=404)
    return audio_response(call, settings)


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    db: DbDep,
    user: OptionalUserDep,
    settings: SettingsDep,
    llm: LLMDep,
    transcriber: TranscriberDep,
) -> Response:
    if user is None:
        return _login()
    total = db.scalar(select(func.count()).select_from(models.Call)) or 0
    model = getattr(llm, "model", None)
    return _render(
        request,
        "settings.html",
        {
            "user": user,
            "active": "settings",
            "version": __version__,
            "env": settings.ENV,
            "provider_name": f"{llm.name} ({model})" if model else llm.name,
            "asr_name": transcriber.name if transcriber else "disabled",
            "total_calls": total,
        },
    )
