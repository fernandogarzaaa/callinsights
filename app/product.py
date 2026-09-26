"""CallInsights product router: dashboard, call intake, scoring, details, audio."""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models, scoring
from app.ai.providers import get_provider
from app.core.config import settings
from app.core.db import get_db
from app.core.deps import get_current_user, page_or_login

router = APIRouter()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = os.path.join("data", "uploads")
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"}
TRANSCRIPT_EXTS = {".txt", ".vtt", ".srt", ".md"}


def _base_ctx(request: Request) -> dict:
    return {"request": request, "app_name": settings.APP_NAME}


def _is_upload(value) -> bool:
    """Duck-typed file check: Starlette's form parser returns its own
    UploadFile class, which is not identical to fastapi.UploadFile."""
    return hasattr(value, "filename") and hasattr(value, "read")


def _save_upload(upload: UploadFile, allowed: set[str]) -> tuple[str, str]:
    """Save an uploaded file under data/uploads. Returns (rel_path, filename)."""
    filename = (upload.filename or "upload").strip().replace("\\", "/").split("/")[-1]
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {ext!r}. Allowed: {sorted(allowed)}",
        )
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe = f"{uuid.uuid4().hex}{ext}"
    rel_path = os.path.join(UPLOAD_DIR, safe)
    with open(rel_path, "wb") as fh:
        while chunk := upload.file.read(1024 * 1024):
            fh.write(chunk)
    return rel_path, filename


def _score_and_build(
    transcript_text: str,
    contact_name: str,
    contact_phone: str,
    agent_name: str,
    started_at,
    duration_sec: int,
    audio_rel: str | None,
    audio_name: str | None,
) -> models.Call:
    turns = scoring.parse_turns(transcript_text)
    if not turns:
        raise HTTPException(status_code=400, detail="No transcript turns found.")
    result = scoring.score_call(turns)
    summary = get_provider().summarize(scoring.plain_text(turns), max_sentences=4)
    if duration_sec <= 0:
        words = sum(len(t["text"].split()) for t in turns)
        duration_sec = max(60, round(words / 2.5))
    return models.Call(
        contact_name=contact_name or "Unknown",
        contact_phone=contact_phone or "",
        agent=agent_name or "Unassigned",
        started_at=started_at,
        duration_sec=duration_sec,
        transcript=transcript_text,
        audio_path=audio_rel,
        audio_filename=audio_name,
        score=result["score"],
        outcome=result["outcome"],
        talk_ratio=result["talk_ratio"],
        questions_asked=result["questions_asked"],
        summary=summary,
        highlights_json=json.dumps(result["highlights"]),
        coaching_json=json.dumps(result["coaching"]),
    )


def _parse_started_at(value: str | None):
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    agent: str = "",
    db: Session = Depends(get_db),
    user=Depends(page_or_login),
):
    if not isinstance(user, models.User):
        return user  # RedirectResponse to /login
    q = db.query(models.Call)
    if agent:
        q = q.filter(models.Call.agent == agent)
    calls = q.order_by(models.Call.created_at.desc()).limit(25).all()

    total = q.count()
    avg_score = q.with_entities(func.avg(models.Call.score)).scalar() or 0
    interested = q.filter(models.Call.outcome == "Interested").count()
    coaching_opps = q.filter(models.Call.score < 75).count()
    conversion = round(interested / total * 100, 1) if total else 0.0
    agents = [r[0] for r in db.query(models.Call.agent).distinct().order_by(models.Call.agent).all()]
    outcomes = {
        row[0]: row[1]
        for row in q.with_entities(models.Call.outcome, func.count()).group_by(models.Call.outcome).all()
    }

    ctx = _base_ctx(request)
    ctx.update(
        {
            "user": user,
            "total_calls": total,
            "avg_score": round(avg_score, 1),
            "conversion_rate": conversion,
            "coaching_opps": coaching_opps,
            "calls": calls,
            "agents": agents,
            "agent_filter": agent,
            "outcome_counts": outcomes,
        }
    )
    return templates.TemplateResponse(request, "dashboard.html", ctx)


@router.get("/calls/new", response_class=HTMLResponse)
def call_new(request: Request, db: Session = Depends(get_db), user=Depends(page_or_login)):
    if not isinstance(user, models.User):
        return user
    ctx = _base_ctx(request)
    ctx["user"] = user
    return templates.TemplateResponse(request, "call_new.html", ctx)


@router.get("/calls/{call_id}", response_class=HTMLResponse)
def call_detail(
    call_id: int, request: Request, db: Session = Depends(get_db), user=Depends(page_or_login)
):
    if not isinstance(user, models.User):
        return user
    call = db.query(models.Call).filter(models.Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    turns = scoring.parse_turns(call.transcript)
    analytics = scoring.score_call(turns)  # deterministic recompute
    ctx = _base_ctx(request)
    ctx.update(
        {
            "user": user,
            "call": call,
            "turns": turns,
            "analytics": analytics,
            "highlights": json.loads(call.highlights_json or "[]"),
            "coaching": json.loads(call.coaching_json or "[]"),
            "audio_url": f"/audio/{call.id}" if call.audio_path else None,
        }
    )
    return templates.TemplateResponse(request, "call_detail.html", ctx)


@router.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, db: Session = Depends(get_db), user=Depends(page_or_login)):
    if not isinstance(user, models.User):
        return user
    provider = get_provider()
    ctx = _base_ctx(request)
    ctx.update(
        {
            "user": user,
            "provider_name": getattr(provider, "name", type(provider).__name__),
            "total_calls": db.query(models.Call).count(),
            "upload_dir": UPLOAD_DIR,
            "admin_email": settings.ADMIN_EMAIL,
        }
    )
    return templates.TemplateResponse(request, "settings.html", ctx)


@router.get("/audio/{call_id}")
def audio_stream(call_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    call = db.query(models.Call).filter(models.Call.id == call_id).first()
    if not call or not call.audio_path or not os.path.exists(call.audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    media_type = mimetypes.guess_type(call.audio_path)[0] or "application/octet-stream"
    return FileResponse(call.audio_path, media_type=media_type, filename=call.audio_filename or "call-audio")


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@router.get("/api/calls")
def list_calls(agent: str = "", db: Session = Depends(get_db), user=Depends(get_current_user)):
    q = db.query(models.Call)
    if agent:
        q = q.filter(models.Call.agent == agent)
    calls = q.order_by(models.Call.created_at.desc()).limit(200).all()
    return {
        "calls": [
            {
                "id": c.id,
                "contact_name": c.contact_name,
                "contact_phone": c.contact_phone,
                "agent": c.agent,
                "score": c.score,
                "outcome": c.outcome,
                "talk_ratio": c.talk_ratio,
                "questions_asked": c.questions_asked,
                "duration_sec": c.duration_sec,
                "has_audio": bool(c.audio_path),
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in calls
        ]
    }


@router.post("/api/calls")
async def create_call(request: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ctype = request.headers.get("content-type", "")
    contact_name = contact_phone = agent_name = transcript_text = ""
    started_at = None
    duration_sec = 0
    audio_rel = audio_name = None

    if "multipart/form-data" in ctype:
        form = await request.form()
        contact_name = str(form.get("contact_name") or "")
        contact_phone = str(form.get("contact_phone") or "")
        agent_name = str(form.get("agent") or "")
        transcript_text = str(form.get("transcript_text") or "")
        started_at = _parse_started_at(str(form.get("started_at") or "") or None)
        try:
            duration_sec = int(form.get("duration_sec") or 0)
        except (TypeError, ValueError):
            duration_sec = 0
        tfile = form.get("transcript_file")
        if _is_upload(tfile) and tfile.filename:
            raw = (await tfile.read()).decode("utf-8", errors="ignore")
            transcript_text = scoring.parse_upload(tfile.filename, raw)
        afile = form.get("audio_file")
        if _is_upload(afile) and afile.filename:
            audio_rel, audio_name = _save_upload(afile, AUDIO_EXTS)
    else:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Expected JSON or multipart body")
        contact_name = str(data.get("contact_name") or "")
        contact_phone = str(data.get("contact_phone") or "")
        agent_name = str(data.get("agent") or "")
        transcript_text = str(data.get("transcript_text") or "")
        started_at = _parse_started_at(data.get("started_at"))
        duration_sec = int(data.get("duration_sec") or 0)

    if not transcript_text.strip():
        raise HTTPException(status_code=400, detail="transcript_text or transcript_file is required")

    call = _score_and_build(
        transcript_text, contact_name, contact_phone, agent_name,
        started_at, duration_sec, audio_rel, audio_name,
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    return JSONResponse(
        {
            "id": call.id,
            "score": call.score,
            "outcome": call.outcome,
            "talk_ratio": call.talk_ratio,
            "questions_asked": call.questions_asked,
            "summary": call.summary,
            "url": f"/calls/{call.id}",
        },
        status_code=201,
    )
