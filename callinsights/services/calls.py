"""Call intake, processing pipeline (transcribe -> score -> insights), queries and analytics."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

from sqlalchemy import Select, case, func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from callinsights import models, scoring
from callinsights.ai.asr import Transcriber, TranscriptionError
from callinsights.ai.diarize import segments_to_transcript
from callinsights.ai.llm import Insights, LLMProvider
from callinsights.config import Settings
from callinsights.db import session_scope

logger = logging.getLogger(__name__)

AUDIO_EXTS = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".mp4", ".aac"})
TRANSCRIPT_EXTS = frozenset({".txt", ".vtt", ".srt", ".md"})
WORDS_PER_SECOND = 2.5
CHUNK = 1024 * 1024
IN_PROGRESS = ("queued", "transcribing", "scoring")


class CallError(ValueError):
    """Invalid input for a call operation (maps to HTTP 400)."""


class UploadTooLargeError(CallError):
    """Upload exceeded MAX_UPLOAD_MB (maps to HTTP 413)."""


@dataclass
class Analysis:
    turns: list[scoring.Turn]
    analytics: dict[str, Any]
    insights: Insights


# ---------------------------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------------------------


def clean_transcript(text: str, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "").strip()
    if len(text) > max_chars:
        raise CallError(f"transcript exceeds {max_chars} characters")
    return text


def decode_transcript_upload(filename: str, raw: bytes, max_chars: int) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in TRANSCRIPT_EXTS:
        raise CallError(f"unsupported transcript type {ext!r}; allowed: {sorted(TRANSCRIPT_EXTS)}")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return clean_transcript(scoring.parse_upload(filename, text), max_chars)


def save_audio(settings: Settings, filename: str, stream: BinaryIO) -> tuple[str, str]:
    """Stream an upload to disk under a random name. Returns ``(stored_name, original_name)``."""
    original = (filename or "recording").replace("\\", "/").split("/")[-1][:255] or "recording"
    ext = os.path.splitext(original)[1].lower()
    if ext not in AUDIO_EXTS:
        raise CallError(f"unsupported audio type {ext!r}; allowed: {sorted(AUDIO_EXTS)}")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    stored = f"{uuid.uuid4().hex}{ext}"
    dest = settings.upload_dir / stored
    written = 0
    try:
        with dest.open("wb") as fh:
            while chunk := stream.read(CHUNK):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise UploadTooLargeError(f"audio exceeds {settings.MAX_UPLOAD_MB} MB limit")
                fh.write(chunk)
        if written == 0:
            raise CallError("audio file is empty")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return stored, original


def resolve_audio_path(settings: Settings, stored: str | None) -> Path | None:
    """Map a stored audio reference to a file inside the upload dir (never outside it)."""
    if not stored:
        return None
    path = settings.upload_dir / Path(stored.replace("\\", "/")).name
    return path if path.is_file() else None


def estimate_duration(turns: list[scoring.Turn]) -> int:
    words = sum(len(t["text"].split()) for t in turns)
    return max(1, round(words / WORDS_PER_SECOND)) if words else 0


def to_naive_utc(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


# ---------------------------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------------------------


def analyze(transcript: str, llm: LLMProvider) -> Analysis:
    turns = scoring.parse_turns(transcript)
    if not turns:
        raise CallError(
            "no speaker turns found; use one line per turn like 'Agent: ...' / 'Prospect: ...'"
        )
    analytics = scoring.score_call(turns)
    return Analysis(turns=turns, analytics=analytics, insights=llm.insights(turns, analytics))


def apply_analysis(call: models.Call, transcript: str, analysis: Analysis) -> None:
    a, ins = analysis.analytics, analysis.insights
    call.transcript = transcript
    call.score = int(a["score"])
    call.outcome = str(a["outcome"])
    call.talk_ratio = float(a["talk_ratio"])
    call.questions_asked = int(a["questions_asked"])
    call.highlights_json = json.dumps(a["highlights"])
    call.summary = ins.summary
    call.coaching_json = json.dumps(ins.coaching)
    call.strengths_json = json.dumps(ins.strengths)
    call.risks_json = json.dumps(ins.risks)
    call.next_best_action = ins.next_best_action
    call.insights_source = ins.source
    if not call.duration_sec:
        call.duration_sec = estimate_duration(analysis.turns)
    call.status = "scored"
    call.error = None


def create_call(
    db: Session,
    *,
    llm: LLMProvider,
    user: models.User | None,
    contact_name: str = "",
    contact_phone: str = "",
    agent: str = "",
    started_at: datetime | None = None,
    duration_sec: int = 0,
    transcript: str = "",
    audio: tuple[str, str] | None = None,
    transcript_source: str = "manual",
) -> models.Call:
    """Create a call. With a transcript it is scored now; audio-only calls are queued."""
    if not transcript and audio is None:
        raise CallError("provide transcript_text, a transcript_file or an audio_file")
    call = models.Call(
        contact_name=contact_name.strip() or "Unknown",
        contact_phone=contact_phone.strip(),
        agent=agent.strip() or "Unassigned",
        started_at=to_naive_utc(started_at),
        duration_sec=max(0, duration_sec),
        transcript="",
        transcript_source=transcript_source,
        audio_path=audio[0] if audio else None,
        audio_filename=audio[1] if audio else None,
        created_by_id=user.id if user else None,
        status="queued",
    )
    if transcript:
        apply_analysis(call, transcript, analyze(transcript, llm))
    db.add(call)
    db.flush()
    logger.info(
        "call created",
        extra={"call_id": call.id, "status": call.status, "has_audio": audio is not None},
    )
    return call


def transcribe_call(
    call: models.Call, settings: Settings, llm: LLMProvider, transcriber: Transcriber
) -> str:
    path = resolve_audio_path(settings, call.audio_path)
    if path is None:
        raise TranscriptionError("audio file is missing")
    segments = transcriber.transcribe(path)
    if not segments:
        raise TranscriptionError("no speech detected in the recording")
    llm_roles = llm.label_speakers([s.text for s in segments])
    transcript, method = segments_to_transcript(
        segments,
        llm_roles=llm_roles,
        gap_seconds=settings.DIARIZATION_GAP_SECONDS,
        first=settings.ASR_FIRST_SPEAKER,
    )
    if not call.duration_sec and segments[-1].end > 0:
        call.duration_sec = round(segments[-1].end)
    call.transcript_source = f"asr:{transcriber.name}:{method}"
    return transcript


def process_call(
    factory: sessionmaker[Session],
    call_id: int,
    settings: Settings,
    llm: LLMProvider,
    transcriber: Transcriber | None,
) -> None:
    """Background job: transcribe and score a queued call. Never raises."""
    try:
        with session_scope(factory) as db:
            call = db.get(models.Call, call_id)
            if call is None or call.status not in IN_PROGRESS:
                return
            if transcriber is None:
                call.status = "failed"
                call.error = "Transcription is disabled (ASR_PROVIDER=none); add a transcript."
                return
            call.status = "transcribing"
        with session_scope(factory) as db:
            call = db.get(models.Call, call_id)
            if call is None:  # deleted while queued
                return
            transcript = clean_transcript(
                transcribe_call(call, settings, llm, transcriber), settings.MAX_TRANSCRIPT_CHARS
            )
            call.status = "scoring"
            apply_analysis(call, transcript, analyze(transcript, llm))
        logger.info("call processed", extra={"call_id": call_id})
    except Exception as exc:
        logger.exception("call processing failed", extra={"call_id": call_id})
        try:
            with session_scope(factory) as db:
                failed = db.get(models.Call, call_id)
                if failed is not None:
                    failed.status = "failed"
                    failed.error = str(exc)[:2000] or type(exc).__name__
        except Exception:
            logger.exception("could not record processing failure", extra={"call_id": call_id})


def update_call(
    db: Session, call: models.Call, changes: dict[str, Any], llm: LLMProvider, max_chars: int
) -> models.Call:
    for field in ("contact_name", "contact_phone", "agent", "duration_sec"):
        if changes.get(field) is not None:
            setattr(call, field, changes[field])
    if "started_at" in changes:
        call.started_at = to_naive_utc(changes["started_at"])
    text = changes.get("transcript_text")
    if text is not None:
        transcript = clean_transcript(text, max_chars)
        if not transcript:
            raise CallError("transcript_text cannot be empty")
        apply_analysis(call, transcript, analyze(transcript, llm))
        call.transcript_source = "edited"
    db.flush()
    return call


def rescore_call(db: Session, call: models.Call, llm: LLMProvider) -> models.Call:
    if not call.transcript.strip():
        raise CallError("call has no transcript to score")
    apply_analysis(call, call.transcript, analyze(call.transcript, llm))
    db.flush()
    return call


def delete_call(db: Session, call: models.Call, settings: Settings) -> None:
    path = resolve_audio_path(settings, call.audio_path)
    db.delete(call)
    db.flush()
    if path is not None:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------
# Queries & analytics
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallFilters:
    agent: str = ""
    outcome: str = ""
    status: str = ""
    q: str = ""
    min_score: int | None = None
    max_score: int | None = None
    since: datetime | None = None


def _like(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def apply_filters(stmt: Select[Any], f: CallFilters) -> Select[Any]:
    c = models.Call
    if f.agent:
        stmt = stmt.where(c.agent == f.agent)
    if f.outcome:
        stmt = stmt.where(c.outcome == f.outcome)
    if f.status:
        stmt = stmt.where(c.status == f.status)
    if f.min_score is not None:
        stmt = stmt.where(c.score >= f.min_score)
    if f.max_score is not None:
        stmt = stmt.where(c.score <= f.max_score)
    if f.since is not None:
        stmt = stmt.where(c.created_at >= f.since)
    if f.q:
        like = _like(f.q)
        stmt = stmt.where(
            or_(
                c.contact_name.ilike(like, escape="\\"),
                c.agent.ilike(like, escape="\\"),
                c.transcript.ilike(like, escape="\\"),
                c.summary.ilike(like, escape="\\"),
            )
        )
    return stmt


def list_calls(
    db: Session, filters: CallFilters, limit: int = 50, offset: int = 0
) -> tuple[list[models.Call], int]:
    base = apply_filters(select(models.Call), filters)
    total = int(db.scalar(select(func.count()).select_from(base.subquery())) or 0)
    rows = db.scalars(
        base.order_by(models.Call.created_at.desc(), models.Call.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return list(rows), total


def agents(db: Session) -> list[str]:
    return list(db.scalars(select(models.Call.agent).distinct().order_by(models.Call.agent)))


def analytics_summary(db: Session, filters: CallFilters) -> dict[str, Any]:
    """Aggregate metrics over scored calls matching ``filters``."""
    base = apply_filters(select(models.Call), replace(filters, status="scored")).subquery()
    total, avg_score, avg_talk, avg_q = db.execute(
        select(
            func.count(),
            func.avg(base.c.score),
            func.avg(base.c.talk_ratio),
            func.avg(base.c.questions_asked),
        ).select_from(base)
    ).one()
    total = int(total or 0)
    outcomes = {
        str(o): int(n)
        for o, n in db.execute(
            select(base.c.outcome, func.count()).select_from(base).group_by(base.c.outcome)
        ).all()
    }
    interested = outcomes.get("Interested", 0)
    coaching_opps = int(
        db.scalar(select(func.count()).select_from(base).where(base.c.score < 75)) or 0
    )
    avg_col = func.avg(base.c.score)
    leaderboard = [
        {
            "agent": str(agent),
            "calls": int(n),
            "avg_score": round(float(avg or 0), 1),
            "interested": int(inter or 0),
        }
        for agent, n, avg, inter in db.execute(
            select(
                base.c.agent,
                func.count(),
                avg_col,
                func.sum(case((base.c.outcome == "Interested", 1), else_=0)),
            )
            .select_from(base)
            .group_by(base.c.agent)
            .order_by(avg_col.desc(), base.c.agent)
        ).all()
    ]
    day = func.date(base.c.created_at)
    trend = [
        {"date": str(d), "calls": int(n), "avg_score": round(float(avg or 0), 1)}
        for d, n, avg in db.execute(
            select(day, func.count(), func.avg(base.c.score))
            .select_from(base)
            .group_by(day)
            .order_by(day)
        ).all()
    ]
    pending = int(
        db.scalar(
            select(func.count()).select_from(models.Call).where(models.Call.status.in_(IN_PROGRESS))
        )
        or 0
    )
    return {
        "total_calls": total,
        "avg_score": round(float(avg_score or 0), 1),
        "avg_talk_ratio": round(float(avg_talk or 0), 3),
        "avg_questions": round(float(avg_q or 0), 1),
        "conversion_rate": round(interested / total * 100, 1) if total else 0.0,
        "coaching_opportunities": coaching_opps,
        "outcomes": outcomes,
        "leaderboard": leaderboard,
        "trend": trend,
        "pending": pending,
    }


def since_days(days: int | None, now: datetime | None = None) -> datetime | None:
    if not days:
        return None
    current = now or datetime.now(UTC)
    return (current - timedelta(days=days)).replace(tzinfo=None)


def call_summary(call: models.Call) -> dict[str, Any]:
    return {
        "id": call.id,
        "contact_name": call.contact_name,
        "contact_phone": call.contact_phone,
        "agent": call.agent,
        "status": call.status,
        "score": call.score,
        "outcome": call.outcome,
        "talk_ratio": call.talk_ratio,
        "questions_asked": call.questions_asked,
        "duration_sec": call.duration_sec,
        "has_audio": bool(call.audio_path),
        "started_at": call.started_at,
        "created_at": call.created_at,
    }


def call_detail(call: models.Call) -> dict[str, Any]:
    return {
        **call_summary(call),
        "transcript": call.transcript,
        "transcript_source": call.transcript_source,
        "summary": call.summary,
        "highlights": call.highlights,
        "coaching": call.coaching,
        "strengths": call.strengths,
        "risks": call.risks,
        "next_best_action": call.next_best_action,
        "insights_source": call.insights_source,
        "error": call.error,
        "url": f"/calls/{call.id}",
    }


CSV_FIELDS = (
    "id",
    "created_at",
    "started_at",
    "agent",
    "contact_name",
    "contact_phone",
    "status",
    "score",
    "outcome",
    "talk_ratio",
    "questions_asked",
    "duration_sec",
    "summary",
    "next_best_action",
)
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def csv_safe(value: Any) -> Any:
    """Neutralize spreadsheet formula injection (CWE-1236)."""
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def export_csv(calls: list[models.Call]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_FIELDS)
    for call in calls:
        row = call_summary(call) | {
            "summary": call.summary,
            "next_best_action": call.next_best_action,
        }
        writer.writerow([csv_safe(row[f]) for f in CSV_FIELDS])
    return buf.getvalue()
