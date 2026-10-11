"""Calls JSON API: intake, scoring, retrieval, editing, export and analytics."""

from __future__ import annotations

import logging
import mimetypes
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from callinsights import models, scoring
from callinsights.config import Settings
from callinsights.schemas import CallCreate, CallDetailOut, CallList, CallUpdate, ScoreIn
from callinsights.services import calls as svc
from callinsights.web.deps import DbDep, LLMDep, SettingsDep, TranscriberDep, UserDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["calls"])

Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0)]
Score = Annotated[int | None, Query(ge=0, le=100)]
Days = Annotated[int | None, Query(ge=1, le=3650)]


def make_filters(
    agent: str = "",
    outcome: str = "",
    status_: str = "",
    q: str = "",
    min_score: int | None = None,
    max_score: int | None = None,
    days: int | None = None,
) -> svc.CallFilters:
    return svc.CallFilters(
        agent=agent.strip(),
        outcome=outcome.strip(),
        status=status_.strip(),
        q=q.strip()[:200],
        min_score=min_score,
        max_score=max_score,
        since=svc.since_days(days),
    )


def _get_call(db: DbDep, call_id: int) -> models.Call:
    call = db.get(models.Call, call_id)
    if call is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found")
    return call


def _http_error(exc: svc.CallError) -> HTTPException:
    if isinstance(exc, svc.UploadTooLargeError):
        return HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(exc))
    return HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))


def _validation_error(exc: ValidationError) -> HTTPException:
    return HTTPException(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        jsonable_encoder(exc.errors(include_url=False, include_context=False)),
    )


@router.get("/calls", response_model=CallList)
def list_calls(
    _: UserDep,
    db: DbDep,
    agent: str = "",
    outcome: str = "",
    status_filter: Annotated[str, Query(alias="status")] = "",
    q: str = "",
    min_score: Score = None,
    max_score: Score = None,
    days: Days = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> dict[str, Any]:
    f = make_filters(agent, outcome, status_filter, q, min_score, max_score, days)
    rows, total = svc.list_calls(db, f, limit, offset)
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "calls": [svc.call_summary(c) for c in rows],
    }


@router.get("/calls/export.csv", response_class=Response)
def export_calls(
    _: UserDep,
    db: DbDep,
    agent: str = "",
    outcome: str = "",
    q: str = "",
    days: Days = None,
) -> Response:
    f = make_filters(agent, outcome, "", q, days=days)
    rows, _total = svc.list_calls(db, f, 10_000, 0)
    return Response(
        svc.export_csv(rows),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="calls.csv"'},
    )


@router.get("/agents")
def list_agents(_: UserDep, db: DbDep) -> dict[str, list[str]]:
    return {"agents": svc.agents(db)}


@router.get("/analytics")
def analytics(_: UserDep, db: DbDep, agent: str = "", days: Days = None) -> dict[str, Any]:
    return svc.analytics_summary(db, make_filters(agent=agent, days=days))


@router.post("/score")
def score_transcript(
    payload: ScoreIn, _: UserDep, llm: LLMDep, settings: SettingsDep
) -> dict[str, Any]:
    """Analyze a transcript without storing it."""
    try:
        text = svc.clean_transcript(payload.transcript_text, settings.MAX_TRANSCRIPT_CHARS)
        result = svc.analyze(text, llm)
    except svc.CallError as exc:
        raise _http_error(exc) from exc
    return {**result.analytics, "insights": result.insights.model_dump()}


async def read_create_request(
    request: Request, settings: Settings
) -> tuple[CallCreate, str, UploadFile | None]:
    """Parse JSON or multipart intake. Returns (fields, transcript, audio upload)."""
    ctype = request.headers.get("content-type", "")
    if "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype:
        form = await request.form(max_files=2, max_fields=20)
        raw = {k: v for k, v in form.items() if isinstance(v, str) and v.strip()}
        try:
            fields = CallCreate.model_validate(
                {k: raw[k] for k in CallCreate.model_fields if k in raw}
            )
        except ValidationError as exc:
            raise _validation_error(exc) from exc
        transcript = fields.transcript_text
        tfile = form.get("transcript_file")
        if isinstance(tfile, UploadFile) and tfile.filename:
            limit = settings.MAX_TRANSCRIPT_CHARS * 4
            data = await tfile.read(limit + 1)
            if len(data) > limit:
                raise svc.UploadTooLargeError("transcript file is too large")
            transcript = svc.decode_transcript_upload(
                tfile.filename, data, settings.MAX_TRANSCRIPT_CHARS
            )
        afile = form.get("audio_file")
        audio = afile if isinstance(afile, UploadFile) and afile.filename else None
        return fields, transcript, audio
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Expected a JSON or multipart body"
        ) from exc
    try:
        fields = CallCreate.model_validate(body)
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    return fields, fields.transcript_text, None


@router.post("/calls", status_code=201, response_model=CallDetailOut)
async def create_call(
    request: Request,
    background: BackgroundTasks,
    user: UserDep,
    db: DbDep,
    llm: LLMDep,
    settings: SettingsDep,
    transcriber: TranscriberDep,
) -> Response:
    """Create a call from a transcript (scored immediately) and/or an audio recording.

    Send JSON (``transcript_text``) or multipart form data (``transcript_text`` /
    ``transcript_file`` and/or ``audio_file``). Audio-only calls return **202** and are
    transcribed in the background (``status``: ``queued -> transcribing -> scoring ->
    scored``); poll ``GET /api/calls/{id}``.
    """
    stored: str | None = None
    try:
        fields, transcript, audio = await read_create_request(request, settings)
        transcript = svc.clean_transcript(transcript, settings.MAX_TRANSCRIPT_CHARS)
        if not transcript and audio is not None and transcriber is None:
            raise svc.CallError(
                "audio-only upload needs speech-to-text; set ASR_PROVIDER or include a transcript"
            )
        audio_ref: tuple[str, str] | None = None
        if audio is not None:
            stored, original = svc.save_audio(settings, audio.filename or "", audio.file)
            audio_ref = (stored, original)
        call = svc.create_call(
            db,
            llm=llm,
            user=user,
            contact_name=fields.contact_name,
            contact_phone=fields.contact_phone,
            agent=fields.agent,
            started_at=fields.started_at,
            duration_sec=fields.duration_sec,
            transcript=transcript,
            audio=audio_ref,
            transcript_source="upload" if transcript and audio_ref else "manual",
        )
        db.commit()
    except svc.CallError as exc:
        if stored:
            (settings.upload_dir / stored).unlink(missing_ok=True)
        raise _http_error(exc) from exc
    except BaseException:
        if stored:
            (settings.upload_dir / stored).unlink(missing_ok=True)
        raise
    code = status.HTTP_201_CREATED
    if call.status == "queued":
        code = status.HTTP_202_ACCEPTED
        background.add_task(
            svc.process_call, request.app.state.session_factory, call.id, settings, llm, transcriber
        )
    payload = CallDetailOut.model_validate(svc.call_detail(call)).model_dump(mode="json")
    return JSONResponse(payload, status_code=code, background=background)


@router.get("/calls/{call_id}", response_model=CallDetailOut)
def get_call(call_id: int, _: UserDep, db: DbDep) -> dict[str, Any]:
    return svc.call_detail(_get_call(db, call_id))


@router.get("/calls/{call_id}/analysis")
def get_call_analysis(call_id: int, _: UserDep, db: DbDep) -> dict[str, Any]:
    """Full deterministic analytics (components, sentiment, objection counts) for a call."""
    call = _get_call(db, call_id)
    turns = scoring.parse_turns(call.transcript)
    return {"turns": turns, "analytics": scoring.score_call(turns) if turns else None}


@router.patch("/calls/{call_id}", response_model=CallDetailOut)
def update_call(
    call_id: int,
    payload: CallUpdate,
    _: UserDep,
    db: DbDep,
    llm: LLMDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    """Edit call metadata; sending ``transcript_text`` re-scores the call."""
    call = _get_call(db, call_id)
    try:
        svc.update_call(
            db, call, payload.model_dump(exclude_unset=True), llm, settings.MAX_TRANSCRIPT_CHARS
        )
    except svc.CallError as exc:
        raise _http_error(exc) from exc
    return svc.call_detail(call)


@router.post("/calls/{call_id}/rescore", response_model=CallDetailOut)
def rescore(call_id: int, _: UserDep, db: DbDep, llm: LLMDep) -> dict[str, Any]:
    call = _get_call(db, call_id)
    try:
        svc.rescore_call(db, call, llm)
    except svc.CallError as exc:
        raise _http_error(exc) from exc
    return svc.call_detail(call)


@router.post("/calls/{call_id}/transcribe", status_code=202, response_model=CallDetailOut)
def retranscribe(
    call_id: int,
    request: Request,
    background: BackgroundTasks,
    _: UserDep,
    db: DbDep,
    llm: LLMDep,
    settings: SettingsDep,
    transcriber: TranscriberDep,
) -> dict[str, Any]:
    """Queue (re-)transcription of the call's audio, replacing its transcript and score."""
    call = _get_call(db, call_id)
    if transcriber is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Speech-to-text is disabled (ASR_PROVIDER)")
    if svc.resolve_audio_path(settings, call.audio_path) is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Call has no audio recording")
    if call.status in ("transcribing", "scoring"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Call is already being processed")
    call.status = "queued"
    call.error = None
    db.commit()
    background.add_task(
        svc.process_call, request.app.state.session_factory, call.id, settings, llm, transcriber
    )
    return svc.call_detail(call)


@router.delete("/calls/{call_id}", status_code=204)
def delete_call(call_id: int, user: UserDep, db: DbDep, settings: SettingsDep) -> Response:
    call = _get_call(db, call_id)
    if not user.is_admin and call.created_by_id != user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only admins or the uploader can delete")
    svc.delete_call(db, call, settings)
    logger.info("call deleted", extra={"call_id": call_id, "user_id": user.id})
    return Response(status_code=204)


def audio_response(call: models.Call, settings: Settings) -> FileResponse:
    path = svc.resolve_audio_path(settings, call.audio_path)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Audio not found")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=call.audio_filename or path.name)


@router.get("/calls/{call_id}/audio", response_class=FileResponse)
def get_audio(call_id: int, _: UserDep, db: DbDep, settings: SettingsDep) -> FileResponse:
    return audio_response(_get_call(db, call_id), settings)
