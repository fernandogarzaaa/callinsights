"""Service-layer tests against a real (temporary) SQLite database."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from callinsights import models
from callinsights.ai.asr import MockTranscriber, Segment, TranscriptionError
from callinsights.ai.llm import LocalProvider
from callinsights.config import Settings
from callinsights.db import make_engine, make_session_factory, run_migrations, session_scope
from callinsights.services import calls as svc
from callinsights.services import users as user_service
from tests.conftest import GOOD_TRANSCRIPT, make_settings

LLM = LocalProvider()


@pytest.fixture
def factory(settings: Settings) -> Iterator[sessionmaker[Session]]:
    engine = make_engine(settings.DATABASE_URL)
    run_migrations(engine)
    yield make_session_factory(engine)
    engine.dispose()


@pytest.fixture
def db(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    yield session
    session.close()


# --- input handling ------------------------------------------------------------------------


def test_clean_transcript() -> None:
    assert svc.clean_transcript("  a\r\nb\x00  ", 100) == "a\nb"
    assert svc.clean_transcript("x" * 10, 10) == "x" * 10
    with pytest.raises(svc.CallError, match="exceeds 9"):
        svc.clean_transcript("x" * 10, 9)


def test_decode_transcript_upload() -> None:
    vtt = b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nAgent: Hi\n"
    assert svc.decode_transcript_upload("a.vtt", vtt, 1000) == "Agent: Hi"
    assert svc.decode_transcript_upload("a.txt", "\ufeffAgent: Olá".encode(), 1000) == (
        "Agent: Olá"
    )
    assert svc.decode_transcript_upload("a.TXT", b"Agent: caf\xe9", 1000) == "Agent: café"
    with pytest.raises(svc.CallError, match="unsupported transcript type"):
        svc.decode_transcript_upload("a.pdf", b"x", 1000)
    with pytest.raises(svc.CallError, match="exceeds"):
        svc.decode_transcript_upload("a.txt", b"x" * 20, 10)


def test_save_audio(settings: Settings) -> None:
    stored, original = svc.save_audio(settings, "C:\\calls\\Rec 1.MP3", io.BytesIO(b"abc"))
    assert original == "Rec 1.MP3"
    assert stored.endswith(".mp3")
    assert len(stored) == 36
    assert (settings.upload_dir / stored).read_bytes() == b"abc"
    assert svc.resolve_audio_path(settings, stored) == settings.upload_dir / stored


def test_save_audio_rejects_bad_type_empty_and_oversize(tmp_path: Path) -> None:
    s = make_settings(tmp_path, MAX_UPLOAD_MB=1)
    with pytest.raises(svc.CallError, match="unsupported audio type"):
        svc.save_audio(s, "x.exe", io.BytesIO(b"abc"))
    with pytest.raises(svc.CallError, match="empty"):
        svc.save_audio(s, "x.wav", io.BytesIO(b""))
    with pytest.raises(svc.UploadTooLargeError, match="1 MB"):
        svc.save_audio(s, "x.wav", io.BytesIO(b"0" * (1024 * 1024 + 1)))
    svc.save_audio(s, "x.wav", io.BytesIO(b"0" * (1024 * 1024)))  # exactly at the limit
    assert len(list(s.upload_dir.iterdir())) == 1  # failed uploads leave no files behind


def test_save_audio_default_name(settings: Settings) -> None:
    with pytest.raises(svc.CallError, match="''"):
        svc.save_audio(settings, "", io.BytesIO(b"abc"))


def test_resolve_audio_path_is_confined(settings: Settings) -> None:
    settings.upload_dir.mkdir(parents=True)
    (settings.upload_dir / "a.wav").write_bytes(b"x")
    assert svc.resolve_audio_path(settings, None) is None
    assert svc.resolve_audio_path(settings, "") is None
    assert svc.resolve_audio_path(settings, "missing.wav") is None
    # Beta rows stored "data/uploads/<name>"; traversal attempts collapse to the basename.
    assert svc.resolve_audio_path(settings, "data/uploads/a.wav") == settings.upload_dir / "a.wav"
    assert svc.resolve_audio_path(settings, "..\\..\\a.wav") == settings.upload_dir / "a.wav"
    assert svc.resolve_audio_path(settings, "/etc/passwd") is None


def test_estimate_duration() -> None:
    assert svc.estimate_duration([]) == 0
    one = [{"speaker": "A", "role": "agent", "text": "hi"}]
    assert svc.estimate_duration(one) == 1  # type: ignore[arg-type]
    many = [{"speaker": "A", "role": "agent", "text": " ".join(["w"] * 25)}]
    assert svc.estimate_duration(many) == 10  # type: ignore[arg-type]


def test_to_naive_utc() -> None:
    manila = timezone(timedelta(hours=8))
    assert svc.to_naive_utc(datetime(2026, 1, 1, 8, 0, tzinfo=manila)) == datetime(2026, 1, 1)
    naive = datetime(2026, 1, 1, 8, 0)
    assert svc.to_naive_utc(naive) is naive
    assert svc.to_naive_utc(None) is None


# --- analysis & creation -------------------------------------------------------------------


def test_analyze_requires_turns() -> None:
    with pytest.raises(svc.CallError, match="no speaker turns"):
        svc.analyze("just some text", LLM)


def test_create_scores_transcript(db: Session) -> None:
    call = svc.create_call(
        db,
        llm=LLM,
        user=None,
        contact_name="  Dana ",
        agent="",
        transcript=GOOD_TRANSCRIPT,
        started_at=datetime(2026, 1, 1, 8, tzinfo=timezone(timedelta(hours=8))),
    )
    db.commit()
    a = svc.analyze(GOOD_TRANSCRIPT, LLM)
    assert call.id is not None
    assert call.status == "scored"
    assert call.contact_name == "Dana"
    assert call.agent == "Unassigned"
    assert call.started_at == datetime(2026, 1, 1, 0, 0)
    assert call.score == a.analytics["score"]
    assert call.outcome == a.analytics["outcome"]
    assert call.talk_ratio == a.analytics["talk_ratio"]
    assert call.questions_asked == a.analytics["questions_asked"]
    assert call.highlights == a.analytics["highlights"]
    assert call.coaching == a.insights.coaching
    assert call.strengths == a.insights.strengths
    assert call.risks == a.insights.risks
    assert call.summary == a.insights.summary
    assert call.next_best_action == a.insights.next_best_action
    assert call.insights_source == "local"
    assert call.duration_sec == svc.estimate_duration(a.turns)
    assert call.error is None


def test_create_keeps_explicit_duration_and_defaults(db: Session) -> None:
    call = svc.create_call(db, llm=LLM, user=None, transcript=GOOD_TRANSCRIPT, duration_sec=99)
    assert call.duration_sec == 99
    assert call.contact_name == "Unknown"
    assert call.transcript_source == "manual"
    neg = svc.create_call(db, llm=LLM, user=None, transcript=GOOD_TRANSCRIPT, duration_sec=-5)
    assert neg.duration_sec == svc.estimate_duration(svc.analyze(GOOD_TRANSCRIPT, LLM).turns)


def test_create_requires_input(db: Session) -> None:
    with pytest.raises(svc.CallError, match="provide transcript_text"):
        svc.create_call(db, llm=LLM, user=None)


def test_create_audio_only_is_queued(db: Session) -> None:
    user = user_service.create_user(db, "u@x.io", "Passw0rd-1", rounds=4)
    call = svc.create_call(db, llm=LLM, user=user, audio=("a.wav", "orig.wav"), agent="Z")
    assert call.status == "queued"
    assert call.audio_path == "a.wav"
    assert call.audio_filename == "orig.wav"
    assert call.created_by_id == user.id
    assert call.score == 0


def _queued(factory: sessionmaker[Session], settings: Settings, data: bytes = b"RIFFdata") -> int:
    stored, original = svc.save_audio(settings, "rec.wav", io.BytesIO(data))
    with session_scope(factory) as db:
        call = svc.create_call(db, llm=LLM, user=None, audio=(stored, original))
        return call.id


def test_process_call_transcribes_and_scores(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    cid = _queued(factory, settings)
    svc.process_call(factory, cid, settings, LLM, MockTranscriber())
    with session_scope(factory) as db:
        call = db.get(models.Call, cid)
        assert call is not None
        assert call.status == "scored"
        assert call.transcript.startswith("Agent: ")
        assert "\nProspect: " in call.transcript
        assert call.transcript_source == "asr:mock:heuristic"
        assert call.duration_sec > 0
        assert call.score > 0
        assert call.error is None


class FixedTranscriber:
    name = "fixed"

    def __init__(self, segments: list[Segment]) -> None:
        self.segments = segments

    def transcribe(self, path: Path) -> list[Segment]:
        return self.segments


class LabellingLLM(LocalProvider):
    def label_speakers(self, segments: list[str]) -> list[str]:  # type: ignore[override]
        return ["prospect" if i % 2 else "agent" for i in range(len(segments))]


def test_process_call_uses_llm_labels(factory: sessionmaker[Session], settings: Settings) -> None:
    cid = _queued(factory, settings)
    segs = [Segment(0, 1, "What is your budget?"), Segment(1, 2.6, "About ten grand.")]
    svc.process_call(factory, cid, settings, LabellingLLM(), FixedTranscriber(segs))
    with session_scope(factory) as db:
        call = db.get(models.Call, cid)
        assert call is not None
        assert call.transcript == "Agent: What is your budget?\nProspect: About ten grand."
        assert call.transcript_source == "asr:fixed:llm"
        assert call.duration_sec == 3


@pytest.mark.parametrize(
    ("segments", "error"),
    [([], "no speech detected"), ([Segment(0, 0, "no speaker prefix")], None)],
)
def test_process_call_failures_are_recorded(
    factory: sessionmaker[Session],
    settings: Settings,
    segments: list[Segment],
    error: str | None,
) -> None:
    cid = _queued(factory, settings)
    svc.process_call(factory, cid, settings, LLM, FixedTranscriber(segments))
    with session_scope(factory) as db:
        call = db.get(models.Call, cid)
        assert call is not None
        if error:
            assert call.status == "failed"
            assert error in (call.error or "")
        else:
            assert call.status == "scored"  # heuristic labels make it scorable
            assert call.duration_sec == 1  # estimated from words; segment end was 0


def test_process_call_without_transcriber_fails(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    cid = _queued(factory, settings)
    svc.process_call(factory, cid, settings, LLM, None)
    with session_scope(factory) as db:
        call = db.get(models.Call, cid)
        assert call is not None
        assert call.status == "failed"
        assert "ASR_PROVIDER=none" in (call.error or "")


def test_process_call_missing_audio(factory: sessionmaker[Session], settings: Settings) -> None:
    with session_scope(factory) as db:
        cid = svc.create_call(db, llm=LLM, user=None, audio=("gone.wav", "gone.wav")).id
    svc.process_call(factory, cid, settings, LLM, MockTranscriber())
    with session_scope(factory) as db:
        call = db.get(models.Call, cid)
        assert call is not None
        assert call.status == "failed"
        assert call.error == "audio file is missing"


def test_process_call_skips_scored_or_missing(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as db:
        cid = svc.create_call(db, llm=LLM, user=None, transcript=GOOD_TRANSCRIPT).id

    class Exploding:
        name = "x"

        def transcribe(self, path: Path) -> list[Segment]:
            raise AssertionError("must not be called")

    svc.process_call(factory, cid, settings, LLM, Exploding())
    svc.process_call(factory, 999_999, settings, LLM, Exploding())
    with session_scope(factory) as db:
        call = db.get(models.Call, cid)
        assert call is not None
        assert call.status == "scored"


def test_transcribe_call_error_type(settings: Settings) -> None:
    call = models.Call(audio_path=None)
    with pytest.raises(TranscriptionError):
        svc.transcribe_call(call, settings, LLM, MockTranscriber())


# --- updates, deletes ----------------------------------------------------------------------


def test_update_call_metadata_and_transcript(db: Session) -> None:
    call = svc.create_call(db, llm=LLM, user=None, transcript="Agent: Hi.\nProspect: Bye.")
    old_score = call.score
    svc.update_call(
        db,
        call,
        {
            "agent": "Maya",
            "contact_name": None,
            "duration_sec": 42,
            "started_at": datetime(2026, 2, 1, 10, tzinfo=UTC),
        },
        LLM,
        10_000,
    )
    assert call.agent == "Maya"
    assert call.contact_name == "Unknown"
    assert call.duration_sec == 42
    assert call.started_at == datetime(2026, 2, 1, 10)
    assert call.transcript_source == "manual"
    svc.update_call(db, call, {"transcript_text": GOOD_TRANSCRIPT}, LLM, 10_000)
    assert call.transcript_source == "edited"
    assert call.score != old_score
    assert call.score == svc.analyze(GOOD_TRANSCRIPT, LLM).analytics["score"]
    svc.update_call(db, call, {"started_at": None}, LLM, 10_000)
    assert call.started_at is None


def test_update_call_rejects_empty_transcript(db: Session) -> None:
    call = svc.create_call(db, llm=LLM, user=None, transcript=GOOD_TRANSCRIPT)
    with pytest.raises(svc.CallError, match="cannot be empty"):
        svc.update_call(db, call, {"transcript_text": "   "}, LLM, 10_000)


def test_rescore(db: Session) -> None:
    call = svc.create_call(db, llm=LLM, user=None, transcript=GOOD_TRANSCRIPT)
    call.score = 1
    call.status = "failed"
    svc.rescore_call(db, call, LLM)
    assert call.score == svc.analyze(GOOD_TRANSCRIPT, LLM).analytics["score"]
    assert call.status == "scored"
    empty = svc.create_call(db, llm=LLM, user=None, audio=("a.wav", "a.wav"))
    with pytest.raises(svc.CallError, match="no transcript"):
        svc.rescore_call(db, empty, LLM)


def test_delete_call_removes_audio(db: Session, settings: Settings) -> None:
    stored, original = svc.save_audio(settings, "r.wav", io.BytesIO(b"x"))
    call = svc.create_call(
        db, llm=LLM, user=None, transcript=GOOD_TRANSCRIPT, audio=(stored, original)
    )
    cid = call.id
    svc.delete_call(db, call, settings)
    db.commit()
    assert db.get(models.Call, cid) is None
    assert not (settings.upload_dir / stored).exists()
    no_audio = svc.create_call(db, llm=LLM, user=None, transcript=GOOD_TRANSCRIPT)
    svc.delete_call(db, no_audio, settings)


# --- queries -------------------------------------------------------------------------------


def _mk(db: Session, name: str, agent: str, transcript: str, days_ago: int = 0) -> models.Call:
    call = svc.create_call(
        db, llm=LLM, user=None, contact_name=name, agent=agent, transcript=transcript
    )
    call.created_at = models.utcnow() - timedelta(days=days_ago)
    db.flush()
    return call


WEAK = "Agent: Our product is great.\nProspect: Not interested, too expensive."


def test_list_calls_filters_and_pagination(db: Session) -> None:
    good = _mk(db, "Dana 100%_x", "Maya", GOOD_TRANSCRIPT, days_ago=1)
    weak = _mk(db, "Rob", "Jake", WEAK, days_ago=40)
    queued = svc.create_call(db, llm=LLM, user=None, audio=("a.wav", "a.wav"), agent="Maya")
    db.commit()
    F = svc.CallFilters
    ids = lambda f, **kw: [c.id for c in svc.list_calls(db, f, **kw)[0]]  # noqa: E731
    assert ids(F()) == [queued.id, good.id, weak.id]
    assert svc.list_calls(db, F())[1] == 3
    assert ids(F(agent="Maya")) == [queued.id, good.id]
    assert ids(F(outcome=weak.outcome)) == [weak.id]
    assert ids(F(status="queued")) == [queued.id]
    assert ids(F(min_score=good.score)) == [good.id]
    assert ids(F(max_score=weak.score)) == [queued.id, weak.id]
    assert ids(F(min_score=weak.score + 1, max_score=good.score)) == [good.id]
    assert ids(F(since=svc.since_days(30))) == [queued.id, good.id]
    assert ids(F(q="ticket volume")) == [good.id]
    assert ids(F(q="100%_")) == [good.id]  # LIKE wildcards are escaped
    assert ids(F(q="%")) == [good.id]  # literal percent sign
    assert ids(F(q="1_0")) == []
    assert ids(F(q="JAKE")) == [weak.id]
    assert ids(F(), limit=1, offset=1) == [good.id]
    rows, total = svc.list_calls(db, F(agent="Maya"), limit=1)
    assert (len(rows), total) == (1, 2)


def test_agents_and_since_days(db: Session) -> None:
    _mk(db, "a", "Zed", GOOD_TRANSCRIPT)
    _mk(db, "b", "Amy", WEAK)
    _mk(db, "c", "Amy", WEAK)
    assert svc.agents(db) == ["Amy", "Zed"]
    now = datetime(2026, 3, 10, 12, tzinfo=UTC)
    assert svc.since_days(None) is None
    assert svc.since_days(0) is None
    assert svc.since_days(7, now=now) == datetime(2026, 3, 3, 12)


def test_analytics_summary(db: Session) -> None:
    empty = svc.analytics_summary(db, svc.CallFilters())
    assert empty == {
        "total_calls": 0,
        "avg_score": 0.0,
        "avg_talk_ratio": 0.0,
        "avg_questions": 0.0,
        "conversion_rate": 0.0,
        "coaching_opportunities": 0,
        "outcomes": {},
        "leaderboard": [],
        "trend": [],
        "pending": 0,
    }
    good = _mk(db, "a", "Maya", GOOD_TRANSCRIPT, days_ago=1)
    weak = _mk(db, "b", "Jake", WEAK, days_ago=2)
    weak2 = _mk(db, "c", "Jake", WEAK, days_ago=2)
    svc.create_call(db, llm=LLM, user=None, audio=("a.wav", "a.wav"))
    db.commit()
    assert good.outcome == "Interested"
    s = svc.analytics_summary(db, svc.CallFilters())
    assert s["total_calls"] == 3
    assert s["avg_score"] == round((good.score + weak.score + weak2.score) / 3, 1)
    assert s["conversion_rate"] == round(1 / 3 * 100, 1)
    assert s["coaching_opportunities"] == 2
    assert s["outcomes"] == {"Interested": 1, weak.outcome: 2}
    assert s["pending"] == 1
    assert s["leaderboard"] == [
        {"agent": "Maya", "calls": 1, "avg_score": float(good.score), "interested": 1},
        {"agent": "Jake", "calls": 2, "avg_score": float(weak.score), "interested": 0},
    ]
    assert [t["calls"] for t in s["trend"]] == [2, 1]
    assert s["avg_questions"] == round((good.questions_asked + 0 + 0) / 3, 1)
    only_jake = svc.analytics_summary(db, svc.CallFilters(agent="Jake"))
    assert only_jake["total_calls"] == 2
    assert only_jake["conversion_rate"] == 0.0


def test_call_serialization_and_csv(db: Session) -> None:
    call = _mk(db, "=HYPERLINK(evil)", "+Agent", GOOD_TRANSCRIPT)
    detail = svc.call_detail(call)
    assert detail["url"] == f"/calls/{call.id}"
    assert detail["has_audio"] is False
    assert detail["highlights"] == call.highlights
    assert set(svc.call_summary(call)) <= set(detail)
    csv_text = svc.export_csv([call])
    header, row = csv_text.strip().splitlines()[:2]
    assert header == ",".join(svc.CSV_FIELDS)
    assert "'=HYPERLINK(evil)" in row
    assert "'+Agent" in row


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("=1+1", "'=1+1"),
        ("+x", "'+x"),
        ("-x", "'-x"),
        ("@x", "'@x"),
        ("\tx", "'\tx"),
        ("\rx", "'\rx"),
        ("plain", "plain"),
        (5, 5),
        ("", ""),
    ],
)
def test_csv_safe(value: object, expected: object) -> None:
    assert svc.csv_safe(value) == expected


def test_model_json_properties_are_robust() -> None:
    call = models.Call(
        highlights_json="not json",
        coaching_json='{"a": 1}',
        strengths_json='["a", 2]',
        risks_json=None,  # type: ignore[arg-type]
    )
    assert call.highlights == []
    assert call.coaching == []
    assert call.strengths == ["a", "2"]
    assert call.risks == []
    mixed = models.Call(highlights_json=json.dumps([{"label": "x"}, "junk"]))
    assert mixed.highlights == [{"label": "x"}]


# --- users ---------------------------------------------------------------------------------


def test_user_lifecycle(db: Session) -> None:
    user = user_service.create_user(db, " New@Example.com ", "Passw0rd-1", name=" N ", rounds=4)
    assert user.email == "new@example.com"
    assert user.name == "N"
    assert user.role == "member"
    assert user.is_admin is False
    assert user_service.get_by_email(db, "NEW@example.com ") is user
    assert user_service.authenticate(db, "new@example.com", "Passw0rd-1") is user
    assert user_service.authenticate(db, "new@example.com", "wrong-pass-1") is None
    assert user_service.authenticate(db, "nobody@example.com", "Passw0rd-1") is None
    user.is_active = False
    assert user_service.authenticate(db, "new@example.com", "Passw0rd-1") is None
    with pytest.raises(user_service.DuplicateUserError):
        user_service.create_user(db, "new@example.com", "Passw0rd-1", rounds=4)


@pytest.mark.parametrize(
    ("email", "password", "role", "message"),
    [
        ("not-an-email", "Passw0rd-1", "member", "invalid email"),
        ("a@b.co", "short", "member", "at least 10"),
        ("a@b.co", "Passw0rd-1", "owner", "role must be"),
    ],
)
def test_create_user_validation(
    db: Session, email: str, password: str, role: str, message: str
) -> None:
    with pytest.raises(user_service.UserError, match=message):
        user_service.create_user(db, email, password, role=role, rounds=4)


def test_set_password(db: Session) -> None:
    user = user_service.create_user(db, "p@x.io", "Passw0rd-1", rounds=4)
    user_service.set_password(user, "N3w-password", rounds=4)
    assert user_service.authenticate(db, "p@x.io", "N3w-password") is user
    with pytest.raises(user_service.UserError):
        user_service.set_password(user, "weak", rounds=4)


def test_ensure_admin_and_counts(db: Session) -> None:
    assert user_service.ensure_admin(db, None, "x", 4) is False
    assert user_service.ensure_admin(db, "a@x.io", None, 4) is False
    assert user_service.ensure_admin(db, "a@x.io", "Adm1n-password", 4) is True
    assert user_service.ensure_admin(db, "a@x.io", "Adm1n-password", 4) is False
    admin = user_service.get_by_email(db, "a@x.io")
    assert admin is not None
    assert admin.role == "admin"
    assert admin.is_admin is True
    assert user_service.active_admin_count(db) == 1
    admin.is_active = False
    db.flush()
    assert user_service.active_admin_count(db) == 0


def test_api_keys(db: Session) -> None:
    user = user_service.create_user(db, "k@x.io", "Passw0rd-1", rounds=4)
    other = user_service.create_user(db, "o@x.io", "Passw0rd-1", rounds=4)
    key, raw = user_service.create_api_key(db, user, "  ci  ")
    assert key.name == "ci"
    assert key.prefix == raw[:10]
    assert key.last_used_at is None
    assert user_service.user_for_api_key(db, raw) is user
    assert key.last_used_at is not None
    assert user_service.user_for_api_key(db, raw + "x") is None
    assert user_service.revoke_api_key(db, other, key.id) is False
    assert user_service.revoke_api_key(db, user, 9999) is False
    assert user_service.revoke_api_key(db, user, key.id) is True
    revoked_at = key.revoked_at
    assert revoked_at is not None
    assert user_service.revoke_api_key(db, user, key.id) is True
    assert key.revoked_at == revoked_at  # idempotent
    assert user_service.user_for_api_key(db, raw) is None
    key2, raw2 = user_service.create_api_key(db, user, "two")
    user.is_active = False
    assert user_service.user_for_api_key(db, raw2) is None
    admin = user_service.create_user(db, "ad@x.io", "Passw0rd-1", role="admin", rounds=4)
    assert user_service.revoke_api_key(db, admin, key2.id) is True
