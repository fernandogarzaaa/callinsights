from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from callinsights.config import Settings, get_settings
from callinsights.logging_setup import (
    JsonFormatter,
    TextFormatter,
    configure_logging,
    request_id_var,
)
from tests.conftest import make_settings

STRONG = "s" * 40


def test_defaults(tmp_path: Path) -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.ENV == "development"
    assert s.PORT == 8003
    assert s.llm_mode == "local"
    assert s.cookie_secure is False
    assert s.upload_dir == Path("./data") / "uploads"
    assert s.max_upload_bytes == 200 * 1024 * 1024
    assert s.trusted_origins == []
    assert s.ADMIN_EMAIL is None
    assert s.ADMIN_PASSWORD is None


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1/")
    monkeypatch.setenv("TRUSTED_ORIGINS", " https://a.example/ , ,https://b.example")
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s is get_settings()
        assert s.PORT == 9000
        assert s.LLM_BASE_URL == "https://api.groq.com/openai/v1"
        assert s.llm_mode == "openai"
        assert s.trusted_origins == ["https://a.example", "https://b.example"]
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgres://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_database_url_normalization(url: str, expected: str) -> None:
    assert expected == Settings(_env_file=None, DATABASE_URL=url).DATABASE_URL  # type: ignore[call-arg]


def test_admin_normalization() -> None:
    s = Settings(_env_file=None, ADMIN_EMAIL="  Boss@Example.COM ", ADMIN_PASSWORD="")  # type: ignore[call-arg]
    assert s.ADMIN_EMAIL == "boss@example.com"
    assert s.ADMIN_PASSWORD is None
    assert Settings(_env_file=None, ADMIN_EMAIL="  ").ADMIN_EMAIL is None  # type: ignore[call-arg]


def test_llm_mode_explicit() -> None:
    assert (
        Settings(_env_file=None, LLM_PROVIDER="local", LLM_BASE_URL="http://x").llm_mode == "local"
    )  # type: ignore[call-arg]
    assert (
        Settings(_env_file=None, LLM_PROVIDER="openai", LLM_BASE_URL="http://x").llm_mode
        == "openai"
    )  # type: ignore[call-arg]


def test_provider_requirements() -> None:
    with pytest.raises(ValidationError, match="LLM_PROVIDER=openai requires LLM_BASE_URL"):
        Settings(_env_file=None, LLM_PROVIDER="openai")  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="ASR_PROVIDER=openai requires"):
        Settings(_env_file=None, ASR_PROVIDER="openai")  # type: ignore[call-arg]
    s = Settings(_env_file=None, ASR_PROVIDER="openai", ASR_BASE_URL="http://asr")  # type: ignore[call-arg]
    assert s.asr_base_url == "http://asr"


def test_asr_falls_back_to_llm_credentials() -> None:
    s = Settings(_env_file=None, LLM_BASE_URL="http://llm", LLM_API_KEY="k1")  # type: ignore[call-arg]
    assert (s.asr_base_url, s.asr_api_key) == ("http://llm", "k1")
    s2 = Settings(  # type: ignore[call-arg]
        _env_file=None, LLM_BASE_URL="http://llm", LLM_API_KEY="k1", ASR_API_KEY="k2"
    )
    assert s2.asr_api_key == "k2"


@pytest.mark.parametrize("secret", ["change-me-in-production", "short-but-not-default", ""])
def test_production_requires_strong_secret(secret: str) -> None:
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(_env_file=None, ENV="production", SECRET_KEY=secret)  # type: ignore[call-arg]


def test_production_secret_length_boundary() -> None:
    Settings(_env_file=None, ENV="production", SECRET_KEY="a" * 32)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ENV="production", SECRET_KEY="a" * 31)  # type: ignore[call-arg]


def test_production_rejects_default_admin_password() -> None:
    with pytest.raises(ValidationError, match="ADMIN_PASSWORD"):
        Settings(_env_file=None, ENV="production", SECRET_KEY=STRONG, ADMIN_PASSWORD="ChangeMe123!")  # type: ignore[call-arg]
    s = Settings(_env_file=None, ENV="production", SECRET_KEY=STRONG, ADMIN_PASSWORD="Str0ng-pass!")  # type: ignore[call-arg]
    assert s.cookie_secure is True


def test_development_allows_weak_values() -> None:
    s = Settings(_env_file=None, ADMIN_PASSWORD="ChangeMe123!")  # type: ignore[call-arg]
    assert s.ADMIN_PASSWORD is not None


def test_cookie_secure_override(tmp_path: Path) -> None:
    assert make_settings(tmp_path, COOKIE_SECURE=True).cookie_secure is True
    prod = Settings(_env_file=None, ENV="production", SECRET_KEY=STRONG, COOKIE_SECURE=False)  # type: ignore[call-arg]
    assert prod.cookie_secure is False


def test_upload_limits(tmp_path: Path) -> None:
    s = make_settings(tmp_path, MAX_UPLOAD_MB=3)
    assert s.max_upload_bytes == 3 * 1024 * 1024
    assert s.upload_dir == tmp_path / "data" / "uploads"


def _record(msg: str = "hello", **extra: object) -> logging.LogRecord:
    rec = logging.LogRecord("callinsights.test", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_json_formatter_fields_and_request_id() -> None:
    token = request_id_var.set("rid-1")
    try:
        out = json.loads(JsonFormatter().format(_record(call_id=7)))
    finally:
        request_id_var.reset(token)
    assert out["msg"] == "hello"
    assert out["level"] == "INFO"
    assert out["logger"] == "callinsights.test"
    assert out["request_id"] == "rid-1"
    assert out["call_id"] == 7
    assert out["ts"].endswith("+00:00")
    assert "exc" not in out


def test_json_formatter_exception_and_no_request_id() -> None:
    try:
        raise RuntimeError("bad")
    except RuntimeError:
        import sys

        rec = _record()
        rec.exc_info = sys.exc_info()
    out = json.loads(JsonFormatter().format(rec))
    assert "RuntimeError: bad" in out["exc"]
    assert "request_id" not in out


def test_text_formatter() -> None:
    assert TextFormatter().format(_record()).endswith("INFO    callinsights.test: hello")
    token = request_id_var.set("r9")
    try:
        line = TextFormatter().format(_record(user_id=3))
    finally:
        request_id_var.reset(token)
    assert line.endswith("callinsights.test: hello request_id=r9 user_id=3")


def test_configure_logging_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging("INFO", "json")
        configure_logging("DEBUG", "text")
        ours = [h for h in root.handlers if h.get_name() == "callinsights"]
        assert len(ours) == 1
        assert isinstance(ours[0].formatter, TextFormatter)
        assert root.level == logging.DEBUG
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("uvicorn.access").level == logging.WARNING
        assert logging.getLogger("uvicorn").propagate is True
        stream = io.StringIO()
        ours[0].setStream(stream)  # type: ignore[attr-defined]
        logging.getLogger("callinsights.x").info("visible")
        assert "visible" in stream.getvalue()
    finally:
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)
        root.setLevel(logging.WARNING)
