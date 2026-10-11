"""Pages, health checks, middleware behaviour, OpenAI-compatible wiring and the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from callinsights import __version__, cli
from callinsights.ai.llm import OpenAICompatibleProvider
from callinsights.config import get_settings
from callinsights.main import create_app
from tests.conftest import (
    ADMIN_EMAIL,
    ADMIN_PASSWORD,
    GOOD_TRANSCRIPT,
    build_client,
    login,
    make_settings,
)

pytestmark = pytest.mark.integration


def test_health_endpoints(anon: TestClient) -> None:
    assert anon.get("/healthz").json() == {"status": "ok"}
    assert anon.get("/readyz").json() == {"status": "ok", "database": True}
    h = anon.get("/api/health").json()
    assert h == {
        "status": "ok",
        "app": "CallInsights",
        "version": __version__,
        "database": True,
        "llm": "local",
        "asr": "mock",
    }


def test_readiness_reports_database_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = make_settings(tmp_path)
    with build_client(s, use_default_transcriber=False) as c:
        assert c.get("/readyz").status_code == 200
        monkeypatch.setattr("callinsights.main.check_database", lambda engine: False)
        assert c.get("/readyz").status_code == 503
        r = c.get("/api/health")
        assert r.status_code == 503
        assert r.json()["status"] == "degraded"
        assert r.json()["asr"] == "none"


def test_security_headers_and_request_id(anon: TestClient) -> None:
    r = anon.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "strict-transport-security" not in r.headers
    generated = anon.get("/healthz").headers["x-request-id"]
    assert len(generated) == 32
    docs = anon.get("/docs")
    assert docs.status_code == 200
    assert "cdn.jsdelivr.net" in docs.headers["content-security-policy"]


def test_unhandled_errors_return_json_500(settings: Any) -> None:
    app = create_app(settings)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("kaboom")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom", headers={"X-Request-ID": "req-9"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error", "request_id": "req-9"}


def test_pages_render(seeded: TestClient) -> None:
    dash = seeded.get("/")
    assert dash.status_code == 200
    assert "Call Analytics" in dash.text
    assert "Dana Whitfield" in dash.text
    assert "Agent leaderboard" in dash.text
    assert f"v{__version__}" in dash.text
    assert "cdn.tailwindcss.com" not in dash.text
    filtered = seeded.get("/", params={"agent": "Maya Santos"})
    assert "Dana Whitfield" in filtered.text
    assert "Tom Beck" not in filtered.text
    assert "Tom Beck" in seeded.get("/", params={"q": "Tom"}).text
    assert seeded.get("/", params={"page": 2}).status_code == 200
    assert seeded.get("/", params={"days": 7, "outcome": "Interested"}).status_code == 200
    assert seeded.get("/", params={"page": 0}).status_code == 422

    detail = seeded.get("/calls/1")
    assert detail.status_code == 200
    for text in (
        "Coaching recommendations",
        "Key highlights",
        "Next best action",
        "Edit transcript",
    ):
        assert text in detail.text
    missing = seeded.get("/calls/999")
    assert missing.status_code == 404
    assert "Call not found" in missing.text

    new = seeded.get("/calls/new")
    assert new.status_code == 200
    assert "transcribed automatically" in new.text
    settings_page = seeded.get("/settings")
    assert "AI insights provider" in settings_page.text
    assert "Team" in settings_page.text
    assert "mock" in settings_page.text


def test_pages_redirect_when_anonymous(anon: TestClient) -> None:
    for path in ("/", "/calls/new", "/calls/1", "/settings"):
        r = anon.get(path, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"


def test_static_assets_are_served(anon: TestClient) -> None:
    for asset in (
        "/static/app.css",
        "/static/vendor/alpine.min.js",
        "/static/vendor/chart.umd.min.js",
    ):
        assert anon.get(asset).status_code == 200, asset


def test_openai_compatible_provider_end_to_end(tmp_path: Path) -> None:
    """The whole intake pipeline against a fake OpenAI-compatible server."""
    replies = {
        "summary": "Prospect wants a pilot.",
        "coaching": ["Confirm the decision date."],
        "strengths": ["Good discovery"],
        "risks": ["Price sensitivity"],
        "next_best_action": "Send the pilot proposal.",
    }
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(replies)}}]}
        )

    llm = OpenAICompatibleProvider(
        "https://fake.llm/v1",
        "",
        "fake-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with build_client(make_settings(tmp_path), llm=llm) as c:
        login(c)
        call = c.post("/api/calls", json={"transcript_text": GOOD_TRANSCRIPT}).json()
        assert call["summary"] == "Prospect wants a pilot."
        assert call["next_best_action"] == "Send the pilot proposal."
        assert call["insights_source"] == "llm:fake-model"
        assert "fake-model" in c.get("/settings").text
    assert seen[0]["model"] == "fake-model"


# --- CLI ---------------------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "DATABASE_URL": f"sqlite:///{tmp_path / 'cli.db'}",
        "DATA_DIR": str(tmp_path / "data"),
        "BCRYPT_ROUNDS": "4",
        "LOG_LEVEL": "WARNING",
        "LOG_FORMAT": "text",
        "ADMIN_EMAIL": ADMIN_EMAIL,
        "ADMIN_PASSWORD": ADMIN_PASSWORD,
        "ASR_PROVIDER": "mock",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"callinsights {__version__}"


def test_cli_migrate_seed_and_create_user(
    cli_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(["migrate"]) == 0
    assert "migrated" in capsys.readouterr().out
    assert cli.main(["seed", "--demo"]) == 0
    assert json.loads(capsys.readouterr().out) == {"users": 1, "calls": 5}
    assert cli.main(["seed"]) == 0
    assert json.loads(capsys.readouterr().out) == {"users": 0, "calls": 0}

    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("Rep-passw0rd\n"))
    assert cli.main(["create-user", "rep@example.com", "--password-stdin", "--name", "Rep"]) == 0
    assert "created member rep@example.com" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("Rep-passw0rd\n"))
    assert cli.main(["create-user", "rep@example.com", "--password-stdin"]) == 1
    assert "already exists" in capsys.readouterr().err

    answers = iter(["Adm1n-number-2", "Adm1n-number-2"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    assert cli.main(["create-user", "boss@example.com", "--admin"]) == 0
    assert "created admin boss@example.com" in capsys.readouterr().out
    mismatch = iter(["Adm1n-number-2", "different-1"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(mismatch))
    assert cli.main(["create-user", "x@example.com"]) == 1
    assert "do not match" in capsys.readouterr().err


def test_cli_score(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = cli_env / "call.txt"
    f.write_text(GOOD_TRANSCRIPT)
    assert cli.main(["score", str(f)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Score: ")
    assert "Next best action:" in out
    assert cli.main(["score", str(f), "--json", "--local"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["insights"]["source"] == "local"
    bad = cli_env / "bad.txt"
    bad.write_text("no turns")
    assert cli.main(["score", str(bad)]) == 1
    assert cli.main(["score", str(cli_env / "missing.txt")]) == 1


def test_cli_transcribe(
    cli_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = cli_env / "a.wav"
    audio.write_bytes(b"RIFF1234")
    assert cli.main(["transcribe", str(audio)]) == 0
    assert capsys.readouterr().out.startswith("Agent: ")
    assert cli.main(["transcribe", str(cli_env / "missing.wav")]) == 1
    monkeypatch.setenv("ASR_PROVIDER", "none")
    get_settings.cache_clear()
    assert cli.main(["transcribe", str(audio)]) == 1
    assert "set ASR_PROVIDER" in capsys.readouterr().err


def test_cli_config_errors_are_reported(
    cli_env: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ENV", "production")
    get_settings.cache_clear()
    assert cli.main(["migrate"]) == 1
    assert "SECRET_KEY" in capsys.readouterr().err


def test_cli_healthcheck(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> httpx.Response:
        assert url == "http://127.0.0.1:9999/readyz"
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "get", fake_get)
    assert cli.main(["healthcheck", "--port", "9999"]) == 0
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(503))
    assert cli.main(["healthcheck"]) == 1

    def down(url: str, timeout: float) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    assert cli.main(["healthcheck"]) == 1


def test_cli_serve_invokes_uvicorn(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: captured.update(app=app, **kw))
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert captured["app"] == "callinsights.main:create_app"
    assert captured["factory"] is True
    assert captured["port"] == 8123
    assert captured["host"] == "0.0.0.0"  # noqa: S104
    assert captured["workers"] == 1
    assert captured["proxy_headers"] is True
