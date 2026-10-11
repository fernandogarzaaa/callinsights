"""Shared fixtures. Every test runs offline: local LLM provider, mock ASR, temp SQLite."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from callinsights.ai.asr import MockTranscriber, Transcriber
from callinsights.ai.llm import LocalProvider
from callinsights.config import Settings
from callinsights.main import create_app

ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "Adm1n-password!"
SECRET = "test-secret-key-that-is-long-enough-0123456789"

GOOD_TRANSCRIPT = """Agent: Hi, thanks for taking the time. What is your biggest support challenge right now?
Prospect: Ticket volume. We are drowning.
Agent: How many tickets per week?
Prospect: About three hundred.
Agent: And what budget have you set aside for tooling?
Prospect: Around a thousand a month. But your pricing looks expensive.
Agent: Fair point. What if we start with a pilot so the ROI is proven first?
Prospect: That sounds good. Our COO makes the final decision. When could we start?
Agent: I will send a proposal and we can schedule a follow up on Thursday with your COO."""


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "ENV": "test",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}",
        "DATA_DIR": tmp_path / "data",
        "SECRET_KEY": SECRET,
        "BCRYPT_ROUNDS": 4,
        "ADMIN_EMAIL": ADMIN_EMAIL,
        "ADMIN_PASSWORD": ADMIN_PASSWORD,
        "LOG_LEVEL": "WARNING",
        "LOG_FORMAT": "text",
        "LLM_PROVIDER": "local",
        "ASR_PROVIDER": "mock",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


def build_client(
    settings: Settings,
    transcriber: Transcriber | None = None,
    *,
    use_default_transcriber: bool = True,
    llm: Any = None,
) -> TestClient:
    app = create_app(
        settings,
        llm=llm or LocalProvider(),
        transcriber=transcriber,
        use_default_transcriber=use_default_transcriber,
    )
    return TestClient(app, base_url="http://testserver", headers={"Origin": "http://testserver"})


def login(client: TestClient, email: str = ADMIN_EMAIL, password: str = ADMIN_PASSWORD) -> str:
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    token: str = r.json()["access_token"]
    return token


@pytest.fixture
def anon(settings: Settings) -> Iterator[TestClient]:
    with build_client(settings, MockTranscriber()) as client:
        yield client


@pytest.fixture
def client(anon: TestClient) -> TestClient:
    """Admin session (cookie) on a fresh database."""
    login(anon)
    return anon


@pytest.fixture
def seeded(tmp_path: Path) -> Iterator[TestClient]:
    """Admin session on a database with the five demo calls."""
    with build_client(make_settings(tmp_path, SEED_DEMO_DATA=True), MockTranscriber()) as c:
        login(c)
        yield c
