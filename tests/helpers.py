"""HTTP test helpers (imported by integration tests)."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from callinsights.ai.asr import Transcriber
from callinsights.ai.llm import LocalProvider
from callinsights.config import Settings
from callinsights.main import create_app
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD


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
