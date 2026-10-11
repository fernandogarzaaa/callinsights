"""Runs the migration + API flow against a real PostgreSQL server.

Skipped unless CALLINSIGHTS_TEST_POSTGRES_URL points at a disposable database (CI sets it
using a postgres service container).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from callinsights.ai.asr import MockTranscriber
from tests.conftest import GOOD_TRANSCRIPT, make_settings
from tests.helpers import build_client, login

pytestmark = pytest.mark.integration

PG_URL = os.getenv("CALLINSIGHTS_TEST_POSTGRES_URL", "")


@pytest.mark.skipif(not PG_URL, reason="CALLINSIGHTS_TEST_POSTGRES_URL not set")
def test_full_flow_on_postgres(tmp_path: Path) -> None:
    engine = create_engine(PG_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()

    with build_client(
        make_settings(tmp_path, DATABASE_URL=PG_URL, SEED_DEMO_DATA=True), MockTranscriber()
    ) as c:
        login(c)
        assert c.get("/readyz").json()["database"] is True
        created = c.post("/api/calls", json={"transcript_text": GOOD_TRANSCRIPT, "agent": "PG"})
        assert created.status_code == 201
        audio = c.post("/api/calls", files={"audio_file": ("a.wav", b"RIFF00", "audio/wav")})
        assert audio.status_code == 202
        assert c.get(f"/api/calls/{audio.json()['id']}").json()["status"] == "scored"
        stats = c.get("/api/analytics").json()
        assert stats["total_calls"] == 7
        assert stats["trend"]
        assert c.get("/api/calls", params={"q": "ticket"}).json()["total"] >= 1
        assert c.get("/").status_code == 200
        key = c.post("/api/keys", json={"name": "pg"}).json()["key"]
        assert c.get("/api/calls", headers={"Authorization": f"Bearer {key}"}).status_code == 200
