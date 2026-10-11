"""Fixtures for HTTP-level tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from callinsights.ai.asr import MockTranscriber
from callinsights.config import Settings
from tests.conftest import make_settings
from tests.helpers import build_client, login


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
