"""Shared fixtures. Every test runs offline: local LLM provider, mock ASR, temp SQLite.

Kept import-light so unit tests (and mutation testing) start fast; HTTP fixtures live in
tests/integration/conftest.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from callinsights.config import Settings

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
