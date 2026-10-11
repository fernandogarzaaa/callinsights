"""Pydantic request/response models for the JSON API."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    if len(email) > 255 or not EMAIL_RE.match(email):
        raise ValueError("invalid email address")
    return email


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LoginIn(_Strict):
    email: str = Field(max_length=255)
    password: str = Field(min_length=1, max_length=256)


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_in: int


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    role: str
    is_active: bool
    created_at: datetime


class UserCreate(_Strict):
    email: str
    password: str = Field(max_length=256)
    name: str = Field(default="", max_length=255)
    role: Literal["admin", "member"] = "member"

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        return normalize_email(value)


class UserUpdate(_Strict):
    name: str | None = Field(default=None, max_length=255)
    role: Literal["admin", "member"] | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, max_length=256)


class PasswordChange(_Strict):
    current_password: str = Field(max_length=256)
    new_password: str = Field(max_length=256)


class ApiKeyCreate(_Strict):
    name: str = Field(min_length=1, max_length=100)


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreated(ApiKeyOut):
    key: str = Field(description="Plaintext key. Shown once; store it securely.")


class CallCreate(_Strict):
    contact_name: str = Field(default="", max_length=255)
    contact_phone: str = Field(default="", max_length=64)
    agent: str = Field(default="", max_length=255)
    started_at: datetime | None = None
    duration_sec: int = Field(default=0, ge=0, le=60 * 60 * 24)
    transcript_text: str = ""


class CallUpdate(_Strict):
    contact_name: str | None = Field(default=None, max_length=255)
    contact_phone: str | None = Field(default=None, max_length=64)
    agent: str | None = Field(default=None, max_length=255)
    started_at: datetime | None = None
    duration_sec: int | None = Field(default=None, ge=0, le=60 * 60 * 24)
    transcript_text: str | None = None


class ScoreIn(_Strict):
    transcript_text: str = Field(min_length=1)


class CallSummaryOut(BaseModel):
    id: int
    contact_name: str
    contact_phone: str
    agent: str
    status: str
    score: int
    outcome: str
    talk_ratio: float
    questions_asked: int
    duration_sec: int
    has_audio: bool
    started_at: datetime | None
    created_at: datetime


class CallDetailOut(CallSummaryOut):
    transcript: str
    transcript_source: str
    summary: str
    highlights: list[dict[str, Any]]
    coaching: list[str]
    strengths: list[str]
    risks: list[str]
    next_best_action: str
    insights_source: str
    error: str | None
    url: str


class CallList(BaseModel):
    total: int
    limit: int
    offset: int
    calls: list[CallSummaryOut]
