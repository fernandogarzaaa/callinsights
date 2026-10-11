"""SQLAlchemy ORM models. Schema changes go through Alembic migrations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from callinsights.db import Base

ROLES = ("admin", "member")
CALL_STATUSES = ("queued", "transcribing", "scoring", "scored", "failed")


def utcnow() -> datetime:
    """Naive UTC timestamp (columns are timezone-naive and always hold UTC)."""
    return datetime.now(UTC).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="member")
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    api_keys: Mapped[list[ApiKey]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE", name="fk_api_keys_user_id"), index=True
    )
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User] = relationship(back_populates="api_keys")


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    contact_name: Mapped[str] = mapped_column(String(255), default="")
    contact_phone: Mapped[str] = mapped_column(String(64), default="")
    agent: Mapped[str] = mapped_column(String(255), default="", index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_sec: Mapped[int] = mapped_column(Integer, default=0)
    transcript: Mapped[str] = mapped_column(Text, default="")
    transcript_source: Mapped[str] = mapped_column(String(32), default="manual")
    audio_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    audio_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="scored", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[int] = mapped_column(Integer, default=0)
    outcome: Mapped[str] = mapped_column(String(32), default="", index=True)
    talk_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    questions_asked: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[str] = mapped_column(Text, default="")
    highlights_json: Mapped[str] = mapped_column(Text, default="[]")
    coaching_json: Mapped[str] = mapped_column(Text, default="[]")
    strengths_json: Mapped[str] = mapped_column(Text, default="[]")
    risks_json: Mapped[str] = mapped_column(Text, default="[]")
    next_best_action: Mapped[str] = mapped_column(Text, default="")
    insights_source: Mapped[str] = mapped_column(String(64), default="local")
    created_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL", name="fk_calls_created_by_id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    @staticmethod
    def _load(raw: str | None) -> list[Any]:
        try:
            value = json.loads(raw or "[]")
        except ValueError:
            return []
        return value if isinstance(value, list) else []

    @property
    def highlights(self) -> list[dict[str, Any]]:
        return [h for h in self._load(self.highlights_json) if isinstance(h, dict)]

    @property
    def coaching(self) -> list[str]:
        return [str(x) for x in self._load(self.coaching_json)]

    @property
    def strengths(self) -> list[str]:
        return [str(x) for x in self._load(self.strengths_json)]

    @property
    def risks(self) -> list[str]:
        return [str(x) for x in self._load(self.risks_json)]
