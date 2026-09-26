from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.core.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), default="", nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), default="member", nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)


# --- Product models are appended below by each repo's builder ---


class Call(Base):
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True)
    contact_name = Column(String(255), default="", nullable=False)
    contact_phone = Column(String(64), default="", nullable=False)
    agent = Column(String(255), default="", nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    duration_sec = Column(Integer, default=0, nullable=False)
    transcript = Column(Text, nullable=False)
    audio_path = Column(String(512), nullable=True)
    audio_filename = Column(String(255), nullable=True)
    score = Column(Integer, default=0, nullable=False)
    outcome = Column(String(32), default="", nullable=False, index=True)
    talk_ratio = Column(Float, default=0.0, nullable=False)
    questions_asked = Column(Integer, default=0, nullable=False)
    summary = Column(Text, default="", nullable=False)
    highlights_json = Column(Text, default="[]", nullable=False)
    coaching_json = Column(Text, default="[]", nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)
