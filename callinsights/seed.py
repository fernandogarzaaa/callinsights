"""Startup bootstrap: create the configured admin user and (optionally) demo data."""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from callinsights import models
from callinsights.ai.llm import LLMProvider
from callinsights.config import Settings
from callinsights.db import session_scope
from callinsights.services import users as user_service

logger = logging.getLogger(__name__)


def bootstrap(
    factory: sessionmaker[Session], settings: Settings, llm: LLMProvider
) -> dict[str, int]:
    counts = {"users": 0, "calls": 0}
    with session_scope(factory) as db:
        password = settings.ADMIN_PASSWORD.get_secret_value() if settings.ADMIN_PASSWORD else None
        if user_service.ensure_admin(db, settings.ADMIN_EMAIL, password, settings.BCRYPT_ROUNDS):
            counts["users"] = 1
            logger.info("bootstrap admin created", extra={"email": settings.ADMIN_EMAIL})
        if settings.SEED_DEMO_DATA:
            from callinsights.demo_data import seed_demo_calls

            counts["calls"] = seed_demo_calls(db, llm)
        if not db.scalar(select(func.count()).select_from(models.User)):
            logger.warning(
                "no users exist; set ADMIN_EMAIL/ADMIN_PASSWORD or run "
                "'callinsights create-user --admin EMAIL'"
            )
    return counts
