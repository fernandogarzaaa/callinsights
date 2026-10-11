"""FastAPI dependencies: DB sessions, settings, authentication and CSRF protection."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from callinsights import models, security
from callinsights.ai.asr import Transcriber
from callinsights.ai.llm import LLMProvider
from callinsights.config import Settings
from callinsights.services import users as user_service

COOKIE_NAME = "access_token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_llm(request: Request) -> LLMProvider:
    llm: LLMProvider = request.app.state.llm
    return llm


def get_transcriber(request: Request) -> Transcriber | None:
    transcriber: Transcriber | None = request.app.state.transcriber
    return transcriber


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[Session, Depends(get_db)]
LLMDep = Annotated[LLMProvider, Depends(get_llm)]
TranscriberDep = Annotated[Transcriber | None, Depends(get_transcriber)]


@dataclass(frozen=True)
class Credentials:
    token: str
    via_cookie: bool


def credentials_from(request: Request) -> Credentials | None:
    auth = request.headers.get("Authorization", "")
    if auth[:7].lower() == "bearer " and auth[7:].strip():
        return Credentials(auth[7:].strip(), via_cookie=False)
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        return Credentials(cookie, via_cookie=True)
    return None


def origin_allowed(request: Request, settings: Settings) -> bool:
    """CSRF defence for cookie-authenticated writes: Origin/Referer must be this site."""
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source:
        return False
    parts = urlsplit(source)
    if not parts.netloc:
        return False
    if f"{parts.scheme}://{parts.netloc}" in settings.trusted_origins:
        return True
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return parts.netloc == host


def resolve_user(request: Request, db: Session, settings: Settings) -> models.User | None:
    creds = credentials_from(request)
    if creds is None:
        return None
    if security.is_api_key(creds.token):
        return user_service.user_for_api_key(db, creds.token)
    email = security.decode_access_token(creds.token, settings.SECRET_KEY.get_secret_value())
    if not email:
        return None
    user = db.scalar(select(models.User).where(models.User.email == email))
    if user is None or not user.is_active:
        return None
    if (
        creds.via_cookie
        and request.method not in SAFE_METHODS
        and not origin_allowed(request, settings)
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Cross-site request rejected (CSRF)")
    return user


def optional_user(request: Request, db: DbDep, settings: SettingsDep) -> models.User | None:
    return resolve_user(request, db, settings)


def current_user(request: Request, db: DbDep, settings: SettingsDep) -> models.User:
    user = resolve_user(request, db, settings)
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def admin_user(user: Annotated[models.User, Depends(current_user)]) -> models.User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return user


UserDep = Annotated[models.User, Depends(current_user)]
AdminDep = Annotated[models.User, Depends(admin_user)]
OptionalUserDep = Annotated[models.User | None, Depends(optional_user)]
