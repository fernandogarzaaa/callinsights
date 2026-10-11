"""Login/logout (HTML form + JSON API), current user and password change."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from callinsights import models, security
from callinsights.config import Settings
from callinsights.ratelimit import SlidingWindowLimiter
from callinsights.schemas import LoginIn, PasswordChange, TokenOut, UserOut
from callinsights.services import users as user_service
from callinsights.web.deps import COOKIE_NAME, DbDep, OptionalUserDep, SettingsDep, UserDep

logger = logging.getLogger(__name__)
router = APIRouter()


def _client_host(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _limiter(request: Request) -> SlidingWindowLimiter:
    limiter: SlidingWindowLimiter = request.app.state.login_limiter
    return limiter


def _attempt(request: Request, db: DbDep, email: str, password: str) -> models.User | None:
    host = _client_host(request)
    key = f"{host}|{email.strip().lower()}"
    limiter = _limiter(request)
    if limiter.is_blocked(key):
        logger.warning("login rate limited", extra={"client": host})
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many login attempts; try again later",
            headers={"Retry-After": str(limiter.retry_after(key))},
        )
    user = user_service.authenticate(db, email, password)
    if user is None:
        limiter.hit(key)
        logger.info("login failed", extra={"client": host})
        return None
    limiter.reset(key)
    logger.info("login succeeded", extra={"user_id": user.id})
    return user


def _token(settings: Settings, user: models.User) -> str:
    return security.create_access_token(
        user.email, settings.SECRET_KEY.get_secret_value(), settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )


def _set_cookie(settings: Settings, resp: Response, token: str) -> None:
    resp.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_page(request: Request, user: OptionalUserDep) -> Response:
    if user is not None:
        return RedirectResponse("/", status_code=303)
    templates = request.app.state.templates
    return templates.TemplateResponse(  # type: ignore[no-any-return]
        request, "auth/login.html", {"error": request.query_params.get("error")}
    )


@router.post("/login", include_in_schema=False)
def login_form(
    request: Request,
    db: DbDep,
    settings: SettingsDep,
    email: Annotated[str, Form(max_length=255)],
    password: Annotated[str, Form(max_length=256)],
) -> Response:
    try:
        user = _attempt(request, db, email, password)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            return RedirectResponse("/login?error=rate", status_code=303)
        raise  # pragma: no cover - _attempt raises only 429
    if user is None:
        return RedirectResponse("/login?error=1", status_code=303)
    resp = RedirectResponse("/", status_code=303)
    _set_cookie(settings, resp, _token(settings, user))
    return resp


@router.post("/api/auth/login", response_model=TokenOut, tags=["auth"])
def login_api(request: Request, payload: LoginIn, db: DbDep, settings: SettingsDep) -> Response:
    user = _attempt(request, db, payload.email, payload.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    token = _token(settings, user)
    body = TokenOut(access_token=token, expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    resp = JSONResponse(body.model_dump())
    _set_cookie(settings, resp, token)
    return resp


@router.get("/logout", include_in_schema=False)
def logout_page() -> Response:
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


@router.post("/api/auth/logout", status_code=204, tags=["auth"])
def logout_api() -> Response:
    resp = Response(status_code=204)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


@router.get("/api/auth/me", response_model=UserOut, tags=["auth"])
def me(user: UserDep) -> models.User:
    return user


@router.post("/api/auth/password", status_code=204, tags=["auth"])
def change_password(
    payload: PasswordChange, user: UserDep, db: DbDep, settings: SettingsDep
) -> Response:
    if not security.verify_password(payload.current_password, user.hashed_password):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current password is incorrect")
    try:
        user_service.set_password(user, payload.new_password, settings.BCRYPT_ROUNDS)
    except user_service.UserError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    db.flush()
    return Response(status_code=204)
