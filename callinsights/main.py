"""ASGI application factory.

Run with ``callinsights serve`` or ``uvicorn --factory callinsights.main:create_app``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from callinsights import __version__
from callinsights.ai.asr import Transcriber, build_transcriber
from callinsights.ai.llm import LLMProvider, build_llm_provider
from callinsights.config import Settings, get_settings
from callinsights.db import check_database, make_engine, make_session_factory, run_migrations
from callinsights.logging_setup import configure_logging, request_id_var
from callinsights.ratelimit import SlidingWindowLimiter
from callinsights.seed import bootstrap

logger = logging.getLogger(__name__)
PACKAGE_DIR = Path(__file__).parent

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    ),
}
# FastAPI's interactive docs load their assets from a CDN.
DOCS_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'none'"
)
DOCS_PATHS = frozenset({"/docs", "/redoc"})


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMProvider | None = None,
    transcriber: Transcriber | None = None,
    use_default_transcriber: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
    engine = make_engine(settings.DATABASE_URL)
    session_factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.AUTO_MIGRATE:
            run_migrations(engine)
        bootstrap(session_factory, settings, app.state.llm)
        logger.info(
            "callinsights started",
            extra={
                "version": __version__,
                "env": settings.ENV,
                "llm": app.state.llm.name,
                "asr": app.state.transcriber.name if app.state.transcriber else "none",
            },
        )
        yield
        engine.dispose()

    app = FastAPI(
        title=settings.APP_NAME,
        version=__version__,
        description="AI sales call monitoring: scoring, coaching insights and analytics.",
        lifespan=lifespan,
    )
    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    templates.env.globals.update(app_name=settings.APP_NAME, version=__version__)

    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.templates = templates
    app.state.llm = llm or build_llm_provider(settings)
    if transcriber is None and use_default_transcriber:
        transcriber = build_transcriber(settings)
    app.state.transcriber = transcriber
    app.state.login_limiter = SlidingWindowLimiter(
        settings.LOGIN_MAX_ATTEMPTS, settings.LOGIN_WINDOW_SECONDS
    )

    max_body = settings.max_upload_bytes + 10 * 1024 * 1024  # files + form overhead

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > max_body:
                response: Response = JSONResponse(
                    {"detail": f"Request body exceeds {settings.MAX_UPLOAD_MB} MB limit"},
                    status_code=413,
                )
            else:
                try:
                    response = await call_next(request)
                except Exception:
                    logger.exception("unhandled error", extra={"path": request.url.path})
                    response = JSONResponse(
                        {"detail": "Internal server error", "request_id": rid}, status_code=500
                    )
            response.headers["X-Request-ID"] = rid
            for name, value in SECURITY_HEADERS.items():
                response.headers.setdefault(name, value)
            if request.url.path in DOCS_PATHS:
                response.headers["Content-Security-Policy"] = DOCS_CSP
            if settings.cookie_secure:
                response.headers.setdefault(
                    "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
                )
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
            return response
        finally:
            request_id_var.reset(token)

    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")

    from callinsights.web import admin, auth, calls, pages

    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(calls.router)
    app.include_router(pages.router)

    @app.get("/healthz", tags=["health"])
    def liveness() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["health"])
    def readiness() -> Response:
        """Readiness: the database answers."""
        ok = check_database(engine)
        return JSONResponse(
            {"status": "ok" if ok else "unavailable", "database": ok},
            status_code=200 if ok else 503,
        )

    @app.get("/api/health", tags=["health"])
    def health() -> Response:
        ok = check_database(engine)
        return JSONResponse(
            {
                "status": "ok" if ok else "degraded",
                "app": settings.APP_NAME,
                "version": __version__,
                "database": ok,
                "llm": app.state.llm.name,
                "asr": app.state.transcriber.name if app.state.transcriber else "none",
            },
            status_code=200 if ok else 503,
        )

    return app
