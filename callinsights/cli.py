"""``callinsights`` command-line interface."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from callinsights import __version__

if TYPE_CHECKING:
    from callinsights.config import Settings


def _settings() -> Settings:
    from callinsights.config import get_settings

    return get_settings()


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    s = _settings()
    uvicorn.run(
        "callinsights.main:create_app",
        factory=True,
        host=args.host or s.HOST,
        port=args.port or s.PORT,
        workers=args.workers or s.WORKERS,
        proxy_headers=True,
        forwarded_allow_ips=args.forwarded_allow_ips,
        log_config=None,
        access_log=False,
    )
    return 0


def cmd_migrate(_args: argparse.Namespace) -> int:
    from callinsights.db import make_engine, run_migrations
    from callinsights.logging_setup import configure_logging

    s = _settings()
    configure_logging(s.LOG_LEVEL, s.LOG_FORMAT)
    engine = make_engine(s.DATABASE_URL)
    try:
        run_migrations(engine)
    finally:
        engine.dispose()
    print("database migrated to head")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    from callinsights.ai.llm import build_llm_provider
    from callinsights.db import make_engine, make_session_factory, run_migrations
    from callinsights.seed import bootstrap

    s = _settings()
    if args.demo:
        s = s.model_copy(update={"SEED_DEMO_DATA": True})
    engine = make_engine(s.DATABASE_URL)
    try:
        run_migrations(engine)
        counts = bootstrap(make_session_factory(engine), s, build_llm_provider(s))
    finally:
        engine.dispose()
    print(json.dumps(counts))
    return 0


def _read_password(from_stdin: bool) -> str | None:
    if from_stdin:
        return sys.stdin.readline().rstrip("\n")
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Repeat password: "):
        print("error: passwords do not match", file=sys.stderr)
        return None
    return password


def cmd_create_user(args: argparse.Namespace) -> int:
    from callinsights.db import make_engine, make_session_factory, run_migrations, session_scope
    from callinsights.services import users as user_service

    password = _read_password(args.password_stdin)
    if password is None:
        return 1
    s = _settings()
    engine = make_engine(s.DATABASE_URL)
    try:
        run_migrations(engine)
        with session_scope(make_session_factory(engine)) as db:
            user = user_service.create_user(
                db,
                args.email,
                password,
                name=args.name,
                role="admin" if args.admin else "member",
                rounds=s.BCRYPT_ROUNDS,
            )
            print(f"created {user.role} {user.email} (id={user.id})")
    except user_service.UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from callinsights import scoring
    from callinsights.ai.llm import LLMProvider, LocalProvider, build_llm_provider
    from callinsights.services.calls import CallError, analyze

    path = Path(args.file)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    llm: LLMProvider = LocalProvider() if args.local else build_llm_provider(_settings())
    try:
        result = analyze(scoring.parse_upload(path.name, raw), llm)
    except CallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({**result.analytics, "insights": result.insights.model_dump()}, indent=2))
        return 0
    a, ins = result.analytics, result.insights
    print(f"Score: {a['score']}/100 ({a['outcome']})")
    print(f"Talk share: {a['talk_ratio_agent_pct']}%  Questions: {a['questions_asked']}")
    print(f"\nSummary: {ins.summary}\n")
    for tip in ins.coaching:
        print(f"- {tip}")
    if ins.next_best_action:
        print(f"\nNext best action: {ins.next_best_action}")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    from callinsights.ai.asr import TranscriptionError, build_transcriber
    from callinsights.ai.diarize import segments_to_transcript
    from callinsights.ai.llm import build_llm_provider

    s = _settings()
    transcriber = build_transcriber(s)
    if transcriber is None:
        print("error: set ASR_PROVIDER (mock, faster-whisper or openai)", file=sys.stderr)
        return 1
    try:
        segments = transcriber.transcribe(Path(args.file))
    except TranscriptionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    roles = build_llm_provider(s).label_speakers([seg.text for seg in segments])
    text, _method = segments_to_transcript(
        segments, llm_roles=roles, gap_seconds=s.DIARIZATION_GAP_SECONDS, first=s.ASR_FIRST_SPEAKER
    )
    print(text)
    return 0


def cmd_healthcheck(args: argparse.Namespace) -> int:
    """Exit 0 when the local server reports ready (used by the Docker HEALTHCHECK)."""
    import httpx

    port = args.port or _settings().PORT
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=4)
    except httpx.HTTPError as exc:
        print(f"unhealthy: {exc}", file=sys.stderr)
        return 1
    if resp.status_code != 200:
        print(f"unhealthy: HTTP {resp.status_code}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="callinsights", description="AI sales call monitoring")
    p.add_argument("--version", action="version", version=f"callinsights {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="run the web server")
    sp.add_argument("--host")
    sp.add_argument("--port", type=int)
    sp.add_argument("--workers", type=int)
    sp.add_argument(
        "--forwarded-allow-ips",
        default="127.0.0.1",
        help="proxy IPs trusted for X-Forwarded-* headers (default: 127.0.0.1)",
    )
    sp.set_defaults(func=cmd_serve)

    sub.add_parser("migrate", help="apply database migrations").set_defaults(func=cmd_migrate)

    sp = sub.add_parser("seed", help="create the configured admin (and demo data with --demo)")
    sp.add_argument("--demo", action="store_true", help="also load five demo calls")
    sp.set_defaults(func=cmd_seed)

    sp = sub.add_parser("create-user", help="create a user account")
    sp.add_argument("email")
    sp.add_argument("--name", default="")
    sp.add_argument("--admin", action="store_true", help="grant the admin role")
    sp.add_argument("--password-stdin", action="store_true", help="read the password from stdin")
    sp.set_defaults(func=cmd_create_user)

    sp = sub.add_parser("score", help="score a transcript file offline (nothing is stored)")
    sp.add_argument("file")
    sp.add_argument("--json", action="store_true", help="print the full JSON result")
    sp.add_argument("--local", action="store_true", help="force the local AI provider")
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("transcribe", help="transcribe an audio file with the configured ASR")
    sp.add_argument("file")
    sp.set_defaults(func=cmd_transcribe)

    sp = sub.add_parser("healthcheck", help="probe the local server's /readyz endpoint")
    sp.add_argument("--port", type=int)
    sp.set_defaults(func=cmd_healthcheck)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:  # configuration errors and the like: no traceback for operators
        print(f"error: {exc}", file=sys.stderr)
        return 1
