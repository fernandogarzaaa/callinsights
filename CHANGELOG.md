# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-10-13

### Fixed

- Accessibility: every form control now has an accessible name. "New call" field labels
  are associated with their inputs (`for`/`id`); the login, dashboard filters, transcript
  editor and settings forms get `aria-label`s; form status messages are announced
  (`role="status"`, `aria-live`) and login errors use `role="alert"`.
- Login card no longer overflows on phones narrower than 384px.

## [1.0.0] - 2026-10-11

First production release. The beta app was restructured into an installable, typed
`callinsights` package.

### Added

- **Speech-to-text**: audio-only uploads are transcribed in the background with local
  faster-whisper (`pip install 'callinsights[asr]'`), any OpenAI-compatible
  `/audio/transcriptions` endpoint (OpenAI, Groq, self-hosted), or a deterministic mock.
  Calls move through `queued -> transcribing -> scoring -> scored | failed`.
- **Speaker labelling** for ASR output: LLM-based when configured, otherwise a pause-based
  heuristic; transcripts can be edited in the UI/API and re-scored.
- **AI coaching insights**: summary, coaching tips, strengths, deal risks and next best
  action from any OpenAI-compatible chat model (OpenAI, Groq, OpenRouter, Ollama, vLLM...),
  with retries/backoff, tolerant JSON parsing and automatic fallback to a deterministic
  local engine. Scores always stay deterministic.
- **API**: pagination and filters (agent, outcome, status, full-text `q`, score range,
  date window), `PATCH`/`DELETE` calls, `rescore`, `transcribe`, `analysis`, CSV export,
  `/api/analytics` (totals, conversion, outcomes, leaderboard, daily trend), stateless
  `/api/score`, agents list.
- **Accounts**: user management API and UI (admin), member/admin roles, deactivation,
  last-admin protection, password change, personal API keys (hashed, revocable).
- **Security**: CSRF protection for cookie sessions, login rate limiting, security headers
  and CSP, secure cookies + HSTS in production, production config guards, upload size
  limits, path-confined audio storage, CSV formula-injection protection.
- **Operations**: Alembic migrations (beta databases are detected and upgraded in place),
  PostgreSQL support via psycopg 3, structured JSON logging with request ids, `/healthz`,
  `/readyz` and `/api/health`, `callinsights` CLI (`serve`, `migrate`, `seed`,
  `create-user`, `score`, `transcribe`, `healthcheck`).
- Dashboard: search, outcome/date filters, pagination, agent leaderboard, score trend chart,
  processing status; call page: strengths, risks, next best action, transcript editor,
  re-score/re-transcribe/delete actions.
- Multi-stage, non-root Docker image with hash-pinned dependencies and a healthcheck;
  Compose stack with PostgreSQL 16.
- CI: lint (ruff), types (mypy --strict), tests on Python 3.11-3.13, PostgreSQL job,
  Docker smoke test, testteeth mutation testing; tag-triggered release workflow.

### Changed

- Package renamed from `app` to `callinsights`; run with `callinsights serve` or
  `uvicorn --factory callinsights.main:create_app`.
- Lexicon matching uses whole words/phrases ("may" no longer matches "maybe"), which can
  shift scores of existing calls slightly when re-scored.
- UI assets (Tailwind build, Alpine.js, Chart.js, wavesurfer.js) are vendored; no CDN.
- JWT library switched from python-jose to PyJWT; Postgres driver to psycopg 3.
- The bootstrap admin is only created when `ADMIN_EMAIL` and `ADMIN_PASSWORD` are set
  (no default credentials).

### Removed

- Unused dependencies (numpy, pypdf, reportlab, apscheduler, python-jose, psycopg2).
- Hashed-embedding provider code that no feature used.

## [0.1.0-beta]

- Initial beta: dashboard, deterministic scoring, seed data, Docker deploy, tests, CI.
