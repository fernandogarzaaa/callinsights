# CallInsights

AI sales call monitoring: automatic scoring, coaching insights, and analytics for every conversation.

[![ci](https://github.com/fernandogarzaaa/callinsights/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/callinsights/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/fernandogarzaaa/callinsights)](https://github.com/fernandogarzaaa/callinsights/releases)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

CallInsights takes a sales call (a transcript, a caption file, or just the recording), scores
it 0-100 with a transparent deterministic engine, and writes a coaching narrative: summary,
strengths, deal risks, coaching tips and the next best action. A dashboard tracks scores,
outcomes, conversion and an agent leaderboard over time.

- **Deterministic, explainable scoring**: five 20-point components (discovery questions,
  talk balance, budget/timeline/authority qualification, objection handling, prospect
  sentiment). Same transcript, same score, every time.
- **Bring your own AI, or none**: any OpenAI-compatible LLM (OpenAI, Groq, OpenRouter,
  Ollama, vLLM, LM Studio...) writes the narrative. Without one, a local offline engine does.
  Provider failures fall back to local results; scoring never blocks on an API.
- **Speech-to-text**: upload audio only and it is transcribed in the background with local
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (free) or any
  OpenAI-compatible `/audio/transcriptions` endpoint (Groq's Whisper is free-tier). Speakers
  are labelled by the LLM when available, or by a pause-based heuristic, and can be corrected
  in the UI before re-scoring.
- **Production essentials**: env-based config with production guards, Alembic migrations
  (SQLite or PostgreSQL), structured JSON logs with request ids, JWT sessions + API keys,
  CSRF protection, login rate limiting, security headers, health/readiness probes,
  non-root Docker image with healthcheck, a `callinsights` CLI.

## Quickstart

### Docker Compose (PostgreSQL)

```bash
cp .env.example .env
# edit .env: SECRET_KEY, ADMIN_PASSWORD, POSTGRES_PASSWORD (and ENV=production for real use)
docker compose up -d --build
open http://localhost:8003      # sign in with ADMIN_EMAIL / ADMIN_PASSWORD
```

### Local (SQLite)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # add ,asr for local Whisper
export ADMIN_EMAIL=admin@example.com ADMIN_PASSWORD='Choose-a-strong-1' SEED_DEMO_DATA=1
callinsights serve               # migrates, creates the admin, loads 5 demo calls
```

### Score a transcript from the command line (nothing stored)

```bash
callinsights score call.txt            # human-readable
callinsights score call.vtt --json     # full analytics + insights
```

Transcripts use one turn per line: `Agent: ...` / `Prospect: ...` (aliases such as `Rep`,
`Sales`, `Customer`, `Client` are recognised). `.txt`, `.md`, `.vtt` and `.srt` uploads work.

## Configuration

All configuration is via environment variables (or a `.env` file). `.env.example` documents
every option; the important ones:

| Variable | Default | Notes |
| --- | --- | --- |
| `ENV` | `development` | `production` refuses weak `SECRET_KEY`/`ADMIN_PASSWORD` and enables secure cookies + HSTS |
| `SECRET_KEY` | dev placeholder | **Required in production**, 32+ random chars |
| `DATABASE_URL` | `sqlite:///./data/callinsights.db` | `postgresql+psycopg://...` (or `postgres://...`) for PostgreSQL |
| `DATA_DIR` | `./data` | Audio uploads go to `$DATA_DIR/uploads` |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | unset | Bootstrap admin created on first start |
| `SEED_DEMO_DATA` | `0` | Load five demo calls |
| `LLM_PROVIDER` | `auto` | `auto` (use LLM if `LLM_BASE_URL` set), `local`, `openai` |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | empty / empty / `gpt-4o-mini` | Any OpenAI-compatible endpoint |
| `ASR_PROVIDER` | `none` | `none`, `mock`, `faster-whisper`, `openai` |
| `ASR_BASE_URL` / `ASR_API_KEY` / `ASR_MODEL` | fall back to LLM values / `whisper-1` | For `ASR_PROVIDER=openai` |
| `WHISPER_MODEL_SIZE` | `base` | For `faster-whisper` (`tiny` ... `large-v3`) |
| `MAX_UPLOAD_MB` | `200` | Request/body limit for uploads |
| `LOG_FORMAT` / `LOG_LEVEL` | `json` / `INFO` | `text` for humans |

Free provider recipes:

```bash
# Groq (free tier): LLM + Whisper
LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_API_KEY=gsk_... LLM_MODEL=llama-3.1-8b-instant
ASR_PROVIDER=openai ASR_MODEL=whisper-large-v3
# Fully local: Ollama + faster-whisper
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1 ASR_PROVIDER=faster-whisper
```

## API

Interactive docs are served at `/docs` (OpenAPI at `/openapi.json`). Authenticate with
`POST /api/auth/login` (returns a bearer token and sets an HttpOnly cookie) or with a
personal API key created in **Settings** / `POST /api/keys` (`Authorization: Bearer ci_...`).

| Method & path | Purpose |
| --- | --- |
| `POST /api/calls` | Create a call from JSON (`transcript_text`) or multipart (`transcript_text`/`transcript_file`/`audio_file`). Transcripts are scored immediately (201); audio-only calls return 202 and are transcribed in the background |
| `GET /api/calls` | List with `agent`, `outcome`, `status`, `q`, `min_score`, `max_score`, `days`, `limit`, `offset` |
| `GET /api/calls/{id}` / `PATCH` / `DELETE` | Read, edit (sending `transcript_text` re-scores), delete |
| `POST /api/calls/{id}/rescore` / `/transcribe` | Re-run scoring / speech-to-text |
| `GET /api/calls/{id}/analysis` / `/audio` | Full deterministic breakdown / audio stream |
| `GET /api/calls/export.csv` | CSV export (formula-injection safe) |
| `GET /api/analytics` | Totals, conversion, outcomes, leaderboard, daily trend |
| `POST /api/score` | Stateless scoring of a transcript |
| `GET/POST/PATCH/DELETE /api/users` | User management (admin) |
| `GET/POST/DELETE /api/keys` | Personal API keys |
| `GET /healthz`, `/readyz`, `/api/health` | Liveness, readiness (DB), status |

```bash
TOKEN=$(curl -s -X POST localhost:8003/api/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"..."}' | jq -r .access_token)
curl -s -X POST localhost:8003/api/calls -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"agent":"Maya","contact_name":"Dana","transcript_text":"Agent: What is your budget?\nProspect: About 2k a month."}'
curl -s -X POST localhost:8003/api/calls -H "Authorization: Bearer $TOKEN" \
  -F agent=Maya -F audio_file=@call.mp3          # needs ASR_PROVIDER
```

## Architecture

```
callinsights/
  main.py           app factory: middleware (request ids, security headers, body limit), routers, health
  config.py         pydantic-settings configuration + production guards
  scoring.py        pure, deterministic scoring engine (no I/O)
  ai/llm.py         LocalProvider + OpenAICompatibleProvider (retries, JSON parsing, local fallback)
  ai/asr.py         mock / faster-whisper / OpenAI-compatible transcription
  ai/diarize.py     speaker labelling (LLM or pause heuristic) -> "Agent:/Prospect:" transcript
  services/         call pipeline, queries & analytics, users & API keys
  web/              FastAPI routers: auth, admin, calls API, HTML pages; auth/CSRF dependencies
  migrations/       Alembic (0001 = beta schema, 0002 = v1.0); beta databases upgrade in place
  templates/ static/  server-rendered UI; Tailwind CSS and JS libraries are vendored (no CDN)
```

Request flow for an audio upload: `POST /api/calls` streams the file to `$DATA_DIR/uploads`
under a random name and stores the call as `queued` -> a background task transcribes
(`transcribing`), labels speakers, scores and writes insights (`scoring` -> `scored`, or
`failed` with the error) -> the UI auto-refreshes until done.

Background work runs in-process (FastAPI background tasks). Long transcriptions occupy a
worker thread; size `WORKERS` accordingly, or use the `openai` ASR provider to offload it.

## Deploy

- **Docker**: `docker build -t callinsights .` builds a non-root image (`uid 10001`) that
  installs hash-pinned dependencies from `requirements.lock`, keeps data in the `/data`
  volume and exposes a `HEALTHCHECK` (`callinsights healthcheck`). Build with
  `--build-arg EXTRAS=postgres,asr` to bundle faster-whisper and ffmpeg.
- **Compose**: `docker-compose.yml` runs the app with PostgreSQL 16 and persistent volumes.
- **Any PaaS / VM**: `pip install .[postgres]`, set the env vars, run `callinsights migrate`
  then `callinsights serve` (or `uvicorn --factory callinsights.main:create_app`) behind a
  TLS-terminating reverse proxy. Pass `--forwarded-allow-ips` for your proxy.
- Set `ENV=production`. With several instances, put login rate limiting at the proxy too
  (the built-in limiter is per process) and use a shared volume or object storage for
  `$DATA_DIR`.

CLI reference: `callinsights serve | migrate | seed [--demo] | create-user EMAIL [--admin]
| score FILE [--json] | transcribe AUDIO | healthcheck`.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && python -m pytest
testteeth run --pytest-args "tests/unit -q -p no:cacheprovider -o addopts="   # mutation testing
```

The test-suite is fully offline: the local AI provider, a deterministic mock transcriber and
`httpx.MockTransport` fakes for the OpenAI-compatible providers. Restyling the UI? Rebuild
the stylesheet with the Tailwind v3 standalone CLI (see `tailwind.config.js`).

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
