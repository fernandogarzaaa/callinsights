# CallInsights

AI sales call monitoring: automatic scoring, coaching insights, and analytics for every conversation. (Beta)

## Quickstart

```bash
cp .env.example .env
docker compose up --build
# open http://localhost:8003
# login: admin@clusterx.local / ChangeMe123!
```

## Live demo

**One-click cloud demo (no local setup):** open this repo in GitHub Codespaces
(`Code` -> `Codespaces` -> `Create codespace on main`). The dev container boots
the app + Postgres via docker compose, seeds demo data on first start, and
forwards the app port automatically. Log in with `admin@clusterx.local` /
`ChangeMe123!`.

For a 24/7 public demo, deploy `docker-compose.yml` to any host that runs
Docker (a VPS, Railway, Render, or Fly.io) and point your domain at the app
port.

Local dev (SQLite):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m app.seed
uvicorn app.main:app --port 8003
```

## Tests

```bash
python -m pytest -q
ruff check app
```

## License

MIT
