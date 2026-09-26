# CallInsights

AI sales call monitoring: automatic scoring, coaching insights, and analytics for every conversation. (Beta)

## Quickstart

```bash
cp .env.example .env
docker compose up --build
# open http://localhost:8003
# login: admin@clusterx.local / ChangeMe123!
```

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
