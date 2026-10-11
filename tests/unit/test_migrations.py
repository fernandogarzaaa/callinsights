"""Alembic migrations: fresh install, beta upgrade in place, model/migration parity."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from callinsights import models
from callinsights.db import Base, alembic_config, check_database, make_engine, run_migrations


def test_fresh_database_migrates_to_head(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'nested' / 'fresh.db'}")
    run_migrations(engine)
    run_migrations(engine)  # idempotent
    tables = set(inspect(engine).get_table_names())
    assert {"users", "calls", "api_keys", "alembic_version"} <= tables
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == (
            "0002_v1_schema"
        )
    assert check_database(engine) is True
    engine.dispose()


def test_models_match_migrations(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'parity.db'}")
    run_migrations(engine)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert models.Call.__table__ is not None
    assert diff == [], diff
    engine.dispose()


def test_beta_database_is_upgraded_in_place(tmp_path: Path) -> None:
    path = tmp_path / "beta.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(255) NOT NULL,
            name VARCHAR(255) NOT NULL, hashed_password VARCHAR(255) NOT NULL,
            role VARCHAR(32) NOT NULL, created_at DATETIME NOT NULL);
        CREATE UNIQUE INDEX ix_users_email ON users (email);
        CREATE TABLE calls (id INTEGER PRIMARY KEY, contact_name VARCHAR(255) NOT NULL,
            contact_phone VARCHAR(64) NOT NULL, agent VARCHAR(255) NOT NULL,
            started_at DATETIME, duration_sec INTEGER NOT NULL, transcript TEXT NOT NULL,
            audio_path VARCHAR(512), audio_filename VARCHAR(255), score INTEGER NOT NULL,
            outcome VARCHAR(32) NOT NULL, talk_ratio FLOAT NOT NULL,
            questions_asked INTEGER NOT NULL, summary TEXT NOT NULL,
            highlights_json TEXT NOT NULL, coaching_json TEXT NOT NULL,
            created_at DATETIME NOT NULL);
        CREATE INDEX ix_calls_agent ON calls (agent);
        CREATE INDEX ix_calls_outcome ON calls (outcome);
        INSERT INTO users VALUES (1, 'old@x.io', 'Old', 'hash', 'admin', '2026-01-01 00:00:00');
        INSERT INTO calls VALUES (1, 'Dana', '', 'Maya', NULL, 60, 'Agent: hi', NULL, NULL,
            80, 'Interested', 0.5, 3, 'summary', '[]', '["tip"]', '2026-01-02 00:00:00');
        """
    )
    con.commit()
    con.close()

    engine = make_engine(f"sqlite:///{path}")
    run_migrations(engine)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT status, transcript_source, insights_source, strengths_json, "
                "updated_at, coaching_json FROM calls WHERE id = 1"
            )
        ).one()
        active = conn.execute(text("SELECT is_active FROM users WHERE id = 1")).scalar()
    assert row[0] == "scored"
    assert row[1] == "manual"
    assert row[2] == "local"
    assert row[3] == "[]"
    assert str(row[4]).startswith("2026-01-02")
    assert row[5] == '["tip"]'
    assert active == 1
    engine.dispose()


def test_downgrade_roundtrip(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'down.db'}"
    engine = make_engine(url)
    run_migrations(engine)
    cfg = alembic_config(url)
    command.downgrade(cfg, "0001_beta_schema")
    cols = {c["name"] for c in inspect(engine).get_columns("calls")}
    assert "status" not in cols
    assert "api_keys" not in inspect(engine).get_table_names()
    command.upgrade(cfg, "head")
    assert "status" in {c["name"] for c in inspect(engine).get_columns("calls")}
    engine.dispose()


def test_check_database_failure(tmp_path: Path) -> None:
    bad = tmp_path / "dir.db"
    bad.mkdir()  # a directory cannot be opened as a database
    engine = make_engine(f"sqlite:///{bad}")
    assert check_database(engine) is False
    engine.dispose()


def test_utcnow_is_naive() -> None:
    now = models.utcnow()
    assert now.tzinfo is None
    assert abs((datetime.now(UTC).replace(tzinfo=None) - now).total_seconds()) < 5
