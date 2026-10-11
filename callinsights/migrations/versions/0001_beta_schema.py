"""Beta schema (pre-1.0, previously created with metadata.create_all).

Revision ID: 0001_beta_schema
Revises:
Create Date: 2026-10-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_beta_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table(
        "calls",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("contact_name", sa.String(255), nullable=False),
        sa.Column("contact_phone", sa.String(64), nullable=False),
        sa.Column("agent", sa.String(255), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("duration_sec", sa.Integer(), nullable=False),
        sa.Column("transcript", sa.Text(), nullable=False),
        sa.Column("audio_path", sa.String(512), nullable=True),
        sa.Column("audio_filename", sa.String(255), nullable=True),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("talk_ratio", sa.Float(), nullable=False),
        sa.Column("questions_asked", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("highlights_json", sa.Text(), nullable=False),
        sa.Column("coaching_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_calls_agent", "calls", ["agent"])
    op.create_index("ix_calls_outcome", "calls", ["outcome"])


def downgrade() -> None:
    op.drop_index("ix_calls_outcome", table_name="calls")
    op.drop_index("ix_calls_agent", table_name="calls")
    op.drop_table("calls")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
