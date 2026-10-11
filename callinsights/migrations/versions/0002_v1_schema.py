"""v1.0 schema: API keys, user activation, call processing status and AI insights.

Revision ID: 0002_v1_schema
Revises: 0001_beta_schema
Create Date: 2026-10-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_v1_schema"
down_revision = "0001_beta_schema"
branch_labels = None
depends_on = None

NEW_CALL_COLUMNS = (
    "updated_at",
    "created_by_id",
    "insights_source",
    "next_best_action",
    "risks_json",
    "strengths_json",
    "error",
    "status",
    "transcript_source",
)


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true())
        )

    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE", name="fk_api_keys_user_id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("prefix", sa.String(16), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
    op.create_index("ix_api_keys_prefix", "api_keys", ["prefix"])

    with op.batch_alter_table("calls") as batch:
        batch.add_column(
            sa.Column("transcript_source", sa.String(32), nullable=False, server_default="manual")
        )
        batch.add_column(
            sa.Column("status", sa.String(16), nullable=False, server_default="scored")
        )
        batch.add_column(sa.Column("error", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("strengths_json", sa.Text(), nullable=False, server_default="[]")
        )
        batch.add_column(sa.Column("risks_json", sa.Text(), nullable=False, server_default="[]"))
        batch.add_column(
            sa.Column("next_best_action", sa.Text(), nullable=False, server_default="")
        )
        batch.add_column(
            sa.Column("insights_source", sa.String(64), nullable=False, server_default="local")
        )
        batch.add_column(sa.Column("created_by_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("updated_at", sa.DateTime(), nullable=True))
        batch.create_foreign_key(
            "fk_calls_created_by_id", "users", ["created_by_id"], ["id"], ondelete="SET NULL"
        )
        batch.create_index("ix_calls_status", ["status"])
        batch.create_index("ix_calls_created_at", ["created_at"])

    op.execute("UPDATE calls SET updated_at = created_at WHERE updated_at IS NULL")
    with op.batch_alter_table("calls") as batch:
        batch.alter_column("updated_at", existing_type=sa.DateTime(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("calls") as batch:
        batch.drop_index("ix_calls_created_at")
        batch.drop_index("ix_calls_status")
        batch.drop_constraint("fk_calls_created_by_id", type_="foreignkey")
        for col in NEW_CALL_COLUMNS:
            batch.drop_column(col)
    op.drop_index("ix_api_keys_prefix", table_name="api_keys")
    op.drop_index("ix_api_keys_user_id", table_name="api_keys")
    op.drop_table("api_keys")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("is_active")
