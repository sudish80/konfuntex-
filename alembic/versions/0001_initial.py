"""Initial schema — all ORM tables.

Revision ID: 0001
Revises: None
Create Date: 2026-06-13 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=True, server_default="pending"),
        sa.Column("method", sa.String(), nullable=True),
        sa.Column("base_model", sa.String(), nullable=True),
        sa.Column("dataset", sa.String(), nullable=True),
        sa.Column("runtime", sa.String(), nullable=True),
        sa.Column("metrics", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("colab_notebook_id", sa.String(), nullable=True),
        sa.Column("colab_notebook_url", sa.String(), nullable=True),
        sa.Column("conversation_id", sa.String(), nullable=True),
        sa.Column("model_version_id", sa.String(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=True, server_default="active"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("messages_json", sa.Text(), nullable=True, server_default="[]"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=True),
        sa.Column("base_model", sa.String(), nullable=False),
        sa.Column("finetuned_path", sa.String(), nullable=True),
        sa.Column("hf_repo_id", sa.String(), nullable=True),
        sa.Column("method", sa.String(), nullable=True),
        sa.Column("metrics", sa.Text(), nullable=True),
        sa.Column("runtime_used", sa.String(), nullable=True),
        sa.Column("training_steps", sa.Integer(), nullable=True),
        sa.Column("final_loss", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("tags", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"],),
    )
    op.create_table(
        "runtime_logs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=True),
        sa.Column("runtime_type", sa.String(), nullable=False),
        sa.Column("ram_gb", sa.Float(), nullable=True),
        sa.Column("vram_gb", sa.Float(), nullable=True),
        sa.Column("gpu_name", sa.String(), nullable=True),
        sa.Column("switched_from", sa.String(), nullable=True),
        sa.Column("switched_to", sa.String(), nullable=True),
        sa.Column("switch_reason", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=True, server_default="active"),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"],),
    )
    op.create_table(
        "metrics",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("epoch", sa.Float(), nullable=True),
        sa.Column("global_step", sa.Integer(), nullable=True),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("gpu_mem_gb", sa.Float(), nullable=True),
        sa.Column("tokens_per_second", sa.Float(), nullable=True),
        sa.Column("learning_rate", sa.Float(), nullable=True),
        sa.Column("grad_norm", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("extras_json", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"],),
    )


def downgrade() -> None:
    op.drop_table("metrics")
    op.drop_table("runtime_logs")
    op.drop_table("model_versions")
    op.drop_table("conversations")
    op.drop_table("jobs")
