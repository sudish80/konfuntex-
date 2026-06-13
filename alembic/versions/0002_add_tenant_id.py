"""Add tenant_id columns for multi-tenant isolation.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-13 12:45:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("tenant_id", sa.String(), nullable=True, index=True))
    op.add_column("conversations", sa.Column("tenant_id", sa.String(), nullable=True, index=True))
    op.add_column("model_versions", sa.Column("tenant_id", sa.String(), nullable=True, index=True))
    op.add_column("runtime_logs", sa.Column("tenant_id", sa.String(), nullable=True, index=True))
    op.add_column("metrics", sa.Column("tenant_id", sa.String(), nullable=True, index=True))


def downgrade() -> None:
    op.drop_column("metrics", "tenant_id")
    op.drop_column("runtime_logs", "tenant_id")
    op.drop_column("model_versions", "tenant_id")
    op.drop_column("conversations", "tenant_id")
    op.drop_column("jobs", "tenant_id")
