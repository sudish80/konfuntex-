"""
Database engine initialisation with SQLite and PostgreSQL support.

Auto-detects PostgreSQL vs SQLite from the configured URL.
"""

import os
import logging
import json
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import NullPool

from config.settings import settings


logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id = sa.Column(sa.String, primary_key=True)
    tenant_id = sa.Column(sa.String, nullable=True, index=True)
    goal = sa.Column(sa.Text, nullable=False)
    status = sa.Column(sa.String, default="pending")
    method = sa.Column(sa.String, nullable=True)
    base_model = sa.Column(sa.String, nullable=True)
    dataset = sa.Column(sa.String, nullable=True)
    runtime = sa.Column(sa.String, nullable=True)
    metrics = sa.Column(sa.Text, nullable=True)
    error = sa.Column(sa.Text, nullable=True)
    created_at = sa.Column(sa.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = sa.Column(sa.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    finished_at = sa.Column(sa.DateTime, nullable=True)
    colab_notebook_id = sa.Column(sa.String, nullable=True)
    colab_notebook_url = sa.Column(sa.String, nullable=True)
    conversation_id = sa.Column(sa.String, nullable=True)
    model_version_id = sa.Column(sa.String, nullable=True)
    metadata_json = sa.Column(sa.Text, nullable=True)

    def set_metrics(self, d: dict):
        self.metrics = json.dumps(d)

    def get_metrics(self) -> dict:
        return json.loads(self.metrics) if self.metrics else {}

    def set_metadata(self, d: dict):
        self.metadata_json = json.dumps(d)

    def get_metadata(self) -> dict:
        return json.loads(self.metadata_json) if self.metadata_json else {}


class Conversation(Base):
    __tablename__ = "conversations"

    id = sa.Column(sa.String, primary_key=True)
    tenant_id = sa.Column(sa.String, nullable=True, index=True)
    goal = sa.Column(sa.Text, nullable=False)
    status = sa.Column(sa.String, default="active")
    created_at = sa.Column(sa.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = sa.Column(sa.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    messages_json = sa.Column(sa.Text, default="[]")
    summary = sa.Column(sa.Text, nullable=True)

    def get_messages(self) -> list:
        return json.loads(self.messages_json) if self.messages_json else []

    def set_messages(self, msgs: list):
        self.messages_json = json.dumps(msgs, default=str)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = sa.Column(sa.String, primary_key=True)
    tenant_id = sa.Column(sa.String, nullable=True, index=True)
    job_id = sa.Column(sa.String, sa.ForeignKey("jobs.id"), nullable=True)
    base_model = sa.Column(sa.String, nullable=False)
    finetuned_path = sa.Column(sa.String, nullable=True)
    hf_repo_id = sa.Column(sa.String, nullable=True)
    method = sa.Column(sa.String, nullable=True)
    metrics = sa.Column(sa.Text, nullable=True)
    runtime_used = sa.Column(sa.String, nullable=True)
    training_steps = sa.Column(sa.Integer, nullable=True)
    final_loss = sa.Column(sa.Float, nullable=True)
    created_at = sa.Column(sa.DateTime, default=lambda: datetime.now(timezone.utc))
    tags = sa.Column(sa.Text, nullable=True)
    metadata_json = sa.Column(sa.Text, nullable=True)

    def set_metrics(self, d: dict):
        self.metrics = json.dumps(d)

    def get_metrics(self) -> dict:
        return json.loads(self.metrics) if self.metrics else {}


class RuntimeLog(Base):
    __tablename__ = "runtime_logs"

    id = sa.Column(sa.String, primary_key=True)
    tenant_id = sa.Column(sa.String, nullable=True, index=True)
    job_id = sa.Column(sa.String, sa.ForeignKey("jobs.id"), nullable=True)
    runtime_type = sa.Column(sa.String, nullable=False)
    ram_gb = sa.Column(sa.Float, nullable=True)
    vram_gb = sa.Column(sa.Float, nullable=True)
    gpu_name = sa.Column(sa.String, nullable=True)
    switched_from = sa.Column(sa.String, nullable=True)
    switched_to = sa.Column(sa.String, nullable=True)
    switch_reason = sa.Column(sa.Text, nullable=True)
    status = sa.Column(sa.String, default="active")
    started_at = sa.Column(sa.DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = sa.Column(sa.DateTime, nullable=True)


def get_engine():
    """Create a SQLAlchemy engine for the configured database URL."""
    db_url = settings.get_db_url()
    if db_url.startswith("postgresql"):
        logger.info("Using PostgreSQL database")
        return create_engine(db_url, poolclass=NullPool)

    # SQLite
    os.makedirs(settings.data_dir, exist_ok=True)
    db_path = db_url.replace("sqlite:///", "")
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    return create_engine(
        db_url,
        connect_args={"check_same_thread": False},
    )


def init_db():
    """Initialise database tables and return a sessionmaker."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


SessionLocal = None


def reset_session():
    """Reset the session singleton (used by tests for isolation)."""
    global SessionLocal
    SessionLocal = None


def get_session():
    """Return a singleton SQLAlchemy sessionmaker."""
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = init_db()
    return SessionLocal()


# ── Async Support ──────────────────────────────────────────────────────────

def get_async_engine():
    db_url = settings.get_db_url()
    if db_url.startswith("sqlite"):
        async_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")
    elif db_url.startswith("postgresql"):
        async_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    else:
        async_url = db_url

    if async_url.startswith("postgresql"):
        return create_async_engine(async_url, poolclass=NullPool)
    
    # SQLite
    os.makedirs(settings.data_dir, exist_ok=True)
    return create_async_engine(async_url)

AsyncSessionLocal = None

def get_async_sessionmaker():
    global AsyncSessionLocal
    if AsyncSessionLocal is None:
        engine = get_async_engine()
        AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return AsyncSessionLocal
