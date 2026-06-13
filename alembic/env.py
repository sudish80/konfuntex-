"""Alembic environment configuration.

Imports all ORM models so that ``autogenerate`` can detect schema
changes, and reads the database URL from the project ``settings``.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context

# Ensure the project root is on sys.path for model imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import all models so they are registered on Base.metadata.
from storage.database import Base, Job, Conversation, ModelVersion, RuntimeLog  # noqa: E402, F401
from storage.metrics_store import MetricRecord  # noqa: E402, F401
from config.settings import load_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point Alembic at the project's live database URL.
settings = load_settings()
config.set_main_option("sqlalchemy.url", settings.get_db_url())

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine,
    allowing migrations to be generated without a live database.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live database."""
    from storage.database import get_engine
    engine = get_engine()

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
