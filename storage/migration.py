"""
SQLite → PostgreSQL migration utility.

Usage:
    from storage.migration import migrate_sqlite_to_postgres
    migrate_sqlite_to_postgres("colab_agent.db", "postgresql://user:pass@localhost/db")
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_db_url() -> str:
    """Return the configured database URL.

    Checks ``COLAB_AGENT_DB_URL`` first, then ``DATABASE_URL``,
    falling back to a default SQLite path.
    """
    return (
        os.environ.get("COLAB_AGENT_DB_URL")
        or os.environ.get("DATABASE_URL")
        or "sqlite:///colab_agent.db"
    )


def is_postgres() -> bool:
    """Return True if the configured DB URL points to PostgreSQL."""
    return get_db_url().startswith("postgresql")


def migrate_sqlite_to_postgres(sqlite_path: str, pg_url: str) -> None:
    """One-shot migration of all tables from SQLite to PostgreSQL.

    Args:
        sqlite_path: Path to the SQLite database file.
        pg_url: Full PostgreSQL connection URL.

    Raises:
        ImportError: If sqlalchemy is not installed.
        RuntimeError: If migration fails midway (some rows may have been copied).
    """
    try:
        from sqlalchemy import create_engine, inspect, text
    except ImportError as exc:
        raise ImportError("sqlalchemy is required for migration") from exc

    src = create_engine(f"sqlite:///{sqlite_path}")
    dst = create_engine(pg_url)

    inspector = inspect(src)
    tables = inspector.get_table_names()
    logger.info("Migrating %d tables: %s", len(tables), tables)

    for table in tables:
        try:
            rows = src.execute(text(f"SELECT * FROM {table}")).fetchall()
        except Exception as exc:
            logger.error("Failed to read table %s: %s", table, exc)
            raise RuntimeError(f"Failed to read table {table}") from exc

        columns = [col["name"] for col in inspector.get_columns(table)]
        if not rows:
            logger.info("  Table %s: 0 rows, skipped", table)
            continue

        placeholders = ", ".join([f":{c}" for c in columns])
        col_list = ", ".join(columns)
        insert = text(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})")

        try:
            with dst.begin() as conn:
                for row in rows:
                    conn.execute(insert, dict(zip(columns, row)))
            logger.info("  Table %s: %d rows migrated", table, len(rows))
        except Exception as exc:
            logger.error("Failed to write table %s: %s", table, exc)
            raise RuntimeError(f"Migration failed on table {table}") from exc

    logger.info("Migration complete")


def migrate(revision: str = "head") -> None:
    """Run alembic migrations to the specified revision."""
    from alembic.config import Config
    from alembic import command
    
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, revision)
