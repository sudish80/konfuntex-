"""
Data retention policy — TTL-based cleanup for logs, jobs, conversations,
and LLM cache. Designed to be called periodically (e.g. via scheduler
or on agent startup).
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from config.settings import settings


logger = logging.getLogger(__name__)


# ── Retention config ─────────────────────────────────────────────────

DEFAULT_RETENTION_DAYS = {
    "jobs": 90,
    "conversations": 90,
    "metrics": 180,
    "llm_cache": 7,
    "logs": 30,
    "audit_log": 365,
}


def get_retention_days(category: str) -> int:
    env_key = f"COLAB_AGENT_RETENTION_{category.upper()}_DAYS"
    return int(os.environ.get(env_key, DEFAULT_RETENTION_DAYS.get(category, 90)))


# ── Cleanup functions ────────────────────────────────────────────────

def clean_old_jobs(days: Optional[int] = None) -> int:
    """Delete jobs older than `days`. Returns count deleted."""
    from storage.jobs import JobStore
    days = days or get_retention_days("jobs")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    store = JobStore()
    count = store.delete_before(cutoff)
    logger.info("Data retention: cleaned %d jobs older than %d days", count, days)
    return count


def clean_old_conversations(days: Optional[int] = None) -> int:
    """Delete conversations older than `days`. Returns count deleted."""
    from storage.conversations import ConversationStore
    days = days or get_retention_days("conversations")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    store = ConversationStore()
    count = store.delete_before(cutoff)
    logger.info("Data retention: cleaned %d conversations older than %d days", count, days)
    return count


def clean_old_metrics(days: Optional[int] = None) -> int:
    """Delete metric epochs older than `days`. Returns count deleted."""
    from storage.metrics_store import MetricsStore
    days = days or get_retention_days("metrics")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    store = MetricsStore()
    count = store.delete_before(cutoff)
    logger.info("Data retention: cleaned %d metric records older than %d days", count, days)
    return count


def clean_old_logs(days: Optional[int] = None) -> int:
    """Delete JSONL log files older than `days`. Returns count deleted."""
    days = days or get_retention_days("logs")
    cutoff = time.time() - days * 86400
    count = 0
    data_dir = settings.data_dir
    if not os.path.isdir(data_dir):
        return 0
    for fname in os.listdir(data_dir):
        if fname.endswith(".jsonl"):
            fpath = os.path.join(data_dir, fname)
            try:
                mtime = os.path.getmtime(fpath)
                if mtime < cutoff:
                    os.remove(fpath)
                    count += 1
            except OSError:
                pass
    if count:
        logger.info("Data retention: cleaned %d log files older than %d days", count, days)
    return count


def run_retention_policy():
    """Run all retention cleanup tasks. Called on startup."""
    logger.info("Data retention: running cleanup policy")
    clean_old_jobs()
    clean_old_conversations()
    clean_old_metrics()
    clean_old_logs()
    logger.info("Data retention: cleanup complete")
