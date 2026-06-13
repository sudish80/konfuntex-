"""
LLM response cache with disk persistence.

Cache key = SHA256 hash of (model, messages JSON, temperature, max_tokens, top_p)
Backed by SQLite for durability across restarts.
Thread-safe with per-key locks.
"""

import os
import json
import time
import hashlib
import logging
import threading
import sqlite3
from typing import Optional

from agent.plugin import Plugin, plugin
from config.settings import settings


logger = logging.getLogger(__name__)


def _cache_key(model: str, messages: list, temperature: float = 0.0,
               max_tokens: int = 4096, top_p: float = 1.0) -> str:
    raw = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMCache:
    """Thread-safe, disk-backed LLM response cache.

    Stores (cache_key, response_json, created_at, ttl_seconds).
    Auto-evicts expired entries on read and periodically.
    """

    def __init__(self, db_path: str = "", ttl_seconds: int = 3600):
        self._db_path = db_path or os.path.join(
            settings.data_dir, "llm_cache.sqlite"
        )
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._closed = False
        self._init_db()

    def _init_db(self):
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        except OSError as e:
            logger.error(f"Cannot create cache directory: {e}")
            self._closed = True
            return
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ttl_seconds REAL NOT NULL
                )
            """)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_cache_created "
                "ON llm_cache(created_at)"
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Cannot initialize cache database: {e}")
            self._closed = True

    def _check_open(self):
        if self._closed or self._conn is None:
            raise RuntimeError("LLMCache is closed")

    def get(self, model: str, messages: list, temperature: float = 0.0,
            max_tokens: int = 4096, top_p: float = 1.0) -> Optional[dict]:
        key = _cache_key(model, messages, temperature, max_tokens, top_p)
        with self._lock:
            self._check_open()
            try:
                row = self._conn.execute(
                    "SELECT response, created_at, ttl_seconds FROM llm_cache WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                response_json, created_at, ttl = row
                if time.time() - created_at > ttl:
                    self._conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                    self._conn.commit()
                    return None
                return json.loads(response_json)
            except sqlite3.Error as e:
                logger.error(f"LLMCache get error: {e}")
                return None

    def set(self, response: dict, model: str, messages: list,
            temperature: float = 0.0, max_tokens: int = 4096,
            top_p: float = 1.0, ttl_seconds: Optional[float] = None):
        key = _cache_key(model, messages, temperature, max_tokens, top_p)
        with self._lock:
            self._check_open()
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO llm_cache (key, response, created_at, ttl_seconds) "
                    "VALUES (?, ?, ?, ?)",
                    (key, json.dumps(response), time.time(),
                     ttl_seconds or self._ttl),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                logger.error(f"LLMCache set error: {e}")

    def invalidate(self, model: str = "", messages: list = None,
                   temperature: float = 0.0, max_tokens: int = 4096,
                   top_p: float = 1.0):
        if messages is not None:
            key = _cache_key(model, messages, temperature, max_tokens, top_p)
            with self._lock:
                self._check_open()
                try:
                    self._conn.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                    self._conn.commit()
                except sqlite3.Error as e:
                    logger.error(f"LLMCache invalidate error: {e}")
        else:
            self.clear()

    def clear(self):
        with self._lock:
            self._check_open()
            try:
                self._conn.execute("DELETE FROM llm_cache")
                self._conn.commit()
            except sqlite3.Error as e:
                logger.error(f"LLMCache clear error: {e}")

    def size(self) -> int:
        with self._lock:
            self._check_open()
            try:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM llm_cache"
                ).fetchone()
                return row[0] if row else 0
            except sqlite3.Error as e:
                logger.error(f"LLMCache size error: {e}")
                return 0

    def evict_expired(self):
        with self._lock:
            if self._closed or self._conn is None:
                return
            try:
                self._conn.execute(
                    "DELETE FROM llm_cache WHERE ? - created_at > ttl_seconds",
                    (time.time(),),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                logger.error(f"LLMCache evict error: {e}")

    def close(self):
        with self._lock:
            self._closed = True
            if self._conn:
                try:
                    self._conn.close()
                except sqlite3.Error as e:
                    logger.error(f"LLMCache close error: {e}")
                finally:
                    self._conn = None


@plugin(name="llm_cache", version="1.0.0",
        description="Caches LLM responses to avoid redundant API calls",
        priority=200)
class LLMCachePlugin(Plugin):
    """Plugin wrapper around LLMCache. Caches chat responses."""

    def __init__(self):
        super().__init__()
        self.cache = LLMCache()

    def before_code_gen(self, step: dict, prompt: str, context: dict) -> tuple[str, dict]:
        try:
            cached = self.cache.get(
                context.get("model", "default"),
                [{"role": "user", "content": prompt}],
            )
            if cached:
                context["cached_response"] = cached.get("content", "")
                logger.info(f"LLM cache hit for step {step.get('id')}")
        except Exception as e:
            logger.warning(f"LLMCachePlugin before_code_gen error: {e}")
        return prompt, context

    def on_complete(self, result: dict, context: dict) -> tuple[dict, dict]:
        try:
            self.cache.evict_expired()
        except Exception as e:
            logger.warning(f"LLMCachePlugin evict error: {e}")
        return result, context
