"""
Phase 9 — Safety & Security (items 86-95).

Adds:
  - ImmutableMode         read-only safety wrapper         (89)
  - EmergencyStop         circuit breaker + kill switch    (90)
  - APIKeyRotation        rotate keys on detection         (91)
  - RateLimiter           token / call rate limiting       (92)
  - JobSealer             seal job config post-hoc          (93)
  - TimeBomb              absolute wall-clock stop         (94)
  - TelemetrySink         audit log stream                 (95)
  - AuditLogger           structured audit trail           (enhanced)
  - IntegrityChecker      file/content hash verification   (new)
"""
import os
import json
import time
import hmac
import hashlib
import logging
import threading
from typing import Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass

from config.settings import settings

logger = logging.getLogger(__name__)


# ==================================================================== #
#  89 — ImmutableMode
# ==================================================================== #

class ImmutableMode:
    """
    Read-only safety wrapper. When enabled, all mutations are blocked.
    Useful for evaluation or inspection sessions.
    """

    def __init__(self, enabled: bool = False):
        self._enabled = enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def active(self) -> bool:
        return self._enabled

    def guard(self, operation: str) -> bool:
        if self._enabled:
            logger.warning(f"Immutable mode BLOCKED: {operation}")
            return False
        return True

    def guard_or_raise(self, operation: str):
        if not self.guard(operation):
            raise PermissionError(f"Immutable mode: {operation} not allowed")

    def __enter__(self):
        self._prior = self._enabled
        self._enabled = True
        return self

    def __exit__(self, *args):
        self._enabled = self._prior


# ==================================================================== #
#  90 — EmergencyStop
# ==================================================================== #

class EmergencyStop:
    """
    Circuit breaker pattern for the agent loop.
    Allows a hard stop from any thread.
    """

    def __init__(self):
        self._triggered = False
        self._lock = threading.Lock()
        self._reason = ""
        self._timestamp: Optional[str] = None
        self._callback: Optional[Callable] = None

    def trigger(self, reason: str = "Manual emergency stop"):
        with self._lock:
            if not self._triggered:
                self._triggered = True
                self._reason = reason
                self._timestamp = datetime.now(timezone.utc).isoformat()
                logger.critical(f"EMERGENCY STOP: {reason}")
                if self._callback:
                    try:
                        self._callback(reason)
                    except Exception as e:
                        logger.error(f"Emergency callback failed: {e}")

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    def check(self):
        if self._triggered:
            raise RuntimeError(f"Emergency stop: {self._reason}")

    def on_trigger(self, callback: Callable[[str], None]):
        self._callback = callback

    def reset(self):
        with self._lock:
            self._triggered = False
            self._reason = ""
            self._timestamp = None

    def get_state(self) -> dict:
        return {
            "triggered": self._triggered,
            "reason": self._reason,
            "timestamp": self._timestamp,
        }


# ==================================================================== #
#  91 — APIKeyRotation
# ==================================================================== #

class APIKeyRotation:
    """
    Rotate API keys from a pool when rate-limit or auth errors are detected.
    Keys are loaded from env vars, a JSON file, or provided as a list.
    """

    def __init__(self, key_var_prefix: str = ""):
        self._key_var_prefix = key_var_prefix
        self._keys: list[dict] = []
        self._current_index = 0
        self._lock = threading.Lock()
        self._load_keys()

    def _load_keys(self):
        prefix = self._key_var_prefix or "OPENAI_API_KEY"
        keys = []
        for env_key, val in os.environ.items():
            if env_key.startswith(prefix) and val:
                keys.append({"name": env_key, "value": val})
        if not keys:
            primary = os.environ.get(prefix)
            if primary:
                keys.append({"name": prefix, "value": primary})
        self._keys = keys

    def set_keys(self, keys: list[str]):
        with self._lock:
            self._keys = [{"name": f"key_{i}", "value": k} for i, k in enumerate(keys)]
            self._current_index = 0

    def current_key(self) -> Optional[str]:
        with self._lock:
            if not self._keys:
                return None
            return self._keys[self._current_index]["value"]

    def rotate(self) -> Optional[str]:
        with self._lock:
            if len(self._keys) <= 1:
                return None
            self._current_index = (self._current_index + 1) % len(self._keys)
            key = self._keys[self._current_index]
            logger.info(f"Rotated to key: {key['name']}")
            return key["value"]

    def count(self) -> int:
        return len(self._keys)


# ==================================================================== #
#  92 — RateLimiter
# ==================================================================== #

@dataclass
class RateBucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """
    Token-bucket rate limiter for LLM API calls.
    Supports per-minute and per-second limits.
    """

    def __init__(self, calls_per_minute: int = 60,
                 tokens_per_minute: int = 90000):
        self.cpm = calls_per_minute
        self.tpm = tokens_per_minute
        self._call_bucket = RateBucket(float(calls_per_minute), time.time())
        self._token_bucket = RateBucket(float(tokens_per_minute), time.time())
        self._lock = threading.Lock()

    def _refill(self, bucket: RateBucket, rate: float):
        now = time.time()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(bucket.tokens + elapsed * (rate / 60.0), rate)
        bucket.last_refill = now

    def wait_if_needed(self, estimated_tokens: int = 0):
        with self._lock:
            self._refill(self._call_bucket, self.cpm)
            self._refill(self._token_bucket, self.tpm)
            wait = 0.0
            if self._call_bucket.tokens < 1:
                wait = max(wait, (1 - self._call_bucket.tokens) * 60.0 / self.cpm)
            if self._token_bucket.tokens < estimated_tokens:
                wait = max(wait, (estimated_tokens - self._token_bucket.tokens) * 60.0 / self.tpm)
            if wait > 0:
                time.sleep(wait)
                self._refill(self._call_bucket, self.cpm)
                self._refill(self._token_bucket, self.tpm)
            self._call_bucket.tokens -= 1
            self._token_bucket.tokens -= estimated_tokens

    def consume(self, tokens: int = 0) -> bool:
        with self._lock:
            self._refill(self._call_bucket, self.cpm)
            self._refill(self._token_bucket, self.tpm)
            if self._call_bucket.tokens < 1 or self._token_bucket.tokens < tokens:
                return False
            self._call_bucket.tokens -= 1
            self._token_bucket.tokens -= tokens
            return True


# ==================================================================== #
#  93 — JobSealer
# ==================================================================== #

class JobSealer:
    """
    Seal a job's configuration post-hoc to prevent tampering.
    Computes a hash of the job config and stores it alongside.
    """

    def __init__(self, secret: str = ""):
        self.secret = secret or settings.hf_token or "colab-agent-seed"

    def seal(self, job_config: dict) -> str:
        serialized = json.dumps(job_config, sort_keys=True)
        sig = hmac.new(
            self.secret.encode(),
            serialized.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"{sig[:16]}"

    def verify(self, job_config: dict, signature: str) -> bool:
        expected = self.seal(job_config)
        return hmac.compare_digest(expected, signature)

    def seal_and_store(self, job_id: str, job_config: dict,
                       store: Callable[[str, str, str], None]):
        sig = self.seal(job_config)
        store(job_id, sig, json.dumps(job_config))
        return sig


# ==================================================================== #
#  94 — TimeBomb
# ==================================================================== #

class TimeBomb:
    """
    Absolute wall-clock timeout. When triggered, the agent must stop.
    Use to enforce Colab runtime limits (12h free tier).
    """

    def __init__(self, max_hours: float = 12):
        self.max_seconds = max_hours * 3600
        self._start: Optional[float] = None
        self._expired = False

    def start(self):
        self._start = time.time()
        self._expired = False
        logger.info(f"TimeBomb started: {self.max_seconds}s countdown")

    @property
    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.time() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    @property
    def expired(self) -> bool:
        if self._start is None:
            return False
        if not self._expired:
            self._expired = self.elapsed >= self.max_seconds
        return self._expired

    def check(self):
        if self.expired:
            raise RuntimeError(f"TimeBomb expired after {self.max_seconds}s")

    def fraction_used(self) -> float:
        return min(1.0, self.elapsed / self.max_seconds) if self.max_seconds > 0 else 0.0

    def reset(self):
        self._start = None
        self._expired = False


# ==================================================================== #
#  95 — TelemetrySink
# ==================================================================== #

class TelemetrySink:
    """
    Audit log stream. Writes structured events to a JSONL file.
    """

    def __init__(self, path: str = "telemetry.jsonl"):
        self.path = path
        self._lock = threading.Lock()
        self._session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

    def emit(self, event: str, data: dict = None, level: str = "info"):
        record = {
            "session": self._session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "level": level,
            "data": data or {},
        }
        with self._lock:
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                logger.error(f"Telemetry write failed: {e}")

    def emit_event(self, event_type: str, payload: dict = None):
        self.emit(f"event:{event_type}", payload or {})

    def emit_error(self, error_type: str, detail: str, context: dict = None):
        payload = {"error_type": error_type, "detail": detail, **(context or {})}
        self.emit(f"error:{error_type}", payload, level="error")

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_summary(self) -> dict:
        records = self.read_all()
        return {
            "total_events": len(records),
            "errors": sum(1 for r in records if r.get("level") == "error"),
            "events_by_type": {
                k: sum(1 for r in records if r.get("event") == k)
                for k in set(r.get("event", "") for r in records)
            },
        }


# ==================================================================== #
#  AuditLogger  — structured audit trail
# ==================================================================== #

class AuditLogger:
    """
    Structured audit trail for security-relevant events.
    Writes to a JSONL file and optionally to a callable sink (e.g., HTTP endpoint).
    """

    AUDIT_CATEGORIES = {
        "access": ("login", "logout", "token_refresh", "permission_check"),
        "config": ("setting_change", "key_rotation", "encryption_status"),
        "execution": ("code_sanitized", "code_blocked", "pip_blocked", "sandbox_exec"),
        "data": ("goal_received", "goal_sanitized", "prompt_sent", "response_received"),
        "error": ("api_error", "auth_failure", "rate_limited", "timeout"),
        "admin": ("user_action", "system_start", "system_shutdown"),
    }

    def __init__(self, path: str = "audit.log.jsonl", sink: Optional[Callable] = None):
        self.path = path
        self._sink = sink
        self._lock = threading.Lock()
        self._session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

    def record(self, category: str, action: str, detail: str = "",
               actor: str = "system", context: Optional[dict] = None,
               level: str = "info"):
        self._validate_category_action(category, action)
        entry = {
            "session": self._session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "action": action,
            "detail": detail[:500],
            "actor": actor,
            "level": level,
            "context": context or {},
        }
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.error(f"Audit log write failed: {e}")
        if self._sink:
            try:
                self._sink(entry)
            except Exception as e:
                logger.error(f"Audit sink failed: {e}")

    def _validate_category_action(self, category: str, action: str):
        if category not in self.AUDIT_CATEGORIES:
            raise ValueError(f"Unknown audit category: {category}")
        if action not in self.AUDIT_CATEGORIES[category]:
            allowed = self.AUDIT_CATEGORIES[category]
            raise ValueError(f"Unknown action '{action}' for category '{category}'. Allowed: {allowed}")

    def goal_received(self, goal: str, actor: str = "user"):
        self.record("data", "goal_received", goal[:200], actor=actor,
                     context={"length": len(goal)})

    def goal_sanitized(self, original: str, warnings: list[str], actor: str = "user"):
        self.record("data", "goal_sanitized", str(warnings), actor=actor,
                     context={"original_length": len(original), "warnings": warnings},
                     level="warning")

    def code_blocked(self, code_hash: str, reason: str, actor: str = "agent"):
        self.record("execution", "code_blocked", reason, actor=actor,
                     context={"code_hash": code_hash}, level="warning")

    def pip_blocked(self, package: str, code_hash: str, actor: str = "agent"):
        self.record("execution", "pip_blocked", package, actor=actor,
                     context={"code_hash": code_hash}, level="warning")

    def sandbox_exec(self, success: bool, execution_time: float, actor: str = "agent"):
        self.record("execution", "sandbox_exec", actor=actor,
                     context={"success": success, "execution_time": execution_time})

    def key_rotated(self, key_name: str, actor: str = "system"):
        self.record("config", "key_rotation", key_name, actor=actor, level="warning")

    def auth_failure(self, service: str, detail: str, actor: str = "system"):
        self.record("error", "auth_failure", detail, actor=actor,
                     context={"service": service}, level="error")

    def rate_limited(self, resource: str, retry_after: float, actor: str = "system"):
        self.record("error", "rate_limited", actor=actor,
                     context={"resource": resource, "retry_after": retry_after},
                     level="warning")

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


# ==================================================================== #
#  IntegrityChecker  — file and content hash verification
# ==================================================================== #

class IntegrityChecker:
    """
    Verify integrity of critical files (config, .env, downloaded scripts).
    Stores expected hashes and compares against actuals on check.
    """

    def __init__(self, state_file: str = "integrity_cache.json"):
        self.state_file = state_file
        self._lock = threading.Lock()
        self._cache: dict[str, str] = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self.state_file):
            return {}
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save integrity cache: {e}")

    def register(self, path: str) -> str:
        """Register a file: compute its hash and store it."""
        h = self._hash_file(path)
        with self._lock:
            self._cache[path] = h
            self._save()
        return h

    def check(self, path: str) -> Optional[bool]:
        """Check if a registered file has changed. Returns None if not registered."""
        with self._lock:
            expected = self._cache.get(path)
        if expected is None:
            return None
        actual = self._hash_file(path)
        return actual == expected

    def verify_all(self) -> list[dict]:
        """Check all registered files. Returns list of {path, ok, actual, expected}."""
        results = []
        with self._lock:
            for path, expected in list(self._cache.items()):
                if not os.path.exists(path):
                    results.append({"path": path, "ok": False, "error": "not_found"})
                    continue
                actual = self._hash_file(path)
                results.append({
                    "path": path,
                    "ok": actual == expected,
                    "actual": actual,
                    "expected": expected,
                })
        return results

    def remove(self, path: str):
        with self._lock:
            self._cache.pop(path, None)
            self._save()

    def _hash_file(self, path: str) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except FileNotFoundError:
            return ""
        return h.hexdigest()[:16]
