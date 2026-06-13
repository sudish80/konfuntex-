import json
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


class JSONFormatter(logging.Formatter):
    """Outputs log records as structured JSON lines.

    Usage:
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    """

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            obj["exc_type"] = record.exc_info[0].__name__
        if hasattr(record, "extra_fields"):
            obj.update(record.extra_fields)
        return json.dumps(obj, default=str)


def setup_json_logging(
    target_logger: logging.Logger,
    level: int = logging.INFO,
    log_path: Optional[str] = None,
) -> logging.Logger:
    """Replace handlers on *target_logger* with JSON-formatted ones.

    Args:
        target_logger: Logger to configure.
        level: Logging level (default INFO).
        log_path: Optional file path for persistent JSONL output.

    Returns:
        The configured logger.
    """
    for handler in target_logger.handlers[:]:
        target_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    target_logger.addHandler(handler)
    target_logger.setLevel(level)

    if log_path:
        try:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(JSONFormatter())
            target_logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("Cannot open log file %s: %s", log_path, exc)

    return target_logger


@dataclass
class MetricPoint:
    name: str
    value: float
    labels: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Thread-safe metrics collector supporting counters and gauges.

    Produces Prometheus-format text output via ``prometheus_text()``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}

    def counter(self, name: str, inc: float = 1.0, labels: Optional[dict] = None) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("counter name must be a non-empty string")
        if inc < 0:
            raise ValueError("counter increment must be >= 0")

        with self._lock:
            prev = self._counters.get(name, 0.0)
            self._counters[name] = prev + inc

    def gauge(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("gauge name must be a non-empty string")

        with self._lock:
            self._gauges[name] = value

    def get_counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> Optional[float]:
        with self._lock:
            return self._gauges.get(name)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }

    def prometheus_text(self) -> str:
        """Return metrics in Prometheus exposition format."""
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                sanitised = name.replace("-", "_").replace(".", "_")
                lines.append(f"# HELP {sanitised} Counter metric")
                lines.append(f"# TYPE {sanitised} counter")
                lines.append(f"{sanitised} {value}")
            for name, value in sorted(self._gauges.items()):
                sanitised = name.replace("-", "_").replace(".", "_")
                lines.append(f"# HELP {sanitised} Gauge metric")
                lines.append(f"# TYPE {sanitised} gauge")
                lines.append(f"{sanitised} {value}")
        return "\n".join(lines) + "\n"

    def clear(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
