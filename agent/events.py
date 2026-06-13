"""Event bus for real-time job event streaming to WebSocket clients.

Thread-safe. Designed for sync-plugin → async-WebSocket bridge.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Optional

from agent.plugin import Plugin, plugin


logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe pub/sub event store.

    ``publish()`` is safe to call from any thread.  ``poll()`` lets
    an async WebSocket handler fetch new events since a given index.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._events: dict[str, list[dict]] = defaultdict(list)

    def publish(self, job_id: str, event: dict) -> None:
        event["_ts"] = time.time()
        with self._lock:
            self._events[job_id].append(event)

    def poll(self, job_id: str, since_index: int = 0) -> tuple[list[dict], int]:
        """Return events for *job_id* appended since *since_index*.

        Returns ``(new_events, total_count)``.
        """
        with self._lock:
            events = self._events.get(job_id, [])
            return events[since_index:], len(events)

    def has_events(self, job_id: str) -> bool:
        with self._lock:
            return bool(self._events.get(job_id))


_event_bus = EventBus()


def get_event_bus() -> EventBus:
    return _event_bus


@plugin(
    name="websocket_events",
    version="1.0.0",
    description="Publishes agent lifecycle events to the EventBus for WebSocket streaming",
    priority=70,
)
class WebSocketEventPlugin(Plugin):
    """Publishes step and job events to the shared ``EventBus``."""

    def __init__(self):
        super().__init__()
        self._bus = get_event_bus()

    def before_step(self, step: dict, context: dict) -> tuple[dict, dict]:
        job_id = context.get("job_id")
        if job_id:
            self._bus.publish(job_id, {
                "type": "step.started",
                "step_id": step.get("id"),
                "action": step.get("action"),
                "description": step.get("description"),
            })
        return step, context

    def after_step(self, step: dict, result: dict, context: dict) -> tuple[dict, dict]:
        job_id = context.get("job_id")
        if job_id:
            self._bus.publish(job_id, {
                "type": "step.completed",
                "step_id": step.get("id"),
                "status": result.get("status"),
                "error": result.get("error"),
                "metrics": result.get("metrics"),
            })
        return result, context

    def on_error(self, step: dict, error: str, context: dict) -> tuple[Optional[str], dict]:
        job_id = context.get("job_id")
        if job_id:
            self._bus.publish(job_id, {
                "type": "step.error",
                "step_id": step.get("id"),
                "error": error[:500],
            })
        return None, context

    def on_complete(self, result: dict, context: dict) -> tuple[dict, dict]:
        job_id = result.get("job_id") or context.get("job_id")
        if job_id:
            self._bus.publish(job_id, {
                "type": "job.completed",
                "status": result.get("status"),
                "summary": result.get("summary", "")[:200],
            })
        return result, context

    def on_summary(self, summary: str, context: dict) -> tuple[str, dict]:
        job_id = context.get("job_id")
        if job_id:
            self._bus.publish(job_id, {
                "type": "summary",
                "summary": summary[:200],
            })
        return summary, context
