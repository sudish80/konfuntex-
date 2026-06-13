import time
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Optional


logger = logging.getLogger(__name__)

HEALTH_VERSION = "1.0.0"
_DEGRADED_RETRY_THRESHOLD = 20


@dataclass
class HealthStatus:
    status: str = "ok"
    circuit_state: str = "closed"
    circuit_failures: int = 0
    total_retries: int = 0
    runtime_switches: int = 0
    current_runtime: str = "None"
    uptime_hours: float = 0.0
    active_job_id: Optional[str] = None
    active_conversation_id: Optional[str] = None
    error: str = ""
    version: str = HEALTH_VERSION


class HealthReporter:
    """Tracks and reports agent health status.

    Thread-safe. Automatically derives overall status from circuit
    breaker state, retry count, and runtime switches.
    """

    VALID_CIRCUIT_STATES = {"closed", "open", "half-open"}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.start_time = time.time()
        self._status = HealthStatus()

    def update(
        self,
        *,
        circuit_state: Optional[str] = None,
        circuit_failures: Optional[int] = None,
        total_retries: Optional[int] = None,
        runtime_switches: Optional[int] = None,
        current_runtime: Optional[str] = None,
        active_job_id: Optional[str] = None,
        active_conversation_id: Optional[str] = None,
    ) -> None:
        if circuit_state is not None and circuit_state not in self.VALID_CIRCUIT_STATES:
            raise ValueError(f"Invalid circuit_state: {circuit_state!r}")

        with self._lock:
            if circuit_state is not None:
                self._status.circuit_state = circuit_state
            if circuit_failures is not None:
                self._status.circuit_failures = max(0, circuit_failures)
            if total_retries is not None:
                self._status.total_retries = max(0, total_retries)
            if runtime_switches is not None:
                self._status.runtime_switches = max(0, runtime_switches)
            if current_runtime is not None:
                self._status.current_runtime = current_runtime
            if active_job_id is not None:
                self._status.active_job_id = active_job_id
            if active_conversation_id is not None:
                self._status.active_conversation_id = active_conversation_id

    def report(self) -> dict:
        with self._lock:
            self._status.uptime_hours = (time.time() - self.start_time) / 3600

            if self._status.circuit_state == "open":
                self._status.status = "degraded"
            elif self._status.total_retries > _DEGRADED_RETRY_THRESHOLD:
                self._status.status = "degraded"
            else:
                self._status.status = "ok"

            return asdict(self._status)

    def reset(self) -> None:
        with self._lock:
            self._status = HealthStatus()
            self.start_time = time.time()
            logger.info("HealthReporter reset")
