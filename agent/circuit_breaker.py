import time
import logging
import threading
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitBreakerError(RuntimeError):
    """Raised when the circuit breaker blocks an operation."""


class CircuitBreaker:
    """Thread-safe circuit breaker with automatic recovery.

    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (test) -> CLOSED.

    Args:
        failure_threshold: Consecutive failures before opening (>= 1).
        recovery_timeout: Seconds before attempting half-open recovery (> 0).
        name: Optional label for logging.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        name: Optional[str] = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name or "default"
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = threading.Lock()

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker[%s] OPEN — failures: %d",
                    self.name, self.failure_count,
                )

    def record_success(self) -> None:
        with self._lock:
            old_state = self.state
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            if old_state != CircuitState.CLOSED:
                logger.info(
                    "CircuitBreaker[%s] CLOSED — recovered from %s",
                    self.name, old_state.value,
                )

    def can_attempt(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info(
                        "CircuitBreaker[%s] HALF_OPEN — testing recovery",
                        self.name,
                    )
                    return True
                return False

            return True

    def reset(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = 0.0
            logger.info("CircuitBreaker[%s] manually reset", self.name)

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }

    def __enter__(self) -> "CircuitBreaker":
        if not self.can_attempt():
            raise CircuitBreakerError(
                f"CircuitBreaker[{self.name}] is OPEN — operation blocked "
                f"(failures: {self.failure_count}/{self.failure_threshold})"
            )
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        if exc_type is not None and exc_type is not CircuitBreakerError:
            self.record_failure()
        elif exc_type is None:
            self.record_success()
