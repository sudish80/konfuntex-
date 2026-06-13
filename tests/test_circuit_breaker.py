import pytest
import time
from agent.circuit_breaker import CircuitBreaker, CircuitState

def test_circuit_breaker_flow():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    # 1. Initially closed
    assert cb.state == CircuitState.CLOSED

    # 2. Record failure
    cb.record_failure()
    assert cb.failure_count == 1

    # 3. Record failure -> Open
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # 4. Blocked
    with pytest.raises(RuntimeError):
        with cb:
            pass

    # 5. Wait for recovery
    time.sleep(0.2)

    # 6. Half-open
    assert cb.can_attempt() is True
    assert cb.state == CircuitState.HALF_OPEN

    # 7. Success -> Closed
    with cb:
        pass
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
