import time
from agent.health import HealthReporter


class TestHealthReporter:
    def test_initial_status(self):
        hr = HealthReporter()
        r = hr.report()
        assert r["status"] == "ok"
        assert r["circuit_state"] == "closed"
        assert r["circuit_failures"] == 0
        assert r["total_retries"] == 0

    def test_update_fields(self):
        hr = HealthReporter()
        hr.update(circuit_state="open", circuit_failures=5, total_retries=10,
                  current_runtime="A100", active_job_id="job-1")
        r = hr.report()
        assert r["circuit_state"] == "open"
        assert r["circuit_failures"] == 5
        assert r["total_retries"] == 10
        assert r["current_runtime"] == "A100"
        assert r["active_job_id"] == "job-1"

    def test_degraded_on_open_circuit(self):
        hr = HealthReporter()
        hr.update(circuit_state="open")
        r = hr.report()
        assert r["status"] == "degraded"

    def test_degraded_on_high_retries(self):
        hr = HealthReporter()
        hr.update(total_retries=25)
        r = hr.report()
        assert r["status"] == "degraded"

    def test_uptime_increases(self):
        hr = HealthReporter()
        r1 = hr.report()
        time.sleep(0.01)
        r2 = hr.report()
        assert r2["uptime_hours"] > r1["uptime_hours"]

    def test_reset(self):
        hr = HealthReporter()
        hr.update(circuit_state="open", total_retries=50)
        hr.reset()
        r = hr.report()
        assert r["circuit_state"] == "closed"
        assert r["total_retries"] == 0
        assert r["uptime_hours"] < 0.01

    def test_health_endpoint_live(self):
        from agent.core import health_endpoint
        r = health_endpoint()
        assert "status" in r
        assert "circuit_state" in r
        assert "version" in r
        assert r["circuit_failures"] == 0
