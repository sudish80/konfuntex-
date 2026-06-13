"""Tests for agent/api.py — rate limiting and auth hardening."""
import time
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from agent.api import app
    with TestClient(app) as c:
        yield c


class TestRateLimiter:

    def test_rate_limiter_accepts(self):
        from agent.api import _rate_limiter
        assert _rate_limiter.check("test-ip") is True

    def test_rate_limiter_blocks_by_burst(self):
        from agent.api import RateLimiter
        limiter = RateLimiter(rate=1, burst=2)
        assert limiter.check("burst-test") is True
        assert limiter.check("burst-test") is True
        # Third should be False (burst exhausted, rate=1/min)
        assert limiter.check("burst-test") is False

    def test_rate_limiter_recovers(self):
        from agent.api import RateLimiter
        limiter = RateLimiter(rate=6000, burst=1)  # high rate so it recovers fast
        limiter.check("recover-test")  # consume burst
        # After a small wait, should recover
        time.sleep(0.02)
        assert limiter.check("recover-test") is True


class TestAuth:

    def test_budget_requires_auth_when_key_set(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_API_KEY", "test-key-123")
        import importlib
        from agent import api
        importlib.reload(api)
        with TestClient(api.app) as c:
            r = c.get("/budget")
            assert r.status_code == 401
            r = c.get("/budget", headers={"Authorization": "Bearer test-key-123"})
            assert r.status_code == 200

    def test_root_with_wrong_key(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_API_KEY", "real-key")
        import importlib
        from agent import api
        importlib.reload(api)
        with TestClient(api.app) as c:
            r = c.get("/v1/budget", headers={"Authorization": "Bearer wrong-key"})
            assert r.status_code == 401

    def test_health_unauthed(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_API_KEY", "some-key")
        import importlib
        from agent import api
        importlib.reload(api)
        with TestClient(api.app) as c:
            r = c.get("/health")
            assert r.status_code == 200  # health is exempt from auth


class TestGDPR:

    def _fresh_app(self):
        import importlib
        from agent import api
        importlib.reload(api)
        from storage.database import init_db, reset_session
        reset_session()
        init_db()
        from fastapi.testclient import TestClient
        return TestClient(api.app)

    def test_gdpr_export_empty(self):
        client = self._fresh_app()
        r = client.post("/v1/gdpr/export/test-tenant")
        assert r.status_code == 200
        data = r.json()
        assert data["tenant_id"] == "test-tenant"
        assert data["jobs"] == 0
        assert data["conversations"] == 0

    def test_gdpr_delete_empty(self):
        client = self._fresh_app()
        r = client.post("/v1/gdpr/delete/test-tenant")
        assert r.status_code == 200
        data = r.json()
        assert data["deleted_jobs"] == 0
        assert data["deleted_conversations"] == 0

    def test_gdpr_export_requires_auth(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_API_KEY", "sec-key")
        client = self._fresh_app()
        r = client.post("/v1/gdpr/export/t1")
        assert r.status_code == 401

    def test_gdpr_delete_requires_auth(self, monkeypatch):
        monkeypatch.setenv("COLAB_AGENT_API_KEY", "sec-key")
        client = self._fresh_app()
        r = client.post("/v1/gdpr/delete/t1")
        assert r.status_code == 401
