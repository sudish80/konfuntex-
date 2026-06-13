import json
from agent.service import AgentHandler
from unittest.mock import MagicMock


class TestAgentHandler:
    def _make_handler(self, path):
        handler = AgentHandler.__new__(AgentHandler)
        handler.path = path
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)
        handler.command = "GET"
        handler.requestline = f"GET {path} HTTP/1.1"
        return handler

    def test_root(self):
        h = self._make_handler("/")
        h.do_GET()
        args, _ = h.wfile.write.call_args
        body = json.loads(args[0])
        assert body["service"] == "colab-agent"

    def test_health(self):
        h = self._make_handler("/health")
        h.do_GET()
        args, _ = h.wfile.write.call_args
        body = json.loads(args[0])
        assert "status" in body

    def test_metrics(self):
        h = self._make_handler("/metrics")
        h.do_GET()
        args, _ = h.wfile.write.call_args
        assert isinstance(args[0], bytes)

    def test_404(self):
        h = self._make_handler("/nonexistent")
        h.do_GET()
        args, _ = h.wfile.write.call_args
        body = json.loads(args[0])
        assert body["error"] == "not found"

    def test_migration_module_imports(self):
        from storage.migration import get_db_url
        url = get_db_url()
        assert url is not None

    def test_postgres_detection(self):
        import os
        orig = os.environ.get("COLAB_AGENT_DB_URL")
        os.environ["COLAB_AGENT_DB_URL"] = "postgresql://user:pass@localhost/db"
        try:
            import importlib
            from storage import migration
            importlib.reload(migration)
            assert migration.get_db_url().startswith("postgresql")
            assert migration.is_postgres() is True
        finally:
            if orig:
                os.environ["COLAB_AGENT_DB_URL"] = orig
            else:
                os.environ.pop("COLAB_AGENT_DB_URL", None)
