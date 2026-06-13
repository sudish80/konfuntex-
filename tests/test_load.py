"""Load tests for the API server — concurrent requests, rate limiting, WebSocket."""
import os
import sys
import time
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# Use a dedicated test DB — must be set before any storage imports
_TEST_DB = os.path.join(os.path.dirname(__file__), "..", "test_load.db")
os.environ["COLAB_AGENT_DB_URL"] = f"sqlite:///{os.path.abspath(_TEST_DB)}"
os.environ["COLAB_AGENT_RATE_LIMIT"] = "600"
os.environ["COLAB_AGENT_RATE_LIMIT_BURST"] = "100"


def _start_server(port: int):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from storage.database import reset_session, init_db
    reset_session()
    init_db()
    from agent.api import app
    import uvicorn, threading
    t = threading.Thread(target=uvicorn.run, args=(app,), kwargs={"host": "127.0.0.1", "port": port, "log_level": "error"}, daemon=True)
    t.start()
    time.sleep(3)
    return t


class TestLoad:

    port = 8090

    @classmethod
    def setup_class(cls):
        _start_server(cls.port)

    @classmethod
    def teardown_class(cls):
        import gc
        gc.collect()
        try:
            if os.path.exists(_TEST_DB):
                os.remove(_TEST_DB)
        except PermissionError:
            pass  # DB may still be in use by another test on Windows

    def _get(self, path: str) -> tuple[int, str]:
        try:
            r = urlopen(Request(f"http://127.0.0.1:{self.port}{path}"), timeout=5)
            return r.status, r.read().decode()
        except HTTPError as e:
            return e.code, e.read().decode()

    def _post(self, path: str) -> tuple[int, str]:
        req = Request(f"http://127.0.0.1:{self.port}{path}", method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            r = urlopen(req, timeout=5)
            return r.status, r.read().decode()
        except HTTPError as e:
            return e.code, e.read().decode()

    def test_health_ok(self):
        status, body = self._get("/v1/health")
        assert status == 200
        assert json.loads(body)["status"] == "ok"

    def test_root_ok(self):
        status, body = self._get("/v1/")
        assert status == 200
        assert json.loads(body)["service"] == "colab-agent"

    def test_metrics_ok(self):
        status, body = self._get("/v1/metrics")
        assert status == 200

    def test_gdpr_export_ok(self):
        status, body = self._post("/v1/gdpr/export/load-test-tenant")
        assert status == 200
        data = json.loads(body)
        assert data["tenant_id"] == "load-test-tenant"

    def test_gdpr_delete_ok(self):
        status, body = self._post("/v1/gdpr/delete/load-test-tenant")
        assert status == 200
        data = json.loads(body)
        assert "deleted_jobs" in data

    def test_concurrent_requests(self):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(self._get, "/v1/health") for _ in range(8)]
            results = [f.result() for f in futs]
        ok = sum(1 for s, _ in results if s == 200)
        assert ok >= 1, f"Expected at least 1 accepted, got {ok}"

    def test_rate_limiter_burst(self):
        start = time.time()
        results = [self._get("/v1/health") for _ in range(20)]
        elapsed = time.time() - start
        ok = sum(1 for s, _ in results if s == 200)
        blocked = sum(1 for s, _ in results if s == 429)
        print(f"  Rate limit: {ok} ok, {blocked} blocked in {elapsed:.2f}s")
        assert ok >= 15, f"Expected at least 15 accepted (burst=100), got {ok}"

    def test_websocket_ping(self):
        import asyncio
        async def _ws():
            import websockets
            async with websockets.connect(f"ws://127.0.0.1:{self.port}/v1/ws/load-test", ping_interval=None) as ws:
                await ws.send("ping")
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
                assert json.loads(resp)["type"] == "pong"
        asyncio.run(_ws())

    def test_websocket_event_stream(self):
        import asyncio
        async def _ws():
            import websockets
            async with websockets.connect(f"ws://127.0.0.1:{self.port}/v1/ws/load-event-job", ping_interval=None) as ws:
                await ws.send("ping")
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
                data = json.loads(resp)
                assert data["type"] == "pong"
        asyncio.run(_ws())

    def test_websocket_concurrent(self):
        import asyncio
        async def _one(n: int):
            import websockets
            async with websockets.connect(f"ws://127.0.0.1:{self.port}/v1/ws/conc-{n}", ping_interval=None) as ws:
                await ws.send("ping")
                resp = await asyncio.wait_for(ws.recv(), timeout=3)
                assert json.loads(resp)["type"] == "pong"
        async def _run():
            await asyncio.gather(*[_one(i) for i in range(10)])
        asyncio.run(_run())

    def test_legacy_routes(self):
        status, _ = self._get("/health")
        assert status == 200
        time.sleep(0.05)
        status, _ = self._get("/")
        assert status == 200
        time.sleep(0.05)
        status, _ = self._get("/metrics")
        assert status == 200
        time.sleep(0.05)
        status, _ = self._post("/gdpr/export/legacy-tenant")
        assert status == 200

    def test_auth_when_configured(self):
        import os, importlib
        os.environ["COLAB_AGENT_API_KEY"] = "load-test-key"
        from agent import api
        importlib.reload(api)
        time.sleep(0.5)
        # Unauthed
        status, _ = self._get("/v1/budget")
        assert status == 401
        # Authed
        req = Request(f"http://127.0.0.1:{self.port}/v1/budget")
        req.add_header("Authorization", "Bearer load-test-key")
        try:
            r = urlopen(req, timeout=5)
            assert r.status == 200
        except HTTPError as e:
            assert False, f"Expected 200, got {e.code}"
        finally:
            os.environ.pop("COLAB_AGENT_API_KEY", None)
            importlib.reload(api)
