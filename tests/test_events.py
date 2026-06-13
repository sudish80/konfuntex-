"""Tests for EventBus, WebSocketEventPlugin, and WebSocket endpoint."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from starlette.websockets import WebSocketDisconnect


class TestEventBus:
    def test_publish_and_poll(self):
        from agent.events import EventBus
        bus = EventBus()
        bus.publish("job-1", {"type": "step.started", "step_id": 1})
        bus.publish("job-1", {"type": "step.completed", "step_id": 1})
        events, idx = bus.poll("job-1")
        assert len(events) == 2
        assert events[0]["type"] == "step.started"
        assert events[1]["type"] == "step.completed"
        assert idx == 2

    def test_poll_since_index(self):
        from agent.events import EventBus
        bus = EventBus()
        bus.publish("job-1", {"type": "a"})
        bus.publish("job-1", {"type": "b"})
        bus.publish("job-1", {"type": "c"})
        events, idx = bus.poll("job-1", since_index=1)
        assert len(events) == 2
        assert events[0]["type"] == "b"
        assert idx == 3

    def test_unknown_job_returns_empty(self):
        from agent.events import EventBus
        bus = EventBus()
        events, idx = bus.poll("nonexistent")
        assert events == []
        assert idx == 0

    def test_has_events(self):
        from agent.events import EventBus
        bus = EventBus()
        assert not bus.has_events("j1")
        bus.publish("j1", {"type": "x"})
        assert bus.has_events("j1")

    def test_events_have_timestamp(self):
        from agent.events import EventBus
        bus = EventBus()
        bus.publish("j1", {"type": "x"})
        events, _ = bus.poll("j1")
        assert "_ts" in events[0]

    def test_thread_safety(self):
        from agent.events import EventBus
        import threading
        bus = EventBus()
        errors = []
        def publisher():
            for i in range(100):
                try:
                    bus.publish("j1", {"type": "t", "i": i})
                except Exception as e:
                    errors.append(e)
        threads = [threading.Thread(target=publisher) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        events, idx = bus.poll("j1")
        assert idx == 400

    def test_isolated_jobs(self):
        from agent.events import EventBus
        bus = EventBus()
        bus.publish("j1", {"type": "a"})
        bus.publish("j2", {"type": "b"})
        e1, _ = bus.poll("j1")
        e2, _ = bus.poll("j2")
        assert len(e1) == 1 and e1[0]["type"] == "a"
        assert len(e2) == 1 and e2[0]["type"] == "b"
        assert not bus.has_events("j3")


class TestWebSocketEventPlugin:
    def test_before_step_publishes(self):
        from agent.events import WebSocketEventPlugin, get_event_bus
        bus = get_event_bus()
        bus.publish = MagicMock()
        p = WebSocketEventPlugin()
        step = {"id": 1, "action": "train", "description": "Training"}
        context = {"job_id": "j1"}
        p.before_step(step, context)
        bus.publish.assert_called_once()
        args = bus.publish.call_args[0]
        assert args[0] == "j1"
        assert args[1]["type"] == "step.started"

    def test_before_step_noop_without_job_id(self):
        from agent.events import WebSocketEventPlugin, get_event_bus
        bus = get_event_bus()
        bus.publish = MagicMock()
        p = WebSocketEventPlugin()
        p.before_step({"id": 1}, {})
        bus.publish.assert_not_called()

    def test_after_step_publishes(self):
        from agent.events import WebSocketEventPlugin, get_event_bus
        bus = get_event_bus()
        bus.publish = MagicMock()
        p = WebSocketEventPlugin()
        p.after_step({"id": 1}, {"status": "success"}, {"job_id": "j1"})
        bus.publish.assert_called_once()
        args = bus.publish.call_args[0]
        assert args[1]["type"] == "step.completed"
        assert args[1]["status"] == "success"

    def test_on_error_publishes(self):
        from agent.events import WebSocketEventPlugin, get_event_bus
        bus = get_event_bus()
        bus.publish = MagicMock()
        p = WebSocketEventPlugin()
        p.on_error({"id": 1}, "GPU OOM", {"job_id": "j1"})
        bus.publish.assert_called_once()
        args = bus.publish.call_args[0]
        assert args[1]["type"] == "step.error"
        assert "GPU OOM" in args[1]["error"]

    def test_on_complete_publishes(self):
        from agent.events import WebSocketEventPlugin, get_event_bus
        bus = get_event_bus()
        bus.publish = MagicMock()
        p = WebSocketEventPlugin()
        p.on_complete({"status": "completed", "job_id": "j1", "summary": "Done"}, {})
        bus.publish.assert_called_once()
        args = bus.publish.call_args[0]
        assert args[1]["type"] == "job.completed"

    def test_on_summary_publishes(self):
        from agent.events import WebSocketEventPlugin, get_event_bus
        bus = get_event_bus()
        bus.publish = MagicMock()
        p = WebSocketEventPlugin()
        p.on_summary("Training complete", {"job_id": "j1"})
        bus.publish.assert_called_once()
        args = bus.publish.call_args[0]
        assert args[1]["type"] == "summary"

    def test_auto_registered(self):
        from agent.plugin import get_registry
        reg = get_registry()
        names = [p["name"] for p in reg.list_registered()]
        assert "websocket_events" in names


class TestWebSocketEndpoint:
    @pytest.fixture(autouse=True)
    def _reload_api(self):
        import importlib
        from agent import api
        importlib.reload(api)
    @pytest.mark.asyncio
    @patch("agent.api.get_event_bus")
    async def test_sends_events(self, mock_get_bus):
        import asyncio
        from agent.api import ws_job_events
        from agent.events import EventBus

        bus = EventBus()
        bus.publish("j1", {"type": "step.started"})
        mock_get_bus.return_value = bus

        ws = AsyncMock()
        ws.query_params = {}
        ws.receive_text = AsyncMock(side_effect=[asyncio.TimeoutError("poll"), WebSocketDisconnect()])

        await ws_job_events(ws, "j1")

        ws.send_json.assert_called_once()
        args = ws.send_json.call_args[0][0]
        assert args["type"] == "step.started"

    @pytest.mark.asyncio
    @patch("agent.api.get_event_bus")
    async def test_ping_pong(self, mock_get_bus):
        from agent.api import ws_job_events
        from agent.events import EventBus

        bus = EventBus()
        mock_get_bus.return_value = bus

        ws = AsyncMock()
        ws.query_params = {}
        ws.receive_text = AsyncMock(side_effect=["ping", WebSocketDisconnect()])

        await ws_job_events(ws, "j1")

        ws.send_json.assert_called_with({"type": "pong"})

    @pytest.mark.asyncio
    @patch("agent.api.get_event_bus")
    async def test_disconnect_graceful(self, mock_get_bus):
        from agent.api import ws_job_events
        from agent.events import EventBus

        bus = EventBus()
        mock_get_bus.return_value = bus

        ws = AsyncMock()
        ws.query_params = {}
        ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect())

        await ws_job_events(ws, "j1")

    @pytest.mark.asyncio
    async def test_default_event_bus_used(self):
        from agent.api import ws_job_events
        ws = AsyncMock()
        ws.query_params = {}
        ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect())

        await ws_job_events(ws, "nobody")
