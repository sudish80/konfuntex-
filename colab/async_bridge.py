"""
Async Colab Bridge — Async variant of ColabBridge with EventBus integration.

Uses asyncio for non-blocking WebSocket connections and integrates
with the agent's EventBus so Colab events appear on the WebSocket endpoint.

Usage:
    bridge = AsyncColabBridge(token="...")
    await bridge.start(job_id="abc123")
    await bridge.record_loss(0.42, epoch=1)
    await bridge.stop()
"""

import os
import json
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AsyncColabBridge:
    """
    Async WebSocket bridge that pushes Colab events to the agent's EventBus.

    Uses asyncio for non-blocking I/O and integrates with the shared EventBus
    so events appear on the /v1/ws/{job_id} WebSocket endpoint.
    """

    MAX_BUFFER = 2000

    def __init__(self, server_url: str = "", token: str = ""):
        self.server_url = server_url or os.environ.get("COLAB_AGENT_WS_URL",
                                                        "ws://localhost:8080/ws/v1/colab")
        self.token = token or os.environ.get("COLAB_AGENT_API_KEY", "")

        self._ws = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._buffer: list[dict] = []
        self._connected = False
        self._event_count = 0
        self._job_id = ""
        self._event_bus = None

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self, job_id: str = "", interval: float = 2.0):
        """Start the async bridge as a background asyncio.Task."""
        self._job_id = job_id
        self._stop_event.clear()

        # Try to integrate with EventBus
        try:
            from agent.events import get_event_bus
            self._event_bus = get_event_bus()
        except ImportError:
            self._event_bus = None

        self._task = asyncio.create_task(
            self._run(interval),
            name=f"async-bridge-{job_id[:8] if job_id else 'anon'}",
        )
        logger.info(f"AsyncColabBridge started (job={job_id})")

    async def stop(self, timeout: float = 5):
        """Stop the bridge."""
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        await self._disconnect()
        logger.info("AsyncColabBridge stopped")

    # ------------------------------------------------------------------ #
    #  Event Recording
    # ------------------------------------------------------------------ #

    async def record_log(self, message: str, level: str = "info"):
        self._buffer_event({"type": "log", "level": level, "message": message})

    async def record_metric(self, name: str, value: float, step: int = 0):
        self._buffer_event({"type": "metric", "name": name, "value": value, "step": step})

    async def record_loss(self, loss: float, epoch: float, step: int = 0):
        await self.record_metric("loss", loss, step)
        await self.record_metric("epoch", epoch, step)

    async def record_error(self, error: str):
        self._buffer_event({"type": "error", "message": error})

    async def record_status(self, status: str, detail: str = ""):
        self._buffer_event({"type": "status", "status": status, "detail": detail})

    async def record_resource(self, vram_used: float = 0, vram_total: float = 0,
                              ram_pct: float = 0, gpu_name: str = ""):
        self._buffer_event({
            "type": "resource", "vram_used": vram_used, "vram_total": vram_total,
            "ram_pct": ram_pct, "gpu_name": gpu_name,
        })

    async def flush(self):
        await self._send_buffer()

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _buffer_event(self, event: dict):
        """Add event to buffer; optionally push to EventBus immediately."""
        import time
        event["ts"] = __import__("datetime").datetime.now().isoformat()
        self._buffer.append(event)

        # Also publish to EventBus for WebSocket endpoint
        if self._event_bus and self._job_id:
            self._event_bus.publish(self._job_id, event)

        if len(self._buffer) > self.MAX_BUFFER:
            self._buffer.pop(0)

    async def _run(self, interval: float):
        reconnect_delay = 1.0
        while not self._stop_event.is_set():
            try:
                await self._connect()
                reconnect_delay = 1.0
                while not self._stop_event.is_set():
                    await self._send_buffer()
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(), timeout=interval
                        )
                    except asyncio.TimeoutError:
                        pass
            except Exception as e:
                logger.warning(f"Async bridge error: {e}")
                self._connected = False
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    async def _connect(self):
        if self._connected and self._ws:
            return
        try:
            import websocket as ws_lib
            url = self.server_url
            if self.token and "token=" not in url:
                sep = "&" if "?" in url else "?"
                url += f"{sep}token={self.token}"
            loop = asyncio.get_event_loop()
            self._ws = await loop.run_in_executor(None, lambda: ws_lib.WebSocket())
            await loop.run_in_executor(None, lambda: self._ws.connect(url, timeout=10))
            self._connected = True
            logger.info("Async bridge connected")
        except Exception as e:
            self._ws = None
            self._connected = False
            raise

    async def _disconnect(self):
        ws, self._ws = self._ws, None
        self._connected = False
        if ws:
            try:
                ws.close()
            except Exception:
                pass

    async def _send_buffer(self):
        if not self._connected or not self._ws or not self._buffer:
            return
        events = list(self._buffer)
        self._buffer.clear()
        try:
            loop = asyncio.get_event_loop()
            for event in events:
                await loop.run_in_executor(None, lambda: self._ws.send(json.dumps(event)))
                self._event_count += 1
        except Exception as e:
            logger.warning(f"Async send failed: {e}")
            self._connected = False
            self._buffer = events + self._buffer
