"""
Colab WebSocket Bridge — Thread-safe, production-hardened bridge between Colab
execution and the FastAPI backend.

Two modes:
  1. Colab-side: background thread connecting out to a WebSocket server
  2. Agent-side: EventBus integration for receiving Colab events

Uses:
  - RLock for all mutable state
  - Exponential backoff reconnection
  - Bounded buffer (never grows unbounded)
  - Graceful degradation (no crash on send failure)
  - Re-buffering on disconnect (data loss prevention)
"""

import os
import json
import time
import logging
import threading
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ColabBridge:
    """
    Bidirectional bridge for real-time Colab events.

    Colab-side usage:
        bridge = ColabBridge(server_url="ws://host:8080/ws/v1/colab", token="...")
        bridge.start(job_id="abc123")
        bridge.record_loss(0.42, epoch=1)
        bridge.record_log("Epoch complete")
        bridge.stop()

    Agent-side (receiving):
        bridge = ColabBridge()
        bridge.connect_to_colab(job_id="abc123", timeout=300)
        events = bridge.poll_events()
    """

    MAX_BUFFER = 2000
    BASE_RECONNECT_DELAY = 1.0
    MAX_RECONNECT_DELAY = 60.0

    def __init__(self, server_url: str = "", token: str = ""):
        self.server_url = server_url or os.environ.get("COLAB_AGENT_WS_URL", "ws://localhost:8080/ws/v1/colab")
        self.token = token or os.environ.get("COLAB_AGENT_API_KEY", "")

        self._lock = threading.RLock()
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._buffer: list[dict] = []
        self._connected = False
        self._event_count = 0
        self._job_id = ""

    # ------------------------------------------------------------------ #
    #  Colab-side: Start / Stop
    # ------------------------------------------------------------------ #

    def start(self, job_id: str = "", interval: float = 2.0):
        """Start the bridge in a background thread."""
        if not isinstance(job_id, str):
            raise TypeError(f"job_id must be str, got {type(job_id)}")
        if not isinstance(interval, (int, float)) or interval < 0.5:
            raise TypeError(f"interval must be >= 0.5, got {interval}")

        with self._lock:
            if self._thread and self._thread.is_alive():
                logger.warning("Bridge already running, not starting again")
                return
            self._job_id = job_id

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_colab_side,
            args=(interval,),
            daemon=True,
            name=f"cbridge-{job_id[:8] if job_id else 'anon'}",
        )
        self._thread.start()
        logger.info(f"ColabBridge started (server={self.server_url}, job={job_id})")
        return self

    def stop(self, timeout: float = 5):
        """Stop the bridge. Idempotent."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._disconnect()
        logger.info("ColabBridge stopped")
        return self

    # ------------------------------------------------------------------ #
    #  Event Recording (thread-safe, bounded buffer)
    # ------------------------------------------------------------------ #

    def record_log(self, message: str, level: str = "info"):
        if not isinstance(message, str):
            return
        self._buffer_event({"type": "log", "level": level, "message": message,
                            "ts": datetime.now(timezone.utc).isoformat()})

    def record_metric(self, name: str, value: float, step: int = 0):
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            return
        self._buffer_event({"type": "metric", "name": name, "value": value, "step": step,
                            "ts": datetime.now(timezone.utc).isoformat()})

    def record_loss(self, loss: float, epoch: float, step: int = 0):
        self.record_metric("loss", loss, step)
        self.record_metric("epoch", epoch, step)

    def record_error(self, error: str):
        if not isinstance(error, str):
            return
        self._buffer_event({"type": "error", "message": error,
                            "ts": datetime.now(timezone.utc).isoformat()})

    def record_status(self, status: str, detail: str = ""):
        if not isinstance(status, str):
            return
        self._buffer_event({"type": "status", "status": status, "detail": detail,
                            "ts": datetime.now(timezone.utc).isoformat()})

    def record_resource(self, vram_used: float = 0, vram_total: float = 0,
                        ram_pct: float = 0, gpu_name: str = ""):
        self._buffer_event({"type": "resource", "vram_used": vram_used,
                            "vram_total": vram_total, "ram_pct": ram_pct,
                            "gpu_name": gpu_name,
                            "ts": datetime.now(timezone.utc).isoformat()})

    def flush(self):
        """Immediately send all buffered events."""
        self._send_buffer()

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def event_count(self) -> int:
        with self._lock:
            return self._event_count

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._buffer)

    # ------------------------------------------------------------------ #
    #  Agent-side: receive events
    # ------------------------------------------------------------------ #

    def connect_to_colab(self, job_id: str = "", timeout: int = 300) -> bool:
        """Placeholder for agent-side connection. Actual WS handling is in agent/service.py."""
        logger.warning("Agent-side connect_to_colab is a stub; use the FastAPI WebSocket endpoint")
        return False

    def poll_events(self) -> list[dict]:
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
        return events

    # ------------------------------------------------------------------ #
    #  Internal (private)
    # ------------------------------------------------------------------ #

    def _buffer_event(self, event: dict):
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) > self.MAX_BUFFER:
                self._buffer.pop(0)

    def _run_colab_side(self, interval: float):
        reconnect_delay = self.BASE_RECONNECT_DELAY
        while not self._stop_event.is_set():
            try:
                self._connect()
                reconnect_delay = self.BASE_RECONNECT_DELAY
                while not self._stop_event.is_set():
                    self._send_buffer()
                    self._stop_event.wait(timeout=interval)
            except Exception as e:
                if "ConnectionRefused" in str(e) or "10061" in str(e) or "111" in str(e):
                    logger.debug(f"Bridge connection refused (no WS server): {e}")
                else:
                    logger.warning(f"Bridge connection error: {e}")
                with self._lock:
                    self._connected = False
                if self._stop_event.wait(timeout=reconnect_delay):
                    break  # stop requested
                reconnect_delay = min(reconnect_delay * 2, self.MAX_RECONNECT_DELAY)

    def _connect(self):
        with self._lock:
            if self._connected and self._ws:
                return

        try:
            import websocket as ws_lib
        except ImportError:
            logger.warning("websocket-client not installed (pip install websocket-client)")
            return

        try:
            url = self.server_url
            if self.token and "token=" not in url:
                sep = "&" if "?" in url else "?"
                url += f"{sep}token={self.token}"

            w = ws_lib.WebSocket()
            w.connect(url, timeout=10)

            with self._lock:
                self._ws = w
                self._connected = True

            w.send(json.dumps({
                "type": "handshake",
                "job_id": self._job_id,
                "client": "colab-bridge",
                "ts": datetime.now(timezone.utc).isoformat(),
            }))
            logger.info("Bridge WebSocket connected")
        except Exception:
            with self._lock:
                self._ws = None
                self._connected = False
            raise

    def _disconnect(self):
        with self._lock:
            ws = self._ws
            self._ws = None
            self._connected = False
        if ws:
            try:
                ws.close()
            except Exception:
                pass

    def _send_buffer(self):
        with self._lock:
            if not self._connected or not self._ws:
                return
            if not self._buffer:
                return
            events = list(self._buffer)
            self._buffer.clear()

        try:
            for event in events:
                self._ws.send(json.dumps(event))
                with self._lock:
                    self._event_count += 1
        except Exception as e:
            logger.warning(f"Failed to send events: {e}")
            with self._lock:
                self._connected = False
                self._buffer = events + self._buffer

    # ------------------------------------------------------------------ #
    #  Code generation
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_bridge_code(server_url: str = "", token: str = "",
                              job_id: str = "") -> str:
        url = server_url or "ws://localhost:8080/ws/v1/colab"
        return f'''
import json, time, threading, os, sys
try:
    import websocket
except ImportError:
    !pip install -q websocket-client
    import websocket
URL = "{url}"
TOKEN = "{token}"
JID = "{job_id or os.environ.get("COLAB_AGENT_JOB_ID", "")}"
class _Bridge:
    def __init__(self):
        self.ws = None; self.ok = False; self._buf = []; self._lk = threading.Lock()
    def connect(self):
        u = URL + (f"?token={{TOKEN}}" if TOKEN else "")
        try:
            self.ws = websocket.WebSocket(); self.ws.connect(u, timeout=10); self.ok = True
            self.ws.send(json.dumps({{"type":"handshake","job_id":JID,"client":"colab-bridge"}}))
            print(f"Bridge: {{u}}")
        except Exception as e: print(f"Bridge fail: {{e}}")
    def send(self, t, d):
        d["type"]=t; d["ts"]=__import__("datetime").datetime.now().isoformat()
        with self._lk: self._buf.append(d)
    def flush(self):
        if not self.ok or not self.ws: return
        with self._lk: evs=list(self._buf); self._buf.clear()
        for e in evs:
            try: self.ws.send(json.dumps(e))
            except: self.ok=False; self._buf=evs+self._buf; break
    def close(self):
        if self.ws:
            try: self.ws.close()
            except: pass
_b = _Bridge(); _b.connect()
def _loop():
    while True: _b.flush(); time.sleep(2)
threading.Thread(target=_loop, daemon=True).start()
print("Bridge running")
'''
