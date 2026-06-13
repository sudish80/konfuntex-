"""
Agent HTTP service.

Two modes:
  1. ``python -m agent.service`` — stdlib ``http.server`` (zero deps).
  2. ``python -m agent.service --fastapi`` — FastAPI via uvicorn (requires
     ``pip install fastapi uvicorn``).

The FastAPI mode is recommended for production.
"""

import os
import signal
import sys
import logging
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from agent.core import health_endpoint, metrics_endpoint, budget_endpoint
from agent.observability import setup_json_logging

logger = logging.getLogger(__name__)
PORT = int(os.environ.get("PORT", "8080"))

_JSON_HEADERS = [("Content-Type", "application/json; charset=utf-8")]
_TEXT_HEADERS = [("Content-Type", "text/plain; charset=utf-8")]


class AgentHandler(BaseHTTPRequestHandler):
    """Fallback HTTP handler (stdlib). Use FastAPI mode for production."""

    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                self._json(health_endpoint())
            elif self.path == "/metrics":
                self._text(metrics_endpoint())
            elif self.path == "/budget":
                self._json(budget_endpoint())
            elif self.path == "/":
                self._json({"service": "colab-agent", "version": "1.0.0"})
            else:
                self._json({"error": "not found"}, status=404)
        except Exception:
            logger.exception("Unhandled error serving %s", self.path)
            self._json({"error": "internal server error"}, status=500)

    def _json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        for k, v in _JSON_HEADERS:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, text: str, status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        for k, v in _TEXT_HEADERS:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s — %s", self.client_address[0], fmt % args)


def _serve_stdlib() -> None:
    """Start stdlib HTTP server (blocking)."""
    setup_json_logging(logger)
    server = HTTPServer(("0.0.0.0", PORT), AgentHandler)

    def _stop(signum: int, frame: object) -> None:
        logger.info("Signal %d received, shutting down...", signum)
        server.shutdown()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    logger.info("stdlib HTTP server listening on 0.0.0.0:%d", PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        logger.info("Server stopped")


def _serve_fastapi() -> None:
    """Start FastAPI via uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError:
        logger.error("fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    from agent.api import app

    setup_json_logging(logger)
    logger.info("FastAPI server listening on 0.0.0.0:%d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


def serve() -> None:
    """Start the agent HTTP service.

    Uses FastAPI if ``--fastapi`` is in ``sys.argv``, otherwise the
    stdlib ``http.server`` fallback.
    """
    if "--fastapi" in sys.argv:
        _serve_fastapi()
    else:
        _serve_stdlib()


if __name__ == "__main__":
    serve()
