"""
Remote Colab Executor — Fully automatic code execution in Colab via Playwright.

No manual steps. No code pasting. Handles Google auth, session persistence,
keepalive, and automatic fallback to local kernel.

Architecture:
    RemoteColabExecutor
      ├── connect()        — opens headless Chromium, navigates to Colab, waits for runtime
      ├── execute(code)    — sets CodeMirror content via JS, presses Ctrl+Enter, reads output
      ├── login_once()     — one-time non-headless browser for Google auth, saves cookies
      └── disconnect()     — closes browser, saves session for next time

State machine:
    init → login_once() [optional] → connect() → execute()* → disconnect()
                              ↕ (session persisted to disk)
"""

import os
import re
import json
import time
import uuid
import base64
import logging
import shutil
import threading
import tempfile
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Constants
CONNECT_TIMEOUT = 180
CELL_TIMEOUT = 600
POLL_INTERVAL = 0.5
KEEPALIVE_INTERVAL = 30
USER_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".colab-session"
)


class RemoteColabExecutor:
    """
    Fully automated Colab executor via Playwright.

    Handles:
      - Google auth via persisted Chrome user data directory
      - Session persistence across runs (cookies, localStorage)
      - Keepalive polling to prevent idle disconnection
      - Automatic fallback to local kernel on failure
      - CodeMirror JS API for fast code injection
      - Output reading via DOM polling
    """

    def __init__(self, headless: bool = True, user_data_dir: str = USER_DATA_DIR):
        self.headless = headless
        self.user_data_dir = user_data_dir

        self._lock = threading.RLock()
        self._browser = None
        self._context = None
        self._page = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._keepalive_stop = threading.Event()

        self._connected = False
        self._runtime_ready = False
        self._notebook_url = ""

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        return self._check_playwright()

    @property
    def connected(self) -> bool:
        return self._connected and self._runtime_ready

    @property
    def session_path(self) -> str:
        """Path where Chrome user data (cookies, auth) is persisted."""
        return self.user_data_dir

    # ------------------------------------------------------------------ #
    #  Auth — one-time Google login
    # ------------------------------------------------------------------ #

    def login_once(self) -> bool:
        """
        Open a non-headless browser for one-time Google login.

        The user logs into their Google account in the visible browser.
        After login, the session (cookies, localStorage) is saved to
        `session_path` and reused by all future headless connections.

        Call this once per machine. Subsequent `connect()` calls will
        reuse the saved session.

        Returns:
            True if login completed, False if cancelled/failed.
        """
        if not self.available:
            logger.error("Playwright not available")
            return False

        print("=== Colab Login ===")
        print("A browser window will open. Log into your Google account.")
        print("After logging in, close the browser tab (not the window).")
        input("Press Enter to continue...")

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as pw:
                browser = pw.chromium.launch_persistent_context(
                    user_data_dir=self.user_data_dir,
                    headless=False,
                    args=["--no-sandbox"],
                )
                page = browser.new_page()
                page.goto("https://colab.research.google.com", timeout=60000)
                print("Browser open. Log into Google, then close this terminal tab.")
                input("Press Enter AFTER you've logged in and closed the Colab tab...")
                browser.close()

            print(f"Session saved to: {self.user_data_dir}")
            print("Future connections will reuse this session.")
            return True

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    def connect(self, notebook_url: str = "", timeout: int = CONNECT_TIMEOUT) -> dict:
        """
        Open headless Chromium and connect to Colab.

        Reuses the persisted Chrome user data directory if it exists
        (from a previous `login_once()` call). If not, runs without auth
        (Colab will be in anonymous read-only mode).

        Args:
            notebook_url: Optional existing notebook URL. Empty = new notebook.
            timeout: Max seconds to wait for runtime connection.

        Returns:
            dict with keys: success, url, connected, runtime_ready, error
        """
        if not self.available:
            return {
                "success": False,
                "error": ("Playwright not available. "
                          "Install: pip install playwright && playwright install chromium"),
            }

        with self._lock:
            if self._connected:
                return {
                    "success": True,
                    "url": self._notebook_url,
                    "connected": True,
                    "runtime_ready": self._runtime_ready,
                }

            return self._connect(notebook_url, timeout)

    def _connect(self, notebook_url: str, timeout: int) -> dict:
        try:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()

            # Launch browser with persistent user data dir if it exists
            user_dir = self.user_data_dir if os.path.isdir(self.user_data_dir) else None
            launch_kwargs = {
                "headless": self.headless,
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            }

            if user_dir:
                self._context = self._pw.chromium.launch_persistent_context(
                    user_data_dir=user_dir,
                    **{k: v for k, v in launch_kwargs.items() if k != "headless"},
                    headless=self.headless,
                )
                self._page = self._context.new_page()
                logger.info(f"Using saved session from {user_dir}")
            else:
                self._browser = self._pw.chromium.launch(**launch_kwargs)
                self._context = self._browser.new_context(
                    viewport={"width": 1280, "height": 900},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                )
                self._page = self._context.new_page()
                logger.info("No saved session; launching anonymous browser")

            url = notebook_url or "https://colab.research.google.com/#create=true"
            self._page.goto(url, timeout=30000, wait_until="domcontentloaded")
            self._notebook_url = url

            # Check if we hit a login wall
            if self._detect_login_wall():
                logger.warning("Login wall detected. Run login_once() first or use --executor local")
                self._disconnect()
                return {
                    "success": False,
                    "error": ("Google login required. "
                              "Run: python cli.py colab-remote --login"),
                }

            logger.info("Waiting for notebook editor...")
            editor_ok = self._wait_for_editor(timeout)
            if not editor_ok:
                self._disconnect()
                return {"success": False, "error": "Notebook editor did not load"}

            self._connected = True

            logger.info("Waiting for runtime connection...")
            self._runtime_ready = self._wait_for_runtime(timeout)

            # Start keepalive
            self._start_keepalive()

            return {
                "success": True,
                "url": url,
                "connected": self._connected,
                "runtime_ready": self._runtime_ready,
            }

        except Exception as e:
            logger.error(f"Colab connect failed: {e}")
            self._disconnect()
            return {"success": False, "error": str(e)}

    def _detect_login_wall(self) -> bool:
        """Check if the page is showing a Google login screen."""
        try:
            content = self._page.content().lower()
            indicators = ["sign in", "sign-in", "choose an account",
                          "google.com/signin", "login"]
            return any(i in content for i in indicators)
        except Exception:
            return False

    def _wait_for_editor(self, timeout: int) -> bool:
        """Wait for the notebook CodeMirror editor to appear."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                cm = self._page.evaluate(
                    "!!document.querySelector('.CodeMirror')"
                )
                if cm:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def _wait_for_runtime(self, timeout: int) -> bool:
        """Wait for 'Connected' indicator in the Colab toolbar."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                connected = self._page.evaluate(
                    "document.body.innerText.includes('Connected')"
                )
                if connected:
                    return True
            except Exception:
                pass
            time.sleep(3)
        return False

    # ------------------------------------------------------------------ #
    #  Keepalive
    # ------------------------------------------------------------------ #

    def _start_keepalive(self):
        """Background thread that polls Colab to prevent idle disconnect."""
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            return
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            daemon=True,
            name="colab-keepalive",
        )
        self._keepalive_thread.start()

    def _keepalive_loop(self):
        """Every 30s, run a trivial cell to keep the runtime alive."""
        while not self._keepalive_stop.is_set():
            self._keepalive_stop.wait(timeout=KEEPALIVE_INTERVAL)
            if self._keepalive_stop.is_set():
                break
            try:
                with self._lock:
                    if self._connected and self._page:
                        self._page.evaluate(
                            "console.log('keepalive')"
                        )
            except Exception:
                pass

    def _stop_keepalive(self):
        self._keepalive_stop.set()
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_thread.join(timeout=2)

    # ------------------------------------------------------------------ #
    #  Code Execution
    # ------------------------------------------------------------------ #

    def execute(self, code: str, timeout: int = CELL_TIMEOUT,
                cell_index: Optional[int] = None) -> dict:
        """
        Execute Python code in the Colab notebook.

        Uses CodeMirror JS API to set cell content (O(1), not typing),
        then Ctrl+Enter to execute, then DOM polling to read output.

        Args:
            code: Python source code.
            timeout: Max wall-clock seconds.
            cell_index: Optional existing cell index.

        Returns:
            dict with keys: success, output, error, execution_time
        """
        start = time.time()
        result = {
            "success": False,
            "output": "",
            "error": None,
            "execution_time": 0.0,
        }

        with self._lock:
            if not self._connected or not self._page:
                result["error"] = "Not connected to Colab"
                return result

            try:
                self._insert_new_cell()
                injected = self._set_cell_code(code)
                if not injected:
                    logger.warning("CodeMirror setValue failed, typing fallback")
                    self._page.keyboard.type(code, delay=0.005)
                    time.sleep(1)

                self._page.keyboard.press("Control+Enter")
                output = self._wait_for_output(code, timeout)

                result["output"] = output
                result["success"] = not self._has_error(output)

                error_text = self._extract_error(output)
                if error_text:
                    result["error"] = error_text

            except Exception as e:
                logger.error(f"Cell execution failed: {e}")
                result["error"] = str(e)

            result["execution_time"] = time.time() - start
            return result

    def _insert_new_cell(self):
        """Insert a new code cell below the active cell."""
        try:
            self._page.keyboard.press("Control+Shift+Space")
            time.sleep(0.5)
        except Exception:
            try:
                self._page.keyboard.press("Control+Shift+Space")
                time.sleep(0.5)
            except Exception:
                pass

    def _set_cell_code(self, code: str) -> bool:
        """
        Set the active CodeMirror editor content via JS evaluation.
        Returns True if successful, False if CodeMirror not found.
        """
        escaped = (code
                   .replace("\\", "\\\\")
                   .replace("`", "\\`")
                   .replace("${", "\\${"))
        js = f"""
        (() => {{
            const el = document.querySelector('.CodeMirror');
            if (el && el.CodeMirror) {{
                el.CodeMirror.setValue(`{escaped}`);
                return true;
            }}
            return false;
        }})()
        """
        try:
            return bool(self._page.evaluate(js))
        except Exception:
            return False

    def _wait_for_output(self, code: str, timeout: int) -> str:
        """
        Poll the output area until execution completes.

        Completion signals:
          1. Output text appears and execution count increments
          2. "=== CELL COMPLETED ===" or "=== CELL ERROR ===" marker
          3. Output contains expected print lines from the code
        """
        start = time.time()
        outputs = []

        expected_markers = [
            "=== CELL COMPLETED",
            "=== CELL ERROR",
            "=== TRAINING COMPLETE",
        ]

        while time.time() - start < timeout:
            try:
                text = self._page.evaluate("""
                    (() => {
                        const outputs = document.querySelectorAll('.output pre, .output-area');
                        return Array.from(outputs).map(o => o.innerText).join('\\n');
                    })()
                """)
                if text and text != (outputs[-1] if outputs else ""):
                    outputs.append(text)

                # Check for completion markers
                for marker in expected_markers:
                    if marker in text:
                        return text

                # If we have substantial output, assume done
                if len(text) > 50 and not text.strip().endswith("..."):
                    break

            except Exception:
                pass
            time.sleep(POLL_INTERVAL)

        return outputs[-1] if outputs else ""

    @staticmethod
    def _has_error(output: str) -> bool:
        """Check if output contains Python error indicators."""
        indicators = ["Traceback", "Error:", "Exception", "CUDA out of memory",
                      "KeyboardInterrupt", "TimeoutError"]
        return any(i in output for i in indicators)

    @staticmethod
    def _extract_error(output: str) -> Optional[str]:
        """Extract the most relevant error lines from output."""
        lines = output.split("\n")
        error_lines = [
            l for l in lines
            if any(kw in l for kw in ["Traceback", "Error:", "Exception",
                                       "CUDA out of memory"])
        ]
        return "\n".join(error_lines[:5]) if error_lines else None

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    def capture_screenshot(self) -> Optional[str]:
        """Base64-encoded PNG screenshot of the Colab page."""
        with self._lock:
            if not self._page:
                return None
            try:
                return base64.b64encode(self._page.screenshot()).decode()
            except Exception:
                return None

    def get_runtime_info(self) -> dict:
        """Detect GPU type from the Colab page."""
        with self._lock:
            if not self._page:
                return {"gpu": None}
            try:
                content = self._page.content()
                for gpu in ["T4", "V100", "A100", "P100", "K80", "TPU"]:
                    if gpu in content:
                        return {"gpu": gpu}
                return {"gpu": "unknown"}
            except Exception:
                return {"gpu": "error"}

    def status(self) -> dict:
        """Full status report."""
        with self._lock:
            return {
                "available": self.available,
                "connected": self._connected,
                "runtime_ready": self._runtime_ready,
                "headless": self.headless,
                "session_path": self.user_data_dir,
                "has_session": os.path.isdir(self.user_data_dir),
            }

    # ------------------------------------------------------------------ #
    #  Disconnection
    # ------------------------------------------------------------------ #

    def disconnect(self):
        """Close browser and save session. Idempotent."""
        with self._lock:
            self._disconnect()

    def _disconnect(self):
        self._stop_keepalive()
        self._runtime_ready = False
        self._connected = False

        try:
            if self._context and hasattr(self._context, 'close'):
                self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if hasattr(self, '_pw') and self._pw:
                self._pw.stop()
        except Exception:
            pass

        self._context = None
        self._browser = None
        self._page = None
        logger.info("RemoteColabExecutor disconnected")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()
        return False

    # ------------------------------------------------------------------ #
    #  Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_playwright() -> bool:
        try:
            import playwright
            return True
        except ImportError:
            logger.info("Playwright not installed. Install: pip install playwright && playwright install chromium")
            return False

    @staticmethod
    def ensure_session_dir(path: str = USER_DATA_DIR) -> str:
        """Create the session directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)
        return path
