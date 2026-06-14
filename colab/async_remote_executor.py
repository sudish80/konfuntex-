"""
Async Remote Colab Executor — async wrapper for RemoteColabExecutor.

Delegates all Playwright operations to a thread pool via asyncio.to_thread,
keeping the event loop responsive during browser I/O.

Usage:
    async with AsyncRemoteColabExecutor() as executor:
        await executor.connect()
        result = await executor.execute("print('hello')")

"""

import asyncio
import logging
from typing import Optional

from colab.remote_executor import RemoteColabExecutor, USER_DATA_DIR as _DEFAULT_USER_DATA_DIR

logger = logging.getLogger(__name__)


class AsyncRemoteColabExecutor:
    """
    Async wrapper around RemoteColabExecutor.

    All blocking Playwright calls run in a thread pool via asyncio.to_thread,
    preserving event loop responsiveness.
    """

    def __init__(self, headless: bool = True, user_data_dir: Optional[str] = None,
                 browser_path: Optional[str] = None):
        self._sync = RemoteColabExecutor(
            headless=headless,
            user_data_dir=user_data_dir or _DEFAULT_USER_DATA_DIR,
            browser_path=browser_path,
        )
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        return self._sync.available

    @property
    def connected(self) -> bool:
        return self._sync.connected

    @property
    def session_path(self) -> str:
        return self._sync.session_path

    # ------------------------------------------------------------------ #
    #  Auth
    # ------------------------------------------------------------------ #

    async def login_once(self) -> bool:
        return await asyncio.to_thread(self._sync.login_once)

    # ------------------------------------------------------------------ #
    #  Connection
    # ------------------------------------------------------------------ #

    async def connect(self, notebook_url: str = "",
                      timeout: int = 180) -> dict:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync.connect, notebook_url, timeout,
            )

    async def disconnect(self):
        async with self._lock:
            await asyncio.to_thread(self._sync.disconnect)

    # ------------------------------------------------------------------ #
    #  Execution
    # ------------------------------------------------------------------ #

    async def execute(self, code: str, timeout: int = 600,
                      cell_index: Optional[int] = None) -> dict:
        async with self._lock:
            return await asyncio.to_thread(
                self._sync.execute, code, timeout, cell_index,
            )

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    async def capture_screenshot(self) -> Optional[str]:
        return await asyncio.to_thread(self._sync.capture_screenshot)

    async def get_runtime_info(self) -> dict:
        return await asyncio.to_thread(self._sync.get_runtime_info)

    async def status(self) -> dict:
        return await asyncio.to_thread(self._sync.status)

    # ------------------------------------------------------------------ #
    #  Context manager
    # ------------------------------------------------------------------ #

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.disconnect()
        return False
