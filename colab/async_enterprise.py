"""
Async Colab Enterprise — Async wrapper for ColabEnterprise.

Provides async methods that mirror ColabEnterprise using
asyncio.to_thread() for non-blocking I/O.
"""

import asyncio
import logging
from typing import Optional

from colab.enterprise import ColabEnterprise as _SyncEnterprise

logger = logging.getLogger(__name__)


class AsyncColabEnterprise:
    """
    Async wrapper around ColabEnterprise.

    All public methods are async and safe to call from asyncio event loops.
    """

    def __init__(self, project: Optional[str] = None, location: str = "us-central1",
                 credentials_path: Optional[str] = None):
        self._sync = _SyncEnterprise(
            project=project,
            location=location,
            credentials_path=credentials_path,
        )

    async def create_runtime(self, runtime_spec: str = "T4",
                              display_name: str = "colab-agent-runtime",
                              idle_timeout: int = 1800,
                              timeout: int = 600) -> dict:
        return await asyncio.to_thread(
            self._sync.create_runtime, runtime_spec, display_name, idle_timeout, timeout,
        )

    async def get_runtime(self, runtime_name: str) -> dict:
        return await asyncio.to_thread(self._sync.get_runtime, runtime_name)

    async def delete_runtime(self, runtime_name: str) -> dict:
        return await asyncio.to_thread(self._sync.delete_runtime, runtime_name)

    async def list_runtimes(self) -> list[dict]:
        return await asyncio.to_thread(self._sync.list_runtimes)

    async def execute_code(self, runtime_name: str, code: str, timeout: int = 300) -> dict:
        return await asyncio.to_thread(
            self._sync.execute_code, runtime_name, code, timeout,
        )

    def is_available(self) -> bool:
        return self._sync.is_available()

    def list_specs(self) -> list[dict]:
        return self._sync.list_specs()

    @staticmethod
    def best_fit_spec(required_vram_gb: float) -> str:
        return _SyncEnterprise.best_fit_spec(required_vram_gb)

    def generate_setup_code(self) -> str:
        return self._sync.generate_setup_code()
