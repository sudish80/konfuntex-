"""
Colab module — All Colab-related functionality for the agent.

Submodules:
  executor          ColabRunner: notebook creation, cell execution, runtime detection
  runtime           RuntimeManager: runtime switching and tier management
  infrastructure    Phase 1 core infrastructure (NotebookManager, CheckpointManager, etc.)
  setup             Setup & configuration (SecretsManager, DriveAuth, LogManager)
  manager           ColabManager: basic notebook/runtime code generation
  local_kernel      LocalIPythonRunner: persistent local kernel
  automation        ColabAutomation: Playwright-based browser automation
  drive_sync        DriveSyncDaemon: background Drive sync with checkpoint versioning
  resumer           ColabResumer: auto-resume after runtime restart
  bridge            ColabBridge: WebSocket bridge between Colab and FastAPI
  async_bridge      AsyncColabBridge: async variant with EventBus integration
  enterprise        ColabEnterprise: Colab Enterprise API support
  async_enterprise  AsyncColabEnterprise: async wrapper for ColabEnterprise
  remote_executor   RemoteColabExecutor: fully automated Colab via Playwright
  async_remote_executor  AsyncRemoteColabExecutor: async wrapper for RemoteColabExecutor
"""

import importlib.util

from colab.executor import ColabRunner, ColabRuntimeInfo, torch_available
from colab.runtime import RuntimeManager, estimate_needed_vram
from colab.setup import SecretsManager, DriveAuth, LogManager, setup_colab_environment
from colab.manager import ColabManager
from colab.local_kernel import LocalIPythonRunner
from colab.remote_executor import RemoteColabExecutor

__all__ = [
    "ColabRunner",
    "ColabRuntimeInfo",
    "torch_available",
    "RuntimeManager",
    "estimate_needed_vram",
    "SecretsManager",
    "DriveAuth",
    "LogManager",
    "setup_colab_environment",
    "ColabManager",
    "LocalIPythonRunner",
    "RemoteColabExecutor",
]


def _lazy_import(name: str, path: str):
    """Lazy-import a submodule. Returns None if unavailable."""
    try:
        spec = importlib.util.find_spec(path.replace("/", ".").replace("\\", "."))
        if spec is None:
            return None
        mod = importlib.import_module(path.replace("/", ".").replace("\\", "."))
        cls = getattr(mod, name, None)
        if cls is not None:
            __all__.append(name)
        return cls
    except (ImportError, AttributeError):
        return None


# Advanced features (lazy-loaded)
ColabAutomation = _lazy_import("ColabAutomation", "colab/automation")
DriveSyncDaemon = _lazy_import("DriveSyncDaemon", "colab/drive_sync")
ColabResumer = _lazy_import("ColabResumer", "colab/resumer")
ColabBridge = _lazy_import("ColabBridge", "colab/bridge")
AsyncColabBridge = _lazy_import("AsyncColabBridge", "colab/async_bridge")
ColabEnterprise = _lazy_import("ColabEnterprise", "colab/enterprise")
AsyncColabEnterprise = _lazy_import("AsyncColabEnterprise", "colab/async_enterprise")
AsyncRemoteColabExecutor = _lazy_import(
    "AsyncRemoteColabExecutor", "colab/async_remote_executor",
)
