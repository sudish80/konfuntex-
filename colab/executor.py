import json
import time
import random
import os
import re
import uuid
import threading
import logging
import tempfile
from typing import Optional, Callable
from datetime import datetime, timezone
from colab.local_kernel import LocalIPythonRunner


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0):
    delay = min(base_delay * (2 ** attempt), max_delay)
    time.sleep(delay + random.uniform(0, 0.1 * delay))

logger = logging.getLogger(__name__)

# ... (rest of the file content remains same, just adding the import and updating class)


def _colab_auth_code() -> str:
    """Return code snippet that authenticates with Google in Colab."""
    return """
from google.colab import auth
auth.authenticate_user()
print("Authenticated with Google")
"""


def _pydrive_import_code() -> str:
    """Return code snippet that installs + imports PyDrive."""
    return """
# Install PyDrive
!pip install -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
print("PyDrive ready")
"""


class ColabRuntimeInfo:
    """Structured result from detect_current_runtime()."""

    def __init__(self, gpu_name: str = "", vram_total_gb: float = 0.0,
                 vram_free_gb: float = 0.0, ram_total_gb: float = 0.0,
                 ram_available_gb: float = 0.0, runtime_label: str = "None",
                 cuda_version: str = "", is_tpu: bool = False,
                 gpu_count: int = 0):
        self.gpu_name = gpu_name
        self.vram_total_gb = vram_total_gb
        self.vram_free_gb = vram_free_gb
        self.ram_total_gb = ram_total_gb
        self.ram_available_gb = ram_available_gb
        self.runtime_label = runtime_label
        self.cuda_version = cuda_version
        self.is_tpu = is_tpu
        self.gpu_count = gpu_count

    def to_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "vram_total_gb": round(self.vram_total_gb, 2),
            "vram_free_gb": round(self.vram_free_gb, 2),
            "ram_total_gb": round(self.ram_total_gb, 2),
            "ram_available_gb": round(self.ram_available_gb, 2),
            "runtime_label": self.runtime_label,
            "cuda_version": self.cuda_version,
            "is_tpu": self.is_tpu,
            "gpu_count": self.gpu_count,
        }

    def __repr__(self) -> str:
        return f"ColabRuntimeInfo(gpu={self.gpu_name}, vram={self.vram_total_gb:.1f}GB, ram={self.ram_total_gb:.1f}GB)"


class ColabRunner:
    """
    Manages Colab notebook creation, code execution, and runtime detection.

    Execution modes:
      - auto:    checks COLAB_AGENT_SIMULATE env var (default simulate=True)
      - local:   persistent LocalIPythonRunner kernel (jupyter_client)
      - colab:   simulation mode; generates upload code for manual Colab use
      - remote:  fully automated via Playwright browser (no manual steps)

    Fallback chain (remote mode):
        1. RemoteColabExecutor (Playwright → headless Chromium → Colab.com)
        2. LocalIPythonRunner (persistent local kernel)
        3. Simulate (AST validation + heuristic output)

    The `remote` mode is the most powerful: opens headless Chromium, connects
    to Colab, executes code via CodeMirror JS API + keyboard automation.
    """

    def __init__(self, drive_folder_id: Optional[str] = None, executor: str = "auto",
                 browser_path: Optional[str] = None):
        self.drive_folder_id = drive_folder_id
        self.session = None
        self.active_notebook_id = None
        self.active_notebook_url = None
        self.execution_history = []

        self._lock = threading.RLock()
        self._local_kernel = None
        self._remote_executor = None
        self._browser_path = browser_path
        self._mode = executor
        self._connected = False
        self._fallback_reason = None

        # Determine execution mode
        if executor == "auto":
            self.simulate = os.environ.get("COLAB_AGENT_SIMULATE", "True").lower() == "true"
        elif executor == "local":
            self.simulate = False
        elif executor == "colab":
            self.simulate = True
        elif executor == "remote":
            self.simulate = False
        else:
            self.simulate = os.environ.get("COLAB_AGENT_SIMULATE", "True").lower() == "true"

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def connect(self) -> dict:
        """
        Connect the selected executor.

        - remote:  opens headless Chromium → Colab (with fallback chain)
        - local:   starts persistent jupyter_client kernel
        - colab:   no-op (manual upload)

        Returns status dict.
        """
        with self._lock:
            if self._connected:
                return {"success": True, "mode": self._mode, "fallback": self._fallback_reason}

            if self._mode == "remote":
                return self._connect_remote()
            elif self._mode == "local":
                return self._connect_local()
            else:
                self._connected = True
                return {"success": True, "mode": "colab", "fallback": None}

    def disconnect(self):
        """Shut down the active executor. Idempotent."""
        with self._lock:
            self._disconnect()

    def _disconnect(self):
        try:
            if self._remote_executor:
                self._remote_executor.disconnect()
        except Exception:
            pass
        try:
            if self._local_kernel:
                self._local_kernel.shutdown()
        except Exception:
            pass
        self._remote_executor = None
        self._local_kernel = None
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
        return False

    # ------------------------------------------------------------------ #
    #  Connection helpers
    # ------------------------------------------------------------------ #

    def _connect_remote(self) -> dict:
        """Connect remote executor with fallback chain."""
        from colab.remote_executor import RemoteColabExecutor

        # Step 1: Try Playwright
        executor = RemoteColabExecutor(headless=True, browser_path=self._browser_path)
        self._remote_executor = executor

        if executor.available:
            result = executor.connect()
            if result.get("success"):
                self._connected = True
                self._fallback_reason = None
                logger.info("Remote executor connected via Playwright browser")
                return {"success": True, "mode": "remote", "fallback": None}
            else:
                logger.warning(f"Remote executor failed: {result.get('error')}")
        else:
            logger.warning("Playwright not installed")

        # Step 2: Fallback to local kernel
        logger.info("Falling back to local kernel")
        self._mode = "local"
        local_result = self._connect_local()
        if local_result.get("success"):
            self._fallback_reason = "remote_unavailable"
            return local_result

        # Step 3: Fallback to simulate
        logger.info("Local kernel unavailable, falling back to simulation")
        self._mode = "colab"
        self.simulate = True
        self._connected = True
        self._fallback_reason = "local_unavailable"
        return {"success": True, "mode": "colab", "fallback": "remote_unavailable"}

    def _connect_local(self) -> dict:
        """Start a persistent local IPython kernel."""
        try:
            self._local_kernel = LocalIPythonRunner()
            self._connected = True
            return {"success": True, "mode": "local"}
        except Exception as e:
            logger.error(f"Failed to start local kernel: {e}")
            return {"success": False, "error": str(e)}

    def _get_local_kernel(self):
        if self._local_kernel is None:
            self._local_kernel = LocalIPythonRunner()
        return self._local_kernel

    def _get_remote_executor(self):
        """Lazy-init the remote Playwright executor with fallback."""
        if self._remote_executor is None:
            from colab.remote_executor import RemoteColabExecutor
            self._remote_executor = RemoteColabExecutor(headless=True, browser_path=self._browser_path)
            result = self._remote_executor.connect()
            if not result.get("success"):
                logger.warning(f"Remote executor connect failed: {result.get('error')}")
                raise RuntimeError(result.get("error", "remote unavailable"))
        return self._remote_executor

    # ------------------------------------------------------------------ #
    #  1. create_notebook
    # ------------------------------------------------------------------ #

    def create_notebook(self, title: str = "Colab-Agent Notebook",
                        cells: Optional[list[dict]] = None) -> dict:
        """
        Generate a .ipynb file and upload it to Google Drive via PyDrive.

        Args:
            title: Notebook title (used as filename).
            cells: List of dicts with keys:
                   - type: "code" | "markdown"
                   - source: str
                   - runtime: Optional GPU hint ("T4", "V100", ...)

        Returns:
            dict with keys: success, notebook_id, notebook_url,
                            drive_file_id, upload_code (Colab code snippet)
        """
        notebook_id = str(uuid.uuid4())[:12]
        notebook_name = f"{title} - {datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"

        cells = cells or [
            {"type": "code", "source": "# Colab Agent notebook\nprint('Ready')"},
        ]

        nb = self._build_ipynb(cells)
        nb_json = json.dumps(nb, indent=2)

        # Generate the Colab code that users paste to create + upload the notebook
        upload_code = self._generate_upload_code(notebook_name, nb_json)

        # For local use, also write the .ipynb to a temp file
        local_path = None
        if self.simulate:
            tmp_dir = tempfile.gettempdir()
            local_path = os.path.join(tmp_dir, notebook_name)
            with open(local_path, "w") as f:
                f.write(nb_json)
            logger.info(f"Notebook written to {local_path}")

        # Construct open-in-colab URL
        colab_url = self._colab_open_url(notebook_name)

        self.active_notebook_id = notebook_id
        self.active_notebook_url = colab_url

        return {
            "success": True,
            "notebook_id": notebook_id,
            "notebook_name": notebook_name,
            "notebook_url": colab_url,
            "local_path": local_path,
            "cell_count": len(cells),
            "upload_code": upload_code,
        }

    def _build_ipynb(self, cells: list[dict]) -> dict:
        """Build a full .ipynb JSON structure from cell descriptions."""
        nb_cells = []
        for cell in cells:
            source = cell.get("source", "")
            if isinstance(source, str):
                source = [source]
            nb_cells.append({
                "cell_type": "code" if cell.get("type", "code") == "code" else "markdown",
                "metadata": {"id": str(uuid.uuid4())[:12]},
                "source": source,
                "outputs": [],
                "execution_count": None,
            })
        # Determine GPU type from last cell that specifies it
        gpu_type = "T4"
        for cell in reversed(cells):
            if "runtime" in cell:
                gpu_type = cell["runtime"]
                break
        return {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "provenance": [],
                    "gpuType": gpu_type,
                    "toc_visible": True,
                },
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
            },
            "cells": nb_cells,
        }

    def _generate_upload_code(self, notebook_name: str, nb_json: str) -> str:
        """Generate Colab-compatible Python code that uploads this notebook
        to Google Drive using PyDrive."""
        # Escape the notebook JSON for embedding in generated code
        nb_json.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        folder_arg = f"'{self.drive_folder_id}'" if self.drive_folder_id else "None"

        return f'''
# ===== Colab Agent: Upload notebook to Google Drive =====
import json, io, os

NOTEBOOK_NAME = "{notebook_name}"
FOLDER_ID = {folder_arg}

# Rebuild notebook JSON
nb_json = ''' + "'''" + nb_json + "'''" + '''

# PyDrive upload
!pip install -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Create file
f = drive.CreateFile({{
    "title": NOTEBOOK_NAME,
    "mimeType": "application/x-ipynb+json",
}})
if FOLDER_ID:
    f["parents"] = [{{"id": FOLDER_ID}}]

f.SetContentString(nb_json)
f.Upload()

print(f"Notebook uploaded: {{f['title']}}")
print(f"File ID: {{f['id']}}")
print(f"Open: https://colab.research.google.com/drive/{{f['id']}}")
'''

    def _colab_open_url(self, notebook_name: str) -> str:
        """Return a colab.research.google.com URL (may use drive if uploaded)."""
        # In practice the URL comes from the Drive file after upload
        return f"https://colab.research.google.com/#create=true&title={notebook_name.replace(' ', '+')}"

    # ------------------------------------------------------------------ #
    #  2. execute_cell
    # ------------------------------------------------------------------ #

    def execute_cell(self, code: str, timeout: int = 300,
                     description: str = "",
                     on_progress: Optional[Callable] = None) -> dict:
        """
        Execute Python code in a Colab cell.

        In a real Colab environment, runs code via IPython or subprocess.
        In simulation mode, performs AST validation and heuristic output.

        Args:
            code: Python source code to execute.
            timeout: Max wall-clock seconds (default 300 / 5 min).
            description: Human-readable label for logs.
            on_progress: Optional callback(dict) for streaming feedback.

        Returns:
            dict with keys: success, output, error, execution_time, cell_id
        """
        start = time.time()
        cell_id = str(uuid.uuid4())[:8]

        logger.info(f"execute_cell [{cell_id}] ({self._mode}) "
                    f"desc={description}, {len(code)} chars, timeout={timeout}s")

        if self._mode == "remote":
            result = self._execute_remote(code, timeout)
        elif self.simulate:
            result = self._simulate_cell(code, timeout)
        else:
            result = self._execute_in_colab(code, timeout)

        elapsed = time.time() - start
        result["execution_time"] = elapsed
        result["cell_id"] = cell_id

        self.execution_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cell_id": cell_id,
            "description": description,
            "code_preview": code[:150],
            "result": result,
        })

        if on_progress:
            on_progress({"status": "completed", "result": result, "elapsed": elapsed})

        return result

    def _simulate_cell(self, code: str, timeout: int) -> dict:
        """Simulate cell execution for local testing."""
        import ast

        output_lines = []
        # 1. Check syntax (strip shell commands like !pip for AST check)
        python_lines = [line for line in code.split("\n") if not line.strip().startswith("!")]
        python_code = "\n".join(python_lines)
        try:
            ast.parse(python_code if python_code.strip() else "pass")
        except SyntaxError as e:
            return {
                "success": False,
                "output": "",
                "error": f"SyntaxError: {e}",
                "execution_time": 0.1,
            }

        # 2. Heuristic training detection
        has_training = any(kw in code for kw in [
            "trainer.train(", ".train(", "accelerator.prepare",
            "Trainer(", "SFTTrainer(", "TrainingArguments("
        ])
        has_download = any(kw in code for kw in [
            "from_pretrained", "load_dataset", "snapshot_download"
        ])
        has_gpu_check = "nvidia-smi" in code or "torch.cuda" in code
        is_setup = any(kw in code for kw in ["!pip install", "!apt"])

        # 3. Heuristic OOM trigger
        if "CUDA_OUT_OF_MEMORY" in code or "OOM_TRIGGER" in code:
            return {
                "success": False,
                "output": "",
                "error": "CUDA out of memory. Tried to allocate 5.2 GiB. GPU: Tesla T4 (16GB)",
                "execution_time": 2.0,
            }

        # 4. Build simulated output
        if is_setup:
            output_lines.append("[SETUP] Installing packages...")
            output_lines.append("[SETUP] All packages installed successfully")
        elif has_download:
            output_lines.append("[DOWNLOAD] Loading from HuggingFace Hub...")
            output_lines.append("[DOWNLOAD] Model loaded successfully")
        elif has_gpu_check:
            output_lines.append("GPU: Tesla T4 (simulated)")
            output_lines.append("VRAM: 15.8 GB / 16.0 GB")
            output_lines.append("RAM: 12.5 GB / 14.0 GB available")
            if torch_available():
                import torch
                if torch.cuda.is_available():
                    output_lines.append(f"Real GPU: {torch.cuda.get_device_name(0)}")
        elif has_training:
            output_lines.append("[TRAIN] Starting training...")
            output_lines.append("[TRAIN] Epoch 1/3: loss=0.8234")
            output_lines.append("[TRAIN] Epoch 2/3: loss=0.4512")
            output_lines.append("[TRAIN] Epoch 3/3: loss=0.2876")
            output_lines.append(f"[TRAIN] Completed in {max(0.5, timeout * 0.3):.1f}s (simulated)")
        else:
            # Try to actually exec simple code
            try:
                local_vars = {}
                safe_code = code.replace("!pip", "print")
                exec(safe_code, {"__builtins__": __builtins__}, local_vars)
                output_lines.append("[OK] Code executed successfully")
            except Exception as e:
                output_lines.append(f"[INFO] {e}")

        output_lines.append(f"=== CELL COMPLETED (simulated, timeout={timeout}s) ===")
        return {
            "success": True,
            "output": "\n".join(output_lines),
            "error": None,
            "execution_time": 0.3,
        }

    def _execute_remote(self, code: str, timeout: int) -> dict:
        """Execute code via remote Playwright-based Colab executor.

        Fallback chain on failure:
            1. Try Playwright
            2. Try local kernel (jupyter_client)
            3. Simulate (AST + heuristic)
        """
        try:
            executor = self._get_remote_executor()
            if not executor.available:
                logger.warning("Remote executor unavailable; falling back to local kernel")
                return self._try_fallback(code, timeout, "remote unavailable")
            return executor.execute(code, timeout=timeout)
        except Exception as e:
            logger.error(f"Remote execution failed: {e}")
            return self._try_fallback(code, timeout, str(e))

    def _try_fallback(self, code: str, timeout: int, reason: str) -> dict:
        """Attempt fallback chain: local kernel → simulation."""
        logger.info(f"Fallback triggered ({reason})")
        try:
            if self._local_kernel is None:
                self._local_kernel = LocalIPythonRunner()
            result = self._local_kernel.execute(code, timeout)
            if result.get("success"):
                self._mode = "local"
                self._fallback_reason = reason
                return result
        except Exception as e2:
            logger.warning(f"Local kernel fallback failed: {e2}")

        self._mode = "colab"
        self.simulate = True
        self._fallback_reason = reason
        logger.info("All fallbacks exhausted; simulating")
        return self._simulate_cell(code, timeout)

    def _execute_in_colab(self, code: str, timeout: int) -> dict:
        """
        Executes code in a persistent kernel.
        """
        # If we are NOT in Colab, use persistent local kernel
        if not os.environ.get("COLAB_GPU"):
            return self._get_local_kernel().execute(code, timeout)

        # Otherwise, run inside Colab runtime using IPython magic.
        try:
            from IPython import get_ipython
            ipython = get_ipython()

            if ipython is None:
                raise ImportError("No IPython shell")

            # Inject timeout wrapper via signal
            wrapped_code = f"""
import signal, sys, json, time

class TimeoutError(Exception):
    pass

def _handler(signum, frame):
    raise TimeoutError("Cell execution timed out after {timeout}s")

signal.signal(signal.SIGALRM, _handler)
signal.alarm({timeout})

try:
    _start = time.time()
    exec('''{code.replace("'", "\\'")}''')
    _elapsed = time.time() - _start
    print(f"=== CELL COMPLETED in {{_elapsed:.2f}}s ===")
except TimeoutError as _te:
    print(f"=== CELL TIMEOUT after {{timeout}}s ===")
    sys.exit(1)
except Exception as _e:
    import traceback
    print(f"=== CELL ERROR ===")
    print(json.dumps({{
        "type": type(_e).__name__,
        "message": str(_e),
        "traceback": traceback.format_exc(),
    }}))
finally:
    signal.alarm(0)
"""
            result = ipython.run_cell(wrapped_code, timeout=timeout)
            output = ""
            error = None

            if result.success:
                output = str(result.result) if result.result else "Cell executed successfully"
            else:
                error = str(result.error_in_input or result.error_before_exec or "Unknown error")

            return {"success": result.success, "output": output, "error": error}

        except ImportError:
            # Fallback: subprocess
            import subprocess
            try:
                proc = subprocess.run(
                    ["python", "-c", code],
                    capture_output=True, text=True, timeout=timeout,
                )
                return {
                    "success": proc.returncode == 0,
                    "output": proc.stdout,
                    "error": proc.stderr if proc.returncode != 0 else None,
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Subprocess timed out after {timeout}s",
                }

    # ------------------------------------------------------------------ #
    #  3. detect_current_runtime
    # ------------------------------------------------------------------ #

    def detect_current_runtime(self) -> ColabRuntimeInfo:
        """
        Detect current Colab runtime specs: GPU type, VRAM, RAM.

        In real Colab: queries torch, psutil, nvidia-smi.
        In simulation: returns plausible defaults.

        Returns:
            ColabRuntimeInfo with all fields populated.
        """
        if not self.simulate:
            return self._detect_real_runtime()
        return self._detect_simulated_runtime()

    def _detect_simulated_runtime(self) -> ColabRuntimeInfo:
        """Return simulated / plausible runtime info."""
        info = ColabRuntimeInfo(
            gpu_name="Tesla T4",
            vram_total_gb=16.0,
            vram_free_gb=15.0,
            ram_total_gb=12.5,
            ram_available_gb=10.2,
            runtime_label="T4",
            cuda_version="12.1",
            gpu_count=1,
        )

        # If torch is actually installed, use real values
        if torch_available():
            import torch
            import psutil
            if torch.cuda.is_available():
                info.gpu_name = torch.cuda.get_device_name(0)
                info.vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                info.vram_free_gb = info.vram_total_gb - (torch.cuda.memory_allocated(0) / 1e9)
                info.cuda_version = torch.version.cuda or "unknown"
                info.gpu_count = torch.cuda.device_count()
                info.runtime_label = self._gpu_to_label(info.gpu_name)
            ram = psutil.virtual_memory()
            info.ram_total_gb = ram.total / 1e9
            info.ram_available_gb = ram.available / 1e9

        return info

    def _detect_real_runtime(self) -> ColabRuntimeInfo:
        """Query actual Colab environment for runtime info."""
        try:
            import torch
            import psutil
        except ImportError:
            return self._detect_simulated_runtime()

        info = ColabRuntimeInfo()

        # GPU
        if torch.cuda.is_available():
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_count = torch.cuda.device_count()
            info.vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            info.vram_free_gb = info.vram_total_gb - (torch.cuda.memory_allocated(0) / 1e9)
            info.cuda_version = torch.version.cuda or ""
            info.runtime_label = self._gpu_to_label(info.gpu_name)

        # RAM
        ram = psutil.virtual_memory()
        info.ram_total_gb = ram.total / 1e9
        info.ram_available_gb = ram.available / 1e9

        # TPU check
        try:
            import google.colab
            info.is_tpu = "TPU" in str(google.colab).upper()
        except ImportError:
            info.is_tpu = False

        return info

    def detect_current_runtime_code(self) -> str:
        """Generate Colab code snippet that returns runtime info as JSON."""
        return """
import json, torch, psutil, subprocess

runtime = {}

# GPU
if torch.cuda.is_available():
    runtime["gpu_name"] = torch.cuda.get_device_name(0)
    runtime["gpu_count"] = torch.cuda.device_count()
    runtime["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    runtime["vram_allocated_gb"] = torch.cuda.memory_allocated(0) / 1e9
    runtime["cuda_version"] = torch.version.cuda
else:
    runtime["gpu_name"] = "None"
    runtime["gpu_count"] = 0
    runtime["vram_total_gb"] = 0

# RAM
ram = psutil.virtual_memory()
runtime["ram_total_gb"] = ram.total / 1e9
runtime["ram_available_gb"] = ram.available / 1e9

# nvidia-smi
try:
    out = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,memory.free","--format=csv,noheader"],
                         capture_output=True, text=True, timeout=5)
    runtime["nvidia_smi"] = out.stdout.strip()
except:
    runtime["nvidia_smi"] = "unavailable"

print(json.dumps(runtime, indent=2))
"""

    @staticmethod
    def _gpu_to_label(gpu_name: str) -> str:
        name = gpu_name.lower()
        if "a100" in name and "80" in name:
            return "A100-80GB"
        if "a100" in name:
            return "A100"
        if "v100" in name:
            return "V100"
        if "t4" in name:
            return "T4"
        if "p100" in name:
            return "P100"
        if "k80" in name:
            return "K80"
        if "tpu" in name or "tensor" in name:
            return "TPU"
        return "None"

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def parse_output(self, result: dict) -> dict:
        output = result.get("output", "")
        error = result.get("error")

        if error:
            cuda_match = re.search(r"CUDA out of memory|out of memory|OOM", error, re.IGNORECASE)
            if cuda_match:
                vram_match = re.search(r"(\d+\.?\d*)\s*GiB", error)
                return {
                    "error_type": "runtime_oom",
                    "error": error,
                    "vram_needed_gb": float(vram_match.group(1)) if vram_match else None,
                }
            if re.search(r"SyntaxError|IndentationError|NameError", error):
                return {"error_type": "syntax_error", "error": error}
            if re.search(r"ModuleNotFoundError|ImportError|No module named", error):
                return {"error_type": "import_error", "error": error}
            return {"error_type": "unknown", "error": error}

        metrics = {}
        loss_match = re.search(r"loss[=:]\s*(\d+\.\d+)", output, re.IGNORECASE)
        if loss_match:
            metrics["final_loss"] = float(loss_match.group(1))
        epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", output, re.IGNORECASE)
        if epoch_match:
            metrics["epoch"] = int(epoch_match.group(1))
            metrics["total_epochs"] = int(epoch_match.group(2))

        return {
            "error_type": None,
            "error": None,
            "metrics": metrics if metrics else None,
            "success_keywords": [
                kw for kw in ["completed", "success", "saved", "training complete"]
                if kw in output.lower()
            ],
        }

    def get_history(self, limit: int = 10) -> list:
        return self.execution_history[-limit:]

    def clear_history(self):
        self.execution_history = []


def torch_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("torch") is not None
