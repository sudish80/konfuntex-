"""
Phase 1 — Core Infrastructure items 6-15.

Provides:
  - NotebookManager        create / edit / upload .ipynb  (item 6)
  - CellExecutor           real-time output streaming      (item 7)
  - TimeoutHandler         process termination             (item 8)
  - RuntimeProfiler        GPU/CPU/RAM metrics             (item 9)
  - ResourceMonitor        alerts on high usage            (item 10)
  - CheckpointManager      save/load to Drive              (item 11)
  - ResumeHandler          restore after restart           (item 12)
  - ErrorClassifier        categorize errors               (item 13)
  - RetryPolicy            exponential backoff             (item 14)
  - SafeCodeValidator      block dangerous ops             (item 15)
"""
import os
import re
import json
import time
import signal
import shutil
import logging
import subprocess
from typing import Optional, Callable
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


# ==================================================================== #
#  6 — NotebookManager
# ==================================================================== #

class NotebookManager:
    """Create, edit, and upload .ipynb notebooks via nbformat."""

    def __init__(self):
        self.active_notebook = None
        self.cells = []

    def create(self, title: str = "Colab-Agent Notebook",
               cells: Optional[list[dict]] = None,
               runtime: str = "T4") -> dict:
        """Create a .ipynb JSON structure with given cells."""
        import nbformat as nbf
        nb = nbf.v4.new_notebook()
        nb.metadata = {
            "colab": {"provenance": [], "gpuType": runtime, "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        }
        self.cells = cells or [{"type": "code", "source": "# Colab Agent"}]
        for c in self.cells:
            ctype = "code" if c.get("type", "code") == "code" else "markdown"
            source = c.get("source", "")
            if isinstance(source, str):
                source = [source]
            nb.cells.append(nbf.v4.new_code_cell(source) if ctype == "code"
                            else nbf.v4.new_markdown_cell(source))
        self.active_notebook = nb
        return self._serialize(nb)

    def _serialize(self, nb) -> dict:
        import nbformat as nbf
        return json.loads(nbf.writes(nb))

    def add_cell(self, source: str, cell_type: str = "code",
                 after_index: Optional[int] = None):
        import nbformat as nbf
        cell = (nbf.v4.new_code_cell([source]) if cell_type == "code"
                else nbf.v4.new_markdown_cell([source]))
        if after_index is not None:
            self.active_notebook.cells.insert(after_index, cell)
        else:
            self.active_notebook.cells.append(cell)

    def save_to_drive(self, path: str) -> str:
        import nbformat as nbf
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            nbf.write(self.active_notebook, f)
        return path

    def upload_to_drive(self, filename: str,
                        folder_id: Optional[str] = None) -> str:
        """Upload via PyDrive and return the open-in-colab URL."""
        import nbformat as nbf
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from oauth2client.client import GoogleCredentials
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)
        f = drive.CreateFile({
            "title": filename,
            "mimeType": "application/x-ipynb+json",
        })
        if folder_id:
            f["parents"] = [{"id": folder_id}]
        f.SetContentString(nbf.writes(self.active_notebook))
        f.Upload()
        return f"https://colab.research.google.com/drive/{f['id']}"


# ==================================================================== #
#  7 — CellExecutor  (real-time output streaming)
# ==================================================================== #

class CellExecutor:
    """Execute code and stream output in real-time."""

    def __init__(self):
        self.process = None

    def execute(self, code: str, timeout: int = 300,
                on_stdout: Optional[Callable[[str], None]] = None,
                on_stderr: Optional[Callable[[str], None]] = None) -> dict:
        """Execute code in a subprocess with streaming output."""
        start = time.time()
        output_lines = []
        error_lines = []

        self.process = subprocess.Popen(
            ["python", "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        def _read_stream(stream, collector, callback):
            for line in iter(stream.readline, ""):
                collector.append(line)
                if callback:
                    callback(line.rstrip())

        import threading
        t1 = threading.Thread(target=_read_stream,
                              args=(self.process.stdout, output_lines, on_stdout))
        t2 = threading.Thread(target=_read_stream,
                              args=(self.process.stderr, error_lines, on_stderr))
        t1.daemon = True; t2.daemon = True
        t1.start(); t2.start()
        t1.join(timeout=timeout); t2.join(timeout=timeout)

        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._terminate()
            return {"success": False, "output": "".join(output_lines),
                    "error": f"TIMEOUT after {timeout}s",
                    "execution_time": time.time() - start}

        elapsed = time.time() - start
        return {
            "success": self.process.returncode == 0,
            "output": "".join(output_lines),
            "error": "".join(error_lines) if self.process.returncode != 0 else None,
            "execution_time": elapsed,
        }

    def execute_in_colab(self, code: str, timeout: int = 300) -> dict:
        """Execute using IPython (when running inside Colab itself)."""
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            if ipy is None:
                return self.execute(code, timeout)
            result = ipy.run_cell(code, timeout=timeout)
            return {
                "success": result.success,
                "output": str(result.result) if result.result else "",
                "error": str(result.error_in_input or result.error_before_exec or ""),
                "execution_time": 0,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _terminate(self):
        if self.process and self.process.poll() is None:
            self.process.kill()
            self.process.wait()


# ==================================================================== #
#  8 — TimeoutHandler
# ==================================================================== #

class TimeoutHandler:
    """Context manager / decorator for execution timeouts."""

    def __init__(self, seconds: int = 300):
        self.seconds = seconds

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        return exc_type is TimeoutError

    @staticmethod
    def _handler(signum, frame):
        raise TimeoutError("Execution timed out")

    @staticmethod
    def wrap(func, timeout: int = 300):
        """Decorator variant."""
        def _wrapper(*args, **kwargs):
            with TimeoutHandler(timeout):
                return func(*args, **kwargs)
        return _wrapper


# ==================================================================== #
#  9 — RuntimeProfiler
# ==================================================================== #

class RuntimeProfiler:
    """Measure GPU, CPU, RAM before/after actions."""

    @staticmethod
    def snapshot() -> dict:
        data = {"timestamp": datetime.now().isoformat()}

        # CPU
        try:
            import psutil
            data["cpu_percent"] = psutil.cpu_percent(interval=0.5)
            data["cpu_count"] = psutil.cpu_count()
        except Exception:
            data["cpu_percent"] = None

        # RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            data["ram_total_gb"] = round(mem.total / 1e9, 2)
            data["ram_used_gb"] = round(mem.used / 1e9, 2)
            data["ram_percent"] = mem.percent
        except Exception:
            pass

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                data["gpu_name"] = torch.cuda.get_device_name(0)
                data["gpu_count"] = torch.cuda.device_count()
                props = torch.cuda.get_device_properties(0)
                data["vram_total_gb"] = round(props.total_memory / 1e9, 2)
                data["vram_used_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
                data["vram_cached_gb"] = round(torch.cuda.memory_reserved(0) / 1e9, 2)
                data["cuda_version"] = torch.version.cuda or ""
            else:
                data["gpu_name"] = "None"
        except Exception:
            data["gpu_name"] = "error"

        return data

    @staticmethod
    def diff(before: dict, after: dict) -> dict:
        return {
            "ram_used_delta": round(
                (after.get("ram_used_gb", 0) or 0) -
                (before.get("ram_used_gb", 0) or 0), 2),
            "vram_used_delta": round(
                (after.get("vram_used_gb", 0) or 0) -
                (before.get("vram_used_gb", 0) or 0), 2),
            "cpu_load_after": after.get("cpu_percent"),
            "gpu_name": after.get("gpu_name", "?"),
        }

    @staticmethod
    def profiled_code() -> str:
        """Returns Colab code that prints a snapshot JSON."""
        return """
import json, torch, psutil
info = {}
info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
info["vram_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
info["vram_free"] = info["vram_total"] - (torch.cuda.memory_allocated(0) / 1e9) if torch.cuda.is_available() else 0
mem = psutil.virtual_memory()
info["ram_total"] = mem.total / 1e9
info["ram_free"] = mem.available / 1e9
info["ram_used_pct"] = mem.percent
print(json.dumps(info))
"""


# ==================================================================== #
#  10 — ResourceMonitor
# ==================================================================== #

class ResourceMonitor:
    """Raises alerts when resource usage exceeds thresholds."""

    THRESHOLDS = {
        "ram_percent": 90,
        "vram_percent": 95,
        "cpu_percent": 95,
        "disk_percent": 90,
    }

    def __init__(self, thresholds: Optional[dict] = None):
        if thresholds:
            self.THRESHOLDS.update(thresholds)
        self.alerts = []

    def check(self) -> list[str]:
        self.alerts = []
        snapshot = RuntimeProfiler.snapshot()

        ram_pct = snapshot.get("ram_percent", 0)
        if ram_pct and ram_pct > self.THRESHOLDS["ram_percent"]:
            self.alerts.append(
                f"HIGH RAM: {ram_pct:.0f}% used "
                f"(threshold {self.THRESHOLDS['ram_percent']}%)")

        vram_total = snapshot.get("vram_total_gb", 0)
        vram_used = snapshot.get("vram_used_gb", 0)
        if vram_total > 0:
            vram_pct = (vram_used / vram_total) * 100
            if vram_pct > self.THRESHOLDS["vram_percent"]:
                self.alerts.append(
                    f"HIGH VRAM: {vram_pct:.0f}% "
                    f"({vram_used:.1f}/{vram_total:.1f} GB)")

        cpu = snapshot.get("cpu_percent", 0)
        if cpu and cpu > self.THRESHOLDS["cpu_percent"]:
            self.alerts.append(f"HIGH CPU: {cpu:.0f}%")

        return self.alerts

    def is_safe(self) -> bool:
        return len(self.check()) == 0

    def assert_safe(self):
        alerts = self.check()
        if alerts:
            raise ResourceWarning(" | ".join(alerts))

    @staticmethod
    def monitoring_code() -> str:
        """Colab code that calls ResourceMonitor every 30s."""
        return """
import threading, time, json, torch, psutil

def _monitor():
    while True:
        snap = {}
        snap["vram_used"] = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        snap["vram_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        snap["ram_pct"] = psutil.virtual_memory().percent
        if snap["vram_total"] > 0 and (snap["vram_used"]/snap["vram_total"]) > 0.95:
            print(f"[ALERT] VRAM at {snap['vram_used']/snap['vram_total']*100:.0f}%")
        if snap["ram_pct"] > 90:
            print(f"[ALERT] RAM at {snap['ram_pct']}%")
        time.sleep(30)

thread = threading.Thread(target=_monitor, daemon=True)
thread.start()
print("Resource monitor started")
"""


# ==================================================================== #
#  11 — CheckpointManager
# ==================================================================== #

class CheckpointManager:
    """Save / load model checkpoints to/from Google Drive."""

    def __init__(self, drive_dir: str = "/content/drive/MyDrive/colab-checkpoints"):
        self.drive_dir = drive_dir

    def save(self, local_path: str, job_id: str, step: int = 0) -> str:
        dest = os.path.join(self.drive_dir, job_id, f"checkpoint-{step}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isdir(local_path):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(local_path, dest)
        else:
            shutil.copy2(local_path, dest)
        logger.info(f"Checkpoint saved: {local_path} -> {dest}")
        return dest

    def load(self, job_id: str, step: Optional[int] = None) -> Optional[str]:
        job_dir = os.path.join(self.drive_dir, job_id)
        if not os.path.isdir(job_dir):
            return None
        checkpoints = sorted(
            [d for d in os.listdir(job_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]) if "-" in x else 0,
        )
        if not checkpoints:
            return None
        target = (f"checkpoint-{step}" if step is not None else checkpoints[-1])
        path = os.path.join(job_dir, target)
        return path if os.path.exists(path) else None

    def list_checkpoints(self, job_id: str) -> list[str]:
        job_dir = os.path.join(self.drive_dir, job_id)
        if not os.path.isdir(job_dir):
            return []
        return sorted(
            [d for d in os.listdir(job_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )

    def save_code(self, local_path: str, job_id: str, step: int = 0) -> str:
        """Generate Colab code that saves checkpoint to Drive."""
        return f"""
import os, shutil
CHECKPOINT_DIR = "{self.drive_dir}/{job_id}/checkpoint-{step}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
if os.path.isdir("{local_path}"):
    for item in os.listdir("{local_path}"):
        src = os.path.join("{local_path}", item)
        dst = os.path.join(CHECKPOINT_DIR, item)
        if os.path.isdir(src): shutil.copytree(src, dst, dirs_exist_ok=True)
        else: shutil.copy2(src, dst)
else:
    shutil.copy2("{local_path}", CHECKPOINT_DIR)
print(f"Checkpoint saved to {{CHECKPOINT_DIR}}")
"""


# ==================================================================== #
#  12 — ResumeHandler
# ==================================================================== #

class ResumeHandler:
    """Save training state metadata and restore after restart."""

    STATE_FILE = "training_state.json"

    def __init__(self, drive_dir: str = "/content/drive/MyDrive/colab-checkpoints"):
        self.drive_dir = drive_dir

    def save_state(self, job_id: str, state: dict) -> str:
        path = os.path.join(self.drive_dir, job_id, self.STATE_FILE)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Training state saved: {path}")
        return path

    def load_state(self, job_id: str) -> Optional[dict]:
        path = os.path.join(self.drive_dir, job_id, self.STATE_FILE)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def build_resume_code(self, job_id: str,
                          base_model: str, dataset: str,
                          method: str = "qlora") -> str:
        """Generate Colab code that resumes training from the latest checkpoint."""
        return f"""
import os, json
STATE_PATH = "{self.drive_dir}/{job_id}/{self.STATE_FILE}"
CHECKPOINT_DIR = "{self.drive_dir}/{job_id}"

if os.path.exists(STATE_PATH):
    with open(STATE_PATH) as f:
        state = json.load(f)
    print(f"Found saved state: {{json.dumps(state, indent=2)}}")

# Find latest checkpoint
checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
if checkpoints:
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    resume_path = os.path.join(CHECKPOINT_DIR, latest)
    print(f"Resuming from {{resume_path}}")
else:
    resume_path = None
    print("No checkpoint found, starting fresh")

RESUME_CHECKPOINT = resume_path
"""


# ==================================================================== #
#  13 — ErrorClassifier
# ==================================================================== #

class ErrorClassifier:
    """Classify errors into categories for targeted recovery."""

    CATEGORIES = {
        "runtime_oom": ["out of memory", "cuda out of", "oom", "cuda error",
                        "device-side assert", "allocate", "memory"],
        "syntax_error": ["SyntaxError", "IndentationError", "NameError",
                         "invalid syntax", "unexpected indent"],
        "import_error": ["ModuleNotFoundError", "ImportError", "No module named",
                         "cannot import", "no module"],
        "connection_error": ["ConnectionError", "Connection refused", "timeout",
                             "cannot connect", "network", "resolve"],
        "api_error": ["401", "403", "429", "500", "rate limit", "unauthorized",
                      "quota exceeded", "API key"],
        "dataset_error": ["DatasetNotFound", "split not found", "config not found",
                          "not found", "404"],
        "training_diverged": ["NaN", "inf", "loss = nan", "diverged",
                              "loss increased"],
        "disk_full": ["No space left", "Disk quota", "disk full"],
        "colab_disconnect": ["Runtime disconnected", "Session crashed",
                             "reset by peer", "disconnected"],
        "unknown": [],
    }

    @classmethod
    def classify(cls, error: str) -> tuple[str, float]:
        """Returns (category, confidence)."""
        error_lower = error.lower()
        best_cat = "unknown"
        best_score = 0.0

        for cat, keywords in cls.CATEGORIES.items():
            if cat == "unknown":
                continue
            matches = sum(1 for kw in keywords if kw.lower() in error_lower)
            score = matches / max(len(keywords), 1)
            if score > best_score:
                best_score = score
                best_cat = cat

        return best_cat, best_score

    @classmethod
    def actionable(cls, category: str) -> bool:
        """Whether the agent can attempt recovery for this category."""
        return category in ("runtime_oom", "syntax_error", "import_error",
                            "connection_error", "dataset_error",
                            "training_diverged", "disk_full")


# ==================================================================== #
#  14 — RetryPolicy
# ==================================================================== #

class RetryPolicy:
    """Exponential backoff with jitter."""

    def __init__(self, max_retries: int = 3,
                 base_delay: float = 2.0,
                 max_delay: float = 120.0,
                 jitter: float = 0.1):
        self.max_retries = max_retries
        self.initial_base_delay = base_delay
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.attempt = 0

    def record_result(self, success: bool):
        if success:
            # Gradually reduce backoff if successful
            self.base_delay = max(self.initial_base_delay, self.base_delay * 0.9)
        else:
            # Increase backoff if failing
            self.base_delay = min(self.max_delay / 2, self.base_delay * 1.5)

    def reset(self):
        self.attempt = 0

    def should_retry(self) -> bool:
        return self.attempt < self.max_retries

    def next_delay(self) -> float:
        self.attempt += 1
        delay = min(self.base_delay * (2 ** (self.attempt - 1)), self.max_delay)
        import random
        jitter_amount = delay * self.jitter * random.random()
        return delay + jitter_amount

    def sleep(self):
        delay = self.next_delay()
        time.sleep(delay)

    @staticmethod
    def exponential_backoff_code(max_retries: int = 3) -> str:
        return f"""
import time, random
MAX_RETRIES = {max_retries}
for attempt in range(1, MAX_RETRIES + 1):
    try:
        # YOUR CODE HERE
        break
    except Exception as e:
        if attempt == MAX_RETRIES: raise
        delay = min(2 ** attempt, 120) + random.random()
        print(f"Attempt {{attempt}} failed: {{e}}. Retrying in {{delay:.1f}}s...")
        time.sleep(delay)
"""


# ==================================================================== #
#  15 — SafeCodeValidator
# ==================================================================== #

class SafeCodeValidator:
    """Block dangerous operations before execution."""

    BLOCKED_PATTERNS = [
        (r"\brm\s+-rf\b", "rm -rf is blocked"),
        (r"\b(?:shutil\.)?rmtree\b", "rmtree blocked"),
        (r"\bos\.system\s*\(['\"]rm\b", "os.system(rm) blocked"),
        (r"\bsubprocess\.(?:call|Popen|run)\s*\(['\"](?:rm|shutdown|reboot|sudo|dd|mkfs)",
         "dangerous subprocess blocked"),
        (r"\beval\s*\(input\b", "eval(input) blocked"),
        (r"\bexec\s*\(input\b", "exec(input) blocked"),
        (r"\bopen\s*\(['\"](?:/etc|/dev|/proc|/sys)", "system file blocked"),
        (r"\b(?:dd|mkfs|fdisk|format|shutdown|reboot|halt|poweroff)\s",
         "system command blocked"),
        (r"!wget\s+.*?(?:--output-document|-O)\s+/", "wget to root blocked"),
        (r"!curl\s+.*?(?:-o|-O)\s+/", "curl to root blocked"),
        (r"\b__import__\s*\(\s*['\"]os['\"]\)", "dynamic os import blocked"),
    ]

    PIP_ALLOWLIST = {
        "transformers", "datasets", "accelerate", "peft", "trl",
        "bitsandbytes", "bitsandweights", "torch", "torchvision",
        "torchaudio", "safetensors", "huggingface_hub", "sentencepiece",
        "tensorboard", "wandb", "scikit-learn", "pandas", "numpy",
        "matplotlib", "seaborn", "tqdm", "ipywidgets", "gradio",
        "streamlit", "pydrive", "google-api-python-client",
        "openai", "anthropic", "google-generativeai", "groq",
        "python-dotenv", "pyyaml", "sqlalchemy", "nbformat",
        "requests", "httpx", "fire", "psutil",
    }

    def validate(self, code: str) -> tuple[bool, str, str]:
        """
        Returns (is_safe, cleaned_code, warning).
        """
        warnings = []
        for pattern, msg in self.BLOCKED_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                warnings.append(msg)

        cleaned = code
        cleaned = self._sanitize_pip(cleaned)

        if warnings:
            return False, cleaned, "; ".join(warnings)
        return True, cleaned, ""

    def _sanitize_pip(self, code: str) -> str:
        def _replace(m):
            full = m.group(0)
            pkgs = re.findall(r"(\S+)", re.sub(r"!pip\s+install\b", "", full))
            blocked = [
                p for p in pkgs
                if p.split("=")[0].split(">")[0].split("<")[0]
                   .split("[")[0].strip() not in self.PIP_ALLOWLIST
                   and not p.startswith("-")
                   and not p.startswith("git+")
            ]
            if blocked:
                return f"# BLOCKED: {full}"
            return full
        return re.sub(r"!pip\s+install\b[^\n]*", _replace, code)

    def compute_hash(self, code: str) -> str:
        import hashlib
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def check_code_gen(self, code: str) -> dict:
        safe, cleaned, warn = self.validate(code)
        return {
            "safe": safe,
            "hash": self.compute_hash(code),
            "length": len(code),
            "warnings": warn,
            "has_pip": "!pip" in code,
            "has_imports": bool(re.findall(r"^(?:import|from)\s", code, re.MULTILINE)),
        }
