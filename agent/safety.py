"""
Safety limits, code sanitization, and cost tracking — Phase 8.
"""

import re
import hashlib
import time
import logging
import threading


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- #
#  Safety limits
# ---------------------------------------------------------------- #

MAX_RUNTIME_SWITCHES = 5
MAX_ERROR_RETRIES_PER_STEP = 10
MAX_TOTAL_ERROR_RETRIES = 50
MAX_COLAB_HOURS = 12
MAX_VRAM_GB = 80
MAX_TRAINING_STEPS = 100000

# ---------------------------------------------------------------- #
#  Code sanitisation
# ---------------------------------------------------------------- #

DANGEROUS_PATTERNS = [
    (r"\brm\s+-rf\b", "rm -rf is blocked"),
    (r"\bos\.system\s*\(['\"]rm\b", "os.system(rm) is blocked"),
    (r"\bsubprocess\.(call|Popen|run)\s*\(['\"]rm\b",
     "subprocess rm is blocked"),
    (r"\b__import__\s*\(\s*['\"]os['\"]\)", "dynamic os import blocked"),
    (r"\bexec\s*\(input\b", "exec(input()) blocked"),
    (r"\beval\s*\(input\b", "eval(input()) blocked"),
    (r"\bopen\s*\(['\"](?:/etc|/dev|/proc)", "system file access blocked"),
    (r"\b(?:dd|mkfs|fdisk|format)\b", "disk operation blocked"),
    (r"!wget\s+.*?(?:--output-document|-O)\s+/", "wget to root blocked"),
    (r"!curl\s+.*?(?:-o|-O)\s+/", "curl to root blocked"),
]

ALLOWED_PIP_PACKAGES = {
    "transformers", "datasets", "accelerate", "peft", "trl",
    "bitsandbytes", "bitsandweights", "torch", "torchvision",
    "torchaudio", "safetensors", "huggingface_hub", "sentencepiece",
    "tensorboard", "wandb", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "tqdm", "ipywidgets", "gradio",
    "streamlit", "pydrive", "google-api-python-client",
    "google-auth-httplib2", "google-auth-oauthlib",
    "python-dotenv", "pyyaml", "json5", "fire",
}


def sanitize_code(code: str) -> tuple[bool, str, str]:
    """
    Check generated code for dangerous operations.

    Returns:
        (is_safe: bool, cleaned_code: str, warning: str)
    """
    warnings = []
    for pattern, msg in DANGEROUS_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            warnings.append(msg)

    if warnings:
        return False, code, "; ".join(warnings)

    return True, code, ""


def sanitize_pip(code: str) -> str:
    """Restrict pip install to an allowlist."""
    def _replace_pip(m):
        full = m.group(0)
        pkgs = re.findall(r"(\S+)", full.replace("!pip install", "")
                          .replace("-q", "").replace("-U", ""))
        blocked = [p for p in pkgs
                   if p.split("=")[0].split(">")[0].split("<")[0]
                   .split("[")[0].strip() not in ALLOWED_PIP_PACKAGES
                   and not p.startswith("-")]
        if blocked:
            return f"# BLOCKED: {full}  (packages: {blocked})"
        return full
    return re.sub(r"!pip\s+install\b[^\n]*", _replace_pip, code)


def compute_code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]

# ---------------------------------------------------------------- #
#  Cost tracking
# ---------------------------------------------------------------- #

# Rough Colab compute unit estimates
COST_PER_GPU_HOUR: dict[str, float] = {
    "None": 0.0,
    "T4": 1.0,
    "V100": 1.5,
    "A100": 2.5,
    "A100-80GB": 3.0,
    "TPU": 4.0,
}


class CostTracker:
    """Tracks per-job GPU cost, retries, and runtime switches.

    Thread-safe. Each job is keyed by a unique job_id.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def start_job(self, job_id: str, runtime: str = "T4") -> None:
        if not job_id:
            raise ValueError("job_id must be non-empty")
        if runtime not in COST_PER_GPU_HOUR:
            logger.warning("Unknown runtime %r, defaulting cost to 1.0", runtime)

        with self._lock:
            self._jobs[job_id] = {
                "runtime": runtime,
                "start_time": time.time(),
                "gpu_seconds": 0.0,
                "switches": 0,
                "total_retries": 0,
            }

    def record_execution(self, job_id: str, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("execution seconds must be >= 0")
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is not None:
                entry["gpu_seconds"] += seconds

    def record_retry(self, job_id: str) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is not None:
                entry["total_retries"] += 1

    def record_switch(self, job_id: str, new_runtime: str) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is not None:
                entry["switches"] += 1
                entry["runtime"] = new_runtime

    def get_summary(self, job_id: str) -> dict:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return {
                    "job_id": job_id,
                    "gpu_hours": 0.0,
                    "estimated_cost_units": 0.0,
                    "current_runtime": "unknown",
                    "runtime_switches": 0,
                    "total_retries": 0,
                }

            gpu_hours = entry["gpu_seconds"] / 3600
            runtime = entry.get("runtime", "T4")
            cost_per_hour = COST_PER_GPU_HOUR.get(runtime, 1.0)
            return {
                "job_id": job_id,
                "gpu_hours": round(gpu_hours, 3),
                "estimated_cost_units": round(gpu_hours * cost_per_hour, 2),
                "current_runtime": runtime,
                "runtime_switches": entry.get("switches", 0),
                "total_retries": entry.get("total_retries", 0),
            }


class BudgetManager:
    """Per-session budget limits with max-spend alerts.

    Thread-safe. Tracks cumulative spend and raises alerts when
    configurable thresholds are crossed.

    Args:
        max_cost_units: Maximum allowed spend in cost units (> 0).
        warn_threshold: Ratio (0-1) at which a warning alert fires.
    """

    def __init__(self, max_cost_units: float = 100.0, warn_threshold: float = 0.8) -> None:
        if max_cost_units <= 0:
            raise ValueError("max_cost_units must be > 0")
        if not 0 < warn_threshold <= 1:
            raise ValueError("warn_threshold must be in (0, 1]")

        self.max_cost_units = max_cost_units
        self.warn_threshold = warn_threshold
        self._lock = threading.Lock()
        self._spent: float = 0.0
        self._alerts: list[str] = []

    def record_cost(self, cost_units: float) -> None:
        """Record a cost addition and fire alerts if thresholds crossed."""
        if cost_units < 0:
            raise ValueError("cost_units must be >= 0")

        with self._lock:
            self._spent += cost_units
            usage = self._spent / self.max_cost_units

            if usage >= 1.0:
                self._alerts.append(
                    f"Budget EXCEEDED: {self._spent:.1f} / {self.max_cost_units} units"
                )
                logger.warning("Budget exceeded: %.1f / %.1f", self._spent, self.max_cost_units)
            elif usage >= self.warn_threshold:
                self._alerts.append(
                    f"Budget WARNING: {self._spent:.1f} / {self.max_cost_units} units "
                    f"({usage:.0%})"
                )
                logger.info("Budget warning at %.0f%%", usage * 100)

    @property
    def exceeded(self) -> bool:
        with self._lock:
            return self._spent >= self.max_cost_units

    @property
    def usage_ratio(self) -> float:
        with self._lock:
            return self._spent / self.max_cost_units

    @property
    def alerts(self) -> list[str]:
        with self._lock:
            return list(self._alerts)

    def clear_alerts(self) -> None:
        with self._lock:
            self._alerts.clear()

    def reset(self) -> None:
        with self._lock:
            self._spent = 0.0
            self._alerts.clear()
            logger.info("BudgetManager reset")

    def snapshot(self) -> dict:
        with self._lock:
            remaining = max(0.0, self.max_cost_units - self._spent)
            usage_pct = (self._spent / self.max_cost_units) * 100 if self.max_cost_units > 0 else 0.0
            return {
                "max_cost_units": self.max_cost_units,
                "spent": round(self._spent, 2),
                "remaining": round(remaining, 2),
                "usage_pct": round(usage_pct, 1),
                "exceeded": self._spent >= self.max_cost_units,
                "alert_count": len(self._alerts),
            }
