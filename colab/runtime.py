import re
import logging
from typing import Optional
from config.settings import settings

logger = logging.getLogger(__name__)


class RuntimeManager:
    """
    Manages Colab runtime tiers: detection, sufficiency checks, and switching.

    Runtime tiers and their approximate VRAM:
        None       0 GB    (CPU only)
        T4        16 GB    (free Colab GPU)
        P100      16 GB    (sometimes available)
        V100      32 GB    (Colab Pro+)
        A100      40 GB    (Colab Pro+ higher tier)
        A100-80GB 80 GB    (highest tier)
        TPU      128 GB    (TPU v2-8, different architecture)

    Switching is done by manipulating Colab's URL parameters:
        &accelerator=GPU|TPU|None
        &gpuType=T4|V100|A100|A100-80GB|P100|K80
    """

    RUNTIME_ORDER = ["None", "T4", "P100", "V100", "A100", "A100-80GB", "TPU"]

    GPU_NAME_TO_LABEL = {
        "tesla t4": "T4",
        "tesla v100": "V100",
        "tesla p100": "P100",
        "tesla k80": "K80",
        "nvidia a100": "A100",
        "a100-sxm": "A100",
        "a100-80gb": "A100-80GB",
        "tpu": "TPU",
        "tensor processing unit": "TPU",
    }

    def __init__(self):
        self.current_gpu = "None"
        self.current_vram_gb = 0.0
        self.current_ram_gb = 0.0
        self.switch_history = []

    # ------------------------------------------------------------------ #
    #  4. switch_runtime
    # ------------------------------------------------------------------ #

    def switch_runtime(self, target_gpu: str = "V100",
                       save_state: bool = True,
                       checkpoint_path: Optional[str] = None,
                       reason: str = "") -> dict:
        """
        Switch Colab to a different GPU runtime tier.

        Strategy (best-effort):
        1. Save current progress (model checkpoint, data, variables) to Drive
        2. Generate a URL that forces Colab to restart with the new GPU
        3. The URL approach uses colab.research.google.com URL parameters:
           - &accelerator=GPU|TPU|None
           - &gpuType=T4|V100|A100|A100-80GB

        Args:
            target_gpu: Target tier label ("T4", "V100", "A100", "A100-80GB", "TPU", "None").
            save_state: Whether to generate code that saves model/data to Drive first.
            checkpoint_path: Path where the model checkpoint is stored.
            reason: Why the switch is needed (logged for history).

        Returns:
            dict with keys: success, switch_code, switched_from, switched_to,
                            restart_url, note, save_state_code
        """
        target_label = self._normalize_label(target_gpu)
        if not target_label:
            return {"success": False, "error": f"Unknown target GPU: {target_gpu}"}

        old_gpu = self.current_gpu

        # Log the switch
        self.switch_history.append({
            "from": old_gpu,
            "to": target_label,
            "reason": reason,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        })

        # Generate save-state code
        save_code = ""
        if save_state and checkpoint_path:
            save_code = self._generate_save_state_code(checkpoint_path)

        # Generate the runtime switch code
        switch_code = self._generate_switch_code(target_label)

        # Build the Colab restart URL with GPU parameter
        restart_url = self._build_restart_url(target_label)

        self.current_gpu = target_label
        self.current_vram_gb = settings.runtime_tiers.get(target_label, 0)

        logger.info(f"Runtime switch: {old_gpu} -> {target_label} ({reason})")

        return {
            "success": True,
            "switched_from": old_gpu,
            "switched_to": target_label,
            "reason": reason,
            "switch_code": switch_code,
            "restart_url": restart_url,
            "save_state_code": save_code,
            "note": (
                f"Switching to {target_label}. "
                "Execute save_state_code first, then switch_code. "
                "The runtime will restart with the new GPU."
            ),
        }

    def _normalize_label(self, label: str) -> Optional[str]:
        """Match a user-provided string to a canonical runtime label."""
        label_lower = label.strip().lower()
        for canonical, aliases in self._label_aliases().items():
            if label_lower == canonical or label_lower in aliases:
                return canonical
        return None

    @staticmethod
    def _label_aliases() -> dict:
        return {
            "T4": ["t4", "tesla t4", "nvidia t4"],
            "V100": ["v100", "tesla v100", "nvidia v100", "volta"],
            "P100": ["p100", "tesla p100", "nvidia p100", "pascal"],
            "A100": ["a100", "nvidia a100", "amper"],
            "A100-80GB": ["a100-80gb", "a100 80gb", "a100_80gb", "a100 80 gb"],
            "TPU": ["tpu", "tensor", "tensor processing unit", "tpu v2", "tpu v3"],
            "None": ["none", "cpu", "no gpu"],
        }

    def _generate_save_state_code(self, checkpoint_path: str) -> str:
        """Generate Colab code to save model/data to Drive before restart."""
        _TEMPLATE = '''
# ===== Colab Agent: Save State Before Runtime Switch =====
import os, shutil, json, torch
from google.colab import drive

CHECKPOINT_PATH = "{}"

# Mount Drive
drive.mount('/content/drive')

# Save checkpoint
if os.path.exists(CHECKPOINT_PATH):
    save_path = "/content/drive/MyDrive/colab-agent-checkpoints/" + os.path.basename(CHECKPOINT_PATH)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.isdir(CHECKPOINT_PATH):
        shutil.copytree(CHECKPOINT_PATH, save_path, dirs_exist_ok=True)
    else:
        shutil.copy2(CHECKPOINT_PATH, save_path)
    print(f"Checkpoint saved to {{save_path}}")

# Save training state
training_state = {{
    "checkpoint_path": CHECKPOINT_PATH,
    "target_runtime": "TARGET_GPU",
        "timestamp": "{{__import__('datetime').datetime.now().isoformat()}}",
}}
with open("/content/drive/MyDrive/colab-agent-checkpoints/training_state.json", "w") as f:
    json.dump(training_state, f, indent=2)

print("State saved. Proceeding with runtime switch...")
'''
        return _TEMPLATE.format(checkpoint_path)
    def _generate_switch_code(self, target_label: str) -> str:
        """Generate Colab code that triggers a runtime restart with new GPU.

        Uses multiple approaches:
        1. google.colab._runtime.set_accelerator_type() (internal API)
        2. google.colab runtime APIs
        3. URL-based restart as fallback
        """
        accelerator = "GPU"
        if target_label == "TPU":
            accelerator = "TPU"
        elif target_label == "None":
            accelerator = "None"

        return f'''
# ===== Colab Agent: Switch Runtime to {target_label} =====
import sys, json, urllib.parse

TARGET = "{target_label}"
ACCELERATOR = "{accelerator}"

print(f"Switching runtime to {{TARGET}}...")
print(f"This will restart the runtime with a {{TARGET}} GPU.")

# Approach 1: Internal Colab API (most reliable when available)
try:
    from google.colab import _runtime
    _runtime.set_accelerator_type(ACCELERATOR)
    print("Runtime switch requested via _runtime API.")
except ImportError:
    print("_runtime API not available, trying alternative...")
except Exception as e:
    print(f"_runtime API error: {{e}}")

# Approach 2: URL parameter manipulation
# The Colab runtime restart URL includes GPU type as a URL parameter
# You can reconnect with: https://colab.research.google.com/...&accelerator={accelerator}
restart_url_params = {{
    "accelerator": ACCELERATOR,
    "gpuType": TARGET,
}}
print(f"To restart manually, use runtime parameters: {{restart_url_params}}")

# Approach 3: Force runtime restart
try:
    from google.colab import runtime
    runtime.restart()
    print("Runtime restart initiated.")
except Exception as e:
    print(f"Runtime restart error: {{e}}")
    print("Manual restart required: Runtime -> Restart and run all")

print("=== RUNTIME SWITCH INITIATED ===")
print("The notebook will restart. Re-run setup cells after restart.")
'''

    def _build_restart_url(self, target_label: str) -> str:
        """Build a Colab URL that forces the specified GPU on restart.

        URL parameter format:
            https://colab.research.google.com/drive/...
                ?accelerator=GPU|TPU|None
                #gpuType=T4|V100|A100|A100-80GB|P100|K80

        This is the best-known approach for programmatic GPU selection.
        """
        accelerator = "GPU"
        gpu_type = target_label
        if target_label == "TPU":
            accelerator = "TPU"
            gpu_type = ""
        elif target_label == "None":
            accelerator = "None"
            gpu_type = ""

        params = {"accelerator": accelerator}
        if gpu_type:
            params["gpuType"] = gpu_type

        query = urllib_encode(params)
        return f"https://colab.research.google.com/notebook?{query}"

    # ------------------------------------------------------------------ #
    #  5. is_runtime_sufficient
    # ------------------------------------------------------------------ #

    def is_runtime_sufficient(self, required_vram_gb: float,
                              required_ram_gb: float = 0.0) -> bool:
        """
        Check if the current runtime has enough resources.

        Args:
            required_vram_gb: Minimum GPU VRAM needed (GB).
            required_ram_gb: Minimum system RAM needed (GB). 0 = skip check.

        Returns:
            True if current runtime meets or exceeds both requirements.
        """
        vram_ok = self.current_vram_gb >= required_vram_gb * 0.9
        ram_ok = (required_ram_gb == 0) or (self.current_ram_gb >= required_ram_gb * 0.9)

        if not vram_ok:
            logger.warning(
                f"VRAM insufficient: {self.current_vram_gb:.1f}GB < "
                f"{required_vram_gb:.1f}GB (x0.9 margin)"
            )
        if not ram_ok:
            logger.warning(
                f"RAM insufficient: {self.current_ram_gb:.1f}GB < "
                f"{required_ram_gb:.1f}GB"
            )

        return vram_ok and ram_ok

    def best_fit_runtime(self, required_vram_gb: float,
                         required_ram_gb: float = 0.0) -> str:
        """
        Find the cheapest runtime tier that satisfies requirements.

        Returns:
            Runtime label string (e.g., "T4", "V100", "A100").
        """
        for tier in self.RUNTIME_ORDER:
            vram = settings.runtime_tiers.get(tier, 0)
            999  # RAM not tiered; assume sufficient
            if vram >= required_vram_gb:
                return tier
        return "A100-80GB"

    def should_switch(self, required_vram_gb: float) -> tuple[bool, str, str, str]:
        """
        Determine if a runtime switch is needed.

        Returns:
            (should_switch: bool, current: str, recommended: str, reason: str)
        """
        current = self.current_gpu
        if self.is_runtime_sufficient(required_vram_gb):
            return False, current, current, "Runtime sufficient"

        recommended = self.best_fit_runtime(required_vram_gb)
        reason = (
            f"Insufficient: {current} ({self.current_vram_gb:.1f}GB) < "
            f"required {required_vram_gb:.1f}GB. Recommend {recommended}."
        )
        return True, current, recommended, reason

    # ------------------------------------------------------------------ #
    #  Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_from_gpu_name(gpu_name: str) -> str:
        gpu_lower = gpu_name.lower()
        if "a100" in gpu_lower and "80" in gpu_lower:
            return "A100-80GB"
        if "a100" in gpu_lower:
            return "A100"
        if "v100" in gpu_lower:
            return "V100"
        if "t4" in gpu_lower:
            return "T4"
        if "p100" in gpu_lower:
            return "P100"
        if "k80" in gpu_lower:
            return "K80"
        if "tpu" in gpu_lower or "tensor" in gpu_lower:
            return "TPU"
        return "None"

    @staticmethod
    def get_vram_gb(gpu_name: str, nvidia_smi: str = "") -> float:
        name_lower = gpu_name.lower()
        if "a100" in name_lower and "80" in name_lower:
            return 80.0
        if "a100" in name_lower:
            return 40.0
        if "v100" in name_lower:
            return 32.0
        if "t4" in name_lower:
            return 16.0
        if "p100" in name_lower:
            return 16.0
        if "k80" in name_lower:
            return 12.0
        match = re.search(r"(\d+)\s*MiB", nvidia_smi)
        if match:
            return int(match.group(1)) / 1024
        return 0.0

    @staticmethod
    def get_switch_history_code() -> str:
        """Generate Colab code to show switch history."""
        return """
# ===== Runtime Switch History =====
import json
history = []
try:
    with open("/content/drive/MyDrive/colab-agent-checkpoints/training_state.json") as f:
        history.append(json.load(f))
except:
    pass
print(json.dumps(history, indent=2) if history else "No switch history found")
"""


def urllib_encode(params: dict) -> str:
    """Simple URL parameter encoding (avoids requiring urllib import at module level)."""
    import urllib.parse
    return urllib.parse.urlencode(params)


def estimate_needed_vram(model_params_b: float, method: str) -> float:
    if method == "qlora":
        return model_params_b * 0.5 + 2.0
    elif method == "lora":
        return model_params_b * 1.2 + 2.0
    elif method == "full":
        return model_params_b * 3.0 + 4.0
    return model_params_b * 1.5 + 2.0
