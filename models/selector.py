"""
Model Selector — Automatically chooses the best model for available hardware.

Usage:
    selector = ModelSelector()
    best = selector.best_fit(vram_gb=16, ram_gb=12)
    # => {"name": "unsloth/phi-4", "params_b": 14.7, "method": "lora", ...}

    selector = ModelSelector()
    all_fitting = selector.models_that_fit(vram_gb=8, ram_gb=8)
    # => [{"name": "google/gemma-2-2b", ...}, ...]

The selector works in both local and Colab mode by reading runtime info
from ColabRuntimeInfo or directly from torch/nvidia-smi.
"""

import os
import re
import json
import time
import logging
import threading
from typing import Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class HardwareTier(Enum):
    """GPU capability tiers for model matching."""
    NONE = "none"           # CPU only
    LOW = "low"             # ~8GB or less (T4 low mem, K80)
    MEDIUM = "medium"       # ~16GB (T4, P100)
    HIGH = "high"           # ~32-40GB (V100, A10G)
    VERY_HIGH = "very_high" # ~80GB (A100-80GB, H100)
    EXTREME = "extreme"     # 80GB+ (multi-GPU, H200)


@dataclass
class ModelSpec:
    """Spec for a single model variant."""
    name: str                      # HuggingFace model ID
    family: str                    # Model family (llama, phi, gemma, etc.)
    params_b: float                # Parameter count in billions
    vram_gb_lora: float            # Minimum VRAM for LoRA fine-tuning
    vram_gb_qlora: float           # Minimum VRAM for QLoRA fine-tuning
    vram_gb_full: float            # Minimum VRAM for full fine-tuning
    ram_gb_required: float         # Minimum system RAM
    context_window: int = 4096     # Max context length
    requires_auth: bool = False    # Needs HF token
    tier: HardwareTier = HardwareTier.MEDIUM
    description: str = ""

    def min_vram(self, method: str = "qlora") -> float:
        return {
            "lora": self.vram_gb_lora,
            "qlora": self.vram_gb_qlora,
            "full": self.vram_gb_full,
        }.get(method, self.vram_gb_qlora)

    def fits_in(self, vram_gb: float, ram_gb: float, method: str = "qlora") -> bool:
        return vram_gb >= self.min_vram(method) and ram_gb >= self.ram_gb_required


# ------------------------------------------------------------------ #
#  Model Registry — curated list of models ranked by capability
# ------------------------------------------------------------------ #

MODEL_REGISTRY: list[ModelSpec] = [
    # --- Tiny models (< 3B) — fit anywhere ---
    ModelSpec(
        name="google/gemma-2-2b",
        family="gemma",
        params_b=2.6,
        vram_gb_lora=3.0,
        vram_gb_qlora=2.0,
        vram_gb_full=8.0,
        ram_gb_required=4.0,
        context_window=8192,
        tier=HardwareTier.LOW,
        description="Gemma 2 2B — fast, fits in 2GB VRAM with QLoRA",
    ),
    ModelSpec(
        name="microsoft/phi-2",
        family="phi",
        params_b=2.7,
        vram_gb_lora=3.0,
        vram_gb_qlora=2.0,
        vram_gb_full=8.0,
        ram_gb_required=4.0,
        context_window=2048,
        tier=HardwareTier.LOW,
        description="Phi-2 — Microsoft's 2.7B, strong reasoning for size",
    ),
    ModelSpec(
        name="Qwen/Qwen2.5-1.5B",
        family="qwen",
        params_b=1.5,
        vram_gb_lora=2.0,
        vram_gb_qlora=1.5,
        vram_gb_full=5.0,
        ram_gb_required=3.0,
        context_window=32768,
        tier=HardwareTier.LOW,
        description="Qwen 2.5 1.5B — tiny, fast, 32K context",
    ),
    ModelSpec(
        name="microsoft/Phi-3-mini-4k-instruct",
        family="phi",
        params_b=3.8,
        vram_gb_lora=4.0,
        vram_gb_qlora=3.0,
        vram_gb_full=12.0,
        ram_gb_required=6.0,
        context_window=4096,
        tier=HardwareTier.LOW,
        description="Phi-3 Mini 3.8B — excellent reasoning for size",
    ),
    ModelSpec(
        name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        family="smollm",
        params_b=1.7,
        vram_gb_lora=2.0,
        vram_gb_qlora=1.5,
        vram_gb_full=5.0,
        ram_gb_required=3.0,
        context_window=2048,
        tier=HardwareTier.LOW,
        description="SmolLM2 1.7B — smallest instruct model",
    ),

    # --- Small models (3B-7B) — fit in T4/P100 ---
    ModelSpec(
        name="google/gemma-2-9b",
        family="gemma",
        params_b=9.2,
        vram_gb_lora=8.0,
        vram_gb_qlora=5.0,
        vram_gb_full=24.0,
        ram_gb_required=8.0,
        context_window=8192,
        tier=HardwareTier.MEDIUM,
        description="Gemma 2 9B — good quality, fits T4 with QLoRA",
    ),
    ModelSpec(
        name="mistralai/Mistral-7B-Instruct-v0.3",
        family="mistral",
        params_b=7.3,
        vram_gb_lora=7.0,
        vram_gb_qlora=4.5,
        vram_gb_full=20.0,
        ram_gb_required=8.0,
        context_window=32768,
        tier=HardwareTier.MEDIUM,
        description="Mistral 7B — strong baseline, 32K context",
    ),
    ModelSpec(
        name="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        params_b=3.2,
        vram_gb_lora=4.0,
        vram_gb_qlora=2.5,
        vram_gb_full=10.0,
        ram_gb_required=6.0,
        context_window=8192,
        tier=HardwareTier.LOW,
        description="Llama 3.2 3B — Meta's latest small model",
        requires_auth=True,
    ),
    ModelSpec(
        name="meta-llama/Llama-3.2-1B-Instruct",
        family="llama",
        params_b=1.2,
        vram_gb_lora=2.0,
        vram_gb_qlora=1.0,
        vram_gb_full=4.0,
        ram_gb_required=2.0,
        context_window=8192,
        tier=HardwareTier.LOW,
        description="Llama 3.2 1B — smallest Llama, runs anywhere",
        requires_auth=True,
    ),
    ModelSpec(
        name="Qwen/Qwen2.5-7B-Instruct",
        family="qwen",
        params_b=7.6,
        vram_gb_lora=7.0,
        vram_gb_qlora=4.5,
        vram_gb_full=20.0,
        ram_gb_required=8.0,
        context_window=32768,
        tier=HardwareTier.MEDIUM,
        description="Qwen 2.5 7B — strong multilingual, 32K context",
    ),

    # --- Medium models (7B-14B) — need V100/A10G ---
    ModelSpec(
        name="unsloth/phi-4",
        family="phi",
        params_b=14.7,
        vram_gb_lora=12.0,
        vram_gb_qlora=8.0,
        vram_gb_full=40.0,
        ram_gb_required=12.0,
        context_window=16384,
        tier=HardwareTier.HIGH,
        description="Phi-4 14.7B — Microsoft's latest, strong reasoning",
    ),
    ModelSpec(
        name="meta-llama/Llama-3.1-8B-Instruct",
        family="llama",
        params_b=8.0,
        vram_gb_lora=8.0,
        vram_gb_qlora=5.0,
        vram_gb_full=22.0,
        ram_gb_required=8.0,
        context_window=131072,
        tier=HardwareTier.MEDIUM,
        description="Llama 3.1 8B — excellent all-rounder, 128K context",
        requires_auth=True,
    ),
    ModelSpec(
        name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        family="mixtral",
        params_b=46.7,
        vram_gb_lora=32.0,
        vram_gb_qlora=20.0,
        vram_gb_full=96.0,
        ram_gb_required=24.0,
        context_window=32768,
        tier=HardwareTier.VERY_HIGH,
        description="Mixtral 8x7B MoE — strong, needs A100",
    ),
    ModelSpec(
        name="Qwen/Qwen2.5-14B-Instruct",
        family="qwen",
        params_b=14.8,
        vram_gb_lora=12.0,
        vram_gb_qlora=8.0,
        vram_gb_full=40.0,
        ram_gb_required=12.0,
        context_window=32768,
        tier=HardwareTier.HIGH,
        description="Qwen 2.5 14B — strong general model",
    ),
    ModelSpec(
        name="google/gemma-2-27b",
        family="gemma",
        params_b=27.2,
        vram_gb_lora=20.0,
        vram_gb_qlora=12.0,
        vram_gb_full=70.0,
        ram_gb_required=16.0,
        context_window=8192,
        tier=HardwareTier.VERY_HIGH,
        description="Gemma 2 27B — high quality, needs A100",
    ),

    # --- Large models (14B-70B) — need A100-80GB ---
    ModelSpec(
        name="meta-llama/Llama-3.3-70B-Instruct",
        family="llama",
        params_b=70.6,
        vram_gb_lora=48.0,
        vram_gb_qlora=28.0,
        vram_gb_full=160.0,
        ram_gb_required=32.0,
        context_window=131072,
        tier=HardwareTier.EXTREME,
        description="Llama 3.3 70B — best open model, needs big GPU",
        requires_auth=True,
    ),
    ModelSpec(
        name="Qwen/Qwen2.5-72B-Instruct",
        family="qwen",
        params_b=72.7,
        vram_gb_lora=48.0,
        vram_gb_qlora=28.0,
        vram_gb_full=160.0,
        ram_gb_required=32.0,
        context_window=32768,
        tier=HardwareTier.EXTREME,
        description="Qwen 2.5 72B — strongest Qwen, 32K context",
    ),
    ModelSpec(
        name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        family="deepseek",
        params_b=32.0,
        vram_gb_lora=24.0,
        vram_gb_qlora=14.0,
        vram_gb_full=80.0,
        ram_gb_required=16.0,
        context_window=8192,
        tier=HardwareTier.VERY_HIGH,
        description="DeepSeek R1 Distill 32B — strong reasoning via distillation",
    ),
]

# Index by name for fast lookup
_MODEL_INDEX = {spec.name: spec for spec in MODEL_REGISTRY}


# ------------------------------------------------------------------ #
#  Hardware Detection
# ------------------------------------------------------------------ #

def detect_hardware() -> dict:
    """
    Detect available hardware (GPU, VRAM, RAM) using torch + psutil.
    Works in both Colab and local environments.

    Returns:
        dict with keys: gpu_name, vram_total_gb, vram_free_gb,
                        ram_total_gb, ram_available_gb, cuda_version,
                        gpu_count, is_tpu, tier
    """
    import importlib.util
    info = {
        "gpu_name": None,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "ram_total_gb": 0.0,
        "ram_available_gb": 0.0,
        "cuda_version": None,
        "gpu_count": 0,
        "is_tpu": False,
        "tier": HardwareTier.NONE.value,
    }

    # RAM
    if importlib.util.find_spec("psutil"):
        import psutil
        ram = psutil.virtual_memory()
        info["ram_total_gb"] = round(ram.total / 1e9, 1)
        info["ram_available_gb"] = round(ram.available / 1e9, 1)

    # GPU
    if importlib.util.find_spec("torch"):
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            info["vram_free_gb"] = round(
                info["vram_total_gb"] - torch.cuda.memory_allocated(0) / 1e9, 1)
            info["cuda_version"] = torch.version.cuda
            info["tier"] = _tier_from_vram(info["vram_total_gb"]).value

        # TPU check
        try:
            import torch_xla
            info["is_tpu"] = True
            info["tier"] = HardwareTier.HIGH.value
        except ImportError:
            pass

    # nvidia-smi fallback if torch not available
    if not info["gpu_name"]:
        info.update(_detect_via_nvidia_smi())

    if not info["tier"] or info["tier"] == HardwareTier.NONE.value:
        if info["vram_total_gb"] > 0:
            info["tier"] = _tier_from_vram(info["vram_total_gb"]).value
        elif info["gpu_name"]:
            info["tier"] = _tier_from_name(info["gpu_name"]).value
        else:
            info["tier"] = HardwareTier.NONE.value

    return info


def _detect_via_nvidia_smi() -> dict:
    """Fallback GPU detection via nvidia-smi."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = out.stdout.strip().split(", ")
            name = parts[0] if len(parts) > 0 else None
            total_mib = float(parts[1]) if len(parts) > 1 else 0
            free_mib = float(parts[2]) if len(parts) > 2 else 0
            return {
                "gpu_name": name,
                "vram_total_gb": round(total_mib / 1024, 1),
                "vram_free_gb": round(free_mib / 1024, 1),
            }
    except Exception:
        pass
    return {}


def _tier_from_vram(vram_gb: float) -> HardwareTier:
    if vram_gb >= 80:
        return HardwareTier.EXTREME
    if vram_gb >= 40:
        return HardwareTier.VERY_HIGH
    if vram_gb >= 24:
        return HardwareTier.HIGH
    if vram_gb >= 12:
        return HardwareTier.MEDIUM
    if vram_gb >= 4:
        return HardwareTier.LOW
    return HardwareTier.NONE


def _tier_from_name(name: str) -> HardwareTier:
    name_lower = name.lower()
    if any(kw in name_lower for kw in ["h100", "h200", "a100-80", "a100 80"]):
        return HardwareTier.EXTREME
    if any(kw in name_lower for kw in ["a100", "a10g", "a10", "l40s", "l40"]):
        return HardwareTier.VERY_HIGH
    if any(kw in name_lower for kw in ["v100", "v100s", "a6000", "a5000", "rtx 6000"]):
        return HardwareTier.HIGH
    if any(kw in name_lower for kw in ["t4", "p100", "rtx 3080", "rtx 3090",
                                         "rtx 4080", "rtx 4090"]):
        return HardwareTier.MEDIUM
    if any(kw in name_lower for kw in ["k80", "p4", "t2", "gtx"]):
        return HardwareTier.LOW
    return HardwareTier.NONE


# ------------------------------------------------------------------ #
#  Model Selector
# ------------------------------------------------------------------ #

class ModelSelector:
    """
    Selects the best model for available hardware.

    Thread-safe. Supports both local and Colab hardware detection.
    Can be called with explicit specs or auto-detect.

    Usage:
        sel = ModelSelector()
        result = sel.best_fit()  # auto-detect hardware
        result = sel.best_fit(vram_gb=16, ram_gb=12, method="lora")
    """

    def __init__(self, registry: Optional[list[ModelSpec]] = None):
        self._registry = MODEL_REGISTRY if registry is None else list(registry)
        self._lock = threading.RLock()
        self._hardware_cache: Optional[dict] = None
        self._cache_time = 0.0
        self._cache_ttl = 30.0  # re-detect every 30s

    @property
    def registry(self) -> list[ModelSpec]:
        return self._registry

    # ------------------------------------------------------------------ #
    #  Detection
    # ------------------------------------------------------------------ #

    def detect(self, force: bool = False) -> dict:
        """Detect hardware specs. Cached for cache_ttl seconds."""
        if force or not self._hardware_cache or \
           (time.time() - self._cache_time > self._cache_ttl):
            self._hardware_cache = detect_hardware()
            self._cache_time = time.time()
        return dict(self._hardware_cache)

    # ------------------------------------------------------------------ #
    #  Model selection
    # ------------------------------------------------------------------ #

    def best_fit(self, vram_gb: Optional[float] = None,
                 ram_gb: Optional[float] = None,
                 method: str = "qlora",
                 prefer_family: Optional[str] = None,
                 require_context: Optional[int] = None,
                 exclude_auth: bool = False) -> dict:
        """
        Select the best model for available hardware.

        Args:
            vram_gb: Available VRAM. None = auto-detect.
            ram_gb: Available RAM. None = auto-detect.
            method: Fine-tuning method (lora, qlora, full).
            prefer_family: Optional family to prefer (llama, phi, gemma, etc.).
            require_context: Minimum context window required.
            exclude_auth: If True, skip models requiring HF auth.

        Returns:
            dict with keys: name, family, params_b, method, tier,
                            vram_needed, fits, explanation
        """
        hw = self.detect()
        vram = vram_gb if vram_gb is not None else hw.get("vram_total_gb", 0)
        ram = ram_gb if ram_gb is not None else hw.get("ram_total_gb", 0)
        tier = hw.get("tier", HardwareTier.NONE.value)

        candidates = []
        for spec in self._registry:
            if not spec.fits_in(vram, ram, method):
                continue
            if require_context and spec.context_window < require_context:
                continue
            if exclude_auth and spec.requires_auth:
                continue
            if prefer_family and spec.family != prefer_family:
                continue
            candidates.append(spec)

        if not candidates:
            # Try QLoRA if LoRA/full didn't fit
            if method != "qlora":
                return self.best_fit(vram_gb, ram_gb, "qlora",
                                     prefer_family, require_context, exclude_auth)

            # Last resort: no model fits at all
            if self._registry:
                smallest = min(self._registry, key=lambda s: s.params_b)
                return {
                    "name": smallest.name,
                    "family": smallest.family,
                    "params_b": smallest.params_b,
                    "method": "qlora",
                    "tier": tier,
                    "vram_needed": smallest.vram_gb_qlora,
                    "vram_available": vram,
                    "ram_available": ram,
                    "fits": False,
                    "explanation": (
                        f"No model fits in {vram:.0f}GB VRAM / {ram:.0f}GB RAM. "
                        f"Smallest model ({smallest.name}) needs "
                        f"{smallest.vram_gb_qlora:.0f}GB VRAM."
                    ),
                }
            return {
                "name": "none",
                "family": "none",
                "params_b": 0,
                "method": "none",
                "tier": tier,
                "vram_needed": vram,
                "vram_available": vram,
                "ram_available": ram,
                "fits": False,
                "explanation": "No models in registry.",
            }

        # Prefer larger models (sorted by params descending)
        candidates.sort(key=lambda s: s.params_b, reverse=True)

        for spec in candidates:
            result = {
                "name": spec.name,
                "family": spec.family,
                "params_b": spec.params_b,
                "method": method,
                "tier": spec.tier.value,
                "vram_total": vram,
                "vram_needed": spec.min_vram(method),
                "ram_total": ram,
                "context_window": spec.context_window,
                "requires_auth": spec.requires_auth,
                "fits": True,
                "explanation": (
                    f"{spec.name} ({spec.params_b:.1f}B) fits in "
                    f"{vram:.0f}GB VRAM with {method.upper()}. "
                    f"Needs {spec.min_vram(method):.0f}GB VRAM."
                ),
            }
            if prefer_family:
                result["explanation"] += f" Preferred family: {prefer_family}."
            return result

        smallest = candidates[-1]
        return {
            "name": smallest.name,
            "family": smallest.family,
            "params_b": smallest.params_b,
            "method": "qlora",
            "tier": tier,
            "vram_needed": smallest.min_vram("qlora"),
            "vram_available": vram,
            "ram_available": ram,
            "fits": False,
            "explanation": (
                f"Models need {smallest.min_vram('qlora'):.0f}GB VRAM, "
                f"but only {vram:.0f}GB available."
            ),
        }

    def models_that_fit(self, vram_gb: Optional[float] = None,
                        ram_gb: Optional[float] = None,
                        method: str = "qlora") -> list[dict]:
        """List all models that fit in the given hardware."""
        hw = self.detect()
        vram = vram_gb if vram_gb is not None else hw.get("vram_total_gb", 0)
        ram = ram_gb if ram_gb is not None else hw.get("ram_total_gb", 0)

        results = []
        for spec in self._registry:
            results.append({
                "name": spec.name,
                "family": spec.family,
                "params_b": spec.params_b,
                "fits": spec.fits_in(vram, ram, method),
                "vram_needed": spec.min_vram(method),
                "method": method,
            })
        return sorted(results, key=lambda r: -r["params_b"])

    def recommend_method(self, vram_gb: Optional[float] = None,
                         params_b: float = 7.0) -> str:
        """Recommend fine-tuning method based on VRAM and model size."""
        hw = self.detect()
        vram = vram_gb if vram_gb is not None else hw.get("vram_total_gb", 0)

        if vram >= 40 and params_b < 7:
            return "full"
        if vram >= 16 and params_b < 7:
            return "lora"
        if vram >= 16 and params_b < 20:
            return "lora"
        if vram >= 40 and params_b < 70:
            return "qlora"
        return "qlora"

    def get_model(self, name: str) -> Optional[ModelSpec]:
        """Look up a model in the registry by name."""
        return _MODEL_INDEX.get(name)

    def register_model(self, spec: ModelSpec):
        """Add a custom model to the registry (thread-safe)."""
        with self._lock:
            self._registry.append(spec)
            _MODEL_INDEX[spec.name] = spec

    def summary(self, vram_gb: Optional[float] = None,
                ram_gb: Optional[float] = None) -> dict:
        """Full hardware + model recommendation summary."""
        hw = self.detect()
        vram = vram_gb if vram_gb is not None else hw.get("vram_total_gb", 0)
        ram = ram_gb if ram_gb is not None else hw.get("ram_total_gb", 0)

        best = self.best_fit(vram, ram)
        return {
            "hardware": {
                "gpu": hw.get("gpu_name", None),
                "vram_total_gb": hw.get("vram_total_gb", 0),
                "vram_free_gb": hw.get("vram_free_gb", 0),
                "ram_total_gb": hw.get("ram_total_gb", 0),
                "ram_available_gb": hw.get("ram_available_gb", 0),
                "tier": hw.get("tier", "none"),
                "cuda": hw.get("cuda_version", None),
            },
            "recommended_model": best,
            "all_fitting": self.models_that_fit(vram, ram, best.get("method", "qlora")),
        }


# ------------------------------------------------------------------ #
#  CLI helper
# ------------------------------------------------------------------ #

def print_model_summary(selector: Optional[ModelSelector] = None):
    """Print a human-readable hardware + model recommendation."""
    sel = selector or ModelSelector()
    summary = sel.summary()

    hw = summary["hardware"]
    print("=== Hardware ===")
    print(f"  GPU:        {hw.get('gpu') or 'None (CPU)'}")
    print(f"  VRAM:       {hw['vram_total_gb']:.1f} GB total, "
          f"{hw['vram_free_gb']:.1f} GB free")
    print(f"  RAM:        {hw['ram_total_gb']:.1f} GB total, "
          f"{hw['ram_available_gb']:.1f} GB available")
    print(f"  Tier:       {hw['tier']}")
    print(f"  CUDA:       {hw.get('cuda') or 'N/A'}")

    best = summary["recommended_model"]
    print(f"\n=== Recommended Model ===")
    print(f"  Model:  {best['name']}")
    print(f"  Params: {best['params_b']:.1f}B")
    print(f"  Method: {best['method'].upper()}")
    print(f"  Fits:   {best['fits']}")
    print(f"  Why:    {best['explanation']}")

    all_f = summary["all_fitting"]
    fitting = [m for m in all_f if m["fits"]]
    if fitting:
        print(f"\n=== All {len(fitting)} Models That Fit ===")
        for m in fitting:
            print(f"  - {m['name']} ({m['params_b']:.1f}B, "
                  f"needs {m['vram_needed']:.0f}GB VRAM)")
