"""Models module — configs, selector, HuggingFace, and fine-tuning."""

from .configs import (
    FINETUNE_METHODS,
    get_method_config,
    estimate_memory,
    recommend_method,
    get_lora_config,
    get_training_args,
)
from .selector import (
    ModelSelector,
    ModelSpec,
    HardwareTier,
    MODEL_REGISTRY,
    detect_hardware,
    print_model_summary,
)
from .huggingface import HuggingFaceManager
from .finetune import FinetuneCodeGenerator

__all__ = [
    "FINETUNE_METHODS",
    "get_method_config",
    "estimate_memory",
    "recommend_method",
    "get_lora_config",
    "get_training_args",
    "ModelSelector",
    "ModelSpec",
    "HardwareTier",
    "MODEL_REGISTRY",
    "detect_hardware",
    "print_model_summary",
    "HuggingFaceManager",
    "FinetuneCodeGenerator",
]


def _lazy_import(name: str, path: str):
    import importlib.util
    try:
        spec = importlib.util.find_spec(path.replace("/", ".").replace("\\", "."))
        if spec is None:
            return None
        mod = importlib.import_module(path.replace("/", ".").replace("\\", "."))
        return getattr(mod, name, None)
    except (ImportError, AttributeError):
        return None


# Lazy-loaded (avoids circular import with colab)
FineTuneOrchestrator = _lazy_import(
    "FineTuneOrchestrator", "models/finetune_orchestrator",
)
