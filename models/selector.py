"""
Phase 3 — HuggingFace Integration items 26-35.

Provides:
  - HFModelSelector         auto-pick model based on task + memory  (item 26)
  - DatasetLoader           HF datasets, JSON, CSV, Parquet, GDrive (item 27)
  - ModelSizeEstimator      predict VRAM/RAM before loading         (item 28)
  - TokenizerManager        padding/truncation auto-config          (item 29)
  - CacheManager            reuse downloaded models                 (item 30)
  - HFHubPusher             authentication + version tagging        (item 31)
  - ModelCardGenerator      auto-create README                     (item 32)
  - AdapterMerger           LoRA -> full model merge                (item 33)
  - QuantizationSelector    4-bit / 8-bit / none based on runtime   (item 34)
  - ModelDownloadProgressBar with ETA                               (item 35)
"""
import os
import json
from typing import Optional

from config.settings import settings


# ==================================================================== #
#  26 — HFModelSelector
# ==================================================================== #

TASK_MODEL_MAP = {
    "text-generation": [
        "microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct",
        "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b-hf",
        "google/gemma-2b", "google/gemma-7b",
        "HuggingFaceH4/zephyr-7b-beta", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ],
    "text-classification": [
        "distilbert-base-uncased", "bert-base-uncased",
        "roberta-base", "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ],
    "token-classification": [
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "dslim/bert-base-NER", "Jean-Baptiste/roberta-large-ner-english",
    ],
    "question-answering": [
        "distilbert-base-cased-distilled-squad",
        "bert-large-uncased-whole-word-masking-finetuned-squad",
    ],
    "summarization": [
        "facebook/bart-large-cnn", "google/pegasus-xsum",
        "Falconsai/text_summarization",
    ],
    "translation": [
        "Helsinki-NLP/opus-mt-en-fr", "facebook/nllb-200-distilled-600M",
    ],
    "code-generation": [
        "microsoft/phi-2", "codellama/CodeLlama-7b-hf",
        "bigcode/starcoder2-3b", "bigcode/starcoder2-7b",
        "deepseek-ai/deepseek-coder-1.3b-instruct",
    ],
    "image-classification": [
        "google/vit-base-patch16-224", "microsoft/resnet-50",
    ],
}


class HFModelSelector:
    """Auto-pick the best model for a task given runtime constraints."""

    @staticmethod
    def recommend(task: str, vram_gb: float = 16.0,
                  prefer_small: bool = False) -> list[dict]:
        candidates = TASK_MODEL_MAP.get(task, TASK_MODEL_MAP["text-generation"])
        from models.configs import estimate_memory
        results = []
        for model in candidates:
            info = estimate_memory(model)
            vram_needed = info["parameters_b"] * (0.5 if vram_gb < 32 else 1.5)
            fits = vram_needed <= vram_gb * 0.9
            results.append({
                "name": model,
                "params_b": info["parameters_b"],
                "label": info["label"],
                "vram_estimate_gb": round(vram_needed, 1),
                "fits_on_current_runtime": fits,
            })
        results.sort(key=lambda x: (not x["fits_on_current_runtime"],
                                     x["params_b"] if prefer_small else -x["params_b"]))
        return results

    @staticmethod
    def best_fit(task: str, vram_gb: float = 16.0) -> Optional[str]:
        candidates = HFModelSelector.recommend(task, vram_gb)
        for c in candidates:
            if c["fits_on_current_runtime"]:
                return c["name"]
        return candidates[0]["name"] if candidates else "microsoft/phi-2"

    @staticmethod
    def selector_code(task: str, vram_gb: float = 16.0) -> str:
        best = HFModelSelector.best_fit(task, vram_gb)
        return f"""
from models.selector import HFModelSelector
task = "{task}"
candidates = HFModelSelector.recommend(task, vram_gb={vram_gb})
print(f"Best model for {{task}}: {{candidates[0]['name']}}" if candidates else "No model found")
MODEL_NAME = "{best}"
"""


# ==================================================================== #
#  27 — DatasetLoader
# ==================================================================== #

class DatasetLoader:
    """Load datasets from HF Hub, JSON, CSV, Parquet, or Google Drive."""

    SUPPORTED_FORMATS = {".json": "json", ".jsonl": "json",
                         ".csv": "csv", ".tsv": "csv",
                         ".parquet": "parquet"}

    @staticmethod
    def detect_format(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return DatasetLoader.SUPPORTED_FORMATS.get(ext, "hf")

    @staticmethod
    def load_code(path_or_name: str, split: str = "train",
                  text_column: str = "text",
                  val_split: float = 0.0) -> str:
        fmt = DatasetLoader.detect_format(path_or_name)

        if fmt == "hf":
            val_code = f"""
split_dataset = dataset.train_test_split(test_size={val_split}) if {val_split} > 0 else {{"train": dataset}}
""" if val_split > 0 else ""
            return f"""
from datasets import load_dataset
dataset = load_dataset("{path_or_name}", split="{split}")
print(f"Loaded: {{len(dataset)}} samples")
print(f"Columns: {{dataset.column_names}}")
{val_code}
"""

        elif fmt == "csv":
            return f"""
import pandas as pd
from datasets import Dataset
df = pd.read_csv("{path_or_name}")
dataset = Dataset.from_pandas(df)
print(f"Loaded CSV: {{len(dataset)}} rows, columns: {{list(df.columns)}}")
"""

        elif fmt == "json":
            return f"""
import pandas as pd
from datasets import Dataset
df = pd.read_json("{path_or_name}")
dataset = Dataset.from_pandas(df)
print(f"Loaded JSON: {{len(dataset)}} rows")
"""

        elif fmt == "parquet":
            return f"""
import pandas as pd
from datasets import Dataset
df = pd.read_parquet("{path_or_name}")
dataset = Dataset.from_pandas(df)
print(f"Loaded Parquet: {{len(dataset)}} rows")
"""
        return f"print('Unknown format for {path_or_name}')"

    @staticmethod
    def from_drive_code(drive_path: str) -> str:
        return f"""
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
from datasets import Dataset
path = "{drive_path}"
ext = path.split('.')[-1]
if ext == 'csv': df = pd.read_csv(path)
elif ext == 'json': df = pd.read_json(path)
else: raise ValueError(f"Unsupported: {{ext}}")
dataset = Dataset.from_pandas(df)
print(f"Loaded from Drive: {{len(dataset)}} rows")
"""


# ==================================================================== #
#  28 — ModelSizeEstimator
# ==================================================================== #

class ModelSizeEstimator:
    """Predict memory usage before loading a model."""

    @staticmethod
    def estimate_vram(model_params_b: float, method: str,
                      seq_length: int = 512,
                      batch_size: int = 4) -> dict:
        from colab.runtime import estimate_needed_vram
        base = estimate_needed_vram(model_params_b, method)
        overhead = (seq_length * batch_size * model_params_b * 1e9 * 2 * 4) / 1e9 * 0.1
        return {
            "model_params_b": model_params_b,
            "method": method,
            "vram_model_gb": round(base, 2),
            "vram_activation_gb": round(overhead, 2),
            "vram_total_gb": round(base + overhead, 2),
            "ram_overhead_gb": round(model_params_b * 0.5, 2),
        }

    @staticmethod
    def loading_code(model_name: str, method: str = "qlora") -> str:
        from models.configs import estimate_memory
        info = estimate_memory(model_name)
        vram = ModelSizeEstimator.estimate_vram(info["parameters_b"], method)
        return f"""
import json, torch
estimate = {json.dumps(vram)}
print(f"Estimated VRAM needed: {{estimate['vram_total_gb']}} GB")
print(f"Available VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB" if torch.cuda.is_available() else "No GPU")
if estimate['vram_total_gb'] > (torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0):
    print("WARNING: May not fit in VRAM!")
else:
    print("Should fit comfortably")
"""


# ==================================================================== #
#  29 — TokenizerManager
# ==================================================================== #

class TokenizerManager:
    """Auto-configure padding, truncation, special tokens."""

    @staticmethod
    def auto_config_code(model_name: str, max_length: int = 512) -> str:
        return f"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("{model_name}", trust_remote_code=True)
# Auto-configure padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"
print(f"Tokenizer: vocab_size={{tokenizer.vocab_size}}, max_length={{tokenizer.model_max_length}}")
print(f"pad_token={{tokenizer.pad_token}}, eos_token={{tokenizer.eos_token}}")
"""

    @staticmethod
    def estimate_tokens(texts: list[str], model_name: str) -> dict:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            lengths = [len(tokenizer.encode(t)) for t in texts]
            return {
                "min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths),
                "total": sum(lengths), "count": len(lengths),
            }
        except Exception:
            return {"error": "Could not estimate"}


# ==================================================================== #
#  30 — CacheManager
# ==================================================================== #

class CacheManager:
    """Reuse downloaded models across sessions via symlinks or Drive cache."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser(
            settings.hf_cache_dir or "~/.cache/huggingface"
        )

    def get_cached_path(self, model_name: str) -> Optional[str]:
        sanitized = model_name.replace("/", "--")
        snapshot_dir = os.path.join(self.cache_dir, "hub", f"models--{sanitized}")
        if os.path.isdir(snapshot_dir):
            refs = os.path.join(snapshot_dir, "refs")
            if os.path.isdir(refs):
                for fname in os.listdir(refs):
                    blob = os.path.join(snapshot_dir, "snapshots", fname)
                    if os.path.isdir(blob):
                        return blob
        return None

    def is_cached(self, model_name: str) -> bool:
        return self.get_cached_path(model_name) is not None

    def cache_size(self) -> str:
        total = 0
        hub_dir = os.path.join(self.cache_dir, "hub")
        if os.path.isdir(hub_dir):
            for root, dirs, files in os.walk(hub_dir):
                total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return f"{total / 1e9:.2f} GB"

    def clear_cache(self):
        import shutil
        hub_dir = os.path.join(self.cache_dir, "hub")
        if os.path.isdir(hub_dir):
            shutil.rmtree(hub_dir)
            os.makedirs(hub_dir)


# ==================================================================== #
#  31 — HFHubPusher
# ==================================================================== #

class HFHubPusher:
    """Push models to HuggingFace Hub with authentication and version tagging."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or settings.hf_token

    def push_code(self, local_path: str, repo_id: str,
                  commit_message: str = "Fine-tuned model",
                  create_pr: bool = False) -> str:
        pr_block = '\n    create_pr=True,' if create_pr else ''
        return f"""
from huggingface_hub import HfApi, login
login(token="{self.token or 'YOUR_TOKEN'}")
api = HfApi()
api.create_repo(repo_id="{repo_id}", exist_ok=True)
api.upload_folder(
    folder_path="{local_path}",
    repo_id="{repo_id}",
    commit_message="{commit_message}",{pr_block}
)
print(f"Pushed: https://huggingface.co/{{repo_id}}")

# Tag version
try:
    api.create_tag(repo_id="{repo_id}", tag="v1.0", message="Fine-tuned version")
except Exception:
    pass
"""

    @staticmethod
    def generate_model_card(model_name: str, dataset: str,
                            method: str, metrics: dict = None) -> str:
        return ModelCardGenerator.generate(model_name, dataset, method, metrics)


# ==================================================================== #
#  32 — ModelCardGenerator
# ==================================================================== #

class ModelCardGenerator:
    """Auto-create README.md for pushed models."""

    @staticmethod
    def generate(base_model: str, dataset: str, method: str,
                 metrics: dict = None, description: str = "") -> str:
        metrics_str = "\n".join(
            f"  - {k}: {v}" for k, v in (metrics or {}).items()
        ) or "  - Loss: TBD"

        return f"""---
license: mit
base_model: {base_model}
tags:
  - colab-agent
  - fine-tuned
  - {method}
datasets:
  - {dataset}
---

# Fine-Tuned Model

**Base model:** {base_model}
**Fine-tuning method:** {method.upper()}
**Dataset:** {dataset}
**Fine-tuned with:** Colab Agent

## Description
{description or f"Model fine-tuned from {base_model} on {dataset} using {method.upper()}."}

## Metrics
{metrics_str}

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("YOUR_REPO_ID")
model = AutoModelForCausalLM.from_pretrained("YOUR_REPO_ID")
```

## Training Details
- Framework: HuggingFace Transformers + PEFT
- Platform: Google Colab
- Generation: Colab Agent
"""


# ==================================================================== #
#  33 — AdapterMerger
# ==================================================================== #

class AdapterMerger:
    """Merge LoRA/QLoRA adapters into the base model."""

    @staticmethod
    def merge_code(peft_model_path: str, output_path: str,
                   base_model_name: Optional[str] = None) -> str:
        return f"""
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base = "{base_model_name or 'BASE_MODEL'}"
print(f"Loading base model: {{base}}")
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base)

print(f"Loading adapter from: {peft_model_path}")
model = PeftModel.from_pretrained(model, "{peft_model_path}")

print("Merging adapter into base model...")
model = model.merge_and_unload()
print(f"Merge complete! Saving to: {output_path}")

model.save_pretrained("{output_path}")
tokenizer.save_pretrained("{output_path}")
print(f"Full model saved to {{output_path}}")
print(f"Size on disk will be ~{{sum(f.stat().st_size for f in __import__('pathlib').Path('{output_path}').rglob('*')) / 1e9:.2f}} GB")
"""


# ==================================================================== #
#  34 — QuantizationSelector
# ==================================================================== #

class QuantizationSelector:
    """Auto-select 4-bit, 8-bit, or none based on runtime."""

    @staticmethod
    def select(vram_gb: float, model_params_b: float) -> str:
        if vram_gb < 8:
            return "4bit"
        if model_params_b > 7 and vram_gb < 24:
            return "4bit"
        if model_params_b > 3 and vram_gb < 16:
            return "4bit"
        if model_params_b > 7 and vram_gb < 40:
            return "8bit"
        if vram_gb < 8:
            return "4bit"
        return "none"

    @staticmethod
    def config_code(vram_gb: float, model_params_b: float) -> str:
        quant = QuantizationSelector.select(vram_gb, model_params_b)
        if quant == "4bit":
            return """
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
"""
        elif quant == "8bit":
            return """
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_8bit=True)
"""
        return "quant_config = None"


# ==================================================================== #
#  35 — ModelDownloadProgressBar
# ==================================================================== #

class ModelDownloadProgressBar:
    """Show download progress with ETA using tqdm."""

    @staticmethod
    def wrapping_code() -> str:
        return """
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import os

class DownloadProgress:
    def __init__(self, desc="Downloading"):
        self.pbar = None
        self.desc = desc
    def __call__(self, current, total, *args):
        if self.pbar is None:
            self.pbar = tqdm(total=total, desc=self.desc, unit="B", unit_scale=True)
        self.pbar.update(current - self.pbar.n)
        if current >= total:
            self.pbar.close()

# Usage:
# path = snapshot_download("microsoft/phi-2", progress_callback=DownloadProgress())
# print(f"Downloaded to: {path}")
"""
