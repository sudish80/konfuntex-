"""
Phases 36-41 — Model Optimization.

Provides:
  - LoRAAdapterZoo            store/switch/merge adapters    (36)
  - ModelMerger               merge adapters into base      (37)
  - GradientCheckpointer      memory-saving checkpoint       (38)
  - MixedPrecisionSelector    auto fp16/bf16 based on GPU   (39)
  - FlashAttentionIntegration auto-detect + enable FA2      (40)
  - QuantizationAwareTraining QAT simulation during training (41)
"""
import json
import os


# ==================================================================== #
#  36 — LoRAAdapterZoo
# ==================================================================== #

class LoRAAdapterZoo:
    """
    Store, switch, and merge multiple LoRA adapters for different tasks.
    """

    def __init__(self, adapters_dir: str = "./adapters"):
        self.adapters_dir = adapters_dir
        os.makedirs(adapters_dir, exist_ok=True)

    def save_adapter_code(self, task_name: str, base_model: str,
                          rank: int = 16, alpha: int = 32) -> str:
        return f"""
import os, json

task = "{task_name}"
base_model = "{base_model}"
adapters_dir = "{self.adapters_dir}"
task_dir = os.path.join(adapters_dir, task)
os.makedirs(task_dir, exist_ok=True)

# Save metadata
metadata = {{
    "task": task,
    "base_model": base_model,
    "rank": {rank},
    "lora_alpha": {alpha},
    "created": str(__import__("datetime").datetime.now()),
}}
with open(os.path.join(task_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# Save adapter from trained model
model.save_pretrained(task_dir)
print(f"Adapter saved to {{task_dir}}")

# Register in zoo index
index_path = os.path.join(adapters_dir, "zoo_index.json")
if os.path.exists(index_path):
    with open(index_path) as f:
        index = json.load(f)
else:
    index = {{"adapters": []}}

index["adapters"].append(metadata)
with open(index_path, "w") as f:
    json.dump(index, f, indent=2)
"""

    def switch_adapter_code(self) -> str:
        return """
# Load a different adapter
from peft import PeftModel
task = "sentiment"  # or any registered task
task_dir = os.path.join(adapters_dir, task)

# Load base model first
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

# Load adapter
model = PeftModel.from_pretrained(model, task_dir)
model.eval()
print(f"Switched to adapter: {task}")
"""

    def list_adapters_code(self) -> str:
        return """
import json, os
adapters_dir = "./adapters"
index_path = os.path.join(adapters_dir, "zoo_index.json")

if os.path.exists(index_path):
    with open(index_path) as f:
        index = json.load(f)
    print(f"Available adapters ({len(index['adapters'])}):")
    for a in index["adapters"]:
        print(f"  {a['task']}: base={a['base_model']}, rank={a['rank']}")
else:
    print("No adapters registered")

# List directories
for d in os.listdir(adapters_dir):
    meta_path = os.path.join(adapters_dir, d, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  {d}: {meta.get('base_model', 'unknown')}")
"""


# ==================================================================== #
#  37 — ModelMerger
# ==================================================================== #

class ModelMerger:
    """
    Merge LoRA adapters into the base model, including
    TIES-merging and task arithmetic for multiple adapters.
    """

    def __init__(self, output_dir: str = "./merged_models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def merge_code(self, adapter_path: str = "./adapters/sentiment",
                   base_model_name: str = "microsoft/phi-2") -> str:
        return f"""
import torch, os, json
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base_model = "{base_model_name}"
adapter_path = "{adapter_path}"
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

print(f"Loading base model: {{base_model}}")
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True,
)

print(f"Loading adapter from: {{adapter_path}}")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging adapter into base model...")
model = model.merge_and_unload()
print("Merge complete!")

# Save merged model
save_path = os.path.join(output_dir, base_model.replace("/", "_") + "_merged")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Calculate size
total_size = sum(f.stat().st_size for f in __import__("pathlib").Path(save_path).rglob("*"))
print(f"Merged model saved to {{save_path}} ({{total_size / 1e9:.2f}} GB)")
"""

    def ties_merge_code(self, adapters: list[dict]) -> str:
        return f"""
import torch, os, json
import numpy as np
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

adapters = {json.dumps(adapters)}
base_model = adapters[0]["base"]
output_dir = "{self.output_dir}"

# Load base
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16, device_map="auto",
)

# TIES-merging: trim, elect, sum
task_vectors = []
for adapter in adapters:
    peft_model = PeftModel.from_pretrained(model, adapter["path"])
    tv = {{}}
    for name, param in peft_model.named_parameters():
        if "lora" in name:
            tv[name] = param.data.clone()
    task_vectors.append(tv)

merged_tv = {{}}
for name in task_vectors[0]:
    # Stack all task vectors
    stacked = torch.stack([tv[name] for tv in task_vectors], dim=0)
    # Trim: keep top-K% by magnitude
    k = int(stacked.shape[0] * 0.7)
    _, indices = torch.topk(stacked.abs().mean(dim=tuple(range(1, stacked.dim()))), k)
    # Elect: majority sign
    sign = torch.sign(stacked).mean(dim=0)
    # Sum remaining
    merged_tv[name] = (stacked * (sign > 0).float()).sum(dim=0)

# Apply merged
for name, param in peft_model.named_parameters():
    if name in merged_tv:
        param.data = merged_tv[name]

peft_model = peft_model.merge_and_unload()
save_path = os.path.join(output_dir, "ties_merged")
peft_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"TIES-merged model saved to {{save_path}}")
"""


# ==================================================================== #
#  38 — GradientCheckpointer (utility)
# ==================================================================== #

class GradientCheckpointer:
    """
    Enable gradient checkpointing to trade compute for memory.
    Wraps the standard `model.gradient_checkpointing_enable()`.
    """

    @staticmethod
    def enable_code() -> str:
        return """
# Enable gradient checkpointing (reduces VRAM ~30-40%)
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Must disable cache when using checkpointing
print("Gradient checkpointing enabled")

# For manual checkpointing:
from torch.utils.checkpoint import checkpoint
# wrapped_output = checkpoint(layer, input)
"""

    @staticmethod
    def auto_select_code(vram_gb: float, model_params_b: float) -> str:
        enable = "True" if vram_gb < 20 or model_params_b > 7 else "False"
        return f"""
# Auto-select gradient checkpointing based on VRAM
vram_gb = {vram_gb}
model_params_b = {model_params_b}
enable_checkpointing = {enable}

if enable_checkpointing:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("Gradient checkpointing: ENABLED")
else:
    print("Gradient checkpointing: DISABLED (sufficient VRAM)")
"""


# ==================================================================== #
#  39 — MixedPrecisionSelector
# ==================================================================== #

class MixedPrecisionSelector:
    """
    Auto-select fp16, bf16, or fp32 based on GPU capability.
    """

    @staticmethod
    def detect_code() -> str:
        return """
import torch

if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name(0)
    major, minor = capability
    print(f"GPU: {device_name}, Compute Capability: {major}.{minor}")

    if major >= 8:  # A100, V100 (7.0+ has fp16, 8.0+ has bf16)
        can_bf16 = True
        can_fp16 = True
    elif major >= 7:  # V100, T4
        can_bf16 = False
        can_fp16 = True
    else:  # Older GPUs
        can_bf16 = False
        can_fp16 = False

    recommendation = "bf16" if can_bf16 else ("fp16" if can_fp16 else "fp32")
    print(f"Recommended precision: {recommendation}")
else:
    print("No GPU detected, using fp32")
    recommendation = "fp32"
"""

    @staticmethod
    def training_args_code(precision: str = "auto") -> str:
        return f"""
import torch

# Auto-detect precision
capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
use_bf16 = capability[0] >= 8
use_fp16 = capability[0] >= 7 and not use_bf16

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    fp16={str(precision == "fp16" or precision == "auto" and "use_fp16")},
    bf16={str(precision == "bf16" or precision == "auto" and "use_bf16")},
    fp16_full_eval={str(precision == "fp16")},
)
print(f"Training with: fp16={{training_args.fp16}}, bf16={{training_args.bf16}}")
"""


# ==================================================================== #
#  40 — FlashAttentionIntegration
# ==================================================================== #

class FlashAttentionIntegration:
    """
    Auto-detect Flash Attention 2 compatibility and enable it.
    """

    @staticmethod
    def install_code() -> str:
        return """
# Install flash-attn (requires Ampere GPU or newer)
import torch
capability = torch.cuda.get_device_capability()
if capability[0] >= 8:
    print("GPU supports Flash Attention 2 (Compute Capability >= 8.0)")
    import subprocess
    subprocess.run(["pip", "install", "flash-attn", "--no-build-isolation", "-q"])
    print("Flash Attention 2 installed")
else:
    print("GPU does not support Flash Attention 2 (needs Ampere+). "
          "Using eager attention.")
"""

    @staticmethod
    def enable_code(model_name: str) -> str:
        return f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{model_name}"

# Check FA2 support
capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
fa2_supported = capability[0] >= 8

if fa2_supported:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        print(f"Flash Attention 2: ACTIVE ({{model.config._attn_implementation}})")
    except Exception as e:
        print(f"FA2 init failed: {{e}}. Falling back to eager.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    print("Flash Attention 2: NOT SUPPORTED (using eager attention)")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
"""


# ==================================================================== #
#  41 — QuantizationAwareTraining
# ==================================================================== #

class QuantizationAwareTraining:
    """
    Simulate quantization during training for better inference accuracy.
    Recommended for edge deployment scenarios only.
    """

    @staticmethod
    def setup_code() -> str:
        return """
# Quantization-Aware Training with torch.ao.quantization
import torch
import torch.ao.quantization as quant

# Prepare model for QAT
model.train()
model.qconfig = quant.get_default_qat_qconfig("fbgemm")
model = quant.prepare_qat(model, inplace=True)

print("QAT: Model prepared for quantization-aware training")

# Train normally
# trainer.train()

# After training, convert to quantized
# model.eval()
# model = quant.convert(model, inplace=True)
# print("QAT: Model converted to INT8")
"""

    @staticmethod
    def brevitas_code() -> str:
        return """
# Brevitas QAT (more advanced)
# pip install brevitas
import brevitas.nn as qnn
import torch.nn as nn

class QuantModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.quant_in = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.quant_in(input_ids)  # Quantize input
        return self.base(input_ids=x, attention_mask=attention_mask, labels=labels)

# Wrap and train
# qmodel = QuantModel(base_model)
# train with qmodel, export to ONNX after
"""
