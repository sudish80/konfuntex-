"""
FineTuneOrchestrator — Phase 3
Automates method selection, training config generation, checkpointing,
Hub push, and metrics logging for LoRA/QLoRA/full fine-tuning in Colab.
"""
import json
from typing import Optional
from models.configs import estimate_memory, recommend_method, FINETUNE_METHODS
from colab.runtime import estimate_needed_vram


class FineTuneOrchestrator:
    """
    Orchestrates the full fine-tuning lifecycle by generating
    Colab-executable Python code tailored to the model, dataset,
    available runtime, and user goal.
    """

    def verify_checkpoint_load(self, checkpoint_path: str, model_name: str) -> bool:
        """Perform a quick validation by loading the model."""
        try:
            from transformers import AutoModelForCausalLM
            # Load only the config or a very small portion to verify integrity
            AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype="auto")
            return True
        except Exception as e:
            logger.error(f"Checkpoint verification failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Top-level: select method + generate complete training script
    # ------------------------------------------------------------------ #

    def plan_training(self, base_model: str, vram_gb: float,
                      dataset_size: Optional[int] = None,
                      model_params_b: Optional[float] = None) -> dict:
        """
        Auto-select fine-tuning method and hyper-parameters based on
        model size and available VRAM.

        Returns a dict with: method, rank, alpha, lr, batch_size,
        epochs, gradient_accum, optim, target_modules, note.
        """
        if model_params_b is None:
            info = estimate_memory(base_model)
            model_params_b = info["parameters_b"]

        method, reason = recommend_method(model_params_b, vram_gb)
        method_config = FINETUNE_METHODS.get(method, {})

        # Dynamic hyper-parameter scaling
        if method == "qlora":
            rank = min(16, max(4, int(8 * (7 / max(model_params_b, 1)))))
            alpha = rank * 2
            lr = 2e-4
            batch_size = 4 if vram_gb < 32 else 8
            gradient_accum = max(1, int(32 / batch_size))
            optim = "paged_adamw_8bit"
            fp16 = True
            target_modules = method_config.get("target_modules",
                ["q_proj", "v_proj", "k_proj", "o_proj"])
        elif method == "lora":
            rank = min(64, max(8, int(32 * (7 / max(model_params_b, 1)))))
            alpha = rank * 2
            lr = 3e-4
            batch_size = 8 if vram_gb < 32 else 16
            gradient_accum = max(1, int(32 / batch_size))
            optim = "adamw_torch"
            fp16 = True
            target_modules = method_config.get("target_modules",
                ["q_proj", "v_proj"])
        else:  # full
            rank = None
            alpha = None
            lr = 5e-5
            batch_size = 8 if vram_gb < 40 else 16
            gradient_accum = max(1, int(64 / batch_size))
            optim = "adamw_torch"
            fp16 = vram_gb >= 16
            target_modules = None

        epochs = 3 if model_params_b < 13 else 2

        return {
            "method": method,
            "rank": rank,
            "alpha": alpha,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "gradient_accumulation_steps": gradient_accum,
            "optimizer": optim,
            "fp16": fp16,
            "target_modules": target_modules,
            "recommendation_reason": reason,
            "estimated_vram_needed_gb": round(
                estimate_needed_vram(model_params_b, method), 1),
        }

    def generate_script(self, base_model: str, dataset: str,
                        method: str = "qlora",
                        hf_token: Optional[str] = None,
                        dataset_split: str = "train",
                        dataset_config: Optional[str] = None,
                        text_column: str = "text",
                        custom_dataset_path: Optional[str] = None,
                        rank: int = 16, alpha: int = 32,
                        learning_rate: float = 2e-4,
                        batch_size: int = 4,
                        epochs: int = 3,
                        gradient_accumulation_steps: int = 4,
                        max_length: int = 512,
                        output_dir: str = "./finetuned_model",
                        drive_checkpoint_dir: str = "/content/drive/MyDrive/colab-checkpoints",
                        push_to_hub: bool = False,
                        hub_repo_id: Optional[str] = None,
                        resume_from_checkpoint: Optional[str] = None,
                        logging_steps: int = 25,
                        save_steps: int = 500,
                        warmup_steps: int = 100,
                        optimizer: str = "paged_adamw_8bit",
                        fp16: bool = True,
                        target_modules: Optional[list] = None,
                        gradient_checkpointing: bool = True) -> str:
        """
        Generate a complete, self-contained Colab Python script for
        fine-tuning.  The script includes:
          - pip installs
          - HF login
          - model loading with optional 4-bit
          - PEFT wrapping (unless full)
          - dataset loading (HF hub or local CSV/JSON)
          - training with transformers.Trainer
          - checkpoint saving to Drive every N steps
          - final push to HF Hub
          - metrics logging to a JSON file

        Returns the script as a string ready to send to ColabRunner.
        """
        token = hf_token or os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN")
        use_4bit = method == "qlora"
        is_full = method == "full"

        # --- quantization setup ---
        quant_setup = ""
        load_kwargs = "torch_dtype=torch.float16,"
        if use_4bit:
            quant_setup = """
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
"""
            load_kwargs = "quantization_config=bnb_config,"
        elif method == "lora":
            load_kwargs = "torch_dtype=torch.float16,"

        # --- dataset loading ---
        if custom_dataset_path:
            dataset_load = f"""
import pandas as pd
from datasets import Dataset
if "{custom_dataset_path}".endswith('.csv'):
    df = pd.read_csv("{custom_dataset_path}")
elif "{custom_dataset_path}".endswith('.json'):
    df = pd.read_json("{custom_dataset_path}")
else:
    raise ValueError("Unsupported format")
dataset = Dataset.from_pandas(df)
print(f"Loaded custom dataset: {{len(dataset)}} rows")
"""
        else:
            config_str = f", \"{dataset_config}\"" if dataset_config else ""
            dataset_load = f"""
from datasets import load_dataset
dataset = load_dataset("{dataset}", split="{dataset_split}"{config_str})
print(f"Dataset loaded: {{len(dataset)}} samples")
"""

        # --- PEFT config ---
        if not is_full:
            tms = (target_modules or
                   ["q_proj", "v_proj", "k_proj", "o_proj"])
            tms_str = json.dumps(tms)
            peft_config_block = f"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
peft_config = LoraConfig(
    r={rank},
    lora_alpha={alpha},
    lora_dropout=0.05,
    target_modules={tms_str},
    bias="none",
    task_type="CAUSAL_LM",
)
"""
            prepare_block = "model = prepare_model_for_kbit_training(model)"
            peft_apply = "model = get_peft_model(model, peft_config)\nmodel.print_trainable_parameters()"
        else:
            peft_config_block = ""
            prepare_block = ""
            peft_apply = ""

        # --- checkpoint / Drive ---
        drive_mount = f"""
import os
os.makedirs("{drive_checkpoint_dir}", exist_ok=True)
# Mount Drive only if not already mounted
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass
checkpoint_dir = "{drive_checkpoint_dir}/{base_model.replace('/', '_')}"
os.makedirs(checkpoint_dir, exist_ok=True)
"""

        resume_arg = ""
        if resume_from_checkpoint:
            resume_arg = f"""
# Resume from checkpoint
RESUME_CHECKPOINT = "{resume_from_checkpoint}"
print(f"Resuming from {{RESUME_CHECKPOINT}}")
"""

        # --- metrics logging callback ---
        metrics_callback = """
class MetricsCallback:
    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["_step"] = state.global_step
            logs["_epoch"] = state.epoch
            self.history.append(dict(logs))
            with open(self.log_path, "w") as f:
                json.dump(self.history, f, indent=2)
"""

        hub_push = ""
        if push_to_hub and hub_repo_id:
            hub_push = f"""
from huggingface_hub import HfApi, upload_folder
api = HfApi()
api.create_repo(repo_id="{hub_repo_id}", exist_ok=True)
upload_folder(
    folder_path="{output_dir}",
    repo_id="{hub_repo_id}",
    commit_message="Fine-tuned {base_model} on {dataset}",
)
print(f"Model pushed to https://huggingface.co/{{hub_repo_id}}")
if os.path.exists(metrics_path):
    upload_folder(
        folder_path=os.path.dirname(metrics_path),
        repo_id="{hub_repo_id}",
        path_in_repo="metrics",
        commit_message="Training metrics",
    )
"""

        script = f"""
# ======================== Colab Agent Fine-Tuning Script ========================
# Model: {base_model}  |  Method: {method.upper()}  |  Dataset: {dataset}
# Generated by FineTuneOrchestrator
# ===============================================================================

import subprocess, sys

def install_with_retry(packages, retries=3):
    for i in range(retries):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
            print(f"Successfully installed {{packages}}")
            return
        except subprocess.CalledProcessError:
            print(f"Install failed, attempt {{i+1}}/{{retries}}")
    raise Exception(f"Failed to install {{packages}} after {{retries}} attempts")

install_with_retry(["transformers", "datasets", "accelerate", "peft", "trl", "bitsandbytes", "huggingface_hub"])

import os, json, math, torch
from datetime import datetime

HF_TOKEN = "{token}"
if HF_TOKEN and HF_TOKEN != "YOUR_HF_TOKEN":
    from huggingface_hub import login
    login(token=HF_TOKEN)

# ------------------------------------------------------------------ #
#  Metrics
# ------------------------------------------------------------------ #
METRICS_DIR = "{output_dir}/metrics"
os.makedirs(METRICS_DIR, exist_ok=True)
metrics_path = os.path.join(METRICS_DIR, "training_log.json")
{metrics_callback}
metrics_cb = MetricsCallback(metrics_path)

# ------------------------------------------------------------------ #
#  Model
# ------------------------------------------------------------------ #
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
)
{quant_setup}
MODEL_NAME = "{base_model}"
print(f"Loading model: {{MODEL_NAME}}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    {load_kwargs}
    device_map="auto",
    trust_remote_code=True,
)
print(f"Model loaded! Params: {{model.num_parameters() / 1e9:.2f}}B")

{prepare_block}

# ------------------------------------------------------------------ #
#  PEFT
# ------------------------------------------------------------------ #
{peft_config_block}
{peft_apply}

# ------------------------------------------------------------------ #
#  Dataset
# ------------------------------------------------------------------ #
{dataset_load}

# Tokenize
def tokenize_fn(examples):
    texts = examples.get("{text_column}", examples.get("text", ""))
    return tokenizer(texts, truncation=True, padding="max_length", max_length={max_length})

tokenized = dataset.map(
    tokenize_fn, batched=True,
    remove_columns=dataset.column_names,
)

# ------------------------------------------------------------------ #
#  Training Arguments
# ------------------------------------------------------------------ #
{resume_arg}
OUTPUT_DIR = "{output_dir}"
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs={epochs},
    per_device_train_batch_size={batch_size},
    gradient_accumulation_steps={gradient_accumulation_steps},
    learning_rate={learning_rate},
    warmup_steps={warmup_steps},
    logging_steps={logging_steps},
    save_steps={save_steps},
    save_total_limit=2,
    fp16={str(fp16)},
    bf16=False,
    max_grad_norm=0.3,
    gradient_checkpointing={str(gradient_checkpointing)},
    optim="{optimizer}",
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    ddp_find_unused_parameters=False if {"True" if not is_full else "False"} else None,
)

# ------------------------------------------------------------------ #
#  Trainer
# ------------------------------------------------------------------ #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    callbacks=[metrics_cb],
)

# ------------------------------------------------------------------ #
#  Train
# ------------------------------------------------------------------ #
print(f"Starting training — {{datetime.now().isoformat()}}")
trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT if 'RESUME_CHECKPOINT' in dir() else None)
print(f"Training completed — {{datetime.now().isoformat()}}")

# ------------------------------------------------------------------ #
#  Save
# ------------------------------------------------------------------ #
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {{OUTPUT_DIR}}")

# Copy to Drive
{drive_mount}
import shutil
dest = os.path.join(checkpoint_dir, "final")
if os.path.exists(dest):
    shutil.rmtree(dest)
shutil.copytree(OUTPUT_DIR, dest)
print(f"Backup copied to {{dest}}")

# ------------------------------------------------------------------ #
#  Push to Hub
# ------------------------------------------------------------------ #
{hub_push}

# ------------------------------------------------------------------ #
#  Summary
# ------------------------------------------------------------------ #
final_metrics = {{"final_loss": None, "total_steps": None, "runtime": "{method.upper()}"}}
if metrics_cb.history:
    final_metrics["final_loss"] = metrics_cb.history[-1].get("loss")
    final_metrics["total_steps"] = metrics_cb.history[-1].get("_step")
    final_metrics["epochs"] = epochs
    final_metrics["batch_size"] = {batch_size}
    final_metrics["learning_rate"] = {learning_rate}
    final_metrics["rank"] = {rank}
with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=2)
print(json.dumps(final_metrics, indent=2))
print("=== FINE-TUNING COMPLETE ===")
"""
        return script

    def generate_resume_script(self, checkpoint_path: str,
                               base_model: str, dataset: str,
                               **kwargs) -> str:
        """Generate a script that resumes training from a checkpoint."""
        return self.generate_script(
            base_model=base_model,
            dataset=dataset,
            resume_from_checkpoint=checkpoint_path,
            **kwargs,
        )
