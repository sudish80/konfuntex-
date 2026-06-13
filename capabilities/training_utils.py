"""
Phases 42-49 — Training Utilities & Monitoring.

Provides:
  - EarlyStoppingCallback    patience-based stop            (42)
  - LearningRateScheduler    cosine / linear / constant     (43)
  - DistributedTraining      accelerate + DeepSpeed config  (44)
  - CheckpointManager        save/load/resume/cleanup       (45)
  - BestModelSelector        track best by metric           (46)
  - EvaluationPipeline       task-specific metrics          (47)
  - DatasetSplitter          train/val/test split           (48)
  - DataCollatorGenerator    auto-select collator           (49)
"""
import json
import os


# ==================================================================== #
#  42 — EarlyStoppingCallback
# ==================================================================== #

class EarlyStoppingCallback:
    """
    Monitor validation loss and stop training when it plateaus.
    Also detects divergence (loss > previous_best * 2).
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4,
                 monitor: str = "eval_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float("inf")
        self.counter = 0
        self.best_model_path = None

    def __call__(self, current_value: float, model=None, step: int = 0) -> bool:
        """Return True if training should stop."""
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
            if model is not None:
                import torch
                self.best_model_path = f"best_model_step_{step}.pt"
                torch.save(model.state_dict(), self.best_model_path)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping: no improvement for {self.patience} checks")
            return True

        # Divergence detection
        if current_value > self.best_value * 2 and self.best_value < float("inf"):
            print(f"Divergence detected: loss={current_value:.4f}, "
                  f"best={self.best_value:.4f}")
            return True

        return False

    def get_state(self) -> dict:
        return {
            "best_value": self.best_value,
            "counter": self.counter,
            "patience": self.patience,
        }

    def huggingface_code(self) -> str:
        return f"""
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience={self.patience}, min_delta={self.min_delta}):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, **kwargs):
        logs = kwargs.get("logs", {{}})
        loss = logs.get("eval_loss", logs.get("loss"))
        if loss is None:
            return

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping at step {{state.global_step}}")
            control.should_training_stop = True

        if loss > self.best_loss * 2 and self.best_loss < float("inf"):
            print(f"Divergence: loss={{loss:.4f}}, best={{self.best_loss:.4f}}")
            control.should_training_stop = True

# Usage: trainer.add_callback(EarlyStoppingCallback())
"""


# ==================================================================== #
#  43 — LearningRateScheduler
# ==================================================================== #

class LearningRateScheduler:
    """
    Generate various LR schedules: cosine, linear, constant with warmup.
    """

    SCHEDULE_TYPES = ["cosine", "linear", "constant", "polynomial",
                      "cosine_with_restarts"]

    @staticmethod
    def setup_code(schedule_type: str = "cosine",
                   warmup_ratio: float = 0.05,
                   num_epochs: int = 3,
                   train_steps_per_epoch: int = 500) -> str:
        return f"""
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
import torch

num_epochs = {num_epochs}
steps_per_epoch = {train_steps_per_epoch}
total_steps = num_epochs * steps_per_epoch
warmup_steps = int(total_steps * {warmup_ratio})

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

schedule_type = "{schedule_type}"
if schedule_type == "cosine":
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
elif schedule_type == "linear":
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
elif schedule_type == "constant":
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps
    )
else:
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

print(f"LR schedule: {{schedule_type}}, warmup={{warmup_steps}}, total={{total_steps}}")

# Log LR at each step in training loop
# for batch in dataloader:
#     ...
#     scheduler.step()
#     current_lr = scheduler.get_last_lr()[0]
"""

    @staticmethod
    def plot_code() -> str:
        return """
import matplotlib.pyplot as plt
import numpy as np

# Plot LR schedule
lrs = []
total_steps = 1000
warmup = 50
for step in range(total_steps):
    # Cosine formula
    if step < warmup:
        lr = step / warmup * 2e-4
    else:
        progress = (step - warmup) / (total_steps - warmup)
        lr = 2e-4 * 0.5 * (1 + np.cos(np.pi * progress))
    lrs.append(lr)

plt.figure(figsize=(10, 4))
plt.plot(lrs)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title(f"Cosine Schedule (warmup={warmup})")
plt.grid(True, alpha=0.3)
plt.savefig("lr_schedule.png", dpi=100, bbox_inches="tight")
plt.show()
"""


# ==================================================================== #
#  44 — DistributedTraining
# ==================================================================== #

class DistributedTraining:
    """
    Configure distributed training with DeepSpeed ZeRO and Accelerate.
    Primarily for memory savings on single GPU (ZeRO-3 offload).
    """

    @staticmethod
    def deepspeed_config_code() -> str:
        return """
# DeepSpeed ZeRO-3 config for memory-efficient training
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
    },
    "bf16": {"enabled": True},
    "fp16": {"enabled": False},
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 4,
    "wall_clock_breakdown": False,
}

import json
with open("ds_config.json", "w") as f:
    json.dump(deepspeed_config, f, indent=2)
print("DeepSpeed config saved to ds_config.json")
"""

    @staticmethod
    def accelerate_launch_code() -> str:
        return """
# For multi-GPU: accelerate launch --num_processes=2 train.py
# For single GPU with DeepSpeed:
# accelerate launch --use_deepspeed --deepspeed_config_file ds_config.json train.py

# In-script usage with transformers:
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    deepspeed="ds_config.json",  # Enable DeepSpeed
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=False,
    bf16=True,
    dataloader_num_workers=2,
)
print(f"Distributed backend: {training_args.ddp_backend or 'none'}")
"""

    @staticmethod
    def data_parallel_code() -> str:
        return """
# Simple DataParallel (single process, multi-GPU)
import torch

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = torch.nn.DataParallel(model)
else:
    print(f"Single GPU: {torch.cuda.device_count()}")
"""


# ==================================================================== #
#  45 — CheckpointManager
# ==================================================================== #

class CheckpointManager:
    """
    Save/load/resume checkpoints with automatic cleanup.
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints",
                 keep_last_n: int = 3, max_age_days: int = 7):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.max_age_days = max_age_days
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_code(self) -> str:
        return f"""
import os, torch, json
from datetime import datetime, timedelta

checkpoint_dir = "{self.checkpoint_dir}"
keep_last_n = {self.keep_last_n}
max_age_days = {self.max_age_days}
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, scheduler, step, loss, metrics=None):
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{{step}}")
    os.makedirs(ckpt_path, exist_ok=True)

    # Save model, optimizer, scheduler
    model.save_pretrained(ckpt_path)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

    # Save state
    state = {{
        "step": step,
        "loss": loss,
        "metrics": metrics or {{}},
        "timestamp": datetime.now().isoformat(),
    }}
    with open(os.path.join(ckpt_path, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    # Cleanup old checkpoints
    checkpoints = sorted([
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-")
    ], key=lambda x: int(x.split("-")[1]))
    for old in checkpoints[:-keep_last_n]:
        import shutil
        shutil.rmtree(os.path.join(checkpoint_dir, old))
        print(f"Removed old checkpoint: {{old}}")

    # Cleanup old by age
    cutoff = datetime.now() - timedelta(days=max_age_days)
    for d in os.listdir(checkpoint_dir):
        dpath = os.path.join(checkpoint_dir, d)
        if os.path.isdir(dpath) and datetime.fromtimestamp(os.path.getctime(dpath)) < cutoff:
            import shutil
            shutil.rmtree(dpath)
            print(f"Removed expired checkpoint: {{d}}")

    print(f"Checkpoint saved: {{ckpt_path}}")
    return ckpt_path

def load_checkpoint(model_class, tokenizer_class, checkpoint_path):
    model = model_class.from_pretrained(checkpoint_path)
    tokenizer = tokenizer_class.from_pretrained(checkpoint_path)
    state_path = os.path.join(checkpoint_path, "training_state.json")
    state = {{}}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    return model, tokenizer, state
"""


# ==================================================================== #
#  46 — BestModelSelector
# ==================================================================== #

class BestModelSelector:
    """
    Track and save the best model based on a monitored metric.
    """

    def __init__(self, monitor: str = "eval_loss", mode: str = "min",
                 min_delta: float = 1e-4, output_dir: str = "./best_model"):
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.output_dir = output_dir
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_step = -1
        os.makedirs(output_dir, exist_ok=True)

    def step(self, current_value: float, model=None,
             tokenizer=None, step: int = 0) -> bool:
        improved = False
        if self.mode == "min":
            if current_value < self.best_value - self.min_delta:
                improved = True
        else:
            if current_value > self.best_value + self.min_delta:
                improved = True

        if improved:
            self.best_value = current_value
            self.best_step = step
            if model is not None:
                model.save_pretrained(self.output_dir)
                if tokenizer:
                    tokenizer.save_pretrained(self.output_dir)
                with open(f"{self.output_dir}/best_metrics.json", "w") as f:
                    json.dump({"best_value": self.best_value, "step": step,
                               "monitor": self.monitor}, f, indent=2)
        return improved

    def huggingface_code(self) -> str:
        return f"""
from transformers import TrainerCallback

class BestModelCallback(TrainerCallback):
    def __init__(self, output_dir="{self.output_dir}"):
        self.output_dir = output_dir
        self.best_loss = float("inf")

    def on_log(self, args, state, control, **kwargs):
        loss = kwargs.get("logs", {{}}).get("eval_loss", kwargs.get("logs", {{}}).get("loss"))
        if loss and loss < self.best_loss:
            self.best_loss = loss
            kwargs["model"].save_pretrained(self.output_dir)
            print(f"New best model at step {{state.global_step}}: loss={{loss:.4f}}")

# Or use built-in:
# training_args = TrainingArguments(
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
# )
"""


# ==================================================================== #
#  47 — EvaluationPipeline
# ==================================================================== #

class EvaluationPipeline:
    """
    Compute task-specific metrics using HuggingFace evaluate library.
    """

    TASK_METRICS = {
        "classification": ["accuracy", "precision", "recall", "f1"],
        "multiclass": ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        "regression": ["mse", "mae", "r2"],
        "generation": ["bleu", "rouge", "perplexity"],
        "summarization": ["rouge"],
        "translation": ["bleu", "comet"],
        "code": ["pass_at_k"],
    }

    @staticmethod
    def compute_code(task_type: str = "classification") -> str:
        return f"""
import evaluate
import numpy as np
from transformers import EvalPrediction

task_type = "{task_type}"

metrics_map = {json.dumps(EvaluationPipeline.TASK_METRICS)}

def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    results = {{}}

    if task_type in ("classification", "multiclass"):
        predictions = np.argmax(predictions, axis=1)
        for metric_name in metrics_map.get(task_type, ["accuracy"]):
            metric = evaluate.load(metric_name)
            if metric_name == "accuracy":
                result = metric.compute(predictions=predictions, references=labels)
            else:
                result = metric.compute(predictions=predictions, references=labels, average="macro")
            results.update(result)

    elif task_type == "regression":
        for metric_name in metrics_map.get(task_type, ["mse"]):
            metric = evaluate.load(metric_name)
            result = metric.compute(predictions=predictions.flatten(), references=labels.flatten())
            results.update(result)

    elif task_type in ("generation", "summarization", "translation"):
        # predictions/labels are lists of strings
        for metric_name in metrics_map.get(task_type, ["rouge"]):
            metric = evaluate.load(metric_name)
            result = metric.compute(predictions=predictions, references=labels)
            results.update(result)

    return results

# For perplexity:
# ppl = evaluate.load("perplexity", module_type="metric")
# results["perplexity"] = ppl.compute(model_id=model_name, input_texts=texts)["mean_perplexity"]

print(f"Metrics configured: {{metrics_map.get(task_type, ['accuracy'])}}")
"""

    @staticmethod
    def confusion_matrix_code() -> str:
        return """
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.show()
"""


# ==================================================================== #
#  48 — DatasetSplitter
# ==================================================================== #

class DatasetSplitter:
    """
    Split datasets into train/validation/test with stratification.
    """

    def __init__(self, output_dir: str = "./data_splits",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 seed: int = 42):
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)

    def split_code(self, dataset_path: str = "./data/raw.parquet",
                   label_column: str = "label",
                   stratified: bool = True) -> str:
        return f"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

df = pd.read_parquet("{dataset_path}")
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

train_ratio = {self.train_ratio}
val_ratio = {self.val_ratio}
test_ratio = {self.test_ratio}
seed = {self.seed}
stratify_col = df["{label_column}"] if "{label_column}" in df.columns and {str(stratified)} else None

# First split: train vs temp
train, temp = train_test_split(
    df, test_size=(1 - train_ratio), random_state=seed,
    stratify=stratify_col,
)

# Second split: val vs test
val_size = val_ratio / (val_ratio + test_ratio)
stratify_temp = temp["{label_column}"] if stratify_col is not None else None
val, test = train_test_split(
    temp, test_size=(1 - val_size), random_state=seed,
    stratify=stratify_temp,
)

# Save splits
train.to_parquet(os.path.join(output_dir, "train.parquet"))
val.to_parquet(os.path.join(output_dir, "val.parquet"))
test.to_parquet(os.path.join(output_dir, "test.parquet"))

print(f"Splits: train={{len(train)}}, val={{len(val)}}, test={{len(test)}}")
if stratify_col is not None:
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"  {{name}} distribution: {{split['{label_column}'].value_counts().to_dict()}}")
"""


# ==================================================================== #
#  49 — DataCollatorGenerator
# ==================================================================== #

class DataCollatorGenerator:
    """
    Auto-select the appropriate data collator based on task type.
    """

    @staticmethod
    def select_code(task_type: str = "causal_lm") -> str:
        return f"""
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    DataCollatorForTokenClassification,
    default_data_collator,
)

task_type = "{task_type}"

if task_type == "causal_lm":
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    pad_type = "dynamic (causal LM)"

elif task_type == "masked_lm":
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
    pad_type = "dynamic (masked LM)"

elif task_type in ("classification", "regression"):
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, return_tensors="pt",
    )
    pad_type = "dynamic (classification)"

elif task_type in ("seq2seq", "summarization", "translation"):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt",
    )
    pad_type = "dynamic (seq2seq)"

elif task_type == "token_classification":
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, return_tensors="pt",
    )
    pad_type = "dynamic (token classification)"

else:
    data_collator = default_data_collator
    pad_type = "default"

print(f"Data collator: {{pad_type}}")
print(f"Padding side: {{tokenizer.padding_side}}")
"""
