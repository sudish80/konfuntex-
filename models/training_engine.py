"""
Phase 4 — Fine-Tuning Engine (items 36-45).

Provides:
  - HyperparameterTuner         grid / optuna search          (36)
  - EarlyStoppingScheduler      patience-based stop           (37)
  - MetricsCallback             HF-compatible logging          (38)
  - EpochTrainer                full epoch training loop       (39)
  - LossLogger                  streaming loss to JSON         (40)
  - BatchSizeTuner              find max safe batch size       (41)
  - GradientAccumulationScheduler  schedule grad accum steps   (42)
  - BestModelCheckpointer       save best by val metric        (43)
  - ValidationSplitter          auto train/val split           (44)
  - TrainingVisualizer          matplotlib loss curves         (45)
"""
import json
import os
import random
from typing import Optional
from dataclasses import dataclass


# ==================================================================== #
#  36 — HyperparameterTuner
# ==================================================================== #

@dataclass
class HyperparameterTuner:
    """Grid or random search over hyper-parameters."""

    method: str = "grid"  # "grid" | "random" | "optuna"
    n_trials: int = 8

    def grid(self, param_grid: dict) -> list[dict]:
        """Product of param lists."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        def _product(idx: int, current: dict):
            if idx == len(keys):
                return [dict(current)]
            result = []
            for v in values[idx]:
                current[keys[idx]] = v
                result.extend(_product(idx + 1, current))
            return result

        return _product(0, {})

    def random_search(self, param_dist: dict) -> list[dict]:
        """Random combinations from param lists."""
        all_combos = self.grid(param_dist)
        sampled = random.sample(all_combos, min(self.n_trials, len(all_combos)))
        return sampled

    def generate_code(self, base_model: str, dataset: str,
                      method: str = "qlora",
                      param_grid: Optional[dict] = None) -> str:
        grid = param_grid or {
            "learning_rate": [1e-4, 2e-4, 5e-4],
            "rank": [8, 16, 32],
            "batch_size": [4, 8],
        }
        combos = self.grid(grid)
        combos_json = json.dumps(combos)
        return f"""
import itertools, json, os
from models.finetune_orchestrator import FineTuneOrchestrator

orchestrator = FineTuneOrchestrator()
combos = {combos_json}

results = []
for i, params in enumerate(combos):
    print(f"=== Trial {{i+1}}/{{len(combos)}}: {{params}} ===")
    script = orchestrator.generate_script(
        base_model="{base_model}",
        dataset="{dataset}",
        method="{method}",
        learning_rate=params["learning_rate"],
        rank=params.get("rank", 16),
        batch_size=params.get("batch_size", 4),
        epochs=1,
        output_dir=f"./trial_{{i}}",
    )
    # In Colab, execute script and capture final loss
    results.append({{"params": params, "script": script}})

with open("hparam_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Generated {{len(combos)}} trial scripts -> hparam_results.json")
"""


# ==================================================================== #
#  37 — EarlyStoppingScheduler
# ==================================================================== #

class EarlyStoppingScheduler:
    """Stop training when loss stops improving (patience-based)."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, current_loss: float) -> bool:
        """Returns True if training should stop."""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

    def reset(self):
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    @staticmethod
    def callback_code(patience: int = 3) -> str:
        return f"""
class EarlyStoppingCallback:
    def __init__(self, patience={patience}):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            if loss < self.best_loss - 1e-4:
                self.best_loss = loss
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at step {{state.global_step}} (loss={{loss:.4f}})")
                control.should_training_stop = True
"""


# ==================================================================== #
#  38 — MetricsCallback (HF-compatible)
# ==================================================================== #

class MetricsCallback:
    """
    HuggingFace Trainer-compatible callback.
    Logs per-step metrics to a JSON file and optional callback.
    """

    def __init__(self, log_path: str = "training_metrics.json"):
        self.log_path = log_path
        self.history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = dict(logs)
            entry["_step"] = state.global_step
            entry["_epoch"] = state.epoch
            entry["_timestamp"] = str(kwargs.get("start_time", ""))
            self.history.append(entry)
            with open(self.log_path, "w") as f:
                json.dump(self.history, f, indent=2)

    def get_final_metrics(self) -> dict:
        if not self.history:
            return {"final_loss": None, "total_steps": 0}
        last = self.history[-1]
        return {
            "final_loss": last.get("loss"),
            "total_steps": last.get("_step", 0),
            "best_loss": min(h.get("loss", float("inf")) for h in self.history if "loss" in h),
        }


# ==================================================================== #
#  39 — EpochTrainer
# ==================================================================== #

class EpochTrainer:
    """Manual epoch-by-epoch training loop with per-epoch metrics."""

    def __init__(self, model, tokenizer, train_dataset, val_dataset=None,
                 batch_size: int = 4, learning_rate: float = 2e-4,
                 max_length: int = 512, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device

    def train_epoch(self) -> dict:
        """Run one epoch, return metrics dict."""
        import torch
        from torch.utils.data import DataLoader
        self.model.train()
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_loss = 0.0
        steps = 0
        for batch in loader:
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(self.device)
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            steps += 1
        return {"epoch_loss": total_loss / max(steps, 1), "steps": steps}

    def eval_epoch(self) -> dict:
        """Run one eval pass, return metrics."""
        import torch
        from torch.utils.data import DataLoader
        self.model.eval()
        loader = DataLoader(
            self.val_dataset or self.train_dataset,
            batch_size=self.batch_size,
        )
        total_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                outputs = self.model(**inputs, labels=labels)
                total_loss += outputs.loss.item()
                steps += 1
        return {"val_loss": total_loss / max(steps, 1), "val_steps": steps}

    @staticmethod
    def training_loop_code(model_name: str, method: str = "qlora") -> str:
        return f"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.training_engine import EpochTrainer, EarlyStoppingScheduler, BestModelCheckpointer

tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModelForCausalLM.from_pretrained("{model_name}", torch_dtype=torch.float16, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

trainer = EpochTrainer(model, tokenizer, tokenized)
stopper = EarlyStoppingScheduler(patience=3)
checkpointer = BestModelCheckpointer("best_model")

for epoch in range(10):
    train_metrics = trainer.train_epoch()
    val_metrics = trainer.eval_epoch()
    loss = val_metrics["val_loss"]
    print(f"Epoch {{epoch+1}}: train_loss={{train_metrics['epoch_loss']:.4f}} val_loss={{loss:.4f}}")
    checkpointer.step(loss, model, tokenizer, epoch)
    if stopper.step(loss):
        print("Early stopping triggered")
        break
"""


# ==================================================================== #
#  40 — LossLogger
# ==================================================================== #

class LossLogger:
    """Stream loss values to a JSONL file in real-time."""

    def __init__(self, path: str = "loss_log.jsonl"):
        self.path = path

    def log(self, epoch: int, step: int, loss: float,
            learning_rate: float = None, extra: dict = None):
        entry = {"epoch": epoch, "step": step, "loss": loss}
        if learning_rate is not None:
            entry["lr"] = learning_rate
        if extra:
            entry.update(extra)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def plot_loss(self, output_path: str = "loss_curve.png"):
        data = self.read_all()
        if not data:
            return None
        steps = [d.get("step", i) for i, d in enumerate(data)]
        losses = [d["loss"] for d in data]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(steps, losses, marker=".")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True)
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            return output_path
        except ImportError:
            return None

    def generate_code(self) -> str:
        return f"""
from models.training_engine import LossLogger
logger = LossLogger("{self.path}")

# Inside training loop:
# logger.log(epoch, step, loss, learning_rate=lr)
# At end:
# logger.plot_loss("loss_curve.png")
"""


# ==================================================================== #
#  41 — BatchSizeTuner
# ==================================================================== #

class BatchSizeTuner:
    """Find the largest batch size that fits in VRAM."""

    @staticmethod
    def find_max_batch(model_name: str, max_length: int = 512,
                       start: int = 1, max_batch: int = 128) -> str:
        return f"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{model_name}")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "{model_name}", torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# Create dummy input
dummy = tokenizer(["test " * {max_length}] * 1, return_tensors="pt",
                  padding=True, truncation=True, max_length={max_length})
dummy = {{k: v.to("cuda") for k, v in dummy.items()}}

max_batch = {max_batch}
for batch in range({start}, max_batch + 1):
    try:
        inputs = {{k: v.repeat(batch, 1) for k, v in dummy.items()}}
        with torch.no_grad():
            _ = model(**inputs)
        print(f"Batch size {{batch}}: OK")
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"Batch size {{batch}}: OOM (max safe = {{batch-1}})")
        break
"""

    @staticmethod
    def sweep_code(model_name: str, methods: list[str] = None) -> str:
        methods = methods or ["qlora", "lora", "full"]
        return f"""
for method in {json.dumps(methods)}:
    print(f"=== Testing batch sizes for method: {{method}} ===")
    # Each method loads differently; test accordingly
"""


# ==================================================================== #
#  42 — GradientAccumulationScheduler
# ==================================================================== #

class GradientAccumulationScheduler:
    """Schedule gradient accumulation steps: increase early, decrease late."""

    def __init__(self, initial: int = 4, min_steps: int = 1, max_steps: int = 16):
        self.initial = initial
        self.current = initial
        self.min_steps = min_steps
        self.max_steps = max_steps

    def step(self, progress: float, loss_volatility: float = 0.0) -> int:
        """Adjust based on training progress (0.0 = start, 1.0 = end)."""
        if progress < 0.2:
            self.current = min(self.max_steps, self.initial * 2)
        elif progress > 0.8:
            self.current = max(self.min_steps, self.current // 2)
        else:
            if loss_volatility > 1.0:
                self.current = min(self.max_steps, self.current + 2)
            else:
                self.current = max(self.min_steps, self.current - 1)
        return self.current

    @staticmethod
    def scheduler_code() -> str:
        return """
from models.training_engine import GradientAccumulationScheduler
ga_scheduler = GradientAccumulationScheduler(initial=4)

# Inside training loop:
# for epoch in range(epochs):
#     grad_accum = ga_scheduler.step(epoch / epochs)
#     training_args.gradient_accumulation_steps = grad_accum
"""


# ==================================================================== #
#  43 — BestModelCheckpointer
# ==================================================================== #

class BestModelCheckpointer:
    """Save model when validation loss improves."""

    def __init__(self, output_dir: str = "best_model", metric: str = "val_loss",
                 mode: str = "min"):
        self.output_dir = output_dir
        self.metric = metric
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1
        self.improved = False

    def step(self, current_value: float, model=None, tokenizer=None,
             epoch: int = 0) -> bool:
        self.improved = False
        if self.mode == "min" and current_value < self.best_value:
            self.improved = True
        elif self.mode == "max" and current_value > self.best_value:
            self.improved = True
        if self.improved:
            self.best_value = current_value
            self.best_epoch = epoch
            if model is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                model.save_pretrained(self.output_dir)
                if tokenizer:
                    tokenizer.save_pretrained(self.output_dir)
                with open(os.path.join(self.output_dir, "checkpoint_info.json"), "w") as f:
                    json.dump({"best_value": self.best_value, "epoch": epoch, "metric": self.metric}, f)
        return self.improved

    @staticmethod
    def checkpointer_code() -> str:
        return """
from models.training_engine import BestModelCheckpointer
checkpointer = BestModelCheckpointer("best_model")

# After each validation:
# checkpointer.step(val_loss, model, tokenizer, epoch)
"""


# ==================================================================== #
#  44 — ValidationSplitter
# ==================================================================== #

class ValidationSplitter:
    """Auto-split dataset into train/val."""

    @staticmethod
    def split(dataset, val_size: float = 0.1, seed: int = 42):
        split_ds = dataset.train_test_split(test_size=val_size, seed=seed)
        return split_ds["train"], split_ds["test"]

    @staticmethod
    def split_code(dataset_name: str, val_size: float = 0.1) -> str:
        return f"""
from datasets import load_dataset
dataset = load_dataset("{dataset_name}", split="train")
split = dataset.train_test_split(test_size={val_size}, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]
print(f"Train: {{len(train_dataset)}}, Val: {{len(val_dataset)}}")
"""

    @staticmethod
    def kfold_code(dataset_name: str, k: int = 5) -> str:
        return f"""
from datasets import load_dataset, Dataset
from sklearn.model_selection import KFold
import numpy as np
dataset = load_dataset("{dataset_name}", split="train")
kf = KFold(n_splits={k}, shuffle=True, random_state=42)
folds = list(kf.split(np.arange(len(dataset))))
# folds[i][0] = train indices, folds[i][1] = val indices
"""


# ==================================================================== #
#  45 — TrainingVisualizer
# ==================================================================== #

class TrainingVisualizer:
    """Generate matplotlib plots of training metrics."""

    @staticmethod
    def plot_loss_curve(history: list[dict], output_path: str = "loss_curve.png"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            steps = [h.get("_step", i) for i, h in enumerate(history)]
            losses = [h.get("loss") for h in history if "loss" in h]
            if not losses:
                return None
            plt.figure(figsize=(10, 5))
            plt.plot(steps[:len(losses)], losses, "b-", linewidth=1.5)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            return output_path
        except ImportError:
            return None

    @staticmethod
    def plot_lr_schedule(history: list[dict], output_path: str = "lr_schedule.png"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            steps = [h.get("_step", i) for i, h in enumerate(history)]
            lrs = [h.get("learning_rate") for h in history if "learning_rate" in h]
            if not lrs:
                return None
            plt.figure(figsize=(10, 4))
            plt.plot(steps[:len(lrs)], lrs, "r-", linewidth=1.5)
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            return output_path
        except ImportError:
            return None

    @staticmethod
    def plot_gpu_memory(history: list[dict], output_path: str = "gpu_mem.png"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            steps = [h.get("_step", i) for i, h in enumerate(history)]
            mems = [h.get("gpu_mem_gb") for h in history if "gpu_mem_gb" in h]
            if not mems:
                return None
            plt.figure(figsize=(10, 4))
            plt.fill_between(steps[:len(mems)], mems, alpha=0.3, color="green")
            plt.plot(steps[:len(mems)], mems, "g-", linewidth=1.5)
            plt.xlabel("Step")
            plt.ylabel("GPU Memory (GB)")
            plt.title("GPU Memory Usage")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            return output_path
        except ImportError:
            return None

    @staticmethod
    def generate_all_code(metrics_path: str = "training_metrics.json") -> str:
        return f"""
import json
from models.training_engine import TrainingVisualizer
with open("{metrics_path}") as f:
    history = json.load(f)
TrainingVisualizer.plot_loss_curve(history, "loss_curve.png")
TrainingVisualizer.plot_lr_schedule(history, "lr_schedule.png")
TrainingVisualizer.plot_gpu_memory(history, "gpu_mem.png")
print("Plots saved: loss_curve.png, lr_schedule.png, gpu_mem.png")
"""
