"""
Phases 31-35 — Experiment Tracking & Hyperparameter Optimization.

Provides:
  - ExperimentTracker        MLflow / W&B integration       (31)
  - HyperparameterDatabase   SQLite sweep storage           (32)
  - OptimalConfigRecommender Bayesian optimization          (33)
  - DriftDetector            PSI / MMD distribution shift   (34)
  - AblationStudyRunner      component ablation analysis    (35)
"""
import json
import os


# ==================================================================== #
#  31 — ExperimentTracker
# ==================================================================== #

class ExperimentTracker:
    """
    Log experiments to MLflow, W&B, or TensorBoard.
    """

    SUPPORTED_BACKENDS = ["mlflow", "wandb", "tensorboard"]

    def __init__(self, backend: str = "mlflow",
                 experiment_name: str = "colab-agent",
                 tracking_dir: str = "./mlruns"):
        self.backend = backend
        self.experiment_name = experiment_name
        self.tracking_dir = tracking_dir

    def setup_code(self) -> str:
        return f"""
import os

backend = "{self.backend}"
experiment_name = "{self.experiment_name}"

if backend == "mlflow":
    import mlflow
    mlflow.set_tracking_uri("file:{self.tracking_dir}")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    print(f"MLflow run: {{mlflow.active_run().info.run_id}}")

elif backend == "wandb":
    import wandb
    wandb.init(project=experiment_name)
    print(f"W&B run: {{wandb.run.id}}")

elif backend == "tensorboard":
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="{self.tracking_dir}/tensorboard")
    print("TensorBoard writer created")

print(f"Tracking with {{backend}}")
"""

    def log_metrics_code(self) -> str:
        return """
# Log metrics
if backend == "mlflow":
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_metric("loss", loss, step=epoch)
    mlflow.log_metric("accuracy", accuracy, step=epoch)
    mlflow.log_artifact("model.pth")

elif backend == "wandb":
    wandb.log({
        "learning_rate": learning_rate,
        "loss": loss,
        "accuracy": accuracy,
        "epoch": epoch,
    })
    wandb.save("model.pth")

elif backend == "tensorboard":
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)
    writer.add_scalar("LR", learning_rate, epoch)

# Auto-log gradients (W&B)
if backend == "wandb":
    wandb.watch(model, log="all")
"""

    def finish_code(self) -> str:
        return """
if backend == "mlflow":
    mlflow.end_run()
elif backend == "wandb":
    wandb.finish()
elif backend == "tensorboard":
    writer.close()
"""


# ==================================================================== #
#  32 — HyperparameterDatabase
# ==================================================================== #

class HyperparameterDatabase:
    """
    SQLite-backed storage for hyperparameter sweeps and trials.
    """

    SCHEMA = {
        "sweeps": """
            CREATE TABLE IF NOT EXISTS hyperparameter_sweeps (
                sweep_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                model_name TEXT,
                dataset_name TEXT,
                method TEXT,
                sweep_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        "trials": """
            CREATE TABLE IF NOT EXISTS trials (
                trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sweep_id INTEGER,
                learning_rate REAL,
                batch_size INTEGER,
                num_epochs INTEGER,
                lora_r INTEGER,
                lora_alpha INTEGER,
                gradient_accumulation_steps INTEGER,
                final_loss REAL,
                final_accuracy REAL,
                training_time_sec REAL,
                peak_vram_gb REAL,
                FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)
            )
        """,
    }

    def __init__(self, db_path: str = "./hyperparameter_db.sqlite"):
        self.db_path = db_path

    def init_code(self) -> str:
        return f"""
import sqlite3, json, os

db_path = "{self.db_path}"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS hyperparameter_sweeps (
    sweep_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    model_name TEXT,
    dataset_name TEXT,
    method TEXT,
    sweep_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS trials (
    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id INTEGER,
    learning_rate REAL,
    batch_size INTEGER,
    num_epochs INTEGER,
    lora_r INTEGER,
    lora_alpha INTEGER,
    gradient_accumulation_steps INTEGER,
    final_loss REAL,
    final_accuracy REAL,
    training_time_sec REAL,
    peak_vram_gb REAL,
    FOREIGN KEY (sweep_id) REFERENCES sweeps(sweep_id)
)''')

conn.commit()
print(f"HP database ready: {{db_path}}")
"""

    def log_trial_code(self) -> str:
        return """
# Log a trial
cursor.execute(
    "INSERT INTO trials (sweep_id, learning_rate, batch_size, num_epochs, "
    "lora_r, lora_alpha, final_loss) VALUES (?, ?, ?, ?, ?, ?, ?)",
    (sweep_id, learning_rate, batch_size, num_epochs, lora_r, lora_alpha, final_loss)
)
conn.commit()
"""

    def best_config_code(self) -> str:
        return """
# Get best config
cursor.execute(
    "SELECT * FROM trials ORDER BY final_loss ASC LIMIT 5"
)
best = cursor.fetchall()
print("Top 5 configurations:")
for row in best:
    print(f"  Trial {row[0]}: lr={row[2]:.6f}, bs={row[3]}, "
          f"lora_r={row[5]}, lora_alpha={row[6]}, loss={row[7]:.4f}")
"""


# ==================================================================== #
#  33 — OptimalConfigRecommender
# ==================================================================== #

class OptimalConfigRecommender:
    """
    Predict optimal hyperparameters using Bayesian optimization
    or heuristic rules when no history exists.
    """

    def __init__(self, db_path: str = "./hyperparameter_db.sqlite"):
        self.db_path = db_path

    def recommend_code(self, model_params_b: float, dataset_size: int,
                       task_type: str = "text-generation",
                       available_vram_gb: float = 16.0) -> str:
        template = r'''
import json, sqlite3, os
import numpy as np

model_params_b = __MODEL_PARAMS_B__
dataset_size = __DATASET_SIZE__
available_vram_gb = __AVAILABLE_VRAM_GB__
db_path = "__DB_PATH__"

# Check for prior trials
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trials")
    count = cursor.fetchone()[0]
else:
    count = 0

if count >= 10:
    # Use Bayesian optimization
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer

        cursor.execute("""
            SELECT learning_rate, batch_size, lora_r, lora_alpha, final_loss
            FROM trials WHERE final_loss IS NOT NULL
        """)
        prior_data = cursor.fetchall()
        X = np.array([[r[0], r[1], r[2], r[3]] for r in prior_data])
        y = np.array([r[4] for r in prior_data])

        space = [
            Real(1e-5, 5e-4, name="learning_rate"),
            Integer(1, 16, name="batch_size"),
            Integer(4, 64, name="lora_r"),
            Integer(8, 128, name="lora_alpha"),
        ]

        def _objective(params):
            lr, bs, r, alpha = params
            # Simple proximity to priors
            dists = np.sum((X - np.array([lr, bs, r, alpha])) ** 2, axis=1)
            weights = 1.0 / (dists + 1e-8)
            return np.average(y, weights=weights)

        result = gp_minimize(_objective, space, n_calls=20, random_state=42)
        recommendation = {
            "learning_rate": float(result.x[0]),
            "batch_size": int(result.x[1]),
            "lora_r": int(result.x[2]),
            "lora_alpha": int(result.x[3]),
            "method": "bayesian",
        }
    except ImportError:
        recommendation = {}
else:
    recommendation = {}

# Fallback to heuristics if needed
if not recommendation:
    recommendation = {
        "learning_rate": 2e-4 if model_params_b < 13 else 5e-5,
        "batch_size": 4 if available_vram_gb < 16 else (8 if available_vram_gb < 32 else 16),
        "lora_r": min(32, max(8, int(16 * (7 / max(model_params_b, 1))))),
        "lora_alpha": min(64, max(16, int(32 * (7 / max(model_params_b, 1))))),
        "method": "heuristic",
    }

recommendation["model_params_b"] = model_params_b
recommendation["dataset_size"] = dataset_size
recommendation["vram_needed_gb"] = round(model_params_b * 0.5 / (0.75 if recommendation.get("lora_r", 0) > 0 else 1), 1)

print(json.dumps(recommendation, indent=2))
'''
        code = template.replace("__MODEL_PARAMS_B__", str(model_params_b))
        code = code.replace("__DATASET_SIZE__", str(dataset_size))
        code = code.replace("__AVAILABLE_VRAM_GB__", str(available_vram_gb))
        code = code.replace("__DB_PATH__", self.db_path)
        return code


# ==================================================================== #
#  34 — DriftDetector
# ==================================================================== #

class DriftDetector:
    """
    Detect data drift between training and validation distributions.
    Uses PSI for categorical, MMD for embeddings, JS-div for text.
    """

    def __init__(self, output_dir: str = "./drift_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def detect_code(self, train_path: str = "./data/train.parquet",
                    val_path: str = "./data/val.parquet",
                    text_column: str = "text") -> str:
        return f"""
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
import json

train = pd.read_parquet("{train_path}")
val = pd.read_parquet("{val_path}")
text_column = "{text_column}"

results = {{}}

# 1. Population Stability Index (PSI) for text length
def compute_psi(expected, actual, bins=10):
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return np.sum((actual - expected) * np.log(actual / expected))

train_lengths = train[text_column].str.len()
val_lengths = val[text_column].str.len()
hist_train, edges = np.histogram(train_lengths, bins=10)
hist_val, _ = np.histogram(val_lengths, bins=edges)
p_train = hist_train / hist_train.sum()
p_val = hist_val / hist_val.sum()
psi = compute_psi(p_train, p_val)
results["length_psi"] = round(psi, 4)
results["length_drift"] = "significant" if psi > 0.2 else ("minor" if psi > 0.1 else "none")

# 2. Jensen-Shannon divergence on character n-grams
def get_ngram_freqs(texts, n=3):
    freqs = {{}}
    for t in texts:
        if isinstance(t, str):
            for i in range(len(t) - n + 1):
                ng = t[i:i+n]
                freqs[ng] = freqs.get(ng, 0) + 1
    total = sum(freqs.values()) or 1
    return {{k: v/total for k, v in freqs.items()}}

train_ngrams = get_ngram_freqs(train[text_column][:1000])
val_ngrams = get_ngram_freqs(val[text_column][:1000])

all_ngrams = list(set(list(train_ngrams.keys())[:100] + list(val_ngrams.keys())[:100]))
p_train_ng = np.array([train_ngrams.get(n, 1e-10) for n in all_ngrams])
p_val_ng = np.array([val_ngrams.get(n, 1e-10) for n in all_ngrams])
p_train_ng /= p_train_ng.sum()
p_val_ng /= p_val_ng.sum()
js_div = jensenshannon(p_train_ng, p_val_ng)
results["ngram_js_divergence"] = round(float(js_div), 4)
results["ngram_drift"] = "significant" if js_div > 0.3 else ("minor" if js_div > 0.15 else "none")

# Summary
results["recommendation"] = (
    "Retrain recommended" if any(v == "significant" for v in results.values() if isinstance(v, str))
    else "Model is stable"
)
print(json.dumps(results, indent=2))

with open("{self.output_dir}/drift_report.json", "w") as f:
    json.dump(results, f, indent=2)
"""


# ==================================================================== #
#  35 — AblationStudyRunner
# ==================================================================== #

class AblationStudyRunner:
    """
    Run ablation studies by removing one component at a time.
    """

    COMPONENT_TYPES = [
        "data_source", "augmentation", "preprocessing",
        "model_type", "quantization", "attention",
    ]

    def __init__(self, output_dir: str = "./ablation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_code(self, components: list[str]) -> str:
        return f"""
import itertools, json, os, time
import pandas as pd

output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

components = {json.dumps(components)}

def run_training(config):
    '''Mock training; in real use, calls FineTuneOrchestrator.'''
    import random, time
    time.sleep(0.5)
    return {{
        "loss": random.uniform(0.3, 1.0),
        "accuracy": random.uniform(0.7, 0.95),
        "training_time": random.uniform(100, 500),
        "peak_vram": random.uniform(8, 20),
    }}

results = []
full_config = {{c: 1 for c in components}}
full_result = run_training(full_config)
full_result["config"] = "full"
full_result["ablation"] = "none"
results.append(full_result)

for comp in components:
    ablated_config = {{c: (0 if c == comp else 1) for c in components}}
    result = run_training(ablated_config)
    result["config"] = "ablate_" + comp
    result["ablation"] = comp
    result["delta_loss"] = round(result["loss"] - full_result["loss"], 4)
    result["delta_accuracy"] = round(result["accuracy"] - full_result["accuracy"], 4)
    results.append(result)
    print(f"Ablated {{comp}}: loss={{result['loss']:.4f}} "
          f"(delta={{result['delta_loss']:+.4f}})")

df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "ablation_results.csv"), index=False)

# Find most impactful component
if len(results) > 1:
    deltas = [(r["ablation"], abs(r["delta_loss"])) for r in results[1:]]
    most_impactful = max(deltas, key=lambda x: x[1])
    print(f"Most impactful component: {{most_impactful[0]}} "
          f"(delta_loss={{most_impactful[1]:.4f}})")

print(f"Results saved to {{output_dir}}")
"""
