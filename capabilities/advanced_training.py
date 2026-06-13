"""
Phases 91-96 — Advanced Training.

Provides:
  - HyperparameterOptimizer   Optuna-based HP search         (91)
  - MultiModelExperimentRunner compare multiple models        (92)
  - DatasetSynthesizer        Self-Instruct data generation  (93)
  - PromptOptimizer           DSPy-like prompt search         (94)
  - ChainOfThoughtTracing     trace agent reasoning steps    (95)
  - ModelComparisonReport     compare fine-tuned models       (96)
"""
import json
import os


# ==================================================================== #
#  91 — HyperparameterOptimizer (Optuna)
# ==================================================================== #

class HyperparameterOptimizer:
    """
    Bayesian hyperparameter search using Optuna.
    """

    def __init__(self, study_path: str = "./optuna_study.pkl"):
        self.study_path = study_path

    def optimize_code(self, n_trials: int = 20) -> str:
        return f"""
import optuna
import torch
from transformers import TrainingArguments, Trainer
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

n_trials = {n_trials}
study_path = "{self.study_path}"

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    # Build training args
    training_args = TrainingArguments(
        output_dir=f"./optuna_trial_{{trial.number}}",
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        num_train_epochs=1,
        logging_steps=10,
        report_to="none",
        save_strategy="no",
    )

    # Train and return validation loss
    # trainer = Trainer(model=model, args=training_args, ...)
    # trainer.train()
    # val_loss = trainer.evaluate()["eval_loss"]

    # Simulated for demo
    import random
    val_loss = random.uniform(0.3, 1.0) * (1 + 0.1 * abs(learning_rate - 2e-4) / 2e-4)
    return val_loss

# Create study
study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    study_name="colab_agent_hpo",
    storage=f"sqlite:///{{study_path.replace('.pkl', '.db')}}",
    load_if_exists=True,
)

study.optimize(objective, n_trials=n_trials)

print(f"Best trial: {{study.best_trial.number}}")
print(f"Best params: {{study.best_params}}")
print(f"Best loss: {{study.best_value:.4f}}")

# Save study
with open(study_path, "wb") as f:
    import pickle
    pickle.dump(study, f)
"""


# ==================================================================== #
#  92 — MultiModelExperimentRunner
# ==================================================================== #

class MultiModelExperimentRunner:
    """
    Run fine-tuning on multiple base models and compare results.
    """

    def __init__(self, output_dir: str = "./multi_model_comparison"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_code(self, models: list[str], dataset: str,
                 method: str = "qlora") -> str:
        return f"""
import os, json, time, subprocess
import pandas as pd

models = {json.dumps(models)}
dataset = "{dataset}"
method = "{method}"
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

results = []

for model_name in models:
    print(f"\\\\n{'='*60}")
    print(f"Fine-tuning: {{model_name}}")
    print(f"{'='*60}")

    start = time.time()
    # Run agent for this model
    # agent.run(f"Fine-tune {{model_name}} on {{dataset}} using {{method}}")

    # Simulated for demo
    import random
    time.sleep(0.5)
    result = {{
        "model": model_name,
        "final_loss": round(random.uniform(0.2, 0.8), 4),
        "accuracy": round(random.uniform(0.75, 0.95), 3),
        "training_time_sec": round(time.time() - start, 1),
        "peak_vram_gb": round(random.uniform(8, 20), 1),
        "inference_speed_tok_s": round(random.uniform(20, 60), 0),
    }}
    results.append(result)

    print(f"  Loss: {{result['final_loss']}}, Acc: {{result['accuracy']}}")

# Comparison table
df = pd.DataFrame(results)
print(f"\\\\n{'='*60}")
print("MODEL COMPARISON")
print(f"{'='*60}")
print(df.to_string(index=False))

# Save
df.to_csv(os.path.join(output_dir, "comparison.csv"), index=False)

# Find best
best_idx = df["final_loss"].idxmin()
best = df.iloc[best_idx]
print(f"\\\\nBest model: {{best['model']}} (loss={{best['final_loss']}})")

with open(os.path.join(output_dir, "best_model.txt"), "w") as f:
    f.write(f"Best model: {{best['model']}}\\\\n")
    f.write(f"Loss: {{best['final_loss']}}\\\\n")
    f.write(f"Accuracy: {{best['accuracy']}}\\\\n")
"""


# ==================================================================== #
#  93 — DatasetSynthesizer
# ==================================================================== #

class DatasetSynthesizer:
    """
    Generate synthetic datasets using a Self-Instruct pipeline.
    """

    def __init__(self, output_dir: str = "./synthetic_datasets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def synthesize_code(self, task_type: str = "classification",
                        num_examples: int = 100,
                        classes: list[str] = None,
                        model_name: str = "microsoft/phi-2") -> str:
        return f"""
import json, os, random
from transformers import AutoTokenizer, AutoModelForCausalLM

task_type = "{task_type}"
num_examples = {num_examples}
classes = {json.dumps(classes or ["positive", "negative"])}
model_name = "{model_name}"
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True,
)

examples = []
for i in range(num_examples):
    label = random.choice(classes)
    if task_type == "classification":
        prompt = f"Generate a {{label}} {{task_type}} example sentence:"
    elif task_type == "instruction":
        prompt = f"Generate an instruction with {{label}} output:"
    else:
        prompt = f"Generate a {{task_type}} example (label: {{label}}):"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=100, temperature=0.8, do_sample=True,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    examples.append({{
        "instruction": prompt,
        "input": "",
        "output": text.replace(prompt, "").strip(),
        "label": label,
    }})

    if (i + 1) % 10 == 0:
        print(f"Generated {{i+1}}/{{num_examples}} examples")

# Save
output_path = os.path.join(output_dir, f"synthetic_{{task_type}}_{{num_examples}}.jsonl")
with open(output_path, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\\\\n")

print(f"Saved {{len(examples)}} examples to {{output_path}}")

# Validate: check format, remove short outputs
valid = [ex for ex in examples if len(ex["output"].split()) > 3]
print(f"Valid examples: {{len(valid)}}/{{len(examples)}} (removed {{len(examples)-len(valid)}} short)")
"""


# ==================================================================== #
#  94 — PromptOptimizer
# ==================================================================== #

class PromptOptimizer:
    """
    Optimize prompts by generating variations and evaluating on a validation set.
    """

    def __init__(self, output_dir: str = "./prompt_optimization"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def optimize_code(self, base_prompt: str, val_inputs: list[str],
                      val_labels: list[str]) -> str:
        return f"""
import json, os, random
import numpy as np

base_prompt = '''{base_prompt}'''
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

# Generate prompt variations
variations = [
    base_prompt,
    base_prompt + " Please be concise.",
    base_prompt + " Let's think step by step.",
    "Task: " + base_prompt,
    base_prompt.replace("Generate", "Create"),
    base_prompt + " Respond in JSON format.",
]

def evaluate_prompt(prompt, val_inputs, val_labels):
    '''Score a prompt by feeding to model and computing accuracy.'''
    # In real use: call model with prompt, compare outputs to val_labels
    # Simulated for demo
    return random.uniform(0.5, 0.95)

results = []
for variant in variations:
    score = evaluate_prompt(variant, val_inputs, val_labels)
    results.append({{
        "prompt": variant[:100],
        "score": round(score, 3),
    }})

# Sort by score
results.sort(key=lambda x: -x["score"])

# Iterative improvement
best_prompt = results[0]["prompt"]
for generation in range(3):
    # Mutate best prompt
    mutations = [
        best_prompt + ".",
        best_prompt.replace("  ", " "),
        best_prompt + " Be specific.",
    ]
    for mut in mutations:
        score = evaluate_prompt(mut, val_inputs, val_labels)
        results.append({{"prompt": mut[:100], "score": round(score, 3)}})
        if score > results[0]["score"]:
            best_prompt = mut

results.sort(key=lambda x: -x["score"])
print(f"Best prompt (score={{results[0]['score']}}):")
print(results[0]["prompt"])

with open(os.path.join(output_dir, "best_prompt.txt"), "w") as f:
    f.write(results[0]["prompt"])

with open(os.path.join(output_dir, "optimization_trajectory.json"), "w") as f:
    json.dump(results, f, indent=2)
"""


# ==================================================================== #
#  95 — ChainOfThoughtTracing
# ==================================================================== #

class ChainOfThoughtTracing:
    """
    Trace and log agent reasoning steps in structured format.
    """

    def __init__(self, trace_path: str = "./traces"):
        self.trace_path = trace_path
        os.makedirs(trace_path, exist_ok=True)

    def tracing_code(self) -> str:
        return f"""
import json, uuid
from datetime import datetime

trace_path = "{self.trace_path}"
import os; os.makedirs(trace_path, exist_ok=True)

class ChainOfThoughtTracer:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.steps = []
        self._current_step = None

    def start_step(self, reasoning, action, confidence=1.0):
        self._current_step = {{
            "step_number": len(self.steps) + 1,
            "reasoning": reasoning,
            "action": action,
            "confidence": confidence,
            "outcome": None,
            "timestamp": datetime.now().isoformat(),
        }}

    def end_step(self, outcome, error=None):
        if self._current_step:
            self._current_step["outcome"] = outcome
            self._current_step["error"] = error
            self._current_step["duration_ms"] = (
                datetime.now() - datetime.fromisoformat(self._current_step["timestamp"])
            ).total_seconds() * 1000
            self.steps.append(self._current_step)
            self._current_step = None

    def export(self):
        trace = {{
            "session_id": self.session_id,
            "steps": self.steps,
            "total_steps": len(self.steps),
            "total_duration_ms": sum(s.get("duration_ms", 0) for s in self.steps),
        }}
        filepath = os.path.join(trace_path, f"trace_{{self.session_id}}.json")
        with open(filepath, "w") as f:
            json.dump(trace, f, indent=2)
        print(f"Trace saved: {{filepath}}")
        return trace

    def visualize_flowchart(self):
        '''Generate a simple text flowchart.'''
        lines = ["Agent Decision Flow:", "=" * 40]
        for s in self.steps:
            icon = "✅" if s.get("outcome") == "success" else "❌" if s.get("outcome") == "failed" else "⏳"
            lines.append(f"{{icon}} Step {{s['step_number']}}: {{s['action'][:50]}}")
            lines.append(f"   Reasoning: {{s['reasoning'][:80]}}")
        return "\\\\n".join(lines)

tracer = ChainOfThoughtTracer()
"""


# ==================================================================== #
#  96 — ModelComparisonReport
# ==================================================================== #

class ModelComparisonReport:
    """
    Generate an HTML comparison report across fine-tuned models.
    """

    REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Model Comparison</title>
<style>
body {{ font-family: Arial; max-width: 1000px; margin: auto; padding: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
th {{ background: #4CAF50; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.best {{ background: #d4edda; font-weight: bold; }}
img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>Model Comparison Report</h1>
<p>Generated: {date}</p>
<table>
<tr><th>Model</th><th>Loss</th><th>Accuracy</th><th>Training Time</th><th>Peak VRAM</th><th>Inference Speed</th></tr>
{rows}
</table>
<h2>Radar Chart</h2>
<div>{radar_chart}</div>
<h2>Recommendation</h2>
<p>{recommendation}</p>
</body>
</html>
"""

    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_code(self, results: list[dict]) -> str:
        return f"""
import os, json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

results = {json.dumps(results)}
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

# Table rows
html_rows = []
best_idx = min(range(len(results)), key=lambda i: results[i].get("final_loss", float("inf")))
for i, r in enumerate(results):
    cls = ' class="best"' if i == best_idx else ""
    html_rows.append(
        f"<tr{{cls}}><td>{{r['model']}}</td>"
        f"<td>{{r.get('final_loss', 'N/A')}}</td>"
        f"<td>{{r.get('accuracy', 'N/A')}}</td>"
        f"<td>{{r.get('training_time_sec', 'N/A')}}s</td>"
        f"<td>{{r.get('peak_vram_gb', 'N/A')}} GB</td>"
        f"<td>{{r.get('inference_speed_tok_s', 'N/A')}} tok/s</td></tr>"
    )

# Radar chart
categories = ["final_loss", "accuracy", "training_time_sec", "peak_vram_gb", "inference_speed_tok_s"]
fig = go.Figure()
for r in results:
    values = [r.get(c, 0) for c in categories]
    # Normalize loss (lower is better -> invert)
    max_loss = max(r.get("final_loss", 1) for r in results) or 1
    values[0] = (1 - values[0] / max_loss) * 100
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        name=r["model"],
        fill="toself",
    ))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
radar_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

# Recommendation
best = results[best_idx]
recommendation = f"Best model: {{best['model']}} (loss={{best['final_loss']}})"

# Render HTML
html = {json.dumps(self.REPORT_TEMPLATE)}.format(
    date=datetime.now().strftime("%Y-%m-%d %H:%M"),
    rows="\\\\n".join(html_rows),
    radar_chart=radar_html,
    recommendation=recommendation,
)

with open(os.path.join(output_dir, "model_comparison.html"), "w") as f:
    f.write(html)

print(f"Report saved: {{os.path.join(output_dir, 'model_comparison.html')}}")
"""
