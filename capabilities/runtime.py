"""
Phases 51-60 — Runtime & GPU Management.

Provides:
  - GPUDetector              detect GPU type + VRAM         (51)
  - RuntimeSwitcher          switch Colab runtime           (52)
  - FallbackStrategy         OOM recovery tiers             (53)
  - SessionKeepAlive         prevent Colab disconnect       (54)
  - DisconnectionHandler     resume after crash             (55)
  - ColabLimitsDetector      track compute units            (56)
  - RuntimeRecommendation    recommend best runtime          (57)
  - MultiGPUChecker          detect + utilize multiple GPUs (58)
  - ResourceReleaseHandler   free GPU/RAM/disk              (59)
  - RuntimeBenchmark         benchmark GPU performance      (60)
"""


# ==================================================================== #
#  51 — GPUDetector
# ==================================================================== #

class GPUDetector:
    """
    Detect GPU type, VRAM, and compute capability.
    """

    GPU_MAP = {
        "Tesla T4": ("T4", 16),
        "Tesla V100": ("V100", 16),
        "Tesla V100-SXM2": ("V100", 32),
        "Tesla A100": ("A100", 40),
        "Tesla A100-SXM4-80GB": ("A100-80GB", 80),
        "Tesla L4": ("L4", 24),
        "Tesla P100": ("P100", 16),
        "Tesla P4": ("P4", 8),
        "Tesla K80": ("K80", 12),
        "NVIDIA A100 80GB": ("A100-80GB", 80),
    }

    @staticmethod
    def detect_code() -> str:
        return """
import torch, subprocess, re

result = {"gpu_name": None, "vram_gb": 0, "compute_capability": None, "count": 0}

if torch.cuda.is_available():
    result["count"] = torch.cuda.device_count()
    result["gpu_name"] = torch.cuda.get_device_name(0)

    # Get VRAM
    props = torch.cuda.get_device_properties(0)
    result["vram_gb"] = round(props.total_memory / 1e9, 1)
    result["compute_capability"] = f"{props.major}.{props.minor}"

    # Map to known type
    gpu_map = {gpu_map_json}
    for key, (gpu_type, vram) in gpu_map.items():
        if key in result["gpu_name"]:
            result["gpu_type"] = gpu_type
            break
    else:
        result["gpu_type"] = result["gpu_name"]

else:
    result["gpu_type"] = "CPU"
    result["vram_gb"] = 0

# Also get via nvidia-smi
try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5,
    )
    result["nvidia_smi"] = smi.stdout.strip().split("\\n")
except Exception:
    result["nvidia_smi"] = []

# Check for CoLab TPU
try:
    import torch_xla
    result["tpu_available"] = True
except ImportError:
    result["tpu_available"] = False

import json
print(json.dumps(result, indent=2))
"""


# ==================================================================== #
#  52 — RuntimeSwitcher
# ==================================================================== #

class RuntimeSwitcher:
    """
    Generate Colab runtime switch code.
    Note: Colab does not allow programmatic runtime switching via API.
    This generates instructions and link-based workarounds.
    """

    RUNTIME_TIERS = ["T4", "V100", "A100", "A100-80GB"]

    @staticmethod
    def switch_code(target_gpu: str = "V100") -> str:
        return f"""
import os, json, time

target = "{target_gpu}"

# Method 1: Save checkpoint for resume
checkpoint_path = "./checkpoint_for_switch"
os.makedirs(checkpoint_path, exist_ok=True)

# Save training state
if hasattr(model, "save_pretrained"):
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Checkpoint saved to {{checkpoint_path}}")

# Method 2: Generate Colab link with target GPU
# Can't switch programmatically; use this link:
print(f"Switch to {{target}}:")
print(f"  https://colab.research.google.com/?accelerator=gpu&gpu_type={{target.lower()}}")

# Method 3: Use colab-cli if available
# !colab-cli server reconfigure --accelerator {{target}} --high-ram

# Method 4: Instructions for manual switch
print(f"To switch manually:")
print(f"  1. Runtime -> Change runtime type")
print(f"  2. Hardware accelerator -> GPU")
print(f"  3. GPU type -> {{target}}")
print(f"  4. Save, then re-run this notebook from top")

# Save switch state
state = {{
    "target_runtime": target,
    "checkpoint_path": checkpoint_path,
    "requires_restart": True,
    "timestamp": time.time(),
}}
with open(os.path.join(checkpoint_path, "switch_state.json"), "w") as f:
    json.dump(state, f, indent=2)
"""

    @staticmethod
    def resume_after_switch_code() -> str:
        return """
import os, json

# Detect if we switched and need to resume
checkpoint_path = "./checkpoint_for_switch"
state_path = os.path.join(checkpoint_path, "switch_state.json")

if os.path.exists(state_path):
    with open(state_path) as f:
        state = json.load(f)
    
    # Verify we're on the target GPU
    import torch
    current_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Current GPU: {current_gpu}")
    print(f"Target was: {state['target_runtime']}")
    
    # Resume from checkpoint
    if state["requires_restart"]:
        print("Runtime switch detected. Resuming from checkpoint...")
        # Load model from checkpoint
        # model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print(f"Checkpoint: {checkpoint_path}")
        
        # Clean up switch state
        os.remove(state_path)
else:
    print("No pending switch state found")
"""


# ==================================================================== #
#  53 — FallbackStrategy
# ==================================================================== #

class FallbackStrategy:
    """
    Tiered fallback for OOM errors:
    1. Halve batch size
    2. Double gradient accumulation
    3. Switch LoRA -> QLoRA
    4. Switch full -> LoRA
    5. Upgrade runtime
    """

    TIERS = [
        "reduce_batch_size",
        "increase_grad_accum",
        "switch_to_lora",
        "switch_to_qlora",
        "upgrade_runtime",
    ]

    def __init__(self, max_retries_per_tier: int = 3):
        self.max_retries_per_tier = max_retries_per_tier
        self.current_tier = 0
        self.retry_count = 0
        self.total_attempts = 0

    def next_fallback(self, tier_name: str = None) -> str:
        """Get the next fallback action."""
        if tier_name:
            self.current_tier = self.TIERS.index(tier_name) + 1
        else:
            self.current_tier += 1

        if self.current_tier >= len(self.TIERS):
            return "abort"
        return self.TIERS[self.current_tier]

    def apply_code(self, current_config: dict) -> str:
        return f"""
import torch

def apply_fallback(tier, config):
    '''Apply a fallback strategy and return updated config.'''
    if tier == "reduce_batch_size":
        config["per_device_train_batch_size"] = max(1, config.get("batch_size", 4) // 2)
        config["gradient_accumulation_steps"] = config.get("gradient_accum", 4) * 2
        print(f"Fallback: batch_size -> {{config['per_device_train_batch_size']}}, "
              f"grad_accum -> {{config['gradient_accumulation_steps']}}")

    elif tier == "increase_grad_accum":
        config["gradient_accumulation_steps"] = config.get("gradient_accumulation_steps", 4) * 2
        print(f"Fallback: grad_accum -> {{config['gradient_accumulation_steps']}}")

    elif tier == "switch_to_lora":
        config["method"] = "lora"
        config["quantization"] = None
        print("Fallback: full -> LoRA")

    elif tier == "switch_to_qlora":
        config["method"] = "qlora"
        config["quantization"] = "4bit"
        print("Fallback: LoRA -> QLoRA (4-bit)")

    elif tier == "upgrade_runtime":
        print("Fallback: runtime upgrade needed")
        config["requires_upgrade"] = True

    else:
        print(f"Unknown tier: {{tier}}")

    config["fallback_tier"] = tier
    return config

# Usage in agent loop:
config = apply_fallback(tier, config)
"""


# ==================================================================== #
#  54 — SessionKeepAlive
# ==================================================================== #

class SessionKeepAlive:
    """
    Keep Colab session alive by simulating user activity.
    """

    @staticmethod
    def enable_code() -> str:
        return """
from IPython.display import Javascript, display
import warnings

# Ethical warning
print("WARNING: Keepalive prevents Colab from reclaiming idle resources.")
print("Only use for long-running training. Disable when not needed.")

# JavaScript click on connect button every 60s
js_code = '''
function ClickConnect() {
    console.log("Keepalive ping at " + new Date().toISOString());
    document.querySelector("colab-connect-button")?.click();
}
setInterval(ClickConnect, 60000);
'''
display(Javascript(js_code))
print("Keepalive enabled (60s interval)")
"""

    @staticmethod
    def disable_code() -> str:
        return """
from IPython.display import Javascript, display

js_code = '''
// Clear all intervals to stop keepalive
for (var i = 0; i < 99999; i++) {
    window.clearInterval(i);
}
console.log("Keepalive disabled");
'''
display(Javascript(js_code))
print("Keepalive disabled")
"""

    @staticmethod
    def training_heartbeat_code() -> str:
        return """
import threading, time
from datetime import datetime, timezone

def heartbeat(interval=60):
    while True:
        time.sleep(interval)
        print(f"[heartbeat] Training alive at {datetime.now().isoformat()}")

heartbeat_thread = threading.Thread(target=heartbeat, args=(60,), daemon=True)
heartbeat_thread.start()
print("Training heartbeat started (prints every 60s)")
"""


# ==================================================================== #
#  55 — DisconnectionHandler
# ==================================================================== #

class DisconnectionHandler:
    """
    Handle Colab disconnection: save state, detect on restart, resume.
    """

    def __init__(self, state_path: str = "./disconnect_state.json"):
        self.state_path = state_path

    def save_state_code(self) -> str:
        return f"""
import json, os, time
from datetime import datetime, timezone

state_path = "{self.state_path}"

def save_disconnect_state(model_name, dataset, step, checkpoint_path, job_id):
    state = {{
        "model_name": model_name,
        "dataset": dataset,
        "step": step,
        "checkpoint_path": checkpoint_path,
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "runtime": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }}
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Disconnect state saved: {{state_path}}")
"""

    def detect_and_resume_code(self) -> str:
        return f"""
import json, os

state_path = "{self.state_path}"

if os.path.exists(state_path):
    with open(state_path) as f:
        state = json.load(f)
    
    print("=== DISCONNECTION DETECTED ===")
    print(f"Previous state: {{json.dumps(state, indent=2)}}")
    
    # Check if checkpoint exists
    if os.path.exists(state.get("checkpoint_path", "")):
        print(f"Checkpoint found: {{state['checkpoint_path']}}")
        print("Ready to resume training.")
        
        # Generate resume command
        resume_code = f'''
model = AutoModelForCausalLM.from_pretrained("{{state["checkpoint_path"]}}")
tokenizer = AutoTokenizer.from_pretrained("{{state["checkpoint_path"]}}")
trainer.train(resume_from_checkpoint="{{state["checkpoint_path"]}}")
'''
        print("Resume with the following code:")
        print(resume_code)
    else:
        print(f"Checkpoint NOT found: {{state.get('checkpoint_path', 'N/A')}}")
        print("Starting fresh training.")
    
    # Clean up
    os.remove(state_path)
else:
    print("No prior disconnect state found. Starting fresh.")
"""

    @staticmethod
    def signal_handler_code() -> str:
        return r'''
import signal
import json
from datetime import datetime, timezone

def handle_disconnect(signum, frame):
    """SIGTERM handler - save state before disconnect."""
    state = {
        "signal": signum,
        "timestamp": datetime.now().isoformat(),
        "step": getattr(trainer, "state", {}).get("global_step", 0),
    }
    with open("./disconnect_emergency.json", "w") as f:
        json.dump(state, f)
    
    # Save model
    model.save_pretrained("./emergency_checkpoint")
    print("Emergency checkpoint saved!")

signal.signal(signal.SIGTERM, handle_disconnect)
print("Disconnect signal handler registered")
'''


# ==================================================================== #
#  56 — ColabLimitsDetector
# ==================================================================== #

class ColabLimitsDetector:
    """
    Detect Colab usage limits (compute units, session time).
    """

    @staticmethod
    def detect_code() -> str:
        return """
import subprocess, json, os, time
from datetime import datetime, timezone

results = {
    "session_start": datetime.now().isoformat(),
    "gpu_available": False,
    "disk_free_gb": 0,
    "ram_gb": 0,
    "uptime_hours": 0,
}

# GPU info
import torch
if torch.cuda.is_available():
    results["gpu_available"] = True
    results["gpu_name"] = torch.cuda.get_device_name(0)
    results["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    results["gpu_utilization"] = round(torch.cuda.utilization(), 1) if hasattr(torch.cuda, "utilization") else "N/A"

# Disk
import shutil
usage = shutil.disk_usage("/")
results["disk_total_gb"] = round(usage.total / 1e9, 1)
results["disk_used_gb"] = round(usage.used / 1e9, 1)
results["disk_free_gb"] = round(usage.free / 1e9, 1)
results["disk_free_pct"] = round(usage.free / usage.total * 100, 1)

# RAM
import psutil
results["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
results["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 1)

# Uptime
try:
    with open("/proc/uptime") as f:
        uptime_sec = float(f.read().split()[0])
    results["uptime_hours"] = round(uptime_sec / 3600, 2)
except Exception:
    pass

# Colab-specific diagnostics
try:
    from google.colab import _message
    diag = _message.blocking_request("diagnose") or {}
    results["colab_limits"] = {
        "backend": diag.get("backend", {}),
        "runtime": diag.get("runtime", {}),
    }
except Exception:
    pass

# Warn if low resources
warnings = []
if results["disk_free_gb"] < 5:
    warnings.append(f"Low disk space: {results['disk_free_gb']} GB free")
if results.get("ram_available_gb", 999) < 2:
    warnings.append(f"Low RAM: {results.get('ram_available_gb', 0)} GB available")
if results.get("uptime_hours", 0) > 10:
    warnings.append(f"Session active for {results['uptime_hours']} hours")

results["warnings"] = warnings
print(json.dumps(results, indent=2))
"""

    @staticmethod
    def compute_unit_tracker() -> str:
        return """
import json, os
from datetime import datetime, timedelta, timezone

usage_log = "./compute_usage.json"

# Load existing usage
if os.path.exists(usage_log):
    with open(usage_log) as f:
        usage = json.load(f)
else:
    usage = {"units_used": 0, "sessions": []}

# Check if within same UTC day
today = datetime.now(timezone.utc).date()
last_session = usage["sessions"][-1] if usage["sessions"] else None
if last_session and datetime.fromisoformat(last_session["date"]).date() == today:
    # Same day, don't reset
    pass
else:
    # New day, start fresh
    usage = {"units_used": 0, "sessions": []}

# Estimate units used this session (approximate)
# T4 ~1 unit/hr, V100 ~1.5, A100 ~2.5
import torch
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
unit_rates = {"T4": 1.0, "V100": 1.5, "A100": 2.5, "A100 80GB": 3.0}
rate = 0
for key, r in unit_rates.items():
    if key.lower() in gpu_name.lower():
        rate = r
        break

session_units = rate * 1.0  # Assume 1 hour if we can't track precisely
usage["units_used"] += session_units
usage["sessions"].append({
    "date": datetime.now(timezone.utc).isoformat(),
    "gpu": gpu_name,
    "estimated_units": session_units,
    "total_units": usage["units_used"],
})

with open(usage_log, "w") as f:
    json.dump(usage, f, indent=2)

print(f"Estimated compute units used: {usage['units_used']:.1f}")
if usage["units_used"] > 80:
    print("WARNING: Approaching daily limit of 100 compute units!")
    print("Consider downgrading runtime to save units.")
"""


# ==================================================================== #
#  57 — RuntimeRecommendationEngine
# ==================================================================== #

class RuntimeRecommendationEngine:
    """
    Recommend the best Colab runtime based on model and dataset.
    """

    @staticmethod
    def recommend_code() -> str:
        return """
import json, math

# Inputs
model_params_b = float(input("Model params (B): ") or 7)
dataset_size_mb = float(input("Dataset size (MB): ") or 100)
method = input("Fine-tune method (qlora/lora/full): ") or "qlora"
max_session_hours = float(input("Max session hours: ") or 12)

# Estimate VRAM needed
vram_needed = {
    "full": model_params_b * 2.0,
    "lora": model_params_b * 0.5 + 2,
    "qlora": model_params_b * 0.3 + 1,
}.get(method, model_params_b * 0.3 + 1)

# Estimate training time (hours)
training_hours = {
    "full": dataset_size_mb * model_params_b * 0.01,
    "lora": dataset_size_mb * model_params_b * 0.005,
    "qlora": dataset_size_mb * model_params_b * 0.003,
}.get(method, dataset_size_mb * model_params_b * 0.003)

# Runtime tiers
tiers = [
    {"name": "T4", "vram": 16, "unit_cost": 1.0, "speed_factor": 1.0},
    {"name": "V100", "vram": 16, "unit_cost": 1.5, "speed_factor": 1.8},
    {"name": "A100", "vram": 40, "unit_cost": 2.5, "speed_factor": 3.5},
    {"name": "A100-80GB", "vram": 80, "unit_cost": 3.0, "speed_factor": 4.0},
]

recommendations = []
for tier in tiers:
    fits = vram_needed <= tier["vram"] * 0.9
    if not fits:
        continue
    adjusted_hours = training_hours / tier["speed_factor"]
    compute_units = adjusted_hours * tier["unit_cost"]
    within_time = adjusted_hours <= max_session_hours
    recommendations.append({
        "runtime": tier["name"],
        "vram_gb": tier["vram"],
        "estimated_hours": round(adjusted_hours, 2),
        "estimated_compute_units": round(compute_units, 2),
        "fits_in_vram": fits,
        "within_session_limit": within_time,
        "score": round((1.0 / compute_units) if compute_units > 0 else 999, 2),
    })

# Sort by score (higher = better)
recommendations.sort(key=lambda x: -x["score"])
print(json.dumps(recommendations, indent=2))

# Best recommendation
if recommendations:
    best = recommendations[0]
    print(f"=== RECOMMENDATION ===")
    print(f"Best runtime: {best['runtime']}")
    print(f"Estimated: {best['estimated_hours']}h, {best['estimated_compute_units']} CCU")
    print(f"VRAM: {best['vram_gb']} GB (needed: {vram_needed:.1f} GB)")
    if not best["within_session_limit"]:
        print("NOTE: Exceeds session limit. Colab Pro+ recommended.")
    if best["estimated_compute_units"] > 80:
        print("NOTE: High compute unit usage. Monitor your quota.")
"""


# ==================================================================== #
#  58 — MultiGPUChecker
# ==================================================================== #

class MultiGPUChecker:
    """
    Detect multiple GPUs and configure DataParallel / DistributedDataParallel.
    """

    @staticmethod
    def detect_code() -> str:
        return """
import torch, subprocess, json

results = {"device_count": 0, "gpus": [], "multi_gpu_available": False}

# PyTorch detection
if torch.cuda.is_available():
    results["device_count"] = torch.cuda.device_count()
    results["multi_gpu_available"] = results["device_count"] > 1
    for i in range(results["device_count"]):
        props = torch.cuda.get_device_properties(i)
        results["gpus"].append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "vram_gb": round(props.total_memory / 1e9, 1),
            "compute_capability": f"{props.major}.{props.minor}",
        })

# nvidia-smi verification
try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5,
    )
    results["nvidia_smi_output"] = smi.stdout.strip().split("\\n")
except Exception:
    pass

print(json.dumps(results, indent=2))

if results["multi_gpu_available"]:
    print(f"Multiple GPUs detected ({results['device_count']})!")
    print("Using nn.DataParallel or DistributedDataParallel.")
else:
    print("Single GPU detected. Using standard training loop.")
"""

    @staticmethod
    def data_parallel_code() -> str:
        return """
# nn.DataParallel (simpler, less efficient)
if torch.cuda.device_count() > 1:
    print(f"Wrapping model with DataParallel ({torch.cuda.device_count()} GPUs)")
    model = torch.nn.DataParallel(model)

# DistributedDataParallel (more efficient, for multi-process)
# Launch with: torchrun --nproc_per_node=N train.py
# import torch.distributed as dist
# dist.init_process_group("nccl")
# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)
# model = model.to(local_rank)
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
"""


# ==================================================================== #
#  59 — ResourceReleaseHandler
# ==================================================================== #

class ResourceReleaseHandler:
    """
    Free GPU memory, RAM, and disk space after training.
    """

    @staticmethod
    def cleanup_code() -> str:
        return """
import torch, gc, os, shutil
from datetime import datetime, timezone

print(f"[{datetime.now().isoformat()}] Starting resource cleanup...")

# 1. Clear CUDA cache
if torch.cuda.is_available():
    before = torch.cuda.memory_allocated() / 1e9
    torch.cuda.empty_cache()
    gc.collect()
    after = torch.cuda.memory_allocated() / 1e9
    print(f"CUDA cache: {before:.2f} GB -> {after:.2f} GB (freed {before-after:.2f} GB)")

# 2. Unload model from memory
try:
    model.cpu()
    del model
    print("Model unloaded from GPU")
except Exception:
    pass

try:
    del tokenizer
except Exception:
    pass

# 3. Force garbage collection
gc.collect()
print(f"GC done. Process memory: {__import__('psutil').Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")

# 4. Clean up old checkpoints (keep only latest)
ckpt_dir = "./checkpoints"
if os.path.exists(ckpt_dir):
    ckpts = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")])
    for old in ckpts[:-3]:  # Keep last 3
        shutil.rmtree(os.path.join(ckpt_dir, old), ignore_errors=True)
        print(f"Removed old checkpoint: {old}")

# 5. Clean HuggingFace cache (optional)
cache_dir = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(cache_dir):
    size_before = sum(f.stat().st_size for f in __import__("pathlib").Path(cache_dir).rglob("*")) / 1e9
    # Only clear hub cache, not downloaded models in use
    hub_cache = os.path.join(cache_dir, "hub")
    if os.path.exists(hub_cache):
        shutil.rmtree(hub_cache, ignore_errors=True)
        print(f"HuggingFace hub cache cleared")

print(f"[{datetime.now().isoformat()}] Cleanup complete")
"""


# ==================================================================== #
#  60 — RuntimeBenchmark
# ==================================================================== #

class RuntimeBenchmark:
    """
    Benchmark GPU performance: time per step, tokens/second, peak memory.
    """

    @staticmethod
    def benchmark_code() -> str:
        return """
import torch, time, json
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=== GPU Benchmark ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Warmup
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True,
)
model.eval()

# Create dummy input
input_text = "Hello, world! " * 128
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Warmup runs
for _ in range(3):
    with torch.no_grad():
        _ = model(**inputs)

# Benchmark forward pass
torch.cuda.synchronize()
start = time.time()
num_steps = 20
for _ in range(num_steps):
    with torch.no_grad():
        _ = model(**inputs)
torch.cuda.synchronize()
end = time.time()

avg_time = (end - start) / num_steps
tokens_per_step = inputs["input_ids"].shape[1]
tokens_per_sec = tokens_per_step / avg_time

print(f"Input length: {tokens_per_step} tokens")
print(f"Average forward pass: {avg_time*1000:.2f} ms")
print(f"Tokens/second: {tokens_per_sec:.0f}")

# Memory benchmark
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak GPU memory: {peak_memory:.2f} GB")

# Benchmark generation
start = time.time()
gen_ids = model.generate(inputs["input_ids"], max_new_tokens=100, do_sample=False)
torch.cuda.synchronize()
gen_time = time.time() - start
gen_tokens = gen_ids.shape[1] - inputs["input_ids"].shape[1]
print(f"Generation: {gen_tokens} tokens in {gen_time:.2f}s ({gen_tokens/gen_time:.0f} tok/s)")

results = {
    "gpu_name": torch.cuda.get_device_name(0),
    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    "forward_pass_ms": round(avg_time * 1000, 2),
    "tokens_per_sec": round(tokens_per_sec, 0),
    "peak_memory_gb": round(peak_memory, 2),
    "gen_speed_tok_s": round(gen_tokens / gen_time, 0),
}

with open("./benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("=== Benchmark Complete ===")
"""
