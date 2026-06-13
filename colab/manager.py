import json
from datetime import datetime


class ColabManager:
    def __init__(self, drive_folder_id: str = None, credentials_path: str = None):
        self.drive_folder_id = drive_folder_id
        self.credentials_path = credentials_path
        self.active_notebook = None
        self.notebooks = {}

    def create_notebook_code(self, title: str = "Colab-Agent Notebook") -> str:
        """Generate code to create and open a new Colab notebook via Google Drive."""
        notebook_name = f"{title} - {datetime.now().strftime('%Y%m%d_%H%M%S')}"

        code = f"""
# ===== Colab Agent: Notebook Setup =====
# This cell initializes the Colab environment

import sys
import subprocess
import json
from datetime import datetime

print(f"Colab Agent Notebook: {notebook_name}")
print(f"Start time: {{datetime.now().isoformat()}}")
print(f"Python: {{sys.version}}")

# Check GPU
!nvidia-smi 2>/dev/null || echo "No GPU detected"

# Check RAM
import psutil
ram = psutil.virtual_memory()
print(f"Total RAM: {{ram.total / 1e9:.2f}} GB")
print(f"Available RAM: {{ram.available / 1e9:.2f}} GB")

# Install base deps
!pip install -q transformers datasets accelerate peft trl bitsandbytes huggingface_hub

print("=== Environment ready ===")
"""
        return code

    def check_runtime_code(self) -> str:
        """Generate code to check current Colab runtime specs."""
        return """
# ===== Runtime Check =====
import torch
import psutil
import subprocess
import json

# GPU Info
gpu_info = {}
if torch.cuda.is_available():
    gpu_info["available"] = True
    gpu_info["count"] = torch.cuda.device_count()
    gpu_info["name"] = torch.cuda.get_device_name(0)
    gpu_info["vram_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_info["vram_allocated"] = torch.cuda.memory_allocated(0) / 1e9
    gpu_info["vram_cached"] = torch.cuda.memory_reserved(0) / 1e9
    gpu_info["compute_capability"] = f"{torch.cuda.get_device_capability(0)}"
else:
    gpu_info["available"] = False
    gpu_info["name"] = "No GPU"

# RAM Info
ram = psutil.virtual_memory()
gpu_info["ram_total_gb"] = ram.total / 1e9
gpu_info["ram_available_gb"] = ram.available / 1e9
gpu_info["ram_percent"] = ram.percent

# Nvidia driver
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=10)
    gpu_info["nvidia_smi"] = result.stdout.strip()
except:
    gpu_info["nvidia_smi"] = "nvidia-smi not available"

# CUDA version
gpu_info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"

print(json.dumps(gpu_info, indent=2))
"""

    def mount_drive_code(self) -> str:
        return """
from google.colab import drive
drive.mount('/content/drive')
print("Google Drive mounted at /content/drive")
"""

    def execute_cell_code(self, code: str) -> str:
        """Wrap arbitrary code for execution in Colab."""
        return """
# ===== Colab Agent: Executing Cell =====
import json, sys, traceback, time
try:
    start = time.time()
    exec(""" + json.dumps(code) + """)
    elapsed = time.time() - start
    print("\\n=== CELL COMPLETED in {{:.2f}}s ===".format(elapsed))
except Exception as e:
    elapsed = time.time() - start
    error_info = {{
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
        "elapsed": elapsed,
    }}
    print(f"\\n=== CELL ERROR ===")
    print(json.dumps(error_info, indent=2))
"""
