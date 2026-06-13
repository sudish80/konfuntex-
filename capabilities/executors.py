"""
Phases 26-30 — Execution & Upload.

Provides:
  - MultimodalDataLoader    load image-text / video       (26)
  - CodeInterpreterExecutor  sandboxed Python execution   (27)
  - BashCommandRunner        whitelisted shell commands   (28)
  - DockerSandbox            container-like isolation     (29)
  - ArtifactUploader         upload to HF / Drive / S3   (30)
"""
import os


# ==================================================================== #
#  26 — MultimodalDataLoader
# ==================================================================== #

class MultimodalDataLoader:
    """
    Load image-text pairs, resize, normalize, return DataLoader.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, output_dir: str = "./multimodal_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_code(self, image_dir: str = "./images",
                  caption_file: str = "./captions.json",
                  image_size: int = 224,
                  batch_size: int = 32) -> str:
        return f"""
import os, json, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
from PIL import Image

image_dir = "{image_dir}"
caption_file = "{caption_file}"
image_size = {image_size}
batch_size = {batch_size}

# Load captions
with open(caption_file) as f:
    captions = json.load(f)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean={self.IMAGENET_MEAN}, std={self.IMAGENET_STD}),
])

class ImageTextDataset(Dataset):
    def __init__(self, captions, image_dir, transform):
        self.items = list(captions.items()) if isinstance(captions, dict) else captions
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if isinstance(self.items[idx], (list, tuple)):
            img_name, text = self.items[idx]
        else:
            img_name = self.items[idx].get("image", "")
            text = self.items[idx].get("text", "")
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {{"image": image, "text": text}}

dataset = ImageTextDataset(captions, image_dir, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
print(f"Loaded {{len(dataset)}} image-text pairs")

# For video: sample every 10th frame
def extract_video_frames(video_path, sample_rate=10):
    import decord
    vr = decord.VideoReader(video_path)
    frames = [vr[i].asnumpy() for i in range(0, len(vr), sample_rate)]
    return frames
"""


# ==================================================================== #
#  27 — CodeInterpreterExecutor
# ==================================================================== #

class CodeInterpreterExecutor:
    """
    Execute Python code in a sandboxed environment
    with restricted builtins, timeout, and output capture.
    """

    WHITELISTED_BUILTINS = {
        "abs", "all", "any", "bin", "bool", "chr", "complex", "dict",
        "dir", "divmod", "enumerate", "filter", "float", "format",
        "frozenset", "hash", "hex", "id", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min",
        "next", "object", "oct", "ord", "pow", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str",
        "sum", "tuple", "type", "zip", "True", "False", "None",
        "Exception", "ValueError", "TypeError", "KeyError",
        "IndexError", "AttributeError", "ImportError", "RuntimeError",
    }

    WHITELISTED_LIBS = {
        "numpy", "pandas", "torch", "transformers", "sklearn",
        "scipy", "json", "math", "random", "datetime", "re",
        "collections", "itertools", "functools", "typing",
    }

    def __init__(self, timeout_sec: int = 30, memory_limit_mb: int = 2048):
        self.timeout_sec = timeout_sec
        self.memory_limit_mb = memory_limit_mb

    def execute_code(self, code_str: str) -> dict:
        """
        Execute Python code in a restricted environment.
        Returns {"output": str, "error": str, "success": bool}.
        """
        import io
        import contextlib
        from func_timeout import func_timeout, FunctionTimedOut

        # Restricted globals
        safe_globals = {"__builtins__": {}}
        for b in self.WHITELISTED_BUILTINS:
            safe_globals["__builtins__"][b] = __builtins__[b]
        safe_globals["__builtins__"]["__import__"] = self._safe_import

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = {"output": "", "error": "", "success": True}

        def _run():
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    exec(code_str, safe_globals)

        try:
            func_timeout(self.timeout_sec, _run)
            result["output"] = stdout_capture.getvalue()
            result["error"] = stderr_capture.getvalue()
        except FunctionTimedOut:
            result["error"] = f"Timeout after {self.timeout_sec}s"
            result["success"] = False
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            result["success"] = False

        return result

    def _safe_import(self, name, *args, **kwargs):
        if name in self.WHITELISTED_LIBS:
            return __import__(name, *args, **kwargs)
        raise ImportError(f"Module '{name}' is not whitelisted")

    def sandboxed_exec_code(self) -> str:
        return f"""
from capabilities.executors import CodeInterpreterExecutor

executor = CodeInterpreterExecutor(timeout_sec={self.timeout_sec})
result = executor.execute_code(user_code)
print("STDOUT:", result["output"])
print("STDERR:", result["error"])
print("Success:", result["success"])
"""


# ==================================================================== #
#  28 — BashCommandRunner
# ==================================================================== #

class BashCommandRunner:
    """
    Execute shell commands via subprocess with whitelist/blocklist.
    """

    WHITELISTED = {"pip", "git", "wget", "curl", "apt-get", "ls",
                   "cd", "mkdir", "cp", "mv", "cat", "echo", "head",
                   "tail", "wc", "sort", "grep", "find", "tar", "gzip",
                   "unzip", "python", "python3", "which", "nvidia-smi",
                   "df", "du", "free", "ps", "kill", "chmod", "ln",
                   "date", "env", "pwd", "rm"}

    BLOCKED_PATTERNS = ["rm -rf /", "dd if=", "mkfs", "shutdown",
                        "reboot", "> /dev/sda", ":(){ :|:& };:"]

    def __init__(self, log_path: str = "execution_log.json"):
        self.log_path = log_path
        self.history = []

    def execute(self, command: str, timeout: int = 60,
                require_confirmation: bool = False) -> dict:
        import subprocess
        import shlex

        cmd_base = shlex.split(command)[0] if command.strip() else ""
        if cmd_base not in self.WHITELISTED:
            return {"output": "", "error": f"Command '{cmd_base}' not whitelisted",
                    "success": False, "return_code": -1}

        for pat in self.BLOCKED_PATTERNS:
            if pat in command:
                return {"output": "", "error": f"Blocked pattern: {pat}",
                        "success": False, "return_code": -1}

        try:
            result = subprocess.run(
                command, shell=True, timeout=timeout, check=False,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            return {"output": "", "error": f"Timeout after {timeout}s",
                    "success": False, "return_code": -1}

        entry = {"command": command, "return_code": result.returncode,
                 "stdout": result.stdout[:1000], "stderr": result.stderr[:500],
                 "success": result.returncode == 0}
        self.history.append(entry)

        import json
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return {"output": result.stdout, "error": result.stderr,
                "success": result.returncode == 0,
                "return_code": result.returncode}


# ==================================================================== #
#  29 — DockerSandbox
# ==================================================================== #

class DockerSandbox:
    """
    Container-like isolation for code execution.
    Falls back to venv + restricted subprocess when Docker is unavailable.
    """

    def __init__(self, workdir: str = "./sandbox"):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)

    def setup_code(self) -> str:
        return f"""
import os, sys, venv, subprocess

sandbox_dir = "{self.workdir}"

# Check for Docker
docker_available = False
try:
    subprocess.run(["docker", "--version"], capture_output=True, check=True)
    docker_available = True
except (FileNotFoundError, subprocess.CalledProcessError):
    pass

if docker_available:
    # Use Docker container
    dockerfile = \"\"\"
FROM python:3.10-slim
RUN pip install numpy pandas torch transformers scikit-learn
WORKDIR /workspace
\"\"\"
    with open(os.path.join(sandbox_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile)
    subprocess.run(["docker", "build", "-t", "colab-sandbox", sandbox_dir], check=True)
    print("Docker sandbox ready: colab-sandbox")
else:
    # Fallback: Python venv
    venv_dir = os.path.join(sandbox_dir, ".venv")
    if not os.path.exists(venv_dir):
        venv.create(venv_dir, with_pip=True)
        pip_path = os.path.join(venv_dir, "bin", "pip") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "pip")
        subprocess.run([pip_path, "install", "numpy", "pandas"], check=True)
    print(f"Venv sandbox ready: {{venv_dir}}")
"""


# ==================================================================== #
#  30 — ArtifactUploader
# ==================================================================== #

class ArtifactUploader:
    """
    Upload fine-tuned models and artifacts to HuggingFace Hub,
    Google Drive, or S3-compatible storage.
    """

    SUPPORTED_DESTINATIONS = ["huggingface", "drive", "s3"]

    def __init__(self, output_dir: str = "./uploads"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def upload_code(self, local_path: str = "./finetuned_model",
                    destination: str = "huggingface",
                    repo_id: str = "",
                    s3_bucket: str = "") -> str:
        return f"""
import os, json, tarfile, time

local_path = "{local_path}"
output_dir = "{self.output_dir}"
os.makedirs(output_dir, exist_ok=True)

# Create archive
archive_name = f"artifact_{{int(time.time())}}.tar.gz"
archive_path = os.path.join(output_dir, archive_name)
with tarfile.open(archive_path, "w:gz") as tar:
    tar.add(local_path, arcname=os.path.basename(local_path))

# Add metadata
metadata = {{
    "job_id": os.environ.get("COLAB_AGENT_JOB_ID", "unknown"),
    "timestamp": time.time(),
    "model_name": "{repo_id}",
    "local_path": local_path,
    "file_size_mb": round(os.path.getsize(archive_path) / 1e6, 2),
}}
with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Archive: {{archive_name}} ({{metadata['file_size_mb']}} MB)")

# Upload to HuggingFace Hub
if "{destination}" == "huggingface":
    from huggingface_hub import HfApi, login
    token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)
    api = HfApi()
    api.upload_folder(
        folder_path=local_path,
        repo_id="{repo_id}",
        commit_message="Upload fine-tuned model",
    )
    print(f"Uploaded to https://huggingface.co/{{repo_id}}")

# Upload to Google Drive
elif "{destination}" == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    import shutil
    dest = f"/content/drive/MyDrive/colab-artifacts/{{os.path.basename(local_path)}}"
    shutil.copytree(local_path, dest, dirs_exist_ok=True)
    shutil.copy(archive_path, f"/content/drive/MyDrive/colab-artifacts/")
    print(f"Uploaded to Drive: {{dest}}")

# Upload to S3
elif "{destination}" == "s3":
    import boto3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    s3.upload_file(archive_path, "{s3_bucket}", archive_name)
    print(f"Uploaded to s3://{{s3_bucket}}/{{archive_name}}")

print("Upload complete")
"""
