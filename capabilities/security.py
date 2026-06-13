"""
Phases 84-90 — Security & Safety.

Provides:
  - CommandBlocklist         AST-based dangerous code check   (84)
  - NetworkTrafficLogger     log external requests           (85)
  - ResourceQuotaEnforcer    per-job resource limits          (86)
  - DataLeakDetector         scan for secrets in outputs     (87)
  - UserConfirmationOnPush   confirm before external push    (88)
  - SessionEncryption        encrypt sensitive data          (89)
  - AnomalyDetector          detect abnormal agent behavior  (90)
"""
import json
import os
import re


# ==================================================================== #
#  84 — CommandBlocklist
# ==================================================================== #

class CommandBlocklist:
    """
    AST-based detection of dangerous Python code.
    """

    DANGEROUS_COMMANDS = [
        "rm -rf /", "sudo", "dd if=", "mkfs", "shutdown",
        "reboot", "chmod 777", "> /dev/sda",
    ]

    DANGEROUS_IMPORTS = ["os.system", "subprocess.call",
                         "subprocess.Popen", "eval", "exec"]

    @staticmethod
    def check_code() -> str:
        return """
import ast
import re

def check_dangerous(code: str) -> list[str]:
    warnings = []

    # Check for dangerous shell commands
    for cmd in {json.dumps(CommandBlocklist.DANGEROUS_COMMANDS)}:
        if cmd in code.lower():
            warnings.append(f"Dangerous command detected: {cmd}")

    # AST analysis
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Check os.system, subprocess.call, etc.
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    full_name = ""
                    if isinstance(node.func.value, ast.Name):
                        full_name = f"{node.func.value.id}.{node.func.attr}"
                    elif isinstance(node.func.value, ast.Attribute):
                        inner = node.func.value
                        if isinstance(inner.value, ast.Name):
                            full_name = f"{inner.value.id}.{inner.attr}.{node.func.attr}"

                    for dangerous in {json.dumps(CommandBlocklist.DANGEROUS_IMPORTS)}:
                        if dangerous in full_name:
                            warnings.append(f"Dangerous call: {full_name}")

            # Check eval/exec
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "__import__"):
                    warnings.append(f"Dangerous function: {node.func.id}()")

    except SyntaxError:
        warnings.append("Could not parse code (SyntaxError)")

    return warnings

# Example
code = '''
import os
os.system("rm -rf /")
'''
warnings = check_dangerous(code)
print(f"Warnings: {warnings}")
"""


# ==================================================================== #
#  85 — NetworkTrafficLogger
# ==================================================================== #

class NetworkTrafficLogger:
    """
    Log all external network requests via requests hooks.
    """

    def __init__(self, log_path: str = "./network.log"):
        self.log_path = log_path

    def logging_code(self) -> str:
        return f"""
import requests
import logging
from datetime import datetime

log_path = "{self.log_path}"
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format="%(asctime)s [NET] %(message)s")

def log_request(response, *args, **kwargs):
    '''Requests hook to log all requests.'''
    url = response.request.url if hasattr(response, 'request') else "unknown"
    method = response.request.method if hasattr(response, 'request') else "?"
    status = response.status_code if hasattr(response, 'status_code') else "?"
    size = len(response.content) if hasattr(response, 'content') else 0

    # Redact auth headers
    headers = dict(response.request.headers) if hasattr(response, 'request') and hasattr(response.request, 'headers') else {{}}
    for key in list(headers.keys()):
        if key.lower() in ("authorization", "x-api-key", "cookie"):
            headers[key] = "[REDACTED]"

    logging.info(json.dumps({{
        "method": method,
        "url": url,
        "status": status,
        "size_bytes": size,
        "headers": headers,
        "timestamp": datetime.now().isoformat(),
    }}))

# Install hook
requests.hooks["response"] = [log_request]
print(f"Network logger active -> {{log_path}}")
"""


# ==================================================================== #
#  86 — ResourceQuotaEnforcer
# ==================================================================== #

class ResourceQuotaEnforcer:
    """
    Enforce per-job resource quotas.
    """

    DEFAULT_QUOTAS = {
        "max_ram_gb": 25,
        "max_vram_gb": 40,
        "max_runtime_hours": 12,
        "max_llm_api_calls": 1000,
        "max_github_api_calls": 5000,
    }

    def __init__(self, quotas: dict = None):
        self.quotas = quotas or self.DEFAULT_QUOTAS

    def enforce_code(self) -> str:
        return f"""
import psutil, time, json
from datetime import datetime

quotas = {json.dumps(self.quotas)}
usage = {{"ram_gb": 0, "vram_gb": 0, "runtime_hours": 0,
          "llm_calls": 0, "github_calls": 0}}
start_time = time.time()

def check_quota(resource: str, increment: float = 0) -> bool:
    if increment:
        if resource in usage:
            usage[resource] += increment

    current = usage.get(resource, 0)
    limit = quotas.get(resource, float("inf"))

    if current >= limit * 0.8:
        print(f"WARNING: {{resource}} at {{current/limit*100:.0f}}% of quota")
    if current >= limit * 0.95:
        print(f"BLOCKED: {{resource}} quota exceeded ({{current}}/{{limit}})")
        return False
    return True

# Per-action check
def pre_action_check(action_type):
    # RAM
    usage["ram_gb"] = psutil.virtual_memory().used / 1e9

    # VRAM
    import torch
    if torch.cuda.is_available():
        usage["vram_gb"] = torch.cuda.memory_allocated() / 1e9

    # Runtime
    usage["runtime_hours"] = (time.time() - start_time) / 3600

    # Count API calls
    if "api_call" in action_type:
        usage["llm_calls"] += 1

    for resource in quotas:
        if not check_quota(resource):
            return False
    return True
"""


# ==================================================================== #
#  87 — DataLeakDetector
# ==================================================================== #

class DataLeakDetector:
    """
    Scan for secrets, keys, and PII before saving or pushing.
    """

    PATTERNS = {
        "OpenAI API Key": r"sk-[a-zA-Z0-9]{20,}",
        "GitHub Token": r"gh[pousr]_[a-zA-Z0-9_]{36,}",
        "HuggingFace Token": r"hf_[a-zA-Z0-9_]{20,}",
        "Google API Key": r"AIza[0-9A-Za-z_-]{35}",
        "AWS Access Key": r"AKIA[0-9A-Z]{16}",
        "JWT Token": r"eyJ[a-zA-Z0-9_-]{10,}\\.[a-zA-Z0-9_-]{10,}\\.[a-zA-Z0-9_-]{10,}",
        "Email": r"[\\w\\.-]+@[\\w\\.-]+\\.\\w+",
        "IP Address": r"\\b\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}}\\.\\d{{1,3}}\\b",
    }

    @staticmethod
    def scan_code() -> str:
        return """
import re
import json
import math

patterns = {json.dumps(DataLeakDetector.PATTERNS)}

def scan_for_secrets(text: str) -> list[dict]:
    findings = []
    for name, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for m in matches:
            # Entropy check for API keys
            key = m.group()
            entropy = -sum(
                (c / len(key)) * math.log2(c / len(key))
                for c in [key.count(ch) for ch in set(key)]
            )
            if entropy > 3.5 or name == "Email":  # Skip low-entropy (likely false positives)
                findings.append({
                    "type": name,
                    "position": (m.start(), m.end()),
                    "preview": key[:8] + "..." + key[-4:] if len(key) > 12 else key,
                    "entropy": round(entropy, 2),
                })
    return findings

def redact_secrets(text: str) -> str:
    for name, pattern in patterns.items():
        text = re.sub(pattern, f"[REDACTED_{name.upper().replace(' ', '_')}]", text)
    return text

# Scan before push
with open("file_to_check.txt") as f:
    content = f.read()

findings = scan_for_secrets(content)
if findings:
    print(f"WARNING: Found {len(findings)} secrets!")
    for f in findings:
        print(f"  {f['type']}: {f['preview']}")
    print("Redacting...")
    content = redact_secrets(content)

with open("file_to_check.txt", "w") as f:
    f.write(content)
"""


# ==================================================================== #
#  88 — UserConfirmationOnExternalPush
# ==================================================================== #

class UserConfirmationOnExternalPush:
    """
    Require user confirmation before pushing to external services.
    """

    @staticmethod
    def confirmation_code() -> str:
        return """
import os, json
from datetime import datetime

def confirm_push(files: list, destination: str, is_public: bool = False) -> bool:
    print(f"\\n{'='*60}")
    print(f"PUSH CONFIRMATION REQUIRED")
    print(f"{'='*60}")
    print(f"Destination: {destination}")
    print(f"Public: {'YES ⚠' if is_public else 'No'}")
    print(f"Files ({len(files)}):")
    for f in files:
        size = os.path.getsize(f) if os.path.exists(f) else 0
        print(f"  - {f} ({size/1024:.1f} KB)")

    # Check for secrets
    from capabilities.ui_builder import DataLeakDetector
    secrets_found = False
    for f in files:
        if os.path.exists(f):
            with open(f) as fh:
                findings = DataLeakDetector.scan_for_secrets(fh.read())
                if findings:
                    secrets_found = True
                    print(f"  ⚠ SECRETS FOUND in {f}:")
                    for finding in findings:
                        print(f"    - {finding['type']}")

    if secrets_found:
        print("\\n⚠ WARNING: Secrets detected in push!")
        response = input("Redact and continue? [y/N]: ").strip().lower()
        if response != "y":
            return False

    response = input(f"\\nPush {len(files)} files to {destination}? [yes/no/diff]: ").strip().lower()
    if response == "diff":
        import subprocess
        subprocess.run(["git", "diff", "--stat"])
        response = input("Push now? [yes/no]: ").strip().lower()

    approved = response in ("yes", "y")
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "destination": destination,
        "files": files,
        "approved": approved,
        "public": is_public,
    }
    with open("push_history.json", "a") as f:
        f.write(json.dumps(log_entry) + "\\n")

    return approved
"""


# ==================================================================== #
#  89 — SessionEncryption
# ==================================================================== #

class SessionEncryption:
    """
    Encrypt sensitive data using Fernet symmetric encryption.
    """

    @staticmethod
    def encrypt_code() -> str:
        return """
import os
from cryptography.fernet import Fernet
import base64
import hashlib

# Generate or load key
key_file = "./session_key.key"
if os.path.exists(key_file):
    with open(key_file, "rb") as f:
        key = f.read()
else:
    key = Fernet.generate_key()
    with open(key_file, "wb") as f:
        f.write(key)
    os.chmod(key_file, 0o600)  # Owner read/write only

cipher = Fernet(key)

def encrypt_data(data: str) -> bytes:
    return cipher.encrypt(data.encode())

def decrypt_data(token: bytes) -> str:
    return cipher.decrypt(token).decode()

# Master password mode (for long-term storage)
def derive_key_from_password(password: str, salt: bytes = None) -> bytes:
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return base64.urlsafe_b64encode(key)

# Usage
api_key = os.environ.get("OPENAI_API_KEY", "")
encrypted = encrypt_data(api_key)
print(f"Encrypted (len={len(encrypted)}): {encrypted[:20]}...")

# Store encrypted key in env
os.environ["OPENAI_API_KEY_ENCRYPTED"] = encrypted.decode()

# Decrypt when needed
# decrypted = decrypt_data(os.environ["OPENAI_API_KEY_ENCRYPTED"].encode())
"""


# ==================================================================== #
#  90 — AnomalyDetector
# ==================================================================== #

class AnomalyDetector:
    """
    Detect abnormal agent behavior using statistical methods.
    """

    def __init__(self):
        self.actions_history = []
        self.errors_history = []
        self.resource_history = []

    def detect_code(self) -> str:
        return """
import numpy as np
from collections import Counter
import json, time, psutil

class AnomalyDetector:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.actions = []
        self.errors = []
        self.resources = []

    def record_action(self, action_type, error=False):
        self.actions.append({"type": action_type, "error": error, "time": time.time()})
        if error:
            self.errors.append(action_type)
        # Trim
        if len(self.actions) > self.window_size:
            self.actions = self.actions[-self.window_size:]

    def record_resource(self, ram_gb=None, vram_gb=None):
        self.resources.append({"ram": ram_gb, "vram": vram_gb, "time": time.time()})
        if len(self.resources) > self.window_size:
            self.resources = self.resources[-self.window_size:]

    def check_anomalies(self):
        warnings = []

        # Repeated same error
        if len(self.errors) > 3:
            recent = self.errors[-5:]
            if len(set(recent)) == 1 and len(recent) >= 3:
                warnings.append(f"Same error repeating: {recent[0]} (x{len(recent)})")

        # Repeated same action
        if len(self.actions) > 5:
            recent_actions = [a["type"] for a in self.actions[-10:]]
            counts = Counter(recent_actions)
            if counts.most_common(1)[0][1] > 5:
                warnings.append(f"Action loop: {counts.most_common(1)[0][0]} repeated {counts.most_common(1)[0][1]}x")

        # Resource anomaly
        if len(self.resources) > 10:
            recent_ram = [r["ram"] for r in self.resources[-10:] if r["ram"]]
            if recent_ram:
                mean_ram = np.mean(recent_ram)
                std_ram = np.std(recent_ram)
                last_ram = recent_ram[-1]
                if last_ram > mean_ram + 3 * std_ram:
                    warnings.append(f"RAM spike: {last_ram:.1f} GB (mean={mean_ram:.1f}, std={std_ram:.1f})")

        return warnings

    def should_pause(self):
        warnings = self.check_anomalies()
        for w in warnings:
            print(f"[ANOMALY] {w}")
        return len(warnings) >= 2  # Pause if 2+ anomalies

detector = AnomalyDetector()
"""
