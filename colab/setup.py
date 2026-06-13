"""
Phase 0 — Setup & Configuration.

Provides:
  - setup_colab_environment()   one-shot Colab env preparation
  - SecretsManager              reads from Colab userdata / .env / env vars
  - DriveAuth                   Google Drive mount + PyDrive auth
  - LogManager                  timestamped logging with rotation
"""
import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime
from config.settings import settings


# ==================================================================== #
#  SecretsManager (item 4)
# ==================================================================== #

class SecretsManager:
    """
    Reads secrets from (priority order):
      1. Colab userdata secrets  (google.colab.userdata)
      2. .env file               (python-dotenv)
      3. OS environment variables
      4. config.yaml
    """

    REQUIRED = {
        "OPENAI_API_KEY": "",
        "HF_TOKEN": "",
        "GITHUB_TOKEN": "",
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
    }

    def __init__(self, env_path: Optional[str] = None):
        self.env_path = env_path or os.path.join(settings.data_dir, ".env")
        self._secrets = {}
        self._load()

    def _load(self):
        # 1. Try Colab userdata
        try:
            from google.colab import userdata
            for key in self.REQUIRED:
                try:
                    self._secrets[key] = userdata.get(key)
                except Exception:
                    pass
            if self._secrets:
                return
        except ImportError:
            pass

        # 2. Try .env file
        dotenv_path = Path(self.env_path)
        if dotenv_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path)
            except ImportError:
                pass

        # 3. OS env vars
        for key in self.REQUIRED:
            val = os.environ.get(key) or os.environ.get(f"COLAB_AGENT_{key}")
            if val:
                self._secrets[key] = val

    def get(self, key: str, default: str = "") -> str:
        return self._secrets.get(key, default)

    def set(self, key: str, value: str):
        self._secrets[key] = value

    def is_configured(self, key: str) -> bool:
        return bool(self._secrets.get(key))

    def summary(self) -> dict:
        return {k: (v[:8] + "..." if v and len(v) > 8 else bool(v))
                for k, v in self._secrets.items()}

    def generate_colab_code(self) -> str:
        """Generate Colab code to set secrets from userdata."""
        lines = ["# === Colab Secrets Setup ==="]
        for key in self.REQUIRED:
            lines.append(
                f'os.environ["{key}"] = "{self.get(key)}"'
                if self.get(key) else
                f'# {key} not set — add to Colab secrets or .env'
            )
        return "\n".join(lines)


# ==================================================================== #
#  DriveAuth (item 3)
# ==================================================================== #

class DriveAuth:
    """Google Drive mount + PyDrive authentication."""

    @staticmethod
    def mount_code() -> str:
        return """
from google.colab import drive
drive.mount('/content/drive')
print("Drive mounted at /content/drive")
"""

    @staticmethod
    def pydrive_auth_code() -> str:
        return """
from google.colab import auth
auth.authenticate_user()
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
print("PyDrive authenticated")
"""

    @staticmethod
    def check_mounted(path: str = "/content/drive") -> bool:
        return os.path.isdir(path)

    @staticmethod
    def ensure_drive_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path


# ==================================================================== #
#  LogManager (item 5)
# ==================================================================== #

class LogManager:
    """
    Rotating file + console logger with timestamps.
    Rotates every 10 MB, keeps 5 backups.
    """

    def __init__(self, name: str = "colab-agent",
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO):
        self.name = name
        self.log_dir = log_dir or os.path.join(settings.data_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler with rotation
        log_path = os.path.join(self.log_dir, f"{name}.log")
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8",
        )
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def debug(self, msg): self.logger.debug(msg)
    def info(self, msg): self.logger.info(msg)
    def warn(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)

    def get_log_path(self) -> str:
        return os.path.join(self.log_dir, f"{self.name}.log")

    def read_recent(self, n: int = 50) -> str:
        path = self.get_log_path()
        if not os.path.exists(path):
            return ""
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])


# ==================================================================== #
#  setup_colab_environment (items 1-2)
# ==================================================================== #

def setup_colab_environment(secrets: Optional[SecretsManager] = None,
                            log: Optional[LogManager] = None) -> str:
    """One-shot Colab environment setup. Returns setup code."""
    if log is None:
        log = LogManager("setup")
    if secrets is None:
        secrets = SecretsManager()

    log.info("Setting up Colab environment")
    log.info(f"Secrets configured: {secrets.summary()}")

    setup_code = f"""
# ===== Colab Agent — Automated Setup =====
import os, sys, subprocess, json
from datetime import datetime

print(f"Setup started: {{datetime.now().isoformat()}}")

# Install
!pip install -q transformers datasets accelerate peft trl bitsandbytes \
  huggingface_hub gradio nbformat requests pydrive sqlalchemy \
  openai anthropic python-dotenv pyyaml > /dev/null 2>&1

# Env vars
{secrets.generate_colab_code()}

# Data dir
DATA_DIR = "{settings.data_dir}"
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Data dir: {{DATA_DIR}}")
print("Setup complete")
"""
    return setup_code
