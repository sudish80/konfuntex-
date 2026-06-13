"""
Application configuration with environment profiles and startup validation.

Profiles (``COLAB_AGENT_ENV``):
  - ``dev`` (default): verbose logging, SQLite, relaxed safety
  - ``staging``: JSON logging, SQLite, standard safety
  - ``prod``: JSON logging, PostgreSQL, strict safety, API key required

All values overridable via ``COLAB_AGENT_*`` env vars or ``.env`` file.
"""

import sys
import logging
from typing import Optional, Literal
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator


logger = logging.getLogger(__name__)

ENV_PROFILES = ("dev", "staging", "prod")


class Settings(BaseSettings):
    # ── Environment profile ────────────────────────────────────────
    env: Literal["dev", "staging", "prod"] = "dev"

    # ── LLM Provider ───────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "gemini", "local"] = "openai"
    llm_model: str = "meta/llama-3.1-70b-instruct" # Upgraded from gpt-4/8b default
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    local_llm_endpoint: Optional[str] = None

    # ── Google / Colab ─────────────────────────────────────────────
    google_credentials_path: Optional[str] = None
    google_drive_folder_id: Optional[str] = None
    colab_runtime_auto_switch: bool = True

    # ── HuggingFace ────────────────────────────────────────────────
    hf_token: Optional[str] = None
    hf_cache_dir: Optional[str] = None

    # ── GitHub ─────────────────────────────────────────────────────
    github_token: Optional[str] = None
    github_repo: Optional[str] = None

    # ── Storage ────────────────────────────────────────────────────
    data_dir: str = str(Path.home() / ".colab-agent")
    db_url: str = "sqlite:///{data_dir}/colab_agent.db"

    # ── Agent ──────────────────────────────────────────────────────
    agent_max_iterations: int = 50
    agent_temperature: float = 0.2
    agent_verbose: bool = True

    # ── API security ───────────────────────────────────────────────
    api_key: Optional[str] = None

    # ── Budget ─────────────────────────────────────────────────────
    budget_max_units: float = 100.0
    budget_warn_threshold: float = 0.8

    # ── Fine-tuning ────────────────────────────────────────────────
    default_finetune_method: Literal["lora", "qlora", "full"] = "qlora"
    default_base_model: str = "microsoft/phi-2"

    # ── Runtime tiers (GPU -> VRAM in GB) ──────────────────────────
    runtime_tiers: dict = {
        "None": 0,
        "T4": 16,
        "V100": 32,
        "A100": 80,
        "A100-80GB": 80,
        "TPU": 128,
    }

    model_config = SettingsConfigDict(
        env_prefix="COLAB_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ── Validators ─────────────────────────────────────────────────

    @field_validator("env")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        if v not in ENV_PROFILES:
            raise ValueError(f"COLAB_AGENT_ENV must be one of {ENV_PROFILES}, got {v!r}")
        return v

    @field_validator("agent_temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if not 0 <= v <= 2:
            raise ValueError(f"agent_temperature must be 0-2, got {v}")
        return v

    @field_validator("budget_max_units")
    @classmethod
    def _validate_budget(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"budget_max_units must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_production_requirements(self) -> "Settings":
        if self.env == "prod":
            if not self.api_key:
                raise ValueError(
                    "COLAB_AGENT_API_KEY is required in prod environment"
                )
            if not self.db_url.startswith("postgresql"):
                raise ValueError(
                    "COLAB_AGENT_DB_URL must point to PostgreSQL in prod environment"
                )
        return self

    # ── Derived helpers ────────────────────────────────────────────

    def get_db_url(self) -> str:
        return self.db_url.format(data_dir=self.data_dir)

    def profile_config(self) -> dict:
        """Return profile-dependent configuration overrides."""
        return {
            "dev": {"agent_verbose": True},
            "staging": {"agent_verbose": False},
            "prod": {"agent_verbose": False},
        }.get(self.env, {})


# ── Singleton ─────────────────────────────────────────────────────

_settings: Optional["Settings"] = None


def load_settings() -> "Settings":
    """Load and validate settings (cached singleton).

    Prints validation errors to stderr and exits if invalid.
    """
    global _settings
    if _settings is not None:
        return _settings
    try:
        _settings = Settings()
        return _settings
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)


settings = load_settings()
