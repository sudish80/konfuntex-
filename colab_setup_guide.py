"""
Phase 8 — Colab Setup Guide Generator.

Produces a complete Colab notebook setup cell block that users can
paste into a fresh notebook to run the Colab Agent from scratch.
"""


def generate_colab_setup(hf_token: str = "",
                          github_token: str = "",
                          github_repo: str = "",
                          openai_api_key: str = "") -> str:
    """Generate the full Colab setup code block."""
    return f'''
# =====================================================================
#  Colab Agent — Full Setup
#  Paste this into a NEW Colab notebook and run.
# =====================================================================

# ── 1. Clone repo / copy code ────────────────────────────────────────
import os, sys
from google.colab import drive
drive.mount("/content/drive")

# ── 2. Install dependencies ──────────────────────────────────────────
!pip install -q \
    openai anthropic google-generativeai \
    pydantic pydantic-settings \
    sqlalchemy \
    transformers datasets accelerate peft trl bitsandweights \
    huggingface_hub safetensors \
    PyGithub \
    gradio \
    python-dotenv pyyaml \
    google-api-python-client google-auth-httplib2 google-auth-oauthlib \
    nbformat requests httpx \
    rich tabulate streamlit

# ── 3. Environment variables ─────────────────────────────────────────
os.environ["COLAB_AGENT_LLM_PROVIDER"] = "openai"
os.environ["COLAB_AGENT_LLM_MODEL"] = "gpt-4"
os.environ["COLAB_AGENT_OPENAI_API_KEY"] = "{openai_api_key or 'YOUR_KEY'}"
os.environ["COLAB_AGENT_HF_TOKEN"] = "{hf_token or ''}"
os.environ["COLAB_AGENT_GITHUB_TOKEN"] = "{github_token or ''}"
os.environ["COLAB_AGENT_GITHUB_REPO"] = "{github_repo or ''}"
os.environ["COLAB_AGENT_RUNTIME_AUTO_SWITCH"] = "true"
os.environ["COLAB_AGENT_DEFAULT_FINETUNE_METHOD"] = "qlora"
os.environ["COLAB_AGENT_DEFAULT_BASE_MODEL"] = "microsoft/phi-2"
os.environ["COLAB_AGENT_DATA_DIR"] = "/content/drive/MyDrive/colab-agent-data"

# ── 4. Set up project ────────────────────────────────────────────────
import urllib.request, zipfile, io

# Download the agent code
AGENT_URL = "https://github.com/YOUR_USER/colab-agent/archive/main.zip"
# TODO: Replace with actual URL or upload colab-agent/ to Drive

# For now, create stub directories
CODE_DIR = "/content/colab-agent"
os.makedirs(CODE_DIR, exist_ok=True)

# ── 5. Init database ─────────────────────────────────────────────────
sys.path.insert(0, CODE_DIR)
from config.settings import settings
from storage.database import init_db
os.makedirs(settings.data_dir, exist_ok=True)
init_db()
print("Database initialized")

# ── 6. Verify ────────────────────────────────────────────────────────
print(f"Data directory: {{settings.data_dir}}")
print(f"LLM provider: {{settings.llm_provider}}")
print(f"HF token set: {{bool(settings.hf_token)}}")
print(f"GitHub set: {{bool(settings.github_token) and bool(settings.github_repo)}}")
print("=== Setup complete ===")

# ── 7. Launch options ────────────────────────────────────────────────
# Option A: CLI
# !python /content/colab-agent/cli.py interactive

# Option B: Gradio UI (run in a separate cell)
# import gradio as gr
# from ui.gradio_app import build_ui
# demo = build_ui()
# demo.launch(share=True, debug=True)

# Option C: Streamlit UI (run in terminal)
# !streamlit run /content/colab-agent/ui/app.py

# Option D: Headless agent
# from agent.core import run_agent
# result = run_agent("Fine-tune Phi-2 on code generation")
# print(result["summary"])
'''


def print_setup():
    print(generate_colab_setup())


if __name__ == "__main__":
    print_setup()
