#!/bin/bash
# ======================================================================
# Phase 0 — setup_colab.sh
# One-shot dependency installer for Colab Agent
# Usage: bash setup_colab.sh [--dev] [--auth]
# ======================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err() { echo -e "${RED}[x]${NC} $1"; }

DEV_MODE=false
AUTH_MODE=false
for arg in "$@"; do
  case $arg in
    --dev) DEV_MODE=true ;;
    --auth) AUTH_MODE=true ;;
  esac
done

log "Colab Agent — Dependency Installer"

# ── System ────────────────────────────────────────────────────────────
log "Updating package lists..."
apt-get update -qq

log "Installing system dependencies..."
apt-get install -y -qq \
  python3-pip python3-dev \
  git curl wget \
  sqlite3 \
  build-essential \
  ninja-build \
  > /dev/null 2>&1

# ── Core ML ───────────────────────────────────────────────────────────
log "Installing ML libraries..."
pip install -q --upgrade \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118 \
  2>/dev/null || pip install -q torch torchvision torchaudio

pip install -q \
  transformers \
  datasets \
  accelerate \
  peft \
  trl \
  bitsandbytes \
  safetensors \
  scikit-learn \
  wandb \
  tensorboard

# ── HuggingFace ───────────────────────────────────────────────────────
log "Installing HuggingFace Hub..."
pip install -q huggingface_hub sentencepiece

# ── Colab / Google ────────────────────────────────────────────────────
log "Installing Google integrations..."
pip install -q \
  google-api-python-client \
  google-auth-httplib2 \
  google-auth-oauthlib \
  PyDrive \
  nbformat \
  requests

# ── LLM APIs ──────────────────────────────────────────────────────────
log "Installing LLM API clients..."
pip install -q openai anthropic google-generativeai groq

# ── UI ─────────────────────────────────────────────────────────────────
log "Installing UI frameworks..."
pip install -q gradio streamlit ipywidgets rich tabulate

# ── Storage / Utils ───────────────────────────────────────────────────
log "Installing utility packages..."
pip install -q \
  sqlalchemy \
  pydantic pydantic-settings \
  python-dotenv pyyaml \
  httpx \
  tqdm \
  psutil \
  fire

# ── Dev extras ─────────────────────────────────────────────────────────
if [ "$DEV_MODE" = true ]; then
  log "Installing dev extras..."
  pip install -q \
    pytest pytest-cov \
    black isort flake8 \
    mypy \
    pre-commit
fi

# ── Auth setup ────────────────────────────────────────────────────────
if [ "$AUTH_MODE" = true ]; then
  log "Setting up Google auth..."
  pip install -q --upgrade google-colab
  cat << 'AOF' > /content/auth_setup.py
from google.colab import auth
auth.authenticate_user()
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
print("Google Drive authenticated")
AOF
  python /content/auth_setup.py
fi

# ── Verify ─────────────────────────────────────────────────────────────
log "Verifying installations..."
python3 -c "
import torch; print(f'  torch {torch.__version__}')
import transformers; print(f'  transformers {transformers.__version__}')
import peft; print(f'  peft {peft.__version__}')
import datasets; print(f'  datasets {datasets.__version__}')
import accelerate; print(f'  accelerate {accelerate.__version__}')
import gradio; print(f'  gradio {gradio.__version__}')
import sqlalchemy; print(f'  sqlalchemy {sqlalchemy.__version__}')
print('All dependencies verified.')
"

log "Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Set your API keys in .env or Colab secrets"
echo "  2. Run: python cli.py interactive"
echo "  3. Or:   streamlit run ui/app.py"
echo "  4. Or:   python ui/gradio_app.py"
