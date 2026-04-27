#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

if "$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except Exception:
    sys.exit(1)

sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "Reusing base PyTorch from $PYTHON_BIN"
else
  # CUDA 12.1 wheels cover common AutoDL RTX 4090 images when PyTorch is absent.
  python -m pip install \
    torch torchvision torchaudio \
    --index-url "$TORCH_INDEX_URL"
fi

python -m pip install -r "$PROJECT_ROOT/requirements.txt"

python - <<'PY'
import platform
import torch

print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
