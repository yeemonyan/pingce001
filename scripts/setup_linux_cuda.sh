#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

# CUDA 12.1 wheels run well on common RTX 4090/A800 images. If your image
# pins another CUDA runtime, change the index URL according to PyTorch docs.
python -m pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements.txt

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
