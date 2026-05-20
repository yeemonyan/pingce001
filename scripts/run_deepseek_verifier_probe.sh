#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

TRAIN_INPUT="${TRAIN_INPUT:-outputs/sft_train_verifier.jsonl}"
TRAIN_SUBSET="${TRAIN_SUBSET:-outputs/sft_train_verifier_probe_600q.jsonl}"
VALID_INPUT="${VALID_INPUT:-outputs/sft_valid_verifier.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-configs/train_deepseek_verifier_probe.yaml}"
QUESTION_COUNT="${QUESTION_COUNT:-600}"
SEED="${SEED:-2026}"
RUN_TRAIN="${RUN_TRAIN:-1}"

echo "[1/4] Sample verifier subset"
"$PYTHON_BIN" scripts/sample_verifier_subset.py \
  --input "$TRAIN_INPUT" \
  --output "$TRAIN_SUBSET" \
  --questions "$QUESTION_COUNT" \
  --seed "$SEED"

echo "[2/4] Dry-run training config"
"$PYTHON_BIN" scripts/train_lora.py --config "$CONFIG_PATH" --dry-run

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "RUN_TRAIN=$RUN_TRAIN, stop after dry-run."
  exit 0
fi

echo "[3/4] Train DeepSeek verifier probe"
"$PYTHON_BIN" scripts/train_lora.py --config "$CONFIG_PATH"

ADAPTER_PATH="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import yaml

config = yaml.safe_load(Path("configs/train_deepseek_verifier_probe.yaml").read_text(encoding="utf-8"))
print(config["training"]["output_dir"])
PY
)"

echo "[4/4] Dev verifier inference"
"$PYTHON_BIN" scripts/infer_verifier.py \
  --input outputs/dev_prompts.jsonl \
  --output outputs/dev_deepseek_verifier_probe_predictions.jsonl \
  --option-output outputs/dev_deepseek_verifier_probe_option_predictions.jsonl \
  --report outputs/dev_deepseek_verifier_probe_eval.json \
  --backend transformers \
  --model-path models/DeepSeek-R1-Distill-Qwen-7B \
  --adapter-path "$ADAPTER_PATH" \
  --dtype bfloat16 \
  --device-map auto

echo "DeepSeek verifier probe complete."
