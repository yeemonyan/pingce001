#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

CONFIG_PATH="${CONFIG_PATH:-configs/train_qwen_mixed_reasoning.yaml}"
RUN_TRAIN="${RUN_TRAIN:-1}"
PROMPTS_PATH="${PROMPTS_PATH:-configs/system_prompts_v3_mixed_focus.yaml}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026}"
PRED_PATH="${PRED_PATH:-outputs/dev_lora_mixed_reasoning_predictions.jsonl}"
EVAL_PATH="${EVAL_PATH:-outputs/dev_lora_mixed_reasoning_eval.json}"

echo "[1/5] Build SFT data variants"
"$PYTHON_BIN" scripts/build_sft_data.py --prompts "$PROMPTS_PATH" --output-dir outputs --report outputs/sft_data_report.json

echo "[2/5] Dry-run training config"
"$PYTHON_BIN" scripts/train_lora.py --config "$CONFIG_PATH" --dry-run

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "RUN_TRAIN=$RUN_TRAIN, stop after dry-run."
  exit 0
fi

echo "[3/5] Train Qwen mixed-reasoning adapter"
"$PYTHON_BIN" scripts/train_lora.py --config "$CONFIG_PATH"

echo "[4/5] Run dev inference"
"$PYTHON_BIN" scripts/infer_score.py \
  --backend transformers \
  --model-path "$MODEL_PATH" \
  --adapter-path "$TRAIN_OUTPUT_DIR" \
  --prompts "$PROMPTS_PATH" \
  --input outputs/dev_prompts.jsonl \
  --output "$PRED_PATH" \
  --dtype bfloat16 \
  --device-map auto \
  --answer-count-hint

echo "[5/5] Evaluate dev accuracy"
"$PYTHON_BIN" scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred "$PRED_PATH" \
  --report "$EVAL_PATH" \
  --max-mistakes 100

echo "qwen_mixed_reasoning_ready=$EVAL_PATH"
