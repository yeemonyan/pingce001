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

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026}"
PROMPTS_PATH="${PROMPTS_PATH:-configs/system_prompts_v3_mixed_focus.yaml}"
TEST_INPUT="${TEST_INPUT:-outputs/test_prompts.jsonl}"
PRED_OUTPUT="${PRED_OUTPUT:-outputs/test_lora_mixed_reasoning_predictions.jsonl}"
SUBMISSION_OUTPUT="${SUBMISSION_OUTPUT:-outputs/submissions/mixed_reasoning_qwen2p5_7b_submission.json}"
REPORT_OUTPUT="${REPORT_OUTPUT:-outputs/submissions/mixed_reasoning_qwen2p5_7b_submission_report.json}"
USE_ANSWER_COUNT_HINT="${USE_ANSWER_COUNT_HINT:-0}"

if [[ ! -f "$TEST_INPUT" ]]; then
  echo "Missing TEST_INPUT=$TEST_INPUT" >&2
  echo "Set TEST_INPUT to the normalized official test JSONL path." >&2
  exit 1
fi

if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "Missing ADAPTER_PATH=$ADAPTER_PATH" >&2
  echo "Set ADAPTER_PATH to qwen2p5_7b_lora_mixed_reasoning_seed2026." >&2
  exit 1
fi

hint_args=()
if [[ "$USE_ANSWER_COUNT_HINT" == "1" ]]; then
  hint_args+=(--answer-count-hint)
fi

echo "[1/3] Run mixed_reasoning test inference"
"$PYTHON_BIN" scripts/infer_score.py \
  --backend transformers \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --prompts "$PROMPTS_PATH" \
  --input "$TEST_INPUT" \
  --output "$PRED_OUTPUT" \
  --dtype bfloat16 \
  --device-map auto \
  "${hint_args[@]}"

echo "[2/3] Format official submission"
"$PYTHON_BIN" scripts/format_submission.py \
  --input "$PRED_OUTPUT" \
  --output "$SUBMISSION_OUTPUT"

echo "[3/3] Validate official submission"
"$PYTHON_BIN" scripts/validate_submission.py \
  --input "$SUBMISSION_OUTPUT" \
  --report "$REPORT_OUTPUT"

echo "submission_ready=$SUBMISSION_OUTPUT"
