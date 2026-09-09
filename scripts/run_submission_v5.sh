#!/usr/bin/env bash
# ============================================================
# SCoRE2026 Submission Pipeline v5
# 
# MINIMAL fix: just change "ALL" → "option(s)" in prompt.
# No classifier, old system prompts, old adapter.
# ============================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python3}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="python3"

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026}"
PROMPTS_PATH="${PROMPTS_PATH:-configs/system_prompts_v3_mixed_focus.yaml}"
TEST_INPUT="${TEST_INPUT:-outputs/test_prompts.jsonl}"
PRED_OUTPUT="${PRED_OUTPUT:-outputs/test_lora_v5_predictions.jsonl}"
SUBMISSION_OUTPUT="${SUBMISSION_OUTPUT:-outputs/submissions/v5_minimal_fix_submission.json}"
REPORT_OUTPUT="${REPORT_OUTPUT:-outputs/submissions/v5_minimal_fix_report.json}"

echo "================================================"
echo "SCoRE2026 Submission Pipeline v5 (minimal fix)"
echo "================================================"
echo "Change: 'Choose ALL' → 'Choose the correct option(s)'"
echo "================================================"

[[ -f "$TEST_INPUT" ]] || { echo "Missing $TEST_INPUT"; exit 1; }
[[ -d "$ADAPTER_PATH" ]] || { echo "Missing $ADAPTER_PATH"; exit 1; }

echo "[1/3] Run inference (no classifier, no hints, just fixed prompt)"
"$PYTHON_BIN" scripts/infer_score.py \
  --backend transformers \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --prompts "$PROMPTS_PATH" \
  --input "$TEST_INPUT" \
  --output "$PRED_OUTPUT" \
  --dtype bfloat16 \
  --device-map cuda:0

echo "[2/3] Format submission"
"$PYTHON_BIN" scripts/format_submission.py \
  --input "$PRED_OUTPUT" \
  --output "$SUBMISSION_OUTPUT"

echo "[3/3] Validate"
"$PYTHON_BIN" scripts/validate_submission.py \
  --input "$SUBMISSION_OUTPUT" \
  --report "$REPORT_OUTPUT"

echo "DONE: $SUBMISSION_OUTPUT"
