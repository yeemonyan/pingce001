#!/usr/bin/env bash
# Generate a test-safe mixed-reasoning submission without gold answer hints.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026}"
PROMPTS_PATH="${PROMPTS_PATH:-configs/system_prompts_v3_mixed_focus.yaml}"
TEST_INPUT="${TEST_INPUT:-outputs/test_prompts.jsonl}"
TEST_WITH_CARDINALITY="${TEST_WITH_CARDINALITY:-outputs/test_prompts_with_cardinality.jsonl}"
PRED_PATH="${PRED_PATH:-outputs/test_lora_mixed_reasoning_predictions.jsonl}"
SUBMISSION_PATH="${SUBMISSION_PATH:-outputs/submissions/mixed_reasoning_qwen2p5_7b_submission.json}"
VALIDATION_REPORT="${VALIDATION_REPORT:-outputs/submissions/mixed_reasoning_qwen2p5_7b_submission_report.json}"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$PYTHON_BIN" scripts/predict_cardinality.py \
  --input "$TEST_INPUT" \
  --output "$TEST_WITH_CARDINALITY" \
  --stats

"$PYTHON_BIN" scripts/infer_score.py \
  --backend transformers \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --prompts "$PROMPTS_PATH" \
  --input "$TEST_WITH_CARDINALITY" \
  --output "$PRED_PATH" \
  --dtype bfloat16 \
  --device-map auto \
  --auto-count-hint

"$PYTHON_BIN" scripts/format_submission.py \
  --input "$PRED_PATH" \
  --output "$SUBMISSION_PATH"

"$PYTHON_BIN" scripts/validate_submission.py \
  --input "$SUBMISSION_PATH" \
  --report "$VALIDATION_REPORT"

echo "submission_ready=$SUBMISSION_PATH"
