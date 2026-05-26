#!/usr/bin/env bash
# ============================================================
# SCoRE2026 Submission Pipeline v6
#
# Strategy:
# 1. Reuse V5 inference settings (same adapter, same prompts, no classifier)
# 2. Post-process only the over-selected predictions
# 3. Collapse to single-answer only when the question has no strong multi cue
# ============================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python3}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="python3"

MODEL_PATH="${MODEL_PATH:-/home/pj/xukangzhe/Qwen2.5-7B-Instruct/qwen/Qwen2___5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-/home/pj/pingce001-moe/checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026}"
PROMPTS_PATH="${PROMPTS_PATH:-configs/system_prompts_v3_mixed_focus.yaml}"
TEST_INPUT="${TEST_INPUT:-outputs/test_prompts.jsonl}"

RAW_PRED_OUTPUT="${RAW_PRED_OUTPUT:-outputs/test_lora_v6_raw_predictions.jsonl}"
POST_PRED_OUTPUT="${POST_PRED_OUTPUT:-outputs/test_lora_v6_postprocessed.jsonl}"
SUBMISSION_OUTPUT="${SUBMISSION_OUTPUT:-outputs/submissions/v6_single_first_submission.json}"
REPORT_OUTPUT="${REPORT_OUTPUT:-outputs/submissions/v6_single_first_report.json}"

echo "================================================"
echo "SCoRE2026 Submission Pipeline v6"
echo "================================================"
echo "Base: V5 inference"
echo "Fix : single-first postprocess for over-selection"
echo "================================================"

[[ -f "$TEST_INPUT" ]] || { echo "Missing $TEST_INPUT"; exit 1; }
[[ -d "$ADAPTER_PATH" ]] || { echo "Missing $ADAPTER_PATH"; exit 1; }

echo "[1/4] Run V5-style inference"
"$PYTHON_BIN" scripts/infer_score.py \
  --backend transformers \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --prompts "$PROMPTS_PATH" \
  --input "$TEST_INPUT" \
  --output "$RAW_PRED_OUTPUT" \
  --dtype bfloat16 \
  --device-map cuda:0

echo "[2/4] Apply V6 single-first postprocess"
"$PYTHON_BIN" scripts/postprocess_v6_single_first.py \
  --predictions "$RAW_PRED_OUTPUT" \
  --source "$TEST_INPUT" \
  --output "$POST_PRED_OUTPUT"

echo "[3/4] Format submission"
"$PYTHON_BIN" scripts/format_submission.py \
  --input "$POST_PRED_OUTPUT" \
  --output "$SUBMISSION_OUTPUT"

echo "[4/4] Validate"
"$PYTHON_BIN" scripts/validate_submission.py \
  --input "$SUBMISSION_OUTPUT" \
  --report "$REPORT_OUTPUT"

echo "DONE: $SUBMISSION_OUTPUT"
