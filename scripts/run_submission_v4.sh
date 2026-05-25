#!/usr/bin/env bash
# ============================================================
# SCoRE2026 Submission Pipeline v4
# 
# Key improvements over v3 (mixed_reasoning):
# 1. Neutral prompt: "Select every correct option" (not "Choose ALL")
# 2. Auto cardinality prediction: classifier predicts single/multi
# 3. Self-determination system prompts (v4)
# 4. Proper single/multi hint injection
# ============================================================

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

# ---- Configuration ----
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026}"
# Try multiple adapter options
: ${ADAPTER_PATH:=checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026}
: ${ADAPTER_PATH:=checkpoints/qwen2p5_7b_lora_answer_only_seed2026}

# Prompt config: v4 has self-determination instructions
PROMPTS_PATH="${PROMPTS_PATH:-configs/system_prompts_v4_self_determine.yaml}"

# Input/output
TEST_INPUT="${TEST_INPUT:-outputs/test_prompts.jsonl}"
TEST_WITH_HINTS="${TEST_WITH_HINTS:-outputs/test_prompts_with_cardinality.jsonl}"
PRED_OUTPUT="${PRED_OUTPUT:-outputs/test_lora_v4_predictions.jsonl}"
SUBMISSION_OUTPUT="${SUBMISSION_OUTPUT:-outputs/submissions/v4_cardinality_aware_submission.json}"
REPORT_OUTPUT="${REPORT_OUTPUT:-outputs/submissions/v4_cardinality_aware_report.json}"

# Strategy
USE_ANSWER_COUNT_HINT="${USE_ANSWER_COUNT_HINT:-0}"  # Never use gold hint for test
USE_AUTO_COUNT_HINT="${USE_AUTO_COUNT_HINT:-1}"       # Use classifier

echo "================================================"
echo "SCoRE2026 Submission Pipeline v4"
echo "================================================"
echo "Model:      $MODEL_PATH"
echo "Adapter:    $ADAPTER_PATH"
echo "Prompts:    $PROMPTS_PATH"
echo "Test input: $TEST_INPUT"
echo "Auto hint:  $USE_AUTO_COUNT_HINT"
echo "================================================"

# Step 0: Validate inputs
if [[ ! -f "$TEST_INPUT" ]]; then
  echo "ERROR: Missing TEST_INPUT=$TEST_INPUT" >&2
  echo "Set TEST_INPUT to the normalized official test JSONL path." >&2
  exit 1
fi

if [[ ! -d "$ADAPTER_PATH" ]]; then
  echo "ERROR: Missing ADAPTER_PATH=$ADAPTER_PATH" >&2
  echo "Available checkpoints:"
  ls -d checkpoints/*/ 2>/dev/null || echo "  (none found)"
  exit 1
fi

# Step 1: Predict cardinality (single/multi) for each test question
echo ""
echo "[1/4] Predict cardinality (single/multi) for test questions"
"$PYTHON_BIN" scripts/predict_cardinality.py \
  --input "$TEST_INPUT" \
  --output "$TEST_WITH_HINTS" \
  --stats

# Step 2: Run inference with auto-count-hint
echo ""
echo "[2/4] Run inference with auto cardinality hints"
hint_args=()
if [[ "$USE_ANSWER_COUNT_HINT" == "1" ]]; then
  hint_args+=(--answer-count-hint)
fi
if [[ "$USE_AUTO_COUNT_HINT" == "1" ]]; then
  hint_args+=(--auto-count-hint)
fi

"$PYTHON_BIN" scripts/infer_score.py \
  --backend transformers \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --prompts "$PROMPTS_PATH" \
  --input "$TEST_WITH_HINTS" \
  --output "$PRED_OUTPUT" \
  --dtype bfloat16 \
  --device-map auto \
  "${hint_args[@]}"

# Step 3: Format official submission
echo ""
echo "[3/4] Format official submission"
"$PYTHON_BIN" scripts/format_submission.py \
  --input "$PRED_OUTPUT" \
  --output "$SUBMISSION_OUTPUT"

# Step 4: Validate
echo ""
echo "[4/4] Validate submission"
"$PYTHON_BIN" scripts/validate_submission.py \
  --input "$SUBMISSION_OUTPUT" \
  --report "$REPORT_OUTPUT"

echo ""
echo "================================================"
echo "DONE: $SUBMISSION_OUTPUT"
echo "================================================"
