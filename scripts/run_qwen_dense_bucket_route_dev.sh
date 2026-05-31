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
DEV_INPUT="${DEV_INPUT:-outputs/dev_prompts.jsonl}"

ANSWER_ONLY_ADAPTER="${ANSWER_ONLY_ADAPTER:-checkpoints/qwen2p5_7b_lora_answer_only_seed2026}"
ANSWER_ONLY_PROMPTS="${ANSWER_ONLY_PROMPTS:-configs/system_prompts_v2.yaml}"
ANSWER_ONLY_PRED="${ANSWER_ONLY_PRED:-outputs/dev_lora_answer_only_predictions.jsonl}"

MIXED_ADAPTER="${MIXED_ADAPTER:-checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026}"
MIXED_PROMPTS="${MIXED_PROMPTS:-configs/system_prompts_v3_mixed_focus.yaml}"
MIXED_PRED="${MIXED_PRED:-outputs/dev_lora_mixed_reasoning_predictions.jsonl}"

ROUTED_PRED="${ROUTED_PRED:-outputs/dev_lora_dense_bucket_route_predictions.jsonl}"
ROUTED_REPORT="${ROUTED_REPORT:-outputs/dev_lora_dense_bucket_route_report.json}"
ROUTED_EVAL="${ROUTED_EVAL:-outputs/dev_lora_dense_bucket_route_eval.json}"

echo "[1/5] Ensure answer_only dev predictions exist"
if [[ ! -f "$ANSWER_ONLY_PRED" ]]; then
  "$PYTHON_BIN" scripts/infer_score.py \
    --backend transformers \
    --model-path "$MODEL_PATH" \
    --adapter-path "$ANSWER_ONLY_ADAPTER" \
    --prompts "$ANSWER_ONLY_PROMPTS" \
    --input "$DEV_INPUT" \
    --output "$ANSWER_ONLY_PRED" \
    --dtype bfloat16 \
    --device-map auto \
    --answer-count-hint
fi

echo "[2/5] Ensure mixed_reasoning dev predictions exist"
if [[ ! -f "$MIXED_PRED" ]]; then
  "$PYTHON_BIN" scripts/infer_score.py \
    --backend transformers \
    --model-path "$MODEL_PATH" \
    --adapter-path "$MIXED_ADAPTER" \
    --prompts "$MIXED_PROMPTS" \
    --input "$DEV_INPUT" \
    --output "$MIXED_PRED" \
    --dtype bfloat16 \
    --device-map auto \
    --answer-count-hint
fi

echo "[3/5] Route the stronger adapter by bucket"
"$PYTHON_BIN" scripts/route_predictions.py \
  --input "$DEV_INPUT" \
  --base "$ANSWER_ONLY_PRED" \
  --candidate "$MIXED_PRED" \
  --output "$ROUTED_PRED" \
  --kind-source candidate \
  --use-candidate-bucket temporal:* \
  --use-candidate-bucket social:* \
  --use-candidate-bucket hybrid:* \
  --use-candidate-bucket spatial:multi \
  --use-candidate-bucket natural:single \
  --report "$ROUTED_REPORT" \
  --max-mistakes 100

echo "[4/5] Evaluate routed dev predictions"
"$PYTHON_BIN" scripts/evaluate_score.py \
  --gold "$DEV_INPUT" \
  --pred "$ROUTED_PRED" \
  --report "$ROUTED_EVAL" \
  --max-mistakes 100

echo "[5/5] Compare routed predictions against mixed baseline"
"$PYTHON_BIN" scripts/compare_predictions.py \
  --gold "$DEV_INPUT" \
  --base "$MIXED_PRED" \
  --candidate "$ROUTED_PRED" \
  --base-name mixed \
  --candidate-name dense_bucket_route \
  --report outputs/dev_lora_dense_bucket_route_vs_mixed.json \
  --max-flips 100

echo "dense_bucket_route_ready=$ROUTED_EVAL"
