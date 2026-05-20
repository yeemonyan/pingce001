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

INPUT_PATH="${INPUT_PATH:-outputs/dev_prompts.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/dev_qwen_selective_vote_predictions.jsonl}"
REPORT_PATH="${REPORT_PATH:-outputs/dev_qwen_selective_vote_report.json}"
EVAL_PATH="${EVAL_PATH:-outputs/dev_qwen_selective_vote_eval.json}"
PROMPTS_PATH="${PROMPTS_PATH:-configs/system_prompts_v2.yaml}"
VOTE_PROMPTS_PATH="${VOTE_PROMPTS_PATH:-configs/system_prompts_round3_short.yaml}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
BACKEND="${BACKEND:-transformers}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

CMD=(
  "$PYTHON_BIN" scripts/selective_vote_infer.py
  --input "$INPUT_PATH"
  --output "$OUTPUT_PATH"
  --report "$REPORT_PATH"
  --prompts "$PROMPTS_PATH"
  --vote-prompts "$VOTE_PROMPTS_PATH"
  --backend "$BACKEND"
  --model-path "$MODEL_PATH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --dtype "$DTYPE"
  --device-map "$DEVICE_MAP"
  --temperature 0
  --top-p 1
)

if [[ -n "$ADAPTER_PATH" ]]; then
  CMD+=(--adapter-path "$ADAPTER_PATH")
fi

"${CMD[@]}"

"$PYTHON_BIN" scripts/evaluate_score.py \
  --gold "$INPUT_PATH" \
  --pred "$OUTPUT_PATH" \
  --report "$EVAL_PATH" \
  --max-mistakes 100

echo "dev_vote_ready=$OUTPUT_PATH"
