#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python3"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_CACHE="${MODEL_CACHE:-models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_CACHE}/Qwen/Qwen3-30B-A3B-Instruct-2507}"
PROMPTS_PATH="${PROMPTS_PATH:-configs/system_prompts_v3_mixed_focus.yaml}"
PRED_PATH="${PRED_PATH:-outputs/dev_qwen3_30b_a3b_baseline_predictions.jsonl}"
EVAL_PATH="${EVAL_PATH:-outputs/dev_qwen3_30b_a3b_baseline_eval.json}"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "started_at=$(date -Is)"
echo "root=$ROOT_DIR"
echo "model_id=$MODEL_ID"
echo "model_path=$MODEL_PATH"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

"$PYTHON_BIN" - <<PY
from modelscope import snapshot_download
path = snapshot_download("$MODEL_ID", cache_dir="$MODEL_CACHE")
print("downloaded_model_path=" + path)
PY

"$PYTHON_BIN" scripts/infer_score.py \
  --backend transformers \
  --model-path "$MODEL_PATH" \
  --prompts "$PROMPTS_PATH" \
  --input outputs/dev_prompts.jsonl \
  --output "$PRED_PATH" \
  --dtype bfloat16 \
  --device-map auto \
  --max-new-tokens 512 \
  --temperature 0 \
  --top-p 1 \
  --answer-count-hint

"$PYTHON_BIN" scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred "$PRED_PATH" \
  --report "$EVAL_PATH" \
  --max-mistakes 100

echo "finished_at=$(date -Is)"
