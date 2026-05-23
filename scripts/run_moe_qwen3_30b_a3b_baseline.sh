#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_CACHE="${MODEL_CACHE:-${ROOT_DIR}/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_CACHE}/Qwen/Qwen3-30B-A3B-Instruct-2507}"
PROMPTS_PATH="${PROMPTS_PATH:-configs/system_prompts_v3_mixed_focus.yaml}"
PRED_PATH="${PRED_PATH:-outputs/dev_qwen3_30b_a3b_baseline_predictions.jsonl}"
EVAL_PATH="${EVAL_PATH:-outputs/dev_qwen3_30b_a3b_baseline_eval.json}"
MODEL_DOWNLOADER="${MODEL_DOWNLOADER:-auto}"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MODEL_ID MODEL_CACHE MODEL_PATH MODEL_DOWNLOADER

echo "started_at=$(date -Is)"
echo "root=$ROOT_DIR"
echo "model_id=$MODEL_ID"
echo "model_path=$MODEL_PATH"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

mkdir -p "$MODEL_CACHE"

if [[ ! -d "$MODEL_PATH" ]]; then
  "$PYTHON_BIN" - <<PY
import importlib
import os
import sys

model_id = os.environ["MODEL_ID"]
cache_dir = os.environ["MODEL_CACHE"]
mode = os.environ.get("MODEL_DOWNLOADER", "auto")

if mode in {"auto", "modelscope"}:
    try:
        snapshot_download = importlib.import_module("modelscope").snapshot_download
        path = snapshot_download(model_id, cache_dir=cache_dir)
        print("downloaded_model_path=" + path)
        sys.exit(0)
    except Exception as exc:
        if mode == "modelscope":
            raise
        print(f"modelscope download skipped: {exc}", file=sys.stderr)

if mode in {"auto", "huggingface"}:
    try:
        snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
        path = snapshot_download(repo_id=model_id, local_dir=os.path.join(cache_dir, model_id))
        print("downloaded_model_path=" + path)
        sys.exit(0)
    except Exception as exc:
        if mode == "huggingface":
            raise
        print(f"huggingface download skipped: {exc}", file=sys.stderr)

raise SystemExit(
    f"model path not found and downloader failed: {model_id} -> {cache_dir}. "
    "Set MODEL_PATH to an existing directory or install/use modelscope/huggingface_hub."
)
PY
fi

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
