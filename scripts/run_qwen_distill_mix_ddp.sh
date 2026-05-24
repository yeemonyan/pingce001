#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

CONFIG_PATH="${CONFIG_PATH:-configs/train_qwen_distill_mix.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29626}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"

echo "started_at=$(date -Is)"
echo "root=$ROOT_DIR"
echo "config=$CONFIG_PATH"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  scripts/train_lora.py \
  --config "$CONFIG_PATH"

echo "finished_at=$(date -Is)"
