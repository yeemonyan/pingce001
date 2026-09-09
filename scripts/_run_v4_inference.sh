#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:-checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026}"
python3 -u scripts/infer_score.py \
  --backend transformers \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --prompts configs/system_prompts_v4_self_determine.yaml \
  --input outputs/test_prompts_with_cardinality.jsonl \
  --output outputs/test_lora_v4_predictions.jsonl \
  --dtype bfloat16 \
  --device-map cuda:0 \
  --auto-count-hint
