#!/bin/bash
cd /home/pj/pingce001-moe/imports/old_50890/pingce001
export PYTHONPATH=/home/pj/pingce001-moe/imports/old_50890/pingce001
python3 -u scripts/infer_score.py \
  --backend transformers \
  --model-path /home/pj/xukangzhe/Qwen2.5-7B-Instruct/qwen/Qwen2___5-7B-Instruct \
  --adapter-path /home/pj/pingce001-moe/checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026 \
  --prompts configs/system_prompts_v4_self_determine.yaml \
  --input outputs/test_prompts_with_cardinality.jsonl \
  --output outputs/test_lora_v4_predictions.jsonl \
  --dtype bfloat16 \
  --device-map cuda:0 \
  --auto-count-hint
