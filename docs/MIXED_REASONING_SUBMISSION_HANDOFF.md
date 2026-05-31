# Mixed Reasoning Submission Handoff

## What It Is

The current best dense dev run is `mixed_reasoning`:

- dev accuracy: `277/720 = 0.3847222222`
- base model: `Qwen2.5-7B-Instruct`
- adapter/checkpoint name: `qwen2p5_7b_lora_mixed_reasoning_seed2026`
- expected adapter path on a run machine:
  - `checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026`

## Git-Tracked Files

These files are in the git repo and define the run:

- `configs/train_qwen_mixed_reasoning.yaml`
- `configs/system_prompts_v3_mixed_focus.yaml`
- `scripts/run_qwen_mixed_reasoning.sh`
- `scripts/run_mixed_reasoning_submission.sh`
- `scripts/format_submission.py`
- `scripts/validate_submission.py`

The dev eval/prediction files may exist locally but are not all git-tracked:

- `outputs/dev_lora_mixed_reasoning_eval.json`
- `outputs/dev_lora_mixed_reasoning_predictions.jsonl`

The LoRA adapter weights are not stored in git. The teammate needs the checkpoint directory or must rerun training.

## Important Compliance Note

The recorded `0.3847` dev run used `--answer-count-hint` during dev inference. On dev, this hint is derived from the gold answer length, so it is a diagnostic convenience rather than a test-time signal.

For official test submission, do not enable answer-count hints unless the test input already provides a legitimate non-answer metadata field for question cardinality. The submission script defaults to no answer-count hint.

## Generate Official Submission

Set paths for the machine that has the base model, adapter, and normalized official test JSONL:

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
ADAPTER_PATH=/path/to/qwen2p5_7b_lora_mixed_reasoning_seed2026 \
TEST_INPUT=/path/to/test_prompts.jsonl \
bash scripts/run_mixed_reasoning_submission.sh
```

Default outputs:

- predictions: `outputs/test_lora_mixed_reasoning_predictions.jsonl`
- upload JSON: `outputs/submissions/mixed_reasoning_qwen2p5_7b_submission.json`
- validation report: `outputs/submissions/mixed_reasoning_qwen2p5_7b_submission_report.json`

Only upload the JSON file under `outputs/submissions/`.

## Reproduce Dev Eval

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
ADAPTER_PATH=/path/to/qwen2p5_7b_lora_mixed_reasoning_seed2026 \
python scripts/infer_score.py \
  --backend transformers \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --prompts configs/system_prompts_v3_mixed_focus.yaml \
  --input outputs/dev_prompts.jsonl \
  --output outputs/dev_lora_mixed_reasoning_predictions.jsonl \
  --dtype bfloat16 \
  --device-map auto \
  --answer-count-hint

python scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_lora_mixed_reasoning_predictions.jsonl \
  --report outputs/dev_lora_mixed_reasoning_eval.json
```

Expected dev result:

- overall: `0.3847222222`
- natural: `0.455`
- spatial: `0.33`
- temporal: `0.27`
- social: `0.59`
- hybrid: `0.35`
