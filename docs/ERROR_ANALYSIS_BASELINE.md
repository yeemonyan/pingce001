# Baseline Dev Error Analysis

This document records the baseline/dev analysis workflow. The gold dev file is available at `outputs/dev_prompts.jsonl`.

## Current Status

- Dev gold split has been generated from official training data only.
- Test data is not used for filtering, baseline evaluation, examples, or pseudo labels.
- Baseline prediction and error reports should be generated after running Qwen2.5-7B-Instruct on a GPU server.

## Generate Dev Gold

```bash
python scripts/filter_split_records.py \
  --input data/raw/train.json \
  --ids-file data/splits/dev_ids.json \
  --output outputs/dev_prompts.jsonl
```

## Run Baseline On Dev

```bash
python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --input outputs/dev_prompts.jsonl \
  --output outputs/dev_baseline_predictions.jsonl
```

## Evaluate Baseline

```bash
python scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_baseline_predictions.jsonl \
  --report outputs/dev_baseline_eval.json \
  --max-mistakes 100
```

## Analyze Errors

```bash
python scripts/analyze_errors.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_baseline_predictions.jsonl \
  --report outputs/error_report_baseline_dev.json \
  --cases outputs/error_cases_baseline_dev.jsonl \
  --doc docs/ERROR_ANALYSIS_BASELINE.md
```

## Failure Types

The analysis script groups errors into these actionable buckets:

- `single_to_multi`
- `multi_missing`
- `spatial_reference_error`
- `temporal_calculation_error`
- `social_relation_error`
- `natural_property_error`
- `multi_constraint_failure`
- `output_format_error`
- `wrong_single_choice`

## How To Use The Report

- If temporal accuracy is lowest, prioritize temporal prompt and timeline formatting.
- If hybrid errors dominate, force separate domain notes before merging constraints.
- If `multi_missing` is high, strengthen multi-answer SFT examples and decoding checks.
- If `single_to_multi` is high, discourage over-selection on single-answer questions.
- If output format errors appear, tighten JSON extraction and fallback handling before another online submission.
