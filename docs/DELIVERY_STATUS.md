# Delivery Status

## Completed

1. Stable dev split
   - `data/splits/train_ids.json`
   - `data/splits/dev_ids.json`
   - Seed: `2026`
   - Split: `2880 train / 720 dev`

2. SFT train/valid JSONL datasets
   - `outputs/sft_train_answer_only.jsonl`
   - `outputs/sft_valid_answer_only.jsonl`
   - `outputs/sft_train_rationale_json.jsonl`
   - `outputs/sft_valid_rationale_json.jsonl`

3. Data distribution reports
   - `outputs/dev_split_report.json`
   - `outputs/sft_data_report.json`

4. Dev gold file for baseline and LoRA comparison
   - `outputs/dev_prompts.jsonl`

5. Error-analysis tooling
   - `scripts/analyze_errors.py`
   - `docs/ERROR_ANALYSIS_BASELINE.md`

6. Baseline dev predictions and evaluation
   - `outputs/dev_baseline_predictions.jsonl`
   - `outputs/dev_baseline_eval.json`
   - Accuracy: `0.18194444444444444`
   - Correct: `131 / 720`

7. Baseline dev error report
   - `outputs/error_report_baseline_dev.json`
   - `outputs/error_cases_baseline_dev.jsonl`
   - refreshed `docs/ERROR_ANALYSIS_BASELINE.md`
   - Error cases retained: `48`

## Baseline Dev Result

| Domain | Total | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| natural | 200 | 50 | 0.25 |
| spatial | 200 | 36 | 0.18 |
| temporal | 200 | 21 | 0.105 |
| social | 100 | 19 | 0.19 |
| hybrid | 20 | 5 | 0.25 |

Top failure types:

- `spatial_reference_error`: 231
- `single_to_multi`: 132
- `multi_missing`: 76
- `natural_property_error`: 63
- `output_format_error`: 49
- `temporal_calculation_error`: 38

## Pending

1. Prompt improvement versions
   - Not started yet.
   - Future prompt files should be versioned, for example `configs/system_prompts_v2.yaml`.
   - Notes should go in `docs/PROMPT_NOTES.md`.

## Reproduction Commands

Run these on the GPU server to reproduce the baseline/dev reports.

```bash
cd /root/autodl-tmp/pingce001

python scripts/filter_split_records.py \
  --input data/raw/train.json \
  --ids-file data/splits/dev_ids.json \
  --output outputs/dev_prompts.jsonl

python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --input outputs/dev_prompts.jsonl \
  --output outputs/dev_baseline_predictions.jsonl

python scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_baseline_predictions.jsonl \
  --report outputs/dev_baseline_eval.json \
  --max-mistakes 100

python scripts/analyze_errors.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_baseline_predictions.jsonl \
  --report outputs/error_report_baseline_dev.json \
  --cases outputs/error_cases_baseline_dev.jsonl \
  --doc docs/ERROR_ANALYSIS_BASELINE.md
```

## Local Verification

Current local checks:

```text
dev_prompts ok 720
31 tests passed
```
