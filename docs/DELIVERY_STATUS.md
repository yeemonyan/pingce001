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

## Pending

1. Baseline dev predictions
   - Expected output: `outputs/dev_baseline_predictions.jsonl`
   - Blocker: GPU server is currently unreachable at `connect.bjb2.seetacloud.com:50890`.

2. Baseline dev evaluation report
   - Expected output: `outputs/dev_baseline_eval.json`
   - Depends on `outputs/dev_baseline_predictions.jsonl`.

3. Baseline dev error report
   - Expected outputs:
     - `outputs/error_report_baseline_dev.json`
     - `outputs/error_cases_baseline_dev.jsonl`
     - refreshed `docs/ERROR_ANALYSIS_BASELINE.md`
   - Depends on `outputs/dev_baseline_predictions.jsonl`.

4. Prompt improvement versions
   - Not started yet.
   - Future prompt files should be versioned, for example `configs/system_prompts_v2.yaml`.
   - Notes should go in `docs/PROMPT_NOTES.md`.

## Commands To Finish Pending Baseline Work

Run these on the GPU server after it is reachable and the repo branch is synced.

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
