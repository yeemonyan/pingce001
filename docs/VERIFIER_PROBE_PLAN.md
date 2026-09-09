# Round 4 Verifier Probe Plan

Status: historical verifier-probe plan. It is retained as experiment context and
does not define the current paper-facing method.

## Why This Route

Current evidence says the weak point is not only model capacity:

- zero-shot dev: `0.1819`
- LoRA dev: `0.3625`
- online result: `0.149`
- latest 90-question probe: original prompt `38.89%`, short prompt `33.33%`

Prompt variants are not the main lever. The heavier errors are decision-format errors:

- `spatial_reference_error = 249`
- `temporal_calculation_error = 35`
- `single_to_multi + multi_missing = 90`

So the next probe should replace answer-set generation with option-level verification.

## Implemented Probe Pieces

- `scripts/build_verifier_data.py`
  - builds four option-level rows per question
  - label is `yes` if the option is in the official answer set, otherwise `no`
  - uses only official training data and existing train/dev split
- `scripts/infer_verifier.py`
  - asks one yes/no question per option
  - merges yes labels back into the competition answer array
  - writes question-level predictions plus optional option-level verdicts
- `scripts/verifier_utils.py`
  - shared prompt, parser, merge logic, and detailed metrics

## Metrics To Track

- overall accuracy
- single-answer accuracy
- multi-answer accuracy
- per-domain accuracy
- `over_predict`
- `under_predict`
- `empty_prediction`
- `wrong_label_set`

## GPU Experiments To Prepare

1. `Qwen2.5-7B + verifier LoRA`
   - train with `outputs/verifier_train.jsonl`
   - validate with `outputs/verifier_valid.jsonl`
   - compare against current LoRA dev `0.3625`
2. `Qwen3-8B zero-shot / verifier probe`
   - run the same `infer_verifier.py` interface first
   - only train if the verifier interface improves the probe

## Success Gate

Do not start a full new training run unless verifier probe beats the current LoRA dev baseline on at least one priority slice:

- temporal
- spatial
- hybrid
- multi-answer

The most important regression check is `single_to_multi`: verifier should reduce output-set instability, not amplify it.
