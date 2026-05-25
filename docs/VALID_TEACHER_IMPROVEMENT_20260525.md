# Valid Teacher Improvement Report

Date: 2026-05-25

## Scope

This report measures the impact of retrying the `71` retryable teacher failures on the `valid` split.

It is a **teacher-stage score comparison**, not yet a new student LoRA score, because train-side distilled teacher responses are still incomplete.

## Merged Result

Files:

- original: `outputs/distill/teacher_responses_valid.jsonl`
- retry: `outputs/distill/teacher_responses_valid_retry_v3.jsonl`
- merged: `outputs/distill/teacher_responses_valid_merged_v4.jsonl`

After merging:

- accepted: `636`
- incorrect_answer: `64`
- invalid_json: `20`
- total: `720`

## Score Change

Overall valid teacher accuracy:

- before: `595 / 720 = 82.64%`
- after: `636 / 720 = 88.33%`
- delta: `+41` accepted samples, `+5.69` points

## Domain Breakdown

- `social`: `78/100 -> 94/100`, `+16.00` points
- `temporal`: `149/200 -> 160/200`, `+5.50` points
- `spatial`: `162/200 -> 170/200`, `+4.00` points
- `natural`: `190/200 -> 195/200`, `+2.50` points
- `hybrid`: `16/20 -> 17/20`, `+5.00` points

## Interpretation

The retry pass clearly helped most on:

- social polarity / selected-vs-answers consistency
- temporal and spatial cases that were previously lost to transport failures

The main infrastructure gain was reducing retryable parse/network failures:

- old invalid-like bucket: `63 invalid_json + 8 inconsistent = 71`
- new unresolved invalid bucket: `20 invalid_json`

## Important Limitation

This does **not** yet mean we have a new student-model dev score.

Reason:

- `outputs/sft_train_distill_mix.jsonl` is still empty in the current full distillation line
- only valid-side teacher data has been repaired so far

## Next Running Probe

A focused train-side teacher probe has been launched:

- request file: `outputs/distill/teacher_requests_train_probe192_v1.jsonl`
- response file: `outputs/distill/teacher_responses_train_probe192_v1.jsonl`
- log: `outputs/distill/train_probe192_v1.log`

Probe composition:

- social: `48`
- temporal: `56`
- spatial: `48`
- natural: `24`
- hybrid: `16`

Goal:

- obtain a fast partial train teacher set
- build a non-empty `distill_mix` probe
- run a first student-side comparison against the previous baseline
