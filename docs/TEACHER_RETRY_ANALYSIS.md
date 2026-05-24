# Teacher Retry Analysis

Date: 2026-05-24

## What We Checked

We re-audited the `valid` teacher distillation run using:

- `outputs/distill/teacher_requests_valid.jsonl`
- `outputs/distill/teacher_responses_valid.jsonl`

The goal was to understand whether the remaining bad cases should be retried, discarded, or downgraded to `answer_only`.

## Key Finding

The previously reported `186` problematic samples should **not** all be retried.

With the updated cleaner and polarity-aware parsing:

- `accepted`: `595`
- `incorrect_answer`: `54`
- `invalid_json`: `63`
- `inconsistent_selected_answers`: `8`

Only the following statuses are worth retrying:

- `invalid_json`
- `inconsistent_selected_answers`
- `missing_option_judgments`
- `missing_option`
- `invalid_option_label`
- `missing_teacher_output`

For the current `valid` run, this means the true retry queue is:

- `63` infrastructure / parse failures
- `8` schema-consistency failures
- total retryable: `71`

The remaining `54` are genuine teacher reasoning mistakes and should stay as `answer_only` fallback for now instead of consuming API budget repeatedly.

## Why The 63 Invalid Cases Matter

These are mostly transport failures rather than model-quality failures:

- `47` `ReadTimeout`
- `16` DNS / name-resolution failures

That makes them high-value retries because successful reruns can directly enlarge the usable distilled dataset.

## Where Real Teacher Errors Concentrate

Among the true reasoning failures, the main weak areas are:

- `temporal`: `38`
- `spatial`: `12`
- `natural`: `3`
- `social`: `1`

This matches earlier observations that temporal multi-answer questions remain the hardest group.

## Prompt / Schema Fixes Applied

We updated the distillation pipeline so that:

1. Teacher output uses explicit `selected` semantics instead of overloading `label`.
2. Request records carry `question_polarity` awareness.
3. Prompts explicitly state that `selected=true` means "this option should appear in the final answer", even for "incorrect statement" questions.
4. The cleaner rejects `selected` vs `answers` inconsistencies instead of silently trusting ambiguous outputs.
5. Retry generation now focuses on retryable statuses only.

## Operational Decision

Current policy:

- keep every sample in `answer_only`
- keep accepted teacher samples for `short_reasoning` and `option_level`
- retry only the `71` retryable cases
- do not blindly requery the `54` teacher wrong-answer cases

## Expected Benefit

If the retry run resolves most infrastructure failures, the valid distilled pool should grow from:

- `595 / 720 = 82.64%`

toward roughly:

- `(595 + recovered retries) / 720`

Even recovering half of the transport failures would noticeably improve the amount of high-quality supervision available for the full train split.
