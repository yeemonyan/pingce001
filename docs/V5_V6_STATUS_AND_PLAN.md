# V5 / V6 Status And Plan

## V5 status

V5 is the minimal prompt-only fix on top of the existing `mixed_reasoning` adapter:

- keep `configs/system_prompts_v3_mixed_focus.yaml`
- keep `qwen2p5_7b_lora_mixed_reasoning_seed2026`
- do **not** use classifier routing
- do **not** use `answer-count-hint`
- only change:
  - old: `Choose all correct option labels. Output only JSON.`
  - new: `Choose the correct option(s). Output only JSON.`

Server-side V5 artifacts:

- submission: `outputs/submissions/v5_submission.json`
- validation: `outputs/submissions/v5_report.json`

Validation result:

```json
{
  "ok": true,
  "count": 1000,
  "errors": [],
  "missing_ids": [],
  "unexpected_ids": []
}
```

Observed answer-count distribution:

- V4: `{1: 651, 2: 116, 3: 38, 4: 195}`
- V5: `{1: 618, 2: 78, 3: 92, 4: 212}`

Interpretation:

- V5 fixed formatting and removed the explicit `ALL` bias
- but it **did not fully solve over-selection**
- 3-answer and 4-answer outputs are still too frequent
- so V5 is valid as a backup submission, but is not yet a high-confidence improvement over `14.9%`

## Why not use `--answer-count-hint` for V6

`--answer-count-hint` depends on gold answer length and is only valid for dev diagnostics.

That means:

- it can help estimate an upper bound locally
- it must **not** be used on the official test set

So the practical V6 direction should stay fully test-legal.

## V6 design

V6 is a **single-first post-processing** strategy.

Core idea:

1. Keep the V5 generation path unchanged.
2. Only touch records where the model predicted multiple answers.
3. If the question has **no strong multi-answer cue**, collapse the prediction to one label.
4. Leave explicit multi-answer questions untouched.

This is intentionally narrower than V4:

- V4 changed prompting before generation and depended on classifier hints.
- V6 does **not** re-steer every sample.
- V6 only repairs the most damaging failure mode after generation: `single_to_multi`.

## V6 heuristics

Force single only when:

- no explicit multi cue such as:
  - `statement(s)`
  - `which of the following statements`
  - `以下选项中`
  - `哪些`
  - `哪几项`
  - `多选`
- and the question looks likely single because:
  - it has a singular question marker such as `which one`, `who`, `when`, `哪个`, `哪一个`
  - or its domain is `natural`, `social`, `general`
  - or its domain is `spatial`

Leave uncertain `temporal` and `hybrid` multi-output cases untouched unless they show a strong single marker.

## Files for V6

- `scripts/postprocess_v6_single_first.py`
- `scripts/run_submission_v6.sh`

## Suggested next step

Run V6 first before any new training.

Reason:

- zero extra GPU training cost
- directly targets the error mode still visible in V5
- lower risk than reintroducing full classifier hints

If V6 still does not beat `14.9%`, the next branch should not be another prompt micro-edit. It should move to:

1. better single/multi supervision in SFT
2. stronger single/multi routing with measured dev ablation
3. or a separate MoE line if compute budget allows
