# Paper Mainline And Evidence Map

This note connects the paper story to the public repository contents. It is
written as an audit trail for readers who want to inspect what is backed by
tracked code and what requires external runtime artifacts.

## Method Story

The final paper line is a compact dense-model pipeline for SCoRE2026:

1. Normalize the official data and build a fixed stratified train/dev split.
2. Fine-tune `Qwen2.5-7B-Instruct` with LoRA on answer-only supervision.
3. Fine-tune a second LoRA with mixed short reasoning for harder domains.
4. Predict question cardinality from the test prompt without reading answers.
5. Route only temporal/hybrid predicted multi-answer questions to the
   mixed-reasoning model; use answer-only for all remaining questions.

This design came from the dev observation that answer-only behavior is more
stable on natural and social questions, while temporal and hybrid multi-answer
questions benefit more from short reasoning.

## Tracked Evidence

| Claim | Tracked Evidence |
| --- | --- |
| Official data split is fixed and stratified | `data/splits/*.json`, `outputs/dev_split_report.json` |
| Zero-shot baseline is 131/720 on dev | `outputs/dev_baseline_eval.json` |
| Answer-only LoRA is 261/720 on dev | `outputs/dev_lora_answer_only_eval.json` |
| Answer-only temporal result is 42/200 | `outputs/dev_lora_answer_only_eval.json` |
| V7d submission is a 1000-item official-format JSON | `outputs/submissions/v7d_submission.json`, validation scripts |
| V7d routing does not use test gold answers | `scripts/build_v7d.py`, `scripts/predict_cardinality.py` |
| Unit/smoke behavior is covered | `tests/`, 62 passing tests |

## Files To Read First

- `README.md`: public entry point and reproduction commands.
- `scripts/build_v7d.py`: final routed submission builder.
- `scripts/predict_cardinality.py`: test-safe single/multi prediction.
- `scripts/build_sft_data.py`: construction of SFT variants.
- `configs/train_qwen_mixed_reasoning.yaml`: mixed-reasoning LoRA config.
- `docs/MIXED_REASONING_SUBMISSION_HANDOFF.md`: mixed-reasoning handoff notes.
- `docs/ERROR_ANALYSIS_LORA.md`: dev error analysis after LoRA.

## Historical Experiment Notes

Older `docs/` files preserve the path of explored methods. They should be read
as dated experiment notes unless this file or `README.md` explicitly names them
as paper-facing evidence. In particular:

- `docs/DENSE_BUCKET_ROUTE_NOTES.md` records a dev-only routing exploration whose
  generated outputs are not tracked in this public branch.
- `docs/DISTILLATION_PIPELINE.md` records an API-teacher distillation plan, not
  the final paper method.
- `docs/VOTE_RULES_LORA*.md`, `docs/V4_IMPROVEMENT_PLAN.md`, and
  `docs/VERIFIER_*.md` are retained to show negative and exploratory routes.

## Reproducibility Boundaries

The public branch tracks code, configs, small JSON/JSONL reports, split ids, and
submission JSON files. It does not track:

- official raw `train.json` or `test.json`;
- base model weights;
- LoRA adapter checkpoints;
- large training logs;
- every intermediate remote prediction artifact.

Therefore, this branch can verify format, data flow, split consistency, routing
logic, and tracked dev metrics. Exact end-to-end regeneration of all competition
predictions requires either rerunning the GPU training/inference pipeline or
restoring the original checkpoint and prediction artifacts.

The public branch preserves the mainline artifacts available from the historical
project state. It does not, by itself, prove a complete provenance chain for
every generated submission file.

## Paper Number Hygiene

Use the tracked evidence above when editing the manuscript:

- answer-only LoRA dev: `261/720 = 36.25%`;
- answer-only temporal dev: `42/200 = 21.00%`;
- mixed-reasoning dev: `277/720 = 38.47%`, with the diagnostic caveat recorded
  in `docs/MIXED_REASONING_SUBMISSION_HANDOFF.md`;
- V7d official test score: `19.0%`, as an external leaderboard/submission
  result rather than a locally recomputed metric.

Avoid claiming that this repository alone fully reconstructs every historical
online submission. It provides a clean public implementation and evidence map
for the paper's main method, while keeping older exploratory notes labeled as
history.
