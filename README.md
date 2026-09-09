# SCoRE2026 Dense LoRA Routing

This branch contains the public, code-oriented companion for our SCoRE2026
submission. The main research line is a dense `Qwen2.5-7B-Instruct` system with
LoRA fine-tuning and a test-safe routing rule between two prediction styles:

- `answer_only`: concise answer generation, used as the stable default.
- `mixed_reasoning`: short reasoning for harder cases, mainly useful for
  temporal and hybrid multi-answer questions.
- `V7d routing`: use mixed reasoning only when the question is temporal or
  hybrid and a rule-based classifier predicts `multi`; otherwise keep
  answer-only.

The repository is organized for public inspection: small reproducible artifacts,
scripts, configuration files, validation tests, and submission-format outputs are
tracked. Raw official data, model checkpoints, LoRA adapter weights, and large
runtime files are intentionally not committed. Small trainer-state logs for
selected negative runs are tracked under `outputs/training_logs/`.

## Results Snapshot

All dev numbers below use the fixed `2880/720` train/dev split with seed `2026`.

| System | Dev Correct | Dev Accuracy | Evidence |
| --- | ---: | ---: | --- |
| Zero-shot Qwen2.5-7B-Instruct | 131 / 720 | 18.19% | `outputs/dev_baseline_eval.json` |
| Answer-only LoRA | 261 / 720 | 36.25% | `outputs/dev_lora_answer_only_eval.json` |
| Mixed-reasoning LoRA | 277 / 720 | 38.47% | `outputs/dev_lora_mixed_reasoning_eval.json`; run notes in `docs/MIXED_REASONING_SUBMISSION_HANDOFF.md` |
| V7d routed test submission | official score 19.0% | test set | submission artifact in `outputs/submissions/v7d_submission.json`; score is an external official record |

The mixed-reasoning dev number used gold answer-count hints during dev
diagnostics. The official test workflow avoids gold hints and uses predicted
cardinality instead. This distinction matters for reproducing the paper claims.

## Mainline Files

| Purpose | Files |
| --- | --- |
| Train answer-only LoRA | `configs/train_qwen_v2_neutral.yaml`, `scripts/train_lora.py` |
| Train mixed-reasoning LoRA | `configs/train_qwen_mixed_reasoning.yaml`, `scripts/run_qwen_mixed_reasoning.sh` |
| Build SFT data | `scripts/build_sft_data.py`, `outputs/sft_data_report.json` |
| Predict single/multi cardinality | `scripts/predict_cardinality.py` |
| Build V7d routed submission | `scripts/build_v7d.py` |
| Test-safe submission workflow | `scripts/run_mixed_reasoning_submission.sh`, `scripts/validate_submission.py` |
| Error analysis and dev reports | `scripts/analyze_errors.py`, `outputs/*_eval.json`, `docs/ERROR_ANALYSIS_*.md` |

See `docs/PAPER_MAINLINE.md` for the paper-facing method map and evidence
boundaries.

## Document Status

The paper-facing entry points are:

- `README.md`: public overview and reproduction commands.
- `docs/PAPER_MAINLINE.md`: method-to-evidence map and reproducibility boundary.
- `docs/PUBLICATION_AUDIT.md`: explicit publication-scope audit.
- `docs/MIXED_REASONING_SUBMISSION_HANDOFF.md`: mixed-reasoning run notes, with
  the dev-only gold-hint caveat.

Several older documents are retained as experiment history. They are useful for
understanding explored routes, but they are not final paper claims:

- `docs/DENSE_BUCKET_ROUTE_NOTES.md`: dev-only dense bucket routing exploration.
- `docs/DISTILLATION_PIPELINE.md`: API-teacher distillation plan.
- `docs/VOTE_RULES_LORA*.md`: voting experiments and proposed gates.
- `docs/V4_IMPROVEMENT_PLAN.md`: cardinality-hint diagnosis and estimated fixes.
- `docs/VERIFIER_*.md`: option-level verifier probe planning.

## Reproduce Local Checks

The tracked unit and smoke tests are CPU-only:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

Current verification on this branch:

```text
Ran 62 tests
OK
```

## Reproduce The Dev Split

Place the official SCoRE2026 `train.json` under `data/raw/train.json`, then run:

```bash
python3 scripts/make_dev_split.py \
  --input data/raw/train.json \
  --train-ids data/splits/train_ids.json \
  --dev-ids data/splits/dev_ids.json \
  --report outputs/dev_split_report.json \
  --seed 2026

python3 scripts/filter_split_records.py \
  --input data/raw/train.json \
  --ids-file data/splits/dev_ids.json \
  --output outputs/dev_prompts.jsonl
```

The tracked split report records:

- train: 2880 examples
- dev: 720 examples
- dev domains: natural 200, spatial 200, temporal 200, social 100, hybrid 20

## Train And Evaluate LoRA

The training scripts expect a local copy of `Qwen/Qwen2.5-7B-Instruct` and a GPU
environment with the usual Transformers/PEFT stack.

Answer-only and mixed-reasoning data can be regenerated with:

```bash
python3 scripts/build_sft_data.py \
  --prompts configs/system_prompts_v3_mixed_focus.yaml \
  --output-dir outputs \
  --report outputs/sft_data_report.json
```

Run mixed-reasoning training and dev evaluation:

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
PYTHON_BIN=python3 \
bash scripts/run_qwen_mixed_reasoning.sh
```

The LoRA adapter directories are not tracked. To reproduce exact paper-side
predictions, keep the trained adapters under the checkpoint names used in the
configs, for example:

```text
checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026
```

## Build A Test-Safe V7d Submission

V7d needs three aligned prediction files: answer-only predictions,
mixed-reasoning predictions, and cardinality predictions. The final router never
reads gold answers.

```bash
python3 scripts/predict_cardinality.py \
  --input outputs/test_prompts.jsonl \
  --output outputs/test_prompts_with_cardinality.jsonl \
  --stats

python3 scripts/build_v7d.py \
  --input outputs/test_prompts.jsonl \
  --answer-only outputs/test_lora_answer_only_predictions.jsonl \
  --mixed-reasoning outputs/test_lora_mixed_reasoning_predictions.jsonl \
  --cardinality outputs/test_prompts_with_cardinality.jsonl \
  --output outputs/submissions/v7d_submission.json

PYTHONPATH=. python3 scripts/validate_submission.py \
  --input outputs/submissions/v7d_submission.json \
  --report outputs/submissions/v7d_report.json
```

## Public Scope

This branch is intended to show the method, reproducible scripts, tracked dev
evidence, and official submission files. It does not claim to contain every raw
remote experiment artifact from the original competition period. In particular,
some intermediate prediction files and all LoRA adapter weights must be restored
from the original run machine or regenerated.

For paper citation, use this branch as the public implementation and evidence
map for the V7d method. Do not describe it as a complete proof that every
historical artifact was regenerated from training within this checkout.
