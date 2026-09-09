# Publication Repository Audit

Status: current audit for the public showcase branch.

## Verdict

The branch is suitable as a public paper companion for the V7d dense LoRA
routing method if the paper cites it with the same scope used in `README.md` and
`docs/PAPER_MAINLINE.md`.

It should be described as a public implementation and evidence map, not as a
self-contained archive of every competition run.

## Confirmed In This Branch

| Item | Status | Evidence |
| --- | --- | --- |
| Fixed train/dev split | confirmed | `data/splits/*.json`, `outputs/dev_split_report.json` |
| Zero-shot dev result | confirmed | `outputs/dev_baseline_eval.json` |
| Answer-only LoRA dev result | confirmed | `outputs/dev_lora_answer_only_eval.json` |
| V7d submission format | confirmed | `outputs/submissions/v7d_submission.json`, `scripts/validate_submission.py` |
| Test-safe V7d routing logic | confirmed | `scripts/build_v7d.py`, `scripts/predict_cardinality.py` |
| CPU tests | confirmed | 62 tests passed with `PYTHONPATH=. python3 -m unittest discover -s tests -v` |

## Claims That Need External Evidence

| Claim | Why |
| --- | --- |
| Official V7d online score is 19.0% | The local repository stores the submitted JSON, but the score comes from the official evaluation platform. |
| Exact mixed-reasoning 277/720 regeneration | The note records the result, but the mixed-reasoning prediction file and LoRA adapter are not tracked. |
| Exact reconstruction of all V4-V17 online submissions | Submission JSON files are tracked, but several intermediate prediction files and run logs are not. |

## Remaining Caveats

- Historical docs are retained for transparency and are labeled as exploration
  notes when they are not part of the final paper method.
- Raw official `train.json` and `test.json` are not tracked.
- Base model weights and LoRA adapter checkpoints are not tracked.
- Some historical helper scripts under `scripts/build_v*.py` require untracked
  intermediate prediction files. The paper-facing V7d builder is
  `scripts/build_v7d.py`.

## Citation Scope

Safe wording:

> We release the code, configuration, tracked dev reports, fixed split ids, and
> final submission artifact for the V7d dense LoRA routing system.

Avoid wording:

> The repository fully reproduces every online submission and training run from
> scratch without external artifacts.
