# Experiment Log

## 2026-05-02: Prompt v2 Planning

- Branch: `codex/phase-d-prompt-tuning`
- Prompt file: `configs/system_prompts_v2.yaml`
- Baseline report: `outputs/error_report_baseline_dev.json`
- Dev baseline ACC: `0.181944`

### Changes

- Added temporal prompt instructions for:
  - timeline listing,
  - before/after arithmetic,
  - weekday cycle handling,
  - start/end/no-longer-happens distinction,
  - option-by-option verification.
- Added social prompt instructions for:
  - person graph,
  - aliases and role rewrites,
  - inverse relations,
  - option-level relation checks.
- Added hybrid prompt instructions for:
  - subdomain decomposition,
  - local constraint solving,
  - joint constraint merging,
  - rejecting single-clue answers.

### Expected Improvements

- Temporal: reduce `temporal_calculation_error`, `multi_missing`, and `single_to_multi`.
- Social: reduce over-selection and role-alias mistakes.
- Hybrid: reduce multi-constraint failures and single-clue shortcuts.

### Next Suggested Runs

1. Run dev baseline again with `configs/system_prompts_v2.yaml`.
2. Compare against:
   - `outputs/dev_baseline_eval.json`
   - `outputs/error_report_baseline_dev.json`
3. If v2 improves temporal without hurting overall ACC, use it as prompt baseline for LoRA evaluation.
4. Run SFT in this order:
   - `answer_only`
   - `rationale_json`
