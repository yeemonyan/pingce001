# Prompt Notes

## Priority List From Baseline Dev Errors

| Priority | Area | Evidence | Action |
| --- | --- | --- | --- |
| 1 | temporal | Lowest dev accuracy: `21/200`, ACC `0.105`; temporal also has `57` multi-missing and `32` single-to-multi errors. | Optimize temporal prompt first: explicit timeline, before/after arithmetic, weekday cycle, option-by-option verification. |
| 2 | spatial | `164/200` spatial errors; `128` are classified as spatial reference errors. | Keep spatial prompt stable for v2, but next prompt round should focus on reference frames, facing direction, left/right/up/down, and "none of the above". |
| 3 | social / hybrid | social ACC `0.19` with heavy over-selection; hybrid has only 20 dev examples but depends on multi-domain merging. | In v2, strengthen social role rewrites/inverse relations and hybrid local-then-joint solving. |
| Global | multi-answer boundary | `single_to_multi=132`, `multi_missing=76`. | Every prompt should require checking all options but selecting only exactly entailed options. SFT should include both answer-only and structured rationale variants. |

Short conclusion:

- Most urgent: `temporal`
- Second: `spatial`
- Third: `social` and `hybrid`
- Multi-answer boundaries must be handled across all domains.

## Prompt Version: `configs/system_prompts_v2.yaml`

### Changed Prompts

- `temporal`
- `social`
- `hybrid`

### Unchanged Prompts

- `spatial`
- `natural`
- `general`

### Why Temporal Changed

Baseline temporal dev accuracy is the weakest at `0.105`. The prompt now enforces a fixed sequence:

1. list timeline variables,
2. compute before/after constraints,
3. handle weekday cycles,
4. distinguish start/end/no-longer-happens events,
5. verify every option.

Expected improvement:

- `temporal_calculation_error`
- `multi_missing`
- `single_to_multi`

### Why Social Changed

Social dev accuracy is `0.19`, and many errors over-select options. The prompt now requires:

1. person list and relationship graph,
2. alias and role rewrite normalization,
3. inverse relation handling,
4. option-level resolution,
5. exact entailment checks.

Expected improvement:

- `social_relation_error`
- `single_to_multi`
- role-alias mistakes hidden inside `spatial_reference_error` buckets

### Why Hybrid Changed

Hybrid dev accuracy is `0.25`, but the sample size is only 20 and the errors are mostly caused by failing to combine constraints. The prompt now requires:

1. identify subdomains,
2. solve each subdomain locally,
3. merge into a joint constraint table,
4. reject single-clue answers,
5. verify every option after merging.

Expected improvement:

- `multi_constraint_failure`
- spatial/social/temporal errors inside hybrid problems
- over-reliance on one clue

## Training Data Recommendation

Run SFT in this order:

1. First run `answer_only`.
   - It directly teaches the model the target output schema.
   - It is safer for output-format stability.
   - It should help reduce `output_format_error`, `single_to_multi`, and excessive free-form responses.

2. Then run `rationale_json`.
   - It is more useful for temporal, social, and hybrid questions.
   - It explicitly exposes `analysis.domain` and `key_constraints`.
   - It should help errors that require intermediate structure, especially timeline and multi-domain merging.

Do not generate pseudo labels from the test set, and do not add external data.

## Assistant Format Recommendation

- `answer_only` is best for final format alignment and quick baseline comparison.
- `rationale_json` is likely better for:
  - temporal timeline calculations,
  - social alias/inverse relation problems,
  - hybrid multi-constraint joining.
- Future rationale format should explicitly expose:
  - domain,
  - key constraints,
  - option verification result,
  - final answers.
