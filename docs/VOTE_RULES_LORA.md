# Vote Rules for LoRA Dev

Status: historical voting-plan note. It records proposed dev experiments and
should not be read as the final routed submission method.

## Evidence From Current Dev

Source: `outputs/error_report_lora_dev.json` and current LoRA dev predictions.

- Overall LoRA dev accuracy: `261/720 = 0.3625`.
- Worst priority domains:
  - `temporal`: `42/200 = 0.21`
  - `hybrid`: `5/20 = 0.25`
  - `spatial`: `68/200 = 0.34`
- Focus error counts:
  - `spatial_reference_error`: `249`
  - `single_to_multi`: `47`
  - `multi_missing`: `43`
  - `temporal_calculation_error`: `35`
- Multi-answer dev subset:
  - `temporal`: `25/116 = 0.2155`
  - `spatial`: `8/46 = 0.1739`
  - `hybrid`: `0/1 = 0.0`
- Single-answer dev subset:
  - `temporal`: `17/84 = 0.2024`
  - `spatial`: `60/154 = 0.3896`
  - `hybrid`: `5/19 = 0.2632`

Conclusion: vote should target unstable reasoning cases, especially multi-answer and layout/time calculation questions. It should not be enabled globally because `single_to_multi` is already a visible failure mode.

## Recommended Vote Gate

Enable vote only when at least one high-risk domain condition and one uncertainty condition are met.

High-risk domain condition:

- `domain == temporal`
- `domain == spatial`
- `domain == hybrid`

Uncertainty condition:

- The question asks for statement selection, such as `Select the correct statement(s)`, `Select the incorrect statement(s)`, `以下选项`, or similar Chinese variants.
- The first-pass output contains more than one answer label.
- The first-pass output is `D` or an option whose text means `none of the above` on a spatial/hybrid problem.
- The question contains explicit calculation or layout words:
  - temporal: `before`, `after`, `gap`, `days`, `years`, weekday words, `之前`, `之后`, `相差`
  - spatial: `left`, `right`, `above`, `below`, `neighbor`, `same layer`, `左`, `右`, `上方`, `下方`, `同层`, `相邻`
  - hybrid: multiple domain cues in the same text, especially shelf/grid layout plus object category constraints

## Rules By Type

### Multi-Answer-Likely Questions

Use vote.

Trigger:

- Statement-selection question.
- First pass returns two or more labels.
- The question says `correct statement(s)` or `incorrect statement(s)`.

Aggregation draft:

- Run `5` generations or prompt variants.
- Count votes per label independently.
- Select labels with at least `2/5` votes if the question explicitly asks for statement(s).
- If no label reaches `2/5`, fall back to the strongest single label.
- Preserve option order in the final JSON.

Why: `multi_missing` is a real error class, especially in temporal questions. A per-label vote can recover missed labels better than single-sequence majority.

Risk: this can increase `single_to_multi`, so only use the lower `2/5` threshold when the input itself has multi-answer cues.

### Temporal

Use vote for most temporal questions with calculation cues.

Trigger:

- `domain == temporal`
- and question/text includes before/after, year gaps, day gaps, weekday cycles, or statement selection.

Aggregation draft:

- Run `5` samples with low temperature or prompt variants.
- For single-answer-looking questions, choose the label with the highest vote count.
- For statement-selection questions, use per-label voting with threshold `2/5`.
- If the top two labels tie on a single-answer-looking question, run one deterministic temporal prompt pass and use it as tie-breaker.

Why: temporal is the weakest domain (`0.21`) and has both `temporal_calculation_error` and heavy multi-answer misses.

Do not vote:

- Very direct temporal lookup where the question asks one explicit event and first pass returns one label with stable JSON.

### Spatial

Use vote selectively.

Trigger:

- `domain == spatial`
- and the text involves reference frames, circular seating, shelf/grid layers, left/right/up/down, same layer, neighbor, or `none of the above`.

Aggregation draft:

- Prefer prompt-variant vote over pure sampling:
  - normal spatial prompt
  - short reference-frame prompt
  - stricter option-by-option prompt
- For single-answer-looking spatial questions, use majority single-label vote.
- For statement-selection spatial questions, use per-label threshold `2/5`.
- If `D` is `none of the above`, require `D` to win by at least two votes in a 5-vote run before selecting it.

Why: spatial has many reference-frame errors, but single-answer spatial is much better than multi-answer spatial. Vote should improve fragile layouts without turning every single-answer case into multi-answer.

Do not vote:

- Spatial-like natural category mapping where the task is really object/category classification.
- First pass returns a single non-`D` label on a short direct lookup with no left/right/up/down/circle/layer cue.

### Hybrid

Use vote, but do not let the hybrid prompt dominate.

Trigger:

- `domain == hybrid`, or inferred mixed domain cues appear in text.

Aggregation draft:

- Run prompt-variant vote:
  - hybrid prompt
  - spatial prompt if layout terms are present
  - temporal prompt if time terms are present
  - general prompt as fallback
- Use majority single-label vote for normal hybrid questions.
- For statement-selection hybrid questions, use per-label threshold `2/5`.
- If hybrid prompt disagrees with both general and the strongest subdomain prompt, down-weight hybrid to `0.5`.

Why: current hybrid dev accuracy is only `0.25`, and its failures cluster around joint/spatial constraints. The current hybrid prompt may be adding drag; vote should test whether general or subdomain prompts are more reliable.

Do not vote:

- Hybrid detection is weak and there is only one clear domain cue. Route to that single domain instead.

## Recommended First Experiment

Start with a constrained vote experiment on dev:

1. Enable vote for `temporal` statement-selection and before/after/gap questions.
2. Enable vote for `spatial` layout/reference-frame questions only.
3. Enable prompt-variant vote for all `hybrid`, with general fallback included.
4. Keep `social` and `natural` off for now.
5. Compare against current LoRA dev baseline:
   - overall accuracy
   - temporal accuracy
   - spatial accuracy
   - hybrid accuracy
   - `single_to_multi`
   - `multi_missing`

Minimum pass condition:

- `temporal` improves without increasing `single_to_multi` by more than `10%`.
- `spatial` improves or `spatial_reference_error` drops.
- `hybrid` improves above `0.25`, or the experiment proves hybrid prompt should be replaced by general/subdomain routing.

## Pseudocode

```python
def should_vote(record, first_prediction):
    domain = infer_domain(record)
    if domain not in {"temporal", "spatial", "hybrid"}:
        return False

    qtext = f"{record.get('text', '')} {record.get('question', '')}".lower()
    pred_labels = first_prediction.get("answer", [])

    multi_cue = any(x in qtext for x in [
        "select the correct", "select the incorrect", "statement(s)",
        "以下", "正确", "不正确",
    ])
    temporal_cue = any(x in qtext for x in [
        "before", "after", "gap", "day", "year", "weekday",
        "之前", "之后", "相差",
    ])
    spatial_cue = any(x in qtext for x in [
        "left", "right", "above", "below", "neighbor", "same layer",
        "左", "右", "上方", "下方", "同层", "相邻",
    ])
    output_uncertain = len(pred_labels) != 1

    if domain == "temporal":
        return multi_cue or temporal_cue or output_uncertain
    if domain == "spatial":
        return multi_cue or spatial_cue or output_uncertain
    if domain == "hybrid":
        return True
    return False
```
