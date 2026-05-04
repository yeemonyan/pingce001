# Rationale JSON Next Training Notes

Do not start training yet. This is the next-round data/training suggestion based on current LoRA dev errors.

## When Rationale JSON Is Likely Helpful

- `temporal`: high priority. The model needs explicit timeline construction, before/after arithmetic, gap calculation, and weekday modulo handling.
- `spatial`: high priority. The model needs explicit reference frame, facing direction, grid/layer/circle layout, and option-by-option verification.
- `hybrid`: medium-high priority, but keep rationale short. The target should teach the model to split constraints by domain and then merge them, not to produce long free-form reasoning.
- Multi-answer statement selection: high priority. The assistant output should make each option verdict explicit so the model learns not to drop valid labels.

## When Answer Only Is Still Better

- `social`: current LoRA dev is stronger than temporal/spatial. Use answer_only unless relation-chain errors stay high after prompt changes.
- Simple `natural` classification or direct lookup: answer_only is likely enough.
- Short single-answer questions without calculation, reference-frame, or multi-condition cues.

## Suggested Rationale JSON Shape

Use compact structured fields, not long chain-of-thought.

```json
{
  "analysis": {
    "domain": "temporal",
    "steps": [
      "timeline built",
      "before/after checked",
      "options verified"
    ],
    "option_verdicts": {
      "A": "false",
      "B": "true",
      "C": "false",
      "D": "false"
    }
  },
  "answer": ["B"]
}
```

## Data Emphasis

- Oversample temporal multi-answer cases.
- Oversample spatial reference-frame cases.
- Include hybrid cases with both a `subdomains` list and a final merged verdict.
- Keep the final answer key as `answer`, matching the current inference parser.
- Avoid test-set pseudo labels and external data.

## Recommended Next Run Order

1. Run the constrained vote experiment first, because it is cheaper than retraining.
2. If vote mainly improves temporal multi-answer but not spatial, train rationale_json with extra spatial reference-frame examples.
3. If vote worsens `single_to_multi`, keep answer_only for single-answer cases and use rationale_json only for multi-answer-like prompts.
4. If hybrid improves when routed through general/spatial prompts, adjust routing before spending GPU time on hybrid SFT.
