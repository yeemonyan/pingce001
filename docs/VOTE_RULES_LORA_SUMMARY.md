# Vote Enablement Summary

Status: historical voting-plan summary. It is retained for context only; the
paper-facing route is described in `README.md` and `docs/PAPER_MAINLINE.md`.

## Recommended to Enable Vote

- `temporal` multi-answer or statement-selection questions.
- `temporal` questions with `before`, `after`, `gap`, `day`, `year`, or weekday-cycle cues.
- `spatial` questions with clear reference-frame cues: left/right/up/down, circle order, shelf layout, same layer, adjacent.
- `hybrid` questions by default, but route through prompt variants and keep a general fallback.

## Recommended Not to Enable Vote

- Easy single-answer questions with one clear clue and stable JSON output.
- `social` and `natural` for now.
- `spatial` direct category-mapping questions that look like natural property matching rather than true layout reasoning.
- Any case where the first pass is already a single label and the wording does not signal multi-answer or calculation risk.

## Simple Rule Draft

Enable vote only when:

1. `domain in {temporal, spatial, hybrid}`
2. and at least one uncertainty cue is present:
   - statement-selection wording
   - more than one predicted label
   - explicit calculation cue
   - explicit reference-frame cue
   - `hybrid` domain

For `temporal` and `spatial`, prefer vote on multi-answer-like questions first.
For `hybrid`, prefer prompt-variant vote with a general fallback rather than pure self-consistency.

## Next Training Note

If vote helps mostly on multi-answer cases, the next `rationale_json` round should emphasize:

- option-by-option verdicts
- compact timeline / layout / relation-chain notes
- explicit final answer array

If vote hurts `single_to_multi`, keep answer-only for single-answer questions and reserve `rationale_json` for temporal/spatial multi-answer cases.
