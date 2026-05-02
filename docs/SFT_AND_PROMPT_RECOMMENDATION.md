# SFT And Prompt Recommendation

## Recommended Run Order

### 1. Run `answer_only` SFT First

Use:

- `outputs/sft_train_answer_only.jsonl`
- `outputs/sft_valid_answer_only.jsonl`

Why:

- 当前存在明显答案边界问题：
  - `single_to_multi=132`
  - `multi_missing=76`
- `answer_only` 最直接地学习最终输出 schema：`{"answers":["A"]}`。
- 更适合作为第一轮 LoRA，对齐格式和多选边界。

Expected improvements:

- fewer output format errors,
- less over-selection on single-answer questions,
- cleaner final answer JSON.

### 2. Then Run `rationale_json` SFT

Use:

- `outputs/sft_train_rationale_json.jsonl`
- `outputs/sft_valid_rationale_json.jsonl`

Why:

- Temporal、social、hybrid 需要中间结构：
  - timeline,
  - relationship graph,
  - multi-domain constraints.
- `rationale_json` 比自由 CoT 更可控，能保留结构化分析而不鼓励长篇发散。

Expected improvements:

- temporal calculation,
- social alias and inverse relation handling,
- hybrid multi-constraint joining.

### 3. Prompt Experiment Order

First prompt experiment:

- `configs/system_prompts_v2.yaml`

Focus:

- `temporal`
- `social`
- `hybrid`

Do not compare on test set. Use dev:

```bash
python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --prompts configs/system_prompts_v2.yaml \
  --input outputs/dev_prompts.jsonl \
  --output outputs/dev_predictions_prompt_v2.jsonl

python scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_predictions_prompt_v2.jsonl \
  --report outputs/dev_eval_prompt_v2.json \
  --max-mistakes 100
```

Second prompt experiment, only if v2 helps:

- Create `configs/system_prompts_v3.yaml`
- Change only `spatial`
- Keep temporal/social/hybrid from the better version

Why:

- Spatial has many reference-frame errors, but changing it together with temporal/social/hybrid would make attribution messy.

## Which Dataset Fits Which Error Type

| Error Type | Better First Dataset | Reason |
| --- | --- | --- |
| `output_format_error` | `answer_only` | Teaches compact final JSON directly. |
| `single_to_multi` | `answer_only` | Reinforces exact option count and avoids over-answering. |
| `multi_missing` | `answer_only`, then `rationale_json` | First learn final label coverage, then learn why multiple options survive. |
| `temporal_calculation_error` | `rationale_json` | Needs explicit timeline and before/after constraints. |
| `social_relation_error` | `rationale_json` | Benefits from relationship graph and inverse-role structure. |
| `multi_constraint_failure` | `rationale_json` | Needs local constraints plus joint merge. |
| `spatial_reference_error` | `rationale_json` later | Needs structured reference-frame reasoning, but should be tested in a separate spatial prompt round. |

## Recommended Next Step

Run this order:

1. Dev baseline with `configs/system_prompts_v2.yaml`.
2. LoRA with `answer_only`.
3. LoRA with `rationale_json`.
4. If prompt v2 improves temporal, create v3 for spatial only.

Keep every run tied to one config file and one report file in `docs/EXPERIMENT_LOG.md`.
