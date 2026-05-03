# Round 3 Inference Plan

目标：在不重新训练模型的前提下，基于当前 `answer_only` LoRA adapter，优先验证 prompt 和 routing 是否能带来更稳的 dev / leaderboard 提升。

## Current Facts

- 线上第二次提交：`14.9%`
- 当前 LoRA dev ACC：`0.3625`
- 当前最重错误：
  - `spatial_reference_error`: 249
  - `single_to_multi`: 47
  - `multi_missing`: 43
  - `temporal_calculation_error`: 35
- `hybrid` dev：`5 / 20 = 0.25`
- 结论：`hybrid` prompt 可能拖后腿，空间题和多选控制仍是主要瓶颈。

## Round 3 Hypothesis

优先验证推理侧问题，而不是立刻重新训练：

1. 默认 routing 可能把太多题送进 `hybrid` prompt。
2. 更短、更硬的 prompt 可能比当前 prompt 更稳。
3. `hybrid -> general fallback` 可能优于直接使用 `hybrid` prompt。

## Experiment Matrix

统一设置：

- model: `models/Qwen2.5-7B-Instruct`
- adapter: `checkpoints/qwen2p5_7b_lora_answer_only_seed2026`
- input: `outputs/dev_prompts.jsonl`
- dtype: `bfloat16`
- max_new_tokens: `64`
- `OMP_NUM_THREADS=1`

### Exp A: General Only

目的：验证当前 routing 是否在拖后腿。

命令：

```bash
OMP_NUM_THREADS=1 python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --adapter-path checkpoints/qwen2p5_7b_lora_answer_only_seed2026 \
  --input outputs/dev_prompts.jsonl \
  --output outputs/dev_round3_general_only_predictions.jsonl \
  --prompts configs/system_prompts_round3_short.yaml \
  --force-domain general \
  --max-new-tokens 64 \
  --dtype bfloat16

python scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_round3_general_only_predictions.jsonl \
  --report outputs/dev_round3_general_only_eval.json
```

### Exp B: Routed + Short Prompt

目的：验证短硬 prompt 是否优于当前 prompt。

命令：

```bash
OMP_NUM_THREADS=1 python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --adapter-path checkpoints/qwen2p5_7b_lora_answer_only_seed2026 \
  --input outputs/dev_prompts.jsonl \
  --output outputs/dev_round3_short_routed_predictions.jsonl \
  --prompts configs/system_prompts_round3_short.yaml \
  --max-new-tokens 64 \
  --dtype bfloat16

python scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_round3_short_routed_predictions.jsonl \
  --report outputs/dev_round3_short_routed_eval.json
```

### Exp C: Routed + Short Prompt + Hybrid Fallback

目的：验证 `hybrid -> general` 是否优于直接使用 `hybrid` prompt。

命令：

```bash
OMP_NUM_THREADS=1 python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --adapter-path checkpoints/qwen2p5_7b_lora_answer_only_seed2026 \
  --input outputs/dev_prompts.jsonl \
  --output outputs/dev_round3_short_fallback_predictions.jsonl \
  --prompts configs/system_prompts_round3_short.yaml \
  --fallback-general \
  --max-new-tokens 64 \
  --dtype bfloat16

python scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_round3_short_fallback_predictions.jsonl \
  --report outputs/dev_round3_short_fallback_eval.json
```

## Decision Rule

优先级如下：

1. 总 ACC 最高
2. `temporal` 和 `spatial` 是否提升
3. `hybrid` 是否不下降，或者下降很小但总 ACC 明显提高
4. 多选过选/漏选是否减少

如果 `Exp B` 或 `Exp C` 高于当前 `0.3625`，直接用该方案跑测试集并发起第三次提交。

如果三组都没有高于 `0.3625`：

- 再决定是否进入：
  - `rationale_json` LoRA
  - 或 self-consistency / vote

