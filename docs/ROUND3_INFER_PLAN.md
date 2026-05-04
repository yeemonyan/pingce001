# Round 3 Inference Plan

目标：在不重新训练模型的前提下，基于当前 `answer_only` LoRA adapter，先做低成本试探，确认方向后再决定是否进行全量 dev / leaderboard 推理。

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
2. 更短、更硬的 prompt 只有在与现有 LoRA 指令分布兼容时才可能提分。
3. `hybrid -> general fallback` 可能优于直接使用 `hybrid` prompt。
4. 全量 vote 未必适合当前这种系统性推理错误，应该先做更小规模验证。

## Experiment Matrix

统一设置：

- model: `models/Qwen2.5-7B-Instruct`
- adapter: `checkpoints/qwen2p5_7b_lora_answer_only_seed2026`
- input: `outputs/dev_prompts.jsonl`
- dtype: `bfloat16`
- max_new_tokens: `64`
- `OMP_NUM_THREADS=1`

## Probe First

在上卡做全量实验前，先优先做两个轻量试探：

1. 小样本 prompt probe
   - 从 dev 中抽取：
     - 30 题 spatial
     - 30 题 temporal
     - 20 题 natural
     - 10 题 hybrid
   - 对比：
     - 原始 prompt
     - short prompt
     - short prompt + fallback
   - 先看错误类型是否真的改善，而不是先看全量分数。

2. 小样本 selective vote probe
   - 不做全量 vote。
   - 只在以下题上试：
     - 多选题
     - temporal
     - hybrid
   - 先比较：
     - greedy
     - vote(3, low temperature)

如果 probe 没看到明显改善，就不要立刻做全量 GPU 实验。

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
  - 或 selective vote

## Current Lessons

目前已经确认：

- `short routed` 低于当前 best
- `short + fallback` 也低于当前 best
- `global vote(3)` 明显低于当前 best
- 2026-05-04 的 90 题 baseline probe 未产出结果，根因不是 prompt，而是服务器会话里 `torch.cuda.is_available() == False`；7B 推理在无 GPU 可见状态下启动后被系统直接 `Killed`

因此下一步不能再做“全局替换”式实验，必须先通过 probe 找到更窄的有效场景。

## Environment Gate

在继续任何 7B / LoRA 推理前，先做这一条环境闸门：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

只有输出为 `True` 且设备数大于 `0` 时，才继续执行 probe / dev / test 推理。

如果这里仍然是 `False`：

- 先不要继续烧卡跑实验
- 先检查当前 AutoDL 实例是否真的处于带 GPU 的运行态
- 再确认进入的是正确容器/实例，而不是只挂载了项目目录的无卡环境
