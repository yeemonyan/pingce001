# Phase C: First LoRA Run

目标：在 `Qwen2.5-7B-Instruct` 上训练第一轮可复现的 LoRA adapter，并在固定 dev split 上比较是否优于当前 zero-shot baseline。

## 当前基线

- 线上 baseline：`7.8%`
- dev baseline ACC：`0.181944`
- 当前主线分支：`codex/phase-a1-baseline-run`
- 训练工作分支：`codex/phase-c-lora-run`

## 第一轮实验

优先训练 `answer_only` 数据：

- `outputs/sft_train_answer_only.jsonl`
- `outputs/sft_valid_answer_only.jsonl`

先不要直接训练 `rationale_json`，等第一轮 dev 对比后再决定是否继续。

## 训练配置

配置文件：

- `configs/train_lora.yaml`

关键参数：

- base model: `Qwen/Qwen2.5-7B-Instruct`
- LoRA rank: `8`
- lr: `1e-4`
- epoch: `3`
- seed: `2026`
- bf16: `true`
- gradient_accumulation_steps: `8`

## 推荐流程

### 1. 训练前 dry run

在已安装依赖的环境中执行：

```bash
python scripts/train_lora.py --config configs/train_lora.yaml --dry-run
```

确认：

- 训练/验证数据文件存在
- 样本数正确
- 输出目录正确

### 2. 正式训练

```bash
python scripts/train_lora.py --config configs/train_lora.yaml
```

### 3. 训练后评测

```bash
python scripts/run_lora_eval.py --config configs/train_lora.yaml
```

输出：

- dev 预测：`outputs/dev_lora_answer_only_predictions.jsonl`
- dev 报告：`outputs/dev_lora_answer_only_eval.json`

## 通过标准

- 训练能完整跑完
- adapter 成功保存到 `checkpoints/`
- dev ACC 高于 `0.181944`
- 或至少 `temporal / social / hybrid` 中两个领域明显优于 baseline

## 下一步决策

如果 `answer_only` 有提升：

- 再训练 `rationale_json`
- 或在 LoRA 基础上叠加 prompt v2

如果 `answer_only` 没提升：

- 优先试 prompt v2 baseline
- 再决定是否调 LoRA 参数或改训练数据
