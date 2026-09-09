# API Teacher 蒸馏探索路线

状态：历史探索文档。这里记录的是 API teacher 蒸馏方案和落盘计划，不是当前
公开分支的论文主线。当前论文主线见 `README.md` 和
`docs/PAPER_MAINLINE.md`。

更新时间：2026-05-24

## 目标

把官方训练集转换成可复现的蒸馏监督数据，并在 `Qwen2.5-7B-Instruct` 上做混合 SFT/LoRA。

本方案严格遵守当前 organizer 已确认的边界：

- 允许对训练集做蒸馏。
- 必须保留蒸馏 prompt、SFT 数据和微调后的模型。
- 不得使用测试集参与训练、伪标签或提示示例。
- 不得使用 SCoRE2026 之外的其他数据集。

## 目录与文件

### Prompt / 配置

- `configs/teacher_distill_prompts.yaml`
- `configs/train_qwen_distill_mix.yaml`

### 脚本

- `scripts/build_teacher_distill_requests.py`
- `scripts/build_distilled_sft_data.py`
- `scripts/run_qwen_distill_mix_ddp.sh`

## 数据流

### 第一步：构造 teacher 请求

从官方 `train.json` 和既有 `train/dev split` 出发，构造两份蒸馏请求：

- `outputs/distill/teacher_requests_train.jsonl`
- `outputs/distill/teacher_requests_valid.jsonl`

每条请求都包含：

- `id`
- `split`
- `domain`
- `language`
- `answer_kind`
- `messages`
- 原题文本和选项

teacher 的输出目标是统一结构：

```json
{
  "option_judgments": {
    "A": {"label": true, "reason": "..."},
    "B": {"label": false, "reason": "..."}
  },
  "final_reasoning": "...",
  "answers": ["A"]
}
```

### 第二步：收集 teacher 返回并清洗

teacher 返回文件默认路径：

- `outputs/distill/teacher_responses_train.jsonl`
- `outputs/distill/teacher_responses_valid.jsonl`

清洗脚本会做三件事：

1. 校验 JSON 结构是否合法。
2. 校验逐选项判断能否还原出标准答案。
3. 将不合格样本写入 retry 清单。

输出包括：

- `outputs/sft_train_distill_answer_only.jsonl`
- `outputs/sft_train_distill_short_reasoning.jsonl`
- `outputs/sft_train_distill_option_level.jsonl`
- `outputs/sft_train_distill_mix.jsonl`
- valid 对应四份
- `outputs/distill/teacher_retry_requests.jsonl`
- `outputs/distill_data_report.json`

### 第三步：混训

当前默认混训比例：

- `answer_only = 0.4`
- `short_reasoning = 0.3`
- `option_level = 0.3`

并默认对以下 bucket 做 2 倍放大：

- `spatial:single`
- `temporal:single`
- `spatial:multi`
- `temporal:multi`

## 推荐执行命令

### 1. 构造 teacher 请求

```bash
python3 scripts/build_teacher_distill_requests.py \
  --input data/raw/train.json \
  --train-ids data/splits/train_ids.json \
  --dev-ids data/splits/dev_ids.json
```

### 2. 基于 teacher 返回构造蒸馏 SFT

```bash
python3 scripts/build_distilled_sft_data.py \
  --input data/raw/train.json \
  --teacher-train outputs/distill/teacher_responses_train.jsonl \
  --teacher-valid outputs/distill/teacher_responses_valid.jsonl
```

### 3. 8 卡 LoRA 训练

```bash
bash scripts/run_qwen_distill_mix_ddp.sh
```

## 落盘要求

后续必须长期保留：

- teacher prompt 配置
- teacher 请求 jsonl
- teacher 原始返回 jsonl
- retry 请求
- 清洗脚本
- 最终蒸馏 SFT 数据
- 训练配置
- adapter 权重
- 训练日志

## 当时策略

1. `MoE` 暂时退出主工作流。
2. 当时探索路线变成 `API teacher -> 结构化蒸馏 -> Dense student`。
3. 先用高质量逐选项判断解决：
   - `single_to_multi`
   - `multi_missing`
   - `spatial_reference_error`
   - `temporal_calculation_error`
