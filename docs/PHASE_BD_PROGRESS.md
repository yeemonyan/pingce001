# Phase B-D Progress

状态：历史阶段记录。本文记录 baseline、dev split 和早期 SFT 数据构造阶段；
它不是当前公开论文主线的入口。当前入口见 `README.md` 和
`docs/PAPER_MAINLINE.md`。

## Phase B0: Dev Split

已完成稳定 dev split，用于后续 prompt、baseline 和 LoRA 对比。

- 输入：`data/raw/train.json`
- 脚本：`scripts/make_dev_split.py`
- 随机种子：`2026`
- 切分比例：80/20
- 训练集：2880
- Dev：720
- 分层维度：`domain`、`language`、答案个数

产物：

- `data/splits/train_ids.json`
- `data/splits/dev_ids.json`
- `outputs/dev_split_report.json`

## Phase B: SFT 数据构造

已完成两套可用于 Qwen 微调的数据。

- 脚本：`scripts/build_sft_data.py`
- 保留字段：`id`、`text`、`question`、`options`、`answers`、`domain`、`language`
- 数据版本：
  - `answer_only`
  - `rationale_json`

产物：

- `outputs/sft_train_answer_only.jsonl`
- `outputs/sft_valid_answer_only.jsonl`
- `outputs/sft_train_rationale_json.jsonl`
- `outputs/sft_valid_rationale_json.jsonl`
- `outputs/sft_data_report.json`

## Phase D: Dev Baseline 与错题分析

已完成真实 Qwen2.5-7B-Instruct zero-shot dev baseline。

- Dev gold：`outputs/dev_prompts.jsonl`
- 预测文件：`outputs/dev_baseline_predictions.jsonl`
- 评测报告：`outputs/dev_baseline_eval.json`
- 错题报告：`outputs/error_report_baseline_dev.json`
- 错题样例：`outputs/error_cases_baseline_dev.jsonl`
- 文档：`docs/ERROR_ANALYSIS_BASELINE.md`

结果：

- 总数：720
- 正确：131
- ACC：0.18194444444444444

分领域：

| Domain | Total | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| natural | 200 | 50 | 0.25 |
| spatial | 200 | 36 | 0.18 |
| temporal | 200 | 21 | 0.105 |
| social | 100 | 19 | 0.19 |
| hybrid | 20 | 5 | 0.25 |

主要结论：

- Temporal 最弱，优先优化时间线、日期/星期换算和 before/after 约束。
- `single_to_multi` 和 `multi_missing` 都很突出，后续训练需要加强多选边界。
- Spatial reference 错误最多，需要显式记录左右、上下、朝向和观察者参考系。

## Prompt 版本规则

Prompt 优化尚未开始。后续不覆盖 `configs/system_prompts.yaml`，而是新增版本文件：

- `configs/system_prompts_v2.yaml`
- `configs/system_prompts_v3.yaml`

每个版本的变化和原因记录到：

- `docs/PROMPT_NOTES.md`
