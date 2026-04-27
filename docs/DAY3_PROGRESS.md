# Day 3 任务大纲与完成进度

## Day 3 目标

Day 3 的目标是格式对齐与首次测评：将模型输出清洗为官方提交文件，编写本地评测脚本，完成从数据解析、模型推理、ACC 评测到 submission 文件导出的完整链路。

## 已完成

- [x] 新增 `scripts/format_submission.py`，将 `infer_score.py` 输出转换为提交文件。
- [x] 默认提交格式改为更保守的 `jsonl_with_id`：每行一个对象，包含 `id` 和 `answer`。
- [x] 保留 `--official-format system_json`，兼容线上系统若要求 JSON 数组的情况。
- [x] 新增 `scripts/evaluate_score.py`，支持按 `id` 或顺序对齐预测和带答案数据。
- [x] 评测脚本输出整体 ACC、分领域 ACC、缺失预测数和错题样例。
- [x] submission 脚本会清洗非法答案，空答案使用可配置 fallback，默认 `D`。
- [x] 新增单元测试覆盖 submission 格式化和 ACC 评测。
- [x] 使用 Day 2 smoke 数据跑通完整链路。

## 本地验证链路

```bash
python scripts/parse_score_json.py \
  --input data/day2_smoke.json \
  --output outputs/day2_smoke_prompts.jsonl

python scripts/infer_score.py \
  --backend mock \
  --input outputs/day2_smoke_prompts.jsonl \
  --output outputs/day2_smoke_predictions.jsonl

python scripts/evaluate_score.py \
  --gold outputs/day2_smoke_prompts.jsonl \
  --pred outputs/day2_smoke_predictions.jsonl \
  --report outputs/day2_smoke_eval_report.json

python scripts/format_submission.py \
  --input outputs/day2_smoke_predictions.jsonl \
  --output outputs/day2_smoke_submission.jsonl
```

## 官方首次提交流程

在云服务器完成模型下载后，将 mock 后端切换为真实模型：

```bash
python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --input outputs/test_prompts.jsonl \
  --output outputs/test_predictions.jsonl

python scripts/format_submission.py \
  --input outputs/test_predictions.jsonl \
  --output outputs/submission_day3.jsonl
```

默认生成 `jsonl_with_id`：

```jsonl
{"id":"example-1","answer":["A"]}
{"id":"example-2","answer":["A","B"]}
```

如果线上系统要求 JSON 数组，显式切换：

```bash
python scripts/format_submission.py \
  --official-format system_json \
  --input outputs/test_predictions.jsonl \
  --output outputs/submission_day3.json
```
