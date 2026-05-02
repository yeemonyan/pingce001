# Day 3 任务大纲与完成进度

## Day 3 目标

Day 3 的目标是格式对齐与首次线上评测：将模型输出清洗为官方提交 JSON，编写本地验证脚本，并完成从数据解析、模型推理到线上提交的完整闭环。

## 已完成

- [x] 新增 `scripts/format_submission.py`，将 `infer_score.py` 输出转换为官方提交文件。
- [x] 官方平台提交格式已确认：JSON 数组，每条包含 `id` 和 `answers`。
- [x] 新增 `scripts/evaluate_score.py`，支持按 `id` 或顺序对齐预测和带答案数据。
- [x] 评测脚本输出整体 ACC、分领域 ACC、缺失预测数和错题样例。
- [x] submission 脚本会清洗非法答案，空答案使用可配置 fallback，默认 `D`。
- [x] 新增单元测试覆盖 submission 格式化和 ACC 评测。
- [x] 使用 Day 2 smoke 数据跑通完整链路。
- [x] 在服务器上完成 Qwen2.5-7B-Instruct 测试集推理。
- [x] 完成第一次线上提交，ACC 为 7.9%。

## 官方提交格式

```json
[
  {
    "id": "SCoRE2026-test-1",
    "answers": ["A"]
  },
  {
    "id": "SCoRE2026-test-2",
    "answers": ["B", "C"]
  }
]
```

多选题必须完全正确才得分，漏选、错选、多选均不得分。

## 第一次提交记录

- 提交模型：`Qwen/Qwen2.5-7B-Instruct`
- 推理策略：zero-shot prompt + 规则化答案抽取
- 提交文件：`submissions/score2026_first_submission_qwen7b.json`
- 线上 ACC：7.9%

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
  --output outputs/day2_smoke_submission.json
```

## 服务器真实模型流程

```bash
python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --input outputs/test_prompts.jsonl \
  --output outputs/test_predictions_qwen7b.jsonl

python scripts/format_submission.py \
  --input outputs/test_predictions_qwen7b.jsonl \
  --output outputs/submission_qwen7b.json
```

## Day 4 衔接

下一步进入 SFT 数据构造和微调准备。重点是只使用官方训练数据，构造带推理过程的监督微调样本，并建立可复现的训练、评测和提交记录。
