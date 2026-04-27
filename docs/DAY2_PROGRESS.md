# Day 2 任务大纲与完成进度

## Day 2 目标

Day 2 的核心目标是构建 CoT Zero-shot baseline：不训练模型，按空间、时间、社会、自然、融合常识五类任务分别设计 System Prompt，并编写推理脚本跑通验证集预测与 ACC 计算流程。

## Prompt 设计

已在 `configs/system_prompts.yaml` 中完成 6 组提示词：

- `spatial`：强调布局、方向、朝向、邻接、行列层和参考系。
- `temporal`：强调时间轴、相对/绝对时间、开始/结束年份和星期换算。
- `social`：强调人物关系图、角色别名、逆关系和选项逐项核验。
- `natural`：强调物品属性表、类别、功能、感官特征和排除法。
- `hybrid`：强调先拆分领域，再合并约束，处理多领域交互。
- `general`：兜底提示词，用于无法识别类别的数据。

## 推理脚本

已新增 `scripts/infer_score.py`：

- 读取 `parse_score_json.py` 生成的标准 JSONL。
- 根据 `domain/category/type/task_type` 字段选择对应 System Prompt。
- 若没有显式类别，使用关键词启发式识别五类任务。
- 支持 `--backend transformers` 调用本地 7B 模型。
- 支持 `--backend mock` 离线烟测完整流程。
- 自动提取模型输出中的 `answer` 字母列表。
- 对含 `answer` 的验证集计算 Accuracy。

## 验证方式

本地未下载 7B 权重，因此使用 mock 后端验证工程链路；云服务器下载模型后可切换到 Transformers 后端。

```bash
python scripts/parse_score_json.py \
  --input data/day2_smoke.json \
  --output outputs/day2_smoke_prompts.jsonl

python scripts/infer_score.py \
  --backend mock \
  --input outputs/day2_smoke_prompts.jsonl \
  --output outputs/day2_smoke_predictions.jsonl
```

## 已完成

- [x] 五大类 System Prompt 设计完成。
- [x] 推理脚本完成，支持真实模型与 mock 验证。
- [x] 答案抽取与 ACC 计算完成。
- [x] 加入 Day 2 smoke 数据。
- [x] 加入单元测试覆盖答案抽取、类别识别和 mock 推理流程。

## 下一步

Day 3 应重点处理官方提交格式清洗、验证集 ACC 报告和首次线上提交流程。
