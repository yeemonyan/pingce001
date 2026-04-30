# Next Phase Plan

第一次线上提交已经完成，Qwen2.5-7B-Instruct zero-shot baseline 的线上 ACC 为 7.9%。接下来目标是从“流程跑通”转入“系统提分”。

## Phase B: SFT 数据构造

目标：把官方训练集转换为可监督微调的数据格式。

- 读取 `data/raw/train.json`。
- 统一字段：`id`、`text`、`question`、`options`、`answers`。
- 生成对话样本：
  - system：按空间、时间、社会、自然、融合选择任务提示词。
  - user：题干、问题、选项。
  - assistant：推理过程 + 最终答案 JSON。
- 不使用测试集内容作为训练样本、提示示例或伪标签。
- 输出：
  - `outputs/sft_train.jsonl`
  - `outputs/sft_valid.jsonl`
  - `outputs/sft_data_report.json`

## Phase C: LoRA 微调

目标：训练第一个可复现的 Qwen2.5-7B LoRA adapter。

- 推荐底座：`Qwen/Qwen2.5-7B-Instruct`。
- 推荐起始参数：
  - LoRA rank: 8 或 16
  - learning rate: `1e-4`
  - epoch: 3
  - seed: 2026
- 保存训练配置、日志、adapter、评测报告。
- 使用带答案数据评估整体 ACC 和分领域 ACC。

## Phase D: 错题分析

目标：找出主要失分来源并做专项改进。

- 按领域统计：spatial、temporal、social、natural、hybrid、general。
- 按错误类型统计：
  - 单选变多选
  - 多选漏选
  - 方向/左右/上下参考系错误
  - 时间差和星期循环错误
  - 社会关系别名和逆关系错误
  - 自然属性分类错误
- 输出错题样例报告，作为下一轮 prompt 和训练数据优化依据。

## Phase E: 第二次线上提交

目标：提交微调后模型的测试集预测。

- 使用同一份 `test_prompts.jsonl` 推理。
- 生成官方 JSON 数组，字段必须为 `id` 和 `answers`。
- 上传前校验：
  - 1000 条
  - id 从 `SCoRE2026-test-1` 到 `SCoRE2026-test-1000`
  - 每个 `answers` 非空
  - 答案只包含 `A/B/C/D`
- 记录线上 ACC、提交文件、模型版本和复现命令。
