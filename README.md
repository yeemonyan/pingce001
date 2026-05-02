# SCoRE2026 Baseline

本仓库用于第一届基于情景的常识推理评测任务（SCoRE2026）的参赛工程。当前第一阶段已经完成：环境与模型入口、五类任务提示词、推理脚本、官方提交格式、本地链路和第一次线上评测试验。

## 当前状态

- 主模型：`Qwen/Qwen2.5-7B-Instruct`
- 模型规模：7B Dense，满足 Dense 模型不超过 8B 的规则
- 官方训练集：3600 题
- 官方测试集：1000 题
- 第一次线上提交：zero-shot baseline
- 第一次线上 ACC：7.9%
- 第一次提交文件：`submissions/score2026_first_submission_qwen7b.json`

## 任务约束

- Dense 模型总参数量不得超过 8B。
- 训练和微调阶段只使用 SCoRE2026 官方公开数据。
- 测试集不得用于提示词示例、伪标签生成或人工作答。
- 最终结果需要可复现：模型、脚本、随机种子、推理参数和提交文件都要保留。

## 已完成内容

### Day 1: 环境与数据入口

- 建立项目目录：`configs/`、`data/`、`models/`、`outputs/`、`scripts/`、`tests/`、`docs/`。
- 新增 Linux CUDA/PyTorch 环境脚本：`scripts/setup_linux_cuda.sh`。
- 固定默认合规模型：`Qwen/Qwen2.5-7B-Instruct`。
- 新增模型下载脚本：`scripts/download_model.py`。
- 新增 SCoRE JSON/JSONL 解析脚本：`scripts/parse_score_json.py`。
- 添加 `.gitignore`，避免提交官方数据、模型权重和大输出。

### Day 2: Zero-shot 推理链路

- 新增五类 System Prompt：`configs/system_prompts.yaml`。
- 新增推理脚本：`scripts/infer_score.py`。
- 支持 `mock` 后端，用于本地无模型烟测。
- 支持 `transformers` 后端，用于云服务器加载本地 7B 模型推理。
- 支持答案抽取、非法输出清洗、带答案数据的 ACC 计算。
- 新增五类 smoke 样例：`data/day2_smoke.json`。

### Day 3: 提交格式与首次线上评测

- 新增官方提交格式脚本：`scripts/format_submission.py`。
- 平台提交格式为 JSON 数组，每条包含 `id` 和 `answers`：

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

- 新增本地评测脚本：`scripts/evaluate_score.py`。
- 使用服务器上的 Qwen2.5-7B-Instruct 跑完 1000 条测试集。
- 完成第一次线上提交，ACC 为 7.9%。

## 快速开始

### 1. 配置环境

```bash
bash scripts/setup_linux_cuda.sh
```

### 2. 下载模型

```bash
python scripts/download_model.py \
  --repo-id Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct
```

国内服务器可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 解析官方数据

```bash
python scripts/parse_score_json.py \
  --input data/raw/train.json \
  --output outputs/train_prompts.jsonl
```

输出为 JSONL，每行包含 `id`、`text`、`question`、`options`、`answer`、`has_answer`、`prompt` 等字段。
脚本兼容官方训练集里的 `answers` 字段，并统一输出为 JSONL。每行包含 `id`、`prompt`、`answer`、`has_answer` 等字段，可直接作为 Day 2 零样本推理脚本的输入。

### 4. 本地烟测完整链路

python scripts/parse_score_json.py \
  --input data/raw/test.json \
  --output outputs/test_prompts.jsonl
```

### 4. 运行推理

```bash
python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --input outputs/test_prompts.jsonl \
  --output outputs/test_predictions_qwen7b.jsonl
```

### 5. 生成官方提交文件

```bash
python scripts/format_submission.py \
  --input outputs/test_predictions_qwen7b.jsonl \
  --output outputs/submission_qwen7b.json
```

如需保留旧的 JSONL 调试格式：

```bash
python scripts/format_submission.py \
  --official-format jsonl_with_id \
  --input outputs/test_predictions_qwen7b.jsonl \
  --output outputs/submission_qwen7b.jsonl
```

## 项目结构

```text
configs/                 模型配置与五类 System Prompt
data/                    本地数据目录，官方数据不提交 Git
docs/                    阶段任务大纲与完成进度
models/                  本地模型权重目录，不提交 Git
outputs/                 解析、推理、评测、提交输出目录，不提交 Git
scripts/                 环境、下载、解析、推理、评测、提交脚本
submissions/             已完成线上提交的可复现 JSON 文件
tests/                   单元测试与 smoke 链路测试
```

## 测试

```bash
python -m unittest discover -s tests -v
```

## 下一阶段规划

### Phase B: 训练数据与 SFT 准备

- 构造 SFT 数据：将官方训练集转换为 Instruction/Input/Output 格式。
- Output 不只放答案，要包含可复现的推理过程和最终 `answers`。
- 按领域统计样本：空间、时间、社会、自然、融合。
- 建立训练/验证切分，不能使用测试集进行任何训练或提示示例。

### Phase C: LoRA 微调

- 使用 Qwen2.5-7B-Instruct 做 LoRA/SFT。
- 记录训练参数：学习率、epoch、batch size、LoRA rank、seed。
- 每轮训练后在带答案数据上计算整体 ACC 和分领域 ACC。
- 保存 adapter、训练日志、配置和评测报告。

### Phase D: 错题分析与专项优化

- 汇总错误样例，按领域和错误类型分类。
- 优先处理低分领域：时间链、空间方向、社会关系、多选题。
- 调整 prompt、答案抽取和清洗规则。
- 对复杂题尝试 self-consistency 多次采样投票。

### Phase E: 第二次线上提交

- 使用微调后模型重新跑 1000 条测试集。
- 生成平台要求的 `id + answers` JSON 数组。
- 提交前做结构校验、id 顺序校验和答案合法性校验。
- 记录线上 ACC、提交文件、模型版本和完整复现命令。
