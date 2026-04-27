# SCoRE2026 Baseline

本仓库用于第一届基于情景的常识推理评测任务（SCoRE2026）的 12 天落地计划。当前已经完成 Day 1 到 Day 3：环境与模型入口、五类任务提示词与推理脚本、官方提交格式与本地评测链路。

## 任务约束
Day 1 的目标是完成基础环境、合规模型选型/下载入口，以及官方 JSON 数据解析脚本。评测约束来自任务说明：Dense 模型总参数量不超过 8B；训练/微调阶段只能使用 SCoRE2026 官方数据；测试集不得用于提示示例、伪标签或人工作答。

- Dense 模型总参数量不得超过 8B。
- 训练和微调阶段只使用 SCoRE2026 官方公开数据；当前按公开仓库的 `train/test` 文件组织，若后续官方提供验证集再单独接入。
- 测试集不得用于提示词示例、伪标签生成或人工作答。
- 最终结果需要可复现：模型、脚本、随机种子、推理参数和提交文件都要保留。

## Day 1 完成内容

Day 1 目标是完成环境配置、模型选型、模型下载入口和官方数据解析脚本。

已完成：

- 新增 Linux CUDA/PyTorch 环境脚本：`scripts/setup_linux_cuda.sh`
- 固定默认合规模型：`Qwen/Qwen2.5-7B-Instruct`
- 新增模型下载脚本：`scripts/download_model.py`
- 新增 SCoRE JSON/JSONL 解析脚本：`scripts/parse_score_json.py`
- 建立 `configs/`、`data/`、`models/`、`outputs/`、`scripts/`、`tests/`、`docs/` 项目结构
- 添加 `.gitignore`，避免提交官方数据、模型权重和推理输出
- 添加 Day 1 进度文档：`docs/DAY1_PROGRESS.md`

## Day 2 完成内容

Day 2 目标是构建 CoT Zero-shot baseline：针对空间、时间、社会、自然、融合常识五类任务分别设计 System Prompt，并编写推理脚本。

已完成：

- 新增五类 System Prompt：`configs/system_prompts.yaml`
- 新增推理脚本：`scripts/infer_score.py`
- 支持 `mock` 后端，用于本地无模型烟测
- 支持 `transformers` 后端，用于云服务器加载本地 7B 模型推理
- 支持答案抽取、非法输出清洗、带答案数据的 ACC 计算
- 新增五类 smoke 样例：`data/day2_smoke.json`
- 添加 Day 2 测试与进度文档：`tests/test_infer_score.py`、`docs/DAY2_PROGRESS.md`

## Day 3 完成内容

Day 3 目标是完成格式对齐与首次测评链路：把模型输出转为官方提交文件，并编写本地评测脚本。

已完成：

- 新增官方提交格式脚本：`scripts/format_submission.py`，默认生成更保守的 `jsonl_with_id` 格式
- 新增本地评测脚本：`scripts/evaluate_score.py`
- 支持按 `id` 或顺序对齐 gold/prediction
- 输出整体 ACC、分领域 ACC、缺失预测数和错题样例
- 解析脚本保留 `domain/category/type/task_type` 元数据，便于分领域统计
- 添加 Day 3 测试与进度文档：`tests/test_format_submission.py`、`tests/test_evaluate_score.py`、`docs/DAY3_PROGRESS.md`

## 快速开始

### 1. 配置环境

在 4090/A800 Linux 云服务器上执行：

```bash
bash scripts/setup_linux_cuda.sh
```

脚本会创建 `.venv` 并安装 PyTorch、Transformers、Accelerate、Hugging Face Hub 等依赖。

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
  --input data/train.json \
  --output outputs/train_prompts.jsonl
```

输出为 JSONL，每行包含 `id`、`text`、`question`、`options`、`answer`、`has_answer`、`prompt` 等字段。
脚本兼容官方训练集里的 `answers` 字段，并统一输出为 JSONL。每行包含 `id`、`prompt`、`answer`、`has_answer` 等字段，可直接作为 Day 2 零样本推理脚本的输入。

### 4. 本地烟测完整链路

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

当前 smoke 链路预期结果：

- 总样本数：5
- 覆盖领域：空间、时间、社会、自然、融合
- ACC：1.0

### 5. 运行真实模型推理

云服务器下载模型后，将 `--backend mock` 改为 `--backend transformers`：

```bash
python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --input outputs/test_prompts.jsonl \
  --output outputs/test_predictions.jsonl
```

生成官方提交文件。默认输出 `jsonl_with_id`，每行一个对象，包含 `id` 和 `answer`：

```bash
python scripts/format_submission.py \
  --input outputs/test_predictions.jsonl \
  --output outputs/submission_day3.jsonl
```

默认提交格式：

```jsonl
{"id":"example-1","answer":["A"]}
{"id":"example-2","answer":["A","B"]}
```

如果线上系统要求 JSON 数组格式，可显式切换：

```bash
python scripts/format_submission.py \
  --official-format system_json \
  --input outputs/test_predictions.jsonl \
  --output outputs/submission_day3.json
```

## 项目结构

```text
configs/                 模型配置与五类 System Prompt
data/                    本地数据目录，官方数据不提交 Git
docs/                    Day 1-3 进度与服务器部署文档
models/                  本地模型权重目录，不提交 Git
outputs/                 解析、推理、评测、提交输出目录，不提交 Git
scripts/                 环境、下载、解析、推理、评测、提交脚本
tests/                   单元测试与 smoke 链路测试
```

## 测试

```bash
python -m unittest discover -s tests -v
```

当前本地验证通过：17 个测试全部 OK。

## 文档索引

- Day 1 进度：`docs/DAY1_PROGRESS.md`
- Day 2 进度：`docs/DAY2_PROGRESS.md`
- Day 3 进度：`docs/DAY3_PROGRESS.md`
- AutoDL 部署：`docs/PHASE_A0_SERVER_BOOTSTRAP.md`

## 后续计划

- 在服务器上拉取集成分支并接入真实模型推理。
- 用官方带答案数据跑真实模型推理并记录 ACC。
- 用官方测试集生成 `outputs/submission_day3.jsonl`。
- 上传官方评测系统，完成第一次提交。
- Day 4 开始构造 SFT 数据，准备监督微调。
