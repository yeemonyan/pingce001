# SCoRE2026 Baseline

本仓库用于第一届基于情景的常识推理评测任务（SCoRE2026）的参赛工程。当前已经完成基础推理链路、第一次线上提交、稳定 dev split、SFT 数据构造、真实 dev baseline 评测和 baseline 错题分析。

## 当前状态

- 主模型：`Qwen/Qwen2.5-7B-Instruct`
- 模型规模：7B Dense，满足 Dense 模型不超过 8B 的规则
- 官方训练集：3600 题
- 官方测试集：1000 题
- 第一次线上测试提交：ACC `7.8%`
- Dev split：2880 train / 720 dev，seed `2026`
- Qwen zero-shot dev baseline：`131 / 720`，ACC `0.1819444444`
- 最低 dev 领域：`temporal`，ACC `0.105`

## 任务约束

- Dense 模型总参数量不得超过 8B。
- 训练和微调阶段只使用 SCoRE2026 官方公开数据。
- 测试集不得用于提示词示例、伪标签生成或人工作答。
- 最终结果需要可复现：模型、脚本、随机种子、推理参数和提交文件都要保留。

## 已完成

### Phase A: Baseline 链路

- 建立项目结构：`configs/`、`data/`、`models/`、`outputs/`、`scripts/`、`tests/`、`docs/`。
- 固定默认合规模型：`Qwen/Qwen2.5-7B-Instruct`。
- 新增环境与模型下载脚本：
  - `scripts/setup_linux_cuda.sh`
  - `scripts/bootstrap_autodl.sh`
  - `scripts/download_model.py`
- 新增数据解析、推理、评测与提交脚本：
  - `scripts/parse_score_json.py`
  - `scripts/infer_score.py`
  - `scripts/evaluate_score.py`
  - `scripts/format_submission.py`
- 完成第一次线上提交：
  - `submissions/score2026_first_submission_qwen7b.json`
  - 线上 ACC：`7.9%`

### Phase B0: Dev Split

- 新增稳定分层切分脚本：`scripts/make_dev_split.py`
- 切分维度：`domain`、`language`、单选/多选
- 固定随机种子：`2026`
- 产物：
  - `data/splits/train_ids.json`
  - `data/splits/dev_ids.json`
  - `outputs/dev_split_report.json`

### Phase B: SFT 数据

- 新增 SFT 数据构造脚本：`scripts/build_sft_data.py`
- 保留字段：`id`、`text`、`question`、`options`、`answers`、`domain`、`language`
- 生成两套训练/验证数据：
  - `outputs/sft_train_answer_only.jsonl`
  - `outputs/sft_valid_answer_only.jsonl`
  - `outputs/sft_train_rationale_json.jsonl`
  - `outputs/sft_valid_rationale_json.jsonl`
- 数据报告：`outputs/sft_data_report.json`

### Phase D: Dev Baseline 与错题分析

- 新增 dev 过滤脚本：`scripts/filter_split_records.py`
- 新增错题分析脚本：`scripts/analyze_errors.py`
- 新增 LoRA 训练与评测脚本：
  - `scripts/train_lora.py`
  - `scripts/run_lora_eval.py`
- Dev gold：`outputs/dev_prompts.jsonl`
- 真实 Qwen zero-shot dev baseline：
  - `outputs/dev_baseline_predictions.jsonl`
  - `outputs/dev_baseline_eval.json`
- 错题分析：
  - `outputs/error_report_baseline_dev.json`
  - `outputs/error_cases_baseline_dev.jsonl`
  - `docs/ERROR_ANALYSIS_BASELINE.md`
- 交付状态：`docs/DELIVERY_STATUS.md`

## Dev Baseline 结果

| Domain | Total | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| natural | 200 | 50 | 0.25 |
| spatial | 200 | 36 | 0.18 |
| temporal | 200 | 21 | 0.105 |
| social | 100 | 19 | 0.19 |
| hybrid | 20 | 5 | 0.25 |

主要错误类型：

- `single_to_multi`: 132
- `multi_missing`: 76
- `natural_property_error`: 78
- `output_format_error`: 49
- `spatial_reference_error`: 128
- `temporal_calculation_error`: 89
- `social_relation_error`: 23
- `multi_constraint_failure`: 14

## 快速开始

### 配置环境

```bash
bash scripts/setup_linux_cuda.sh
```

### 下载模型

```bash
python scripts/download_model.py \
  --repo-id Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct
```

国内服务器可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 解析官方数据

```bash
python scripts/parse_score_json.py \
  --input data/raw/train.json \
  --output outputs/train_prompts.jsonl

python scripts/parse_score_json.py \
  --input data/raw/test.json \
  --output outputs/test_prompts.jsonl
```

### 生成 dev gold

```bash
python scripts/filter_split_records.py \
  --input data/raw/train.json \
  --ids-file data/splits/dev_ids.json \
  --output outputs/dev_prompts.jsonl
```

### 跑 dev baseline

```bash
python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --input outputs/dev_prompts.jsonl \
  --output outputs/dev_baseline_predictions.jsonl

python scripts/evaluate_score.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_baseline_predictions.jsonl \
  --report outputs/dev_baseline_eval.json \
  --max-mistakes 100

python scripts/analyze_errors.py \
  --gold outputs/dev_prompts.jsonl \
  --pred outputs/dev_baseline_predictions.jsonl \
  --report outputs/error_report_baseline_dev.json \
  --cases outputs/error_cases_baseline_dev.jsonl \
  --doc docs/ERROR_ANALYSIS_BASELINE.md
```

### 生成官方提交文件

平台提交格式为 JSON 数组，每条包含 `id` 和 `answers`：

```json
[
  {
    "id": "SCoRE2026-test-1",
    "answers": ["A"]
  }
]
```

生成命令：

```bash
python scripts/format_submission.py \
  --input outputs/test_predictions_qwen7b.jsonl \
  --output outputs/submission_qwen7b.json
```

### Phase C: LoRA 训练

训练说明见：

- `docs/PHASE_C_LORA_PLAN.md`
- `docs/ROUND3_INFER_PLAN.md`

有卡环境下的最小流程：

```bash
python scripts/train_lora.py --config configs/train_lora.yaml --dry-run
python scripts/train_lora.py --config configs/train_lora.yaml
python scripts/run_lora_eval.py --config configs/train_lora.yaml
```

Round 3 优先不重训，先做推理侧对比实验，入口见：

- `docs/ROUND3_INFER_PLAN.md`
- `configs/system_prompts_round3_short.yaml`
- `scripts/analyze_lora_round3.py`

## 项目结构

```text
configs/                 模型配置与 System Prompt
data/                    本地数据目录，官方原始数据不提交 Git
docs/                    阶段进度、部署、交付和分析文档
models/                  本地模型权重目录，不提交 Git
outputs/                 可复现实验产物和报告
scripts/                 环境、下载、解析、推理、评测、提交、分析脚本
submissions/             已完成线上提交的 JSON 文件
tests/                   单元测试与 smoke 链路测试
```

## 测试

```bash
python -m unittest discover -s tests -v
```

当前本地验证通过：35 个测试全部 OK。

## 文档索引

- `docs/DAY1_PROGRESS.md`
- `docs/DAY2_PROGRESS.md`
- `docs/DAY3_PROGRESS.md`
- `docs/PHASE_BD_PROGRESS.md`
- `docs/DELIVERY_STATUS.md`
- `docs/ERROR_ANALYSIS_BASELINE.md`
- `docs/PHASE_A0_SERVER_BOOTSTRAP.md`
- `docs/PHASE_C_LORA_PLAN.md`

## 下一步

- 新建 `configs/system_prompts_v2.yaml`，优先优化 `temporal`、`social`、`hybrid`。
- 在 `docs/PROMPT_NOTES.md` 记录每次 prompt 版本变化和原因。
- 用 dev baseline 作为对照，比较 prompt v2 和后续 LoRA 模型。
- 开始 LoRA/SFT 训练，优先观察 temporal、多选漏选和过选问题。
