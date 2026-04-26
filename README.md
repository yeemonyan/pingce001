# SCoRE2026 Day 1 Baseline

本仓库用于第一届基于情景的常识推理评测任务（SCoRE2026）的 12 天落地计划。

Day 1 的目标是完成基础环境、合规模型选型/下载入口，以及官方 JSON 数据解析脚本。评测约束来自任务说明：Dense 模型总参数量不超过 8B；训练/微调阶段只使用 SCoRE2026 官方训练集和验证集；测试集不得用于提示示例、伪标签或人工作答。

## Day 1 快速开始

### 1. 配置 Linux/PyTorch 环境

在 4090/A800 云服务器上执行：

```bash
bash scripts/setup_linux_cuda.sh
```

脚本会创建 `.venv` 并安装 PyTorch、Transformers、Accelerate、Hugging Face Hub 等 Day 1 依赖。

### 2. 下载合规模型

默认模型为 `Qwen/Qwen2.5-7B-Instruct`，属于 7B 级 Dense Instruct 模型，满足 Dense ≤ 8B 的参赛约束。

```bash
python scripts/download_model.py \
  --repo-id Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct
```

国内服务器可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 解析 SCoRE JSON/JSONL 数据

```bash
python scripts/parse_score_json.py \
  --input data/train.json \
  --output outputs/train_prompts.jsonl
```

输出为 JSONL，每行包含 `id`、`prompt`、`answer`、`has_answer` 等字段，可直接作为 Day 2 零样本推理脚本的输入。

## 项目结构

```text
configs/                 模型与推理默认配置
data/                    本地数据目录，不提交官方数据
docs/                    任务大纲与进度记录
models/                  本地模型权重目录，不提交 Git
outputs/                 解析/推理输出目录，不提交 Git
scripts/                 环境、下载、数据处理脚本
tests/                   最小单元测试
```

## 当前进度

详见 [docs/DAY1_PROGRESS.md](docs/DAY1_PROGRESS.md)。
