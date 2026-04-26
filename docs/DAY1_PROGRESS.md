# Day 1 任务大纲与完成进度

## 任务背景

SCoRE2026 是基于情景的常识推理评测，题目字段包括 `text`、`question`、`options`，训练/验证集含 `answer`，测试集不含 `answer`。指标为 Accuracy。参赛约束要求 Dense 模型总参数量不超过 8B，且不得在训练、微调、提示示例或伪标签中使用测试集。

## Day 1 目标

1. 配置 Linux/PyTorch 推理环境。
2. 选择符合规则的 7B/8B 级基座模型。
3. 提供可复现的模型下载脚本。
4. 编写 Python 脚本解析官方 JSON/JSONL 数据，抽取 `text`、`question`、`options` 和可选 `answer`。
5. 为 Day 2 的 CoT Zero-shot baseline 生成标准 prompt JSONL。

## 已完成

- [x] 建立项目目录：`configs/`、`data/`、`models/`、`outputs/`、`scripts/`、`tests/`、`docs/`。
- [x] 编写 Linux CUDA 环境脚本：`scripts/setup_linux_cuda.sh`。
- [x] 固定 Day 1 默认模型：`Qwen/Qwen2.5-7B-Instruct`。
- [x] 编写模型下载脚本：`scripts/download_model.py`。
- [x] 编写 SCoRE 数据解析脚本：`scripts/parse_score_json.py`。
- [x] 添加最小单元测试：`tests/test_parse_score_json.py`。
- [x] 添加仓库说明与 Git 忽略规则，避免提交官方数据、模型权重和输出文件。

## 待在云服务器执行

- [ ] 在 AutoDL/阿里云 4090 或 A800 Linux 实例运行 `bash scripts/setup_linux_cuda.sh`。
- [ ] 运行 `python scripts/download_model.py --repo-id Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct` 下载模型权重。
- [ ] 将官方训练/验证 JSON 放入本地 `data/`，运行解析脚本生成 Day 2 推理输入。

## Day 2 衔接

下一步应编写推理脚本，读取 `outputs/*_prompts.jsonl`，调用本地模型生成答案，并用验证集 `answer` 计算 ACC。
