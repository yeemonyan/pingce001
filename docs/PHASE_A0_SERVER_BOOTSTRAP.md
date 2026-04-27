# Phase A0: Server Bootstrap

目标：把 AutoDL 服务器变成一个可复现的 SCoRE2026 工作环境，并完成一次最小冒烟验证。

## 目录规划

建议使用如下目录：

```text
/root/autodl-tmp/pingce001/
├── data/
│   └── raw/
├── models/
├── outputs/
└── .venv/
```

原因：

- `/root/autodl-tmp` 有独立 50G 空间，适合放代码、数据和 7B 模型。
- GitHub 只备份代码与配置，不提交官方数据、模型权重和大输出。

## 建议执行步骤

1. `git clone` 共享仓库到 `/root/autodl-tmp/pingce001`
2. 切到工作分支
3. 运行 `bash scripts/bootstrap_autodl.sh`
4. 下载官方数据到 `data/raw/`
5. 下载主模型 `Qwen/Qwen2.5-7B-Instruct` 到 `models/`
6. 运行解析脚本和最小推理脚本做冒烟验证

## 参考命令

```bash
cd /root/autodl-tmp
git clone https://github.com/yeemonyan/pingce001.git
cd pingce001
git checkout codex/phase-a0-bootstrap
bash scripts/bootstrap_autodl.sh
```

官方数据仓库：

```bash
git clone https://github.com/PKU-SpaCE/SCoRE2026.git /root/autodl-tmp/SCoRE2026
cp /root/autodl-tmp/SCoRE2026/data/SCoRE2026_trainset.json data/raw/train.json
cp /root/autodl-tmp/SCoRE2026/data/SCoRE2026_testset.json data/raw/test.json
```

模型下载：

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_model.py \
  --repo-id Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct
```

数据解析冒烟验证：

```bash
source .venv/bin/activate
python scripts/parse_score_json.py \
  --input data/raw/train.json \
  --output outputs/train_prompts.jsonl
```

## 通过标准

- 虚拟环境创建成功
- `torch.cuda.is_available()` 为 `True`
- 官方训练集解析成功
- 模型权重开始正常下载或已成功下载
- 代码和文档已经 push 到 GitHub 分支，服务器随时可恢复

## AutoDL 备注

如果当前镜像的 `/root/miniconda3/bin/python` 已自带可用的 CUDA 版 PyTorch，
`scripts/bootstrap_autodl.sh` 会优先复用它，而不会重复下载整套 torch 轮子。
这样能明显减少启动时间和计费时长。
