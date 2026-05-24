# MoE 路线操作汇报

更新时间：2026-05-24

## 1. 背景

在 dense 路线中，我们本地 dev 最好大约能到 `36%~38%`，但线上正式提交始终停留在 `14.4%~14.9%`，说明本地提升不能稳定迁移到榜单。因此我们开始把 `Qwen3-30B-A3B` 这条 MoE 路线作为突破方向，目标不是盲目重训，而是先验证：

1. `Qwen3-30B-A3B` 在本任务上是否真的比现有 dense 更有上限。
2. dense 阶段沉淀下来的 prompt / routing / 结构化输出策略，迁到 MoE 后是否有效。
3. 在当前算力约束下，MoE 是不是值得转成主线。

## 2. 我这边已经做过的 MoE 相关操作

### 2.1 增加了 MoE 基线运行脚本

已在仓库加入脚本：

- [scripts/run_moe_qwen3_30b_a3b_baseline.sh](/Users/xiaoyuzhang/Documents/Playground/pingce001/scripts/run_moe_qwen3_30b_a3b_baseline.sh)

这个脚本的作用是：

1. 以 `Qwen/Qwen3-30B-A3B-Instruct-2507` 作为默认模型。
2. 支持自动下载模型，优先尝试 `modelscope`，失败后回退到 `huggingface_hub`。
3. 使用现有 `scripts/infer_score.py` + `scripts/evaluate_score.py` 跑 dev baseline。
4. 默认走 `transformers` 后端、`temperature=0`、`answer-count-hint`，尽量和 dense 评测口径一致。

### 2.2 给现有推理脚本补了 MoE 兼容逻辑

在 [scripts/infer_score.py](/Users/xiaoyuzhang/Documents/Playground/pingce001/scripts/infer_score.py) 里，已经有 MoE 相关兼容处理：

- `transformers.integrations.moe` 的 grouped mm 兜底
- `MOE_EXPERTS_IMPLEMENTATION` 环境变量控制

这一步的目的，是减少 `Qwen3-30B-A3B` 在本地 / 服务器上直接跑 `transformers` 时的兼容性问题。

### 2.3 做了 dense vs MoE 的本地对照

已有对照结果文件：

- [outputs/compare_dense_mixed_vs_moe_fp8_vllm_local.json](/Users/xiaoyuzhang/Documents/Playground/pingce001/outputs/compare_dense_mixed_vs_moe_fp8_vllm_local.json)

核心结果：

- dense mixed：`277 / 720 = 0.3847`
- moe fp8：`211 / 720 = 0.2931`
- delta：`-0.0917`

这个结果说明：

1. 当前这版本地 MoE 方案没有直接超过 dense mixed。
2. 问题不是“MoE 一上来就天然更强”，而是当前 MoE 运行设置、prompt 迁移和推理实现还没调顺。
3. 不能因为排行榜上 MoE 强，就默认我们手里的 MoE 当前配置也一定强。

### 2.4 做了基于题型的 dense/MoE 路由实验

已有结果文件：

- [outputs/dev_route_dense_mixed_moe_goldkind_report.json](/Users/xiaoyuzhang/Documents/Playground/pingce001/outputs/dev_route_dense_mixed_moe_goldkind_report.json)
- [outputs/dev_route_dense_mixed_moe_predkind_report.json](/Users/xiaoyuzhang/Documents/Playground/pingce001/outputs/dev_route_dense_mixed_moe_predkind_report.json)

这组实验的核心想法是：

- baseline 仍然用 dense mixed。
- 只在少数候选 bucket 上切给 MoE。
- 当前候选 bucket 是：
  - `spatial:single`
  - `temporal:single`

结果如下：

1. 使用 gold kind 做理想路由时：
   - `284 / 720 = 0.3944`
   - 比 dense mixed 的 `0.3847` 小幅提升
2. 使用 candidate kind 做实际可落地路由时：
   - `264 / 720 = 0.3667`
   - 低于 dense mixed

这说明：

1. MoE 不是全局都强，只在个别 bucket 可能有局部收益。
2. 理想路由有增益，但真实路由误差一旦引入，就会把收益吃掉。
3. 后面如果继续做 MoE，不应该直接“全量替换 dense”，而应该优先做更稳的专项接管。

## 3. 当前从实验里得到的结论

### 3.1 当前 MoE 版本还不是可直接替换 dense 的主方案

从本地对照看，当前这版 `moe_fp8`：

- 整体低于 dense mixed
- 对 `social`、`natural`、`multi-choice` 还明显掉分

尤其在 `compare_dense_mixed_vs_moe_fp8_vllm_local.json` 里：

- `kind:multi`
  - dense mixed：`0.2339`
  - moe fp8：`0.0351`
- `domain:social`
  - dense mixed：`0.59`
  - moe fp8：`0.32`

这意味着当前 MoE 最大的问题不是“不会做难题”，而是多选和若干常识类题型的输出稳定性、选项约束和提示词迁移没有做好。

### 3.2 MoE 在部分 bucket 上存在局部潜力

从路由实验看，MoE 在以下桶里不是完全没用：

- `spatial:single`
- `temporal:single`

因为 gold 路由时整体是能略涨的。这说明：

1. MoE 的收益更可能出现在时空推理单选题，而不是全局收益。
2. 如果后面继续做，应该围绕“专项接管”而不是“全题替换”。

### 3.3 当前 MoE 的真正瓶颈是工程与配置，不只是模型能力

目前已经暴露出的 MoE 现实问题包括：

1. 推理成本高，不能像 7B dense 那样高频试错。
2. 当前配置下本地分数没有自动优于 dense。
3. 路由一旦依赖预测出来的题型 / 题目种类，就会带来误路由损失。
4. 多选题表现明显差，说明输出格式和答案数量约束没有迁好。

## 4. 服务器与迁移相关操作

### 4.1 做过的迁移工作

为了后续把旧机释放掉，我已经把旧服务器 `50890` 上的核心实验资产迁到了 `3090` 机器。

迁移后的主要目录：

- repo 快照：
  - `/home/pj/pingce001-moe/imports/old_50890/pingce001`
- LoRA 推理权重：
  - `/home/pj/pingce001-moe/imports/old_50890/checkpoints`
- 迁移说明：
  - `/home/pj/pingce001-moe/imports/old_50890/MIGRATION_NOTE.txt`

### 4.2 已迁过去的内容

已迁移：

- 代码仓库快照
- `.git` 元数据
- `data`
- `outputs`
- `docs`
- `scripts`
- `tests`
- `submissions`
- 4 份 LoRA 推理权重 `adapter_model.safetensors`

### 4.3 没有完整搬过去的内容

为了节省迁移时间，没有做 32G 严格全量镜像；未完整搬运的主要是：

- `models/DeepSeek-R1-Distill-Qwen-7B` 基座
- `.venv`
- 完整训练中间态 `checkpoint-*`
- `optimizer.pt` / `scheduler.pt` / `trainer_state.json` 等训练恢复文件

这是有意为之，因为这些文件对“继续推理 / 提交 / 复现实验结论”不是第一优先级。

### 4.4 在 3090 上做的模型路径处理

我已经把 3090 上现有的 Qwen 基座挂到导入目录：

- `/home/pj/pingce001-moe/imports/old_50890/pingce001/models/Qwen2.5-7B-Instruct`

它实际指向：

- `/home/pj/xukangzhe/Qwen2.5-7B-Instruct`

这样后续如果在 3090 上继续复现实验，不需要再重新下这份 Qwen 基座。

## 5. 当前 MoE 路线的真实状态

可以概括成一句话：

> MoE 方向已经完成了“脚本接入、基础实验、局部路由验证、服务器资产迁移”这四步，但还没有形成一条比 dense 明显更强、且可稳定提交的成熟主线。

更具体一点：

1. 有 baseline 脚本了。
2. 有本地对照结果了。
3. 有路由实验结果了。
4. 有服务器迁移和目录落点了。
5. 但当前分数表现还不足以直接证明“MoE 一定比 dense 强”。

## 6. 我建议后续继续 MoE 时的优先顺序

如果后面要继续做 MoE，我建议优先级如下：

1. 先修 MoE 多选题输出约束。
   目前最大掉分点之一就是 multi-choice。
2. 只在 `spatial:single` / `temporal:single` 这类局部有潜力的桶里试专项接管。
3. 不要一上来全量替换 dense。
4. 先把 `prompt + routing + output format` 调顺，再考虑更重的训练或蒸馏。
5. 如果资源紧张，优先让 MoE 作为“老师模型”生成更高质量 reasoning 数据，再反哺 dense。

## 7. 当前仓库里与 MoE 最相关的文件

- [scripts/run_moe_qwen3_30b_a3b_baseline.sh](/Users/xiaoyuzhang/Documents/Playground/pingce001/scripts/run_moe_qwen3_30b_a3b_baseline.sh)
- [outputs/compare_dense_mixed_vs_moe_fp8_vllm_local.json](/Users/xiaoyuzhang/Documents/Playground/pingce001/outputs/compare_dense_mixed_vs_moe_fp8_vllm_local.json)
- [outputs/dev_route_dense_mixed_moe_goldkind_report.json](/Users/xiaoyuzhang/Documents/Playground/pingce001/outputs/dev_route_dense_mixed_moe_goldkind_report.json)
- [outputs/dev_route_dense_mixed_moe_predkind_report.json](/Users/xiaoyuzhang/Documents/Playground/pingce001/outputs/dev_route_dense_mixed_moe_predkind_report.json)
- [docs/MOE_OPERATION_REPORT.md](/Users/xiaoyuzhang/Documents/Playground/pingce001/docs/MOE_OPERATION_REPORT.md)

## 8. 一句话总结

我这边在 MoE 上已经做完的工作，核心不是“训出了一个很强的新模型”，而是把这条路线从口头设想推进到了可执行、可复盘、可迁移的状态；但从现有结果看，MoE 还处在“有局部潜力、整体没打赢 dense”的阶段，后面必须按专项优化的方式继续，而不能把它当成自动提分按钮。
