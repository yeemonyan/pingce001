# SCoRE2026 dense 路线试错时间线汇报

本文档用于给 mqh 小组同步 SCoRE2026 评测中 dense 模型路线的工作。这里的 dense 路线主要包括两部分：一是 DeepSeek-R1-Distill-Qwen-7B 方向，二是 Qwen2.5-7B-Instruct 方向。我们后续主线已经转向 Qwen2.5，因此本文重点保留每一步尝试、失败原因、关键数据和可复用资源。

## 1. 起点：任务理解与官方数据整理

最开始我们先阅读了 SCoRE2026 官方页面和提交要求，明确任务是基于场景的常识推理选择题，输出需要符合官方提交 JSON 格式。任务题型覆盖 spatial、temporal、social、natural、hybrid 等场景类型，既有单选也有多选。

随后在本地和服务器上建立了基础目录：

- `data/`：保存官方训练集、测试集，以及转换后的中间格式。
- `model/`：保存本地模型权重。
- `model/LoRA/`：保存 LoRA 微调结果。
- `submit/`：保存官方测试集推理后的提交文件。
- `scripts/`：保存数据转换、训练、推理、评估脚本。

早期完成了官方训练集和测试集的 JSON/JSONL 转换，目标是统一后续训练、dev 评估和官方提交格式。这个阶段主要解决的是工程链路问题：数据能读、格式能转、提交文件能校验。

## 2. 第一轮 zero-shot：模型候选比较

初始候选模型包括：

- `Qwen3-30B-A3B`
- `Qwen3-8B`
- `DeepSeek-R1-Distill-Qwen-7B`

因为服务器磁盘和显存一开始不稳定，我们先判断是否能同时下载多个模型。如果空间不足，就采用“下载一个、测试一个、记录分数、再删除”的策略。

在 zero-shot 阶段，`DeepSeek-R1-Distill-Qwen-7B` 的表现相对更好，并且模型规模在 7B 级别，符合后续官方说明中 7-8B 模型可用的限制。因此我们先把 DeepSeek 作为第一条主试验线。

这一阶段的重要结论是：直接上更大的 MoE 模型并不现实。`Qwen3-30B-A3B` 虽然参数总量大，但部署成本高、推理慢，对当时资源不友好，因此后续暂时放弃 MoE 路线，把 dense 7B/8B 作为主要路线。

## 3. DeepSeek 路线：从推理数据到 LoRA 微调

### 3.1 推理过程数据构造

因为 SCoRE2026 是场景推理任务，我们一开始希望模型先判断场景类型，再采用对应推理方法，最后输出答案。于是设计了“带推理过程”的训练格式：让样本中包含 rationale 或 scratchpad，再让模型学习在最终答案前进行中间推理。

我们使用 DeepSeek API 为官方训练集生成带推理过程的版本。早期脚本遇到几个问题：

- API 请求失败后 retry 时间太长，单个失败样本可能拖慢 4-5 分钟。
- 有些失败不是偶发问题，重试 5 次仍然失败。
- API key 如果被复制进环境变量时带换行，会报 `Invalid header value b'Bearer ...\n'`。
- 输出有时不稳定，容易出现格式漂移。

针对这些问题，脚本做过几轮修改：

- 对 API key 做 `strip()`，避免尾部换行导致 header 非法。
- 增加实时进度输出，显示成功、失败、平均耗时和 ETA。
- 支持断点续跑，避免长任务中断后覆盖已有结果。
- 后续把 retry 改弱，甚至失败后直接跳过，以免失败样本长时间阻塞。

最初只得到 600 多条成功样本，后来继续跑完整训练集，最终生成约 3600 条带推理过程的训练数据。

### 3.2 DeepSeek rationale LoRA

拿到 600 多条样本后，先做了一版小规模 LoRA smoke test，确认训练链路能跑通。随后用扩展后的约 3600 条 rationale 数据做 DeepSeek LoRA 微调。

训练过程中 loss 从约 14 下降到 0.27，表面上模型已经很好地拟合了训练格式。但官方测试集反馈并没有提升，甚至出现下降：

- 微调后测试集准确率一度只有约 7.2%。

我们复盘后认为，这不是简单的“训练不够”，而更像是训练目标和最终评测目标不一致：

- rationale 质量参差不齐，错误推理会被模型学进去。
- 长自然语言推理容易让模型学会“写过程”，但不一定提高选项判断。
- 最终提交只看答案，训练中过多关注推理文本可能牺牲答案格式和答案准确率。
- 低 loss 只说明模型拟合了训练分布，不代表推理泛化能力提高。

### 3.3 DeepSeek answer-only 对照

在队友建议下，我们又做了 answer-only LoRA 对照实验。这个方向的思路是：先不训练长推理过程，只训练模型稳定输出最终答案 JSON，减少格式漂移。

对应脚本包括：

- `scripts/build_score2026_answer_only_sft.py`
- `scripts/finetune_score2026_answer_only_lora.py`
- `scripts/run_score2026_answer_only_submit.py`
- `scripts/evaluate_score2026_answer_only.py`

DeepSeek answer-only dev 集结果：

- dev accuracy：`0.3333333333333333`

这个结果比 rationale LoRA 的官方测试反馈稳定很多，说明“先稳定输出答案格式”比“直接训练长推理”更可靠。但它仍然没有形成明显优势，因此 DeepSeek 后续没有继续作为主线。

DeepSeek 路线的阶段性结论：

- DeepSeek-R1-Distill-Qwen-7B 可以作为可用 baseline。
- answer-only 训练比长 rationale 训练更稳。
- 直接用外部模型生成长推理再 SFT，收益不稳定，且容易过拟合到推理表述。
- 后续不建议继续把主要算力投入 DeepSeek 路线。

## 4. GitHub 旧仓库资源复用

用户提供了 GitHub 仓库 `yeemonyan/pingce001.git`。我们检查后发现，仓库中已经有一套之前训练 Qwen2.5 的过程文件，尤其是 `short-reasoning-sft` 分支相关内容可以复用。

可复用资源主要包括：

- `configs/system_prompts_v2.yaml`
- `configs/train_star_structured_scratchpad.yaml`
- `scripts/build_sft_data.py`
- `scripts/infer_score.py`
- `scripts/evaluate_score.py`
- `scripts/format_submission.py`
- `docs/ERROR_ANALYSIS_BASELINE.md`

其中最关键的是 `configs/system_prompts_v2.yaml`。它不是让模型自由写长 CoT，而是按题型给出更短、更结构化的解题约束，例如：

- spatial：先建空间布局，注意方向、朝向、左右参照系。
- temporal：列出时间变量、相对日期、星期偏移，再逐项验证。
- social：构建人物关系图，处理角色别名和反向关系。
- natural：建立物品、类别、属性、功能表。
- hybrid：先拆成多个子领域，再合并约束。

这个配置后来成为 Qwen2.5 路线的重要参考。我们从 DeepSeek 的“长推理 SFT”转向 Qwen2.5 的“短推理、结构化 scratchpad、答案格式稳定”路线。

## 5. Qwen2.5 路线：从已有 LoRA 继续推进

### 5.1 服务器和模型资源迁移

Qwen2.5 相关训练资源原先在另一台服务器上。我们检查到已有 checkpoint：

- `checkpoints/qwen2p5_7b_lora_answer_only_seed2026`
- `checkpoints/qwen2p5_7b_lora_reasoning_short_seed2026`
- `checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026`

后续用户指出，STaR + structured scratchpad 应该接在 Qwen2.5 经过 LoRA 微调后的模型上，而不是接在 Qwen3-8B 或未微调模型上。于是我们把需要的资源迁移到原 DeepSeek 服务器：

- `model/Qwen2.5-7B-Instruct`
- `model/LoRA/qwen2p5_7b_lora_reasoning_short_seed2026`
- Qwen2.5 相关脚本和配置

这里的关键调整是：STaR 不是从 base model 直接做，而是从已有 `qwen2p5_7b_lora_reasoning_short_seed2026` 继续做 SFT。

### 5.2 STaR + structured scratchpad 生成

我们实现了脚本：

- `scripts/star_generate_structured_scratchpad.py`

脚本逻辑：

1. 输入官方训练集构造的 answer-only train JSONL。
2. 使用 `Qwen2.5-7B-Instruct + qwen2p5_7b_lora_reasoning_short_seed2026` 生成候选 structured scratchpad。
3. 模型在 prompt 中看不到 gold answer。
4. 解析模型输出的 JSON。
5. 只有当模型生成答案与 gold answer 完全一致时，才把该样本保留为 STaR SFT 数据。

生成结果：

- 输入训练样本：`2880` 条。
- 候选输出：`outputs/star_structured_candidates_qwen25_train.jsonl`，共 `2880` 条。
- 过滤后 SFT 数据：`outputs/star_structured_sft_qwen25_train.jsonl`，共 `660` 条。
- keep rate：`660 / 2880 = 0.2292`。
- JSON ok rate：`0.9128`。
- repaired scratchpads：`53 / 660 = 8%`。

这个结果说明：Qwen2.5 短推理 LoRA 能在约 23% 的训练样本上自己生成正确答案和可解析结构化过程，但覆盖率不高。

### 5.3 在 short-reasoning LoRA 上继续训练

我们实现了 continuation SFT 脚本：

- `scripts/finetune_star_structured_from_lora.py`

训练设置：

- base model：`model/Qwen2.5-7B-Instruct`
- 初始 LoRA：`model/LoRA/qwen2p5_7b_lora_reasoning_short_seed2026`
- 训练数据：`outputs/star_structured_sft_qwen25_train.jsonl`
- 输出 LoRA：`model/LoRA/qwen2p5_7b_lora_star_structured_seed2026`
- 训练样本：`660`
- train/valid：`594 / 66`
- epochs：`2`
- learning rate：`5e-5`

训练跑通后，我们用 dev 集评估。

### 5.4 Qwen2.5 STaR dev 结果

STaR structured scratchpad LoRA 的 dev 评估结果：

- overall：`235 / 720 = 0.3263889`
- natural：`0.38`
- social：`0.52`
- spatial：`0.25`
- temporal：`0.265`
- hybrid：`0.20`
- single-answer：`0.3734`
- multi-answer：`0.1754`

这个结果没有超过预期，也没有明显优于 answer-only 对照。尤其是：

- multi-answer 表现很弱，只有 `0.1754`。
- hybrid 表现较低，只有 `0.20`。
- temporal 和 spatial 仍然是主要短板。
- social 相对最好，说明关系图类 prompt/训练较容易被模型吸收。

因此我们判断：这一版 STaR structured scratchpad 不值得直接跑官方测试集提交，除非只是做实验记录。

## 6. 失败和问题复盘

### 6.1 为什么 rationale LoRA loss 降了但准确率没升

DeepSeek rationale LoRA 中 loss 从 14 降到 0.27，但准确率下降，说明模型学到的是“训练输出形式”，不是更强的任务推理能力。

主要问题：

- 生成的 rationale 不一定真实可靠，错误推理会污染训练。
- 模型可能优先模仿推理文本，而不是优化选项选择。
- 训练目标和评测目标不一致：训练看完整文本 loss，评测只看答案。
- 长输出增加了解析失败和格式漂移概率。

### 6.2 为什么 STaR structured scratchpad 没提升

STaR 的理论优势是让模型学习自己能做对的推理轨迹，但本次效果有限。

可能原因：

- 保留下来的正确样本只有 660 条，数据量偏小。
- 保留样本天然偏向模型已经会的题，不能覆盖真正困难样本。
- 对错误题没有引入外部纠错或 verifier，模型不会从“接近但错误”的轨迹中学习。
- structured scratchpad 仍然是生成式目标，最终答案准确率未必直接受益。
- 多选题和 hybrid 题需要更强的候选比较能力，仅靠 SFT 不够。

### 6.3 为什么 answer-only 反而更稳

answer-only 的优势是目标更贴近提交格式：

- 输出短，解析稳定。
- 训练目标和评测目标一致。
- 不容易被长推理文本带偏。

但它的上限也有限，因为没有显式训练复杂推理能力。当前看它适合作为稳定 baseline，而不是最终最优路线。

## 7. 当前 dense 路线资产清单

当前本地仓库中保留的主要资产：

- `configs/system_prompts_v2.yaml`：按题型划分的推理 prompt。
- `configs/train_star_structured_scratchpad.yaml`：STaR structured scratchpad 训练配置。
- `scripts/build_score2026_answer_only_sft.py`：answer-only SFT 数据构造。
- `scripts/build_score2026_rationale.py`：调用外部 API 生成 rationale 数据。
- `scripts/finetune_score2026_answer_only_lora.py`：answer-only LoRA 微调。
- `scripts/run_score2026_answer_only_submit.py`：LoRA 推理并生成 dev/submit 输出。
- `scripts/evaluate_score2026_answer_only.py`：dev 集评估。
- `scripts/star_generate_structured_scratchpad.py`：STaR structured scratchpad 生成与过滤。
- `scripts/finetune_star_structured_from_lora.py`：从已有 Qwen2.5 LoRA 继续做 STaR SFT。
- `scripts/validate_score2026_submission.py`：提交格式校验。

服务器侧关键模型资产：

- `model/Qwen2.5-7B-Instruct`
- `model/LoRA/qwen2p5_7b_lora_reasoning_short_seed2026`
- `model/LoRA/qwen2p5_7b_lora_star_structured_seed2026`
- DeepSeek answer-only LoRA 和 rationale LoRA 的历史输出

GitHub 旧仓库中最值得继续用的是 Qwen2.5 的 `short-reasoning-sft` 分支相关脚本、prompt 和历史 checkpoint。

## 8. 当前判断和下一步建议

截至目前，dense 路线里最值得保留的结论是：

1. DeepSeek 路线已经跑通完整链路，但不是当前主线。
2. DeepSeek answer-only dev accuracy 为 `0.3333`，可作为对照。
3. Qwen2.5 STaR structured scratchpad dev accuracy 为 `0.3264`，没有明显提升。
4. 直接训练长 rationale 风险较大，容易出现 loss 好看但准确率下降。
5. Qwen2.5 的短推理 LoRA 和题型 prompt 仍然是后续最有价值的基础。

建议下一阶段不要继续盲目加长 CoT，而是优先做以下方向：

### 8.1 selective self-consistency

只在困难题型上做多采样投票，而不是全量增加推理成本。优先覆盖：

- temporal
- spatial
- hybrid
- multi-answer

做法：

- 对简单题保持 greedy 或 temperature 0。
- 对困难题生成 3-5 个候选答案。
- 只对 `answers` 投票，不投票长推理文本。
- 用 dev 集比较是否提升 multi-answer、temporal、hybrid。

### 8.2 outcome verifier / reranker

训练一个轻量 verifier，只判断“题目 + 候选答案”是否合理，不要求生成完整推理。

数据构造方式：

- 正样本：官方 gold answer。
- 负样本：从错误候选、随机错误选项、多选漏选/多选构造。
- 推理阶段：主模型生成 3-5 个候选，verifier 选分最高的答案。

这比训练长 reasoning verifier 更轻量，也更贴近最终准确率。

### 8.3 domain-targeted SFT

不要平均训练所有题型，而是针对 dev 错误多的题型加权：

- temporal
- spatial
- hybrid
- multi-answer

social 当前相对较好，不宜过度训练，避免牺牲已有优势。

### 8.4 保持 answer-only 提交头

即使内部使用 scratchpad 或投票，最终输出仍建议强制压成：

```json
{"answers":["A"]}
```

提交格式稳定性应该作为硬约束，而不是依赖模型自由输出。

## 9. 一句话总结

这条 dense 路线已经验证了 DeepSeek 和 Qwen2.5 两套 7B 级模型的完整数据、训练、推理、评估链路；失败经验表明，直接堆长推理 SFT 不可靠，当前更可行的方向是以 Qwen2.5 short-reasoning LoRA 为底座，结合题型 prompt、answer-only 稳定输出、困难题 selective voting 和轻量 verifier 来提高准确率。
