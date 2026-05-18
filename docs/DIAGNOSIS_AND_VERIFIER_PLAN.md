# Diagnosis And Verifier Plan

## Current Diagnosis

目前不能把瓶颈简单归因为“Qwen2.5-7B 已经到上限”。

已有结果说明：

- zero-shot dev：`131 / 720 = 0.181944`
- LoRA dev：`261 / 720 = 0.3625`
- 第一次线上：`7.8%`
- 第二次线上：`14.9%`

这说明当前 7B 底座并非完全没有提升空间；真正的问题是：

1. dev 提升可以做出来，但线上泛化比例始终偏低
   - zero-shot：`0.078 / 0.181944 ≈ 0.429`
   - LoRA：`0.149 / 0.3625 ≈ 0.411`
   - 结论：目前方法在 dev 上能学到模式，但迁移到测试集时不稳定。

2. 当前主要错误不是“知识完全不会”，而是“任务建模方式不对”
   - LoRA dev 主错误：
     - `spatial_reference_error`: `249`
     - `single_to_multi`: `47`
     - `multi_missing`: `43`
     - `temporal_calculation_error`: `35`
   - baseline dev 中，多选问题尤其严重：
     - single：`120 / 549 = 0.2186`
     - multi：`11 / 171 = 0.0643`
   - 结论：当前“一次性直接生成答案集合”的方式，对多选和精确集合判断非常脆弱。

3. prompt 微调不是当前主线
   - 90 题 probe：
     - baseline prompt：`35 / 90 = 0.3889`
     - short prompt：`30 / 90 = 0.3333`
   - 结论：继续堆新的短 prompt / hard prompt，收益已经不明显，甚至会回退。

4. hybrid 不是唯一根因
   - hybrid 确实低，但真正占大头的仍然是 spatial / temporal / multi-answer。
   - 尤其 temporal 领域在 LoRA 后仍只有 `0.21`。

## Why The Current Route Stalls

当前主线的核心假设是：

- 输入整题
- 模型一次性输出 `{"answers":["A","C"]}`
- 用 exact-match 评估

这个设置有两个结构性问题：

1. 训练目标和评测目标之间没有中间约束
   - 模型不需要逐个验证 A/B/C/D，只需要“猜一组答案”。
   - 这样在单选和多选混合任务里，很容易出现过选和漏选。

2. 时空题需要局部判定，但当前输出形式鼓励整体拍脑袋
   - 很多题本质更像：
     - “A 这个陈述是否成立？”
     - “B 这个选项是否被题干蕴含？”
   - 而不是直接生成整组标签。

## Recommended New Direction

下一轮不建议优先继续：

- 全局 prompt 替换
- 全量 vote
- 在相同 answer-only 数据上继续小修小补

建议切换到 `verifier` 路线：

### Route A: Option-Level Verifier

把每道题拆成四个二分类判断：

- 输入：题干 + 问题 + 单个选项
- 输出：该选项是否成立，格式固定为 `yes` / `no`

然后把四个选项的判断合并回最终 `answers`。

优点：

- 直接打到 `single_to_multi` 和 `multi_missing`
- 更适合时空推理题的逐条验证
- 不需要外部数据，完全合规

### Route B: Verifier LoRA Before Model Swap

优先顺序建议：

1. `Qwen2.5-7B-Instruct + verifier data`
2. 若 verifier 路线仍不行，再试 `Qwen3-8B`
3. 再之后才考虑其他 7B/8B 底座

原因：

- 现有底座已经证明能从 `0.18` 到 `0.36`
- 当前更像“任务表达错误”，不是“模型完全学不会”
- 先改任务形式的投入产出比更高

## No-GPU Work First

无卡阶段优先完成：

1. verifier 数据构造脚本
2. verifier 推理脚手架
3. verifier 评测与合并逻辑
4. 单元测试
5. 训练配置草案

## GPU Work Later

开卡后只做两步：

1. 训练 `Qwen2.5-7B` verifier LoRA
2. 在固定 dev 上与当前 `0.3625` 主线对比

只有当 verifier 路线没有明显改善，才进入换底座实验。
