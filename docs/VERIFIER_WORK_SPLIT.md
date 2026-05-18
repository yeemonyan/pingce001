# Verifier Work Split

## Goal

把当前主线从“整题直接输出答案集合”切到“逐选项判定 verifier”，优先解决：

- `single_to_multi`
- `multi_missing`
- `temporal_calculation_error`
- `spatial_reference_error`

## Why This Is The Next Step

- 当前 LoRA dev：`0.3625`
- 当前线上第二次提交：`14.9%`
- 90 题 probe 中 short prompt 低于 baseline
- baseline 多选准确率只有 `11 / 171 = 0.0643`

结论：当前最需要改的是任务形式，而不是继续全局换 prompt。

## Your Work

1. 准备本地/服务器数据可用性
   - 确认官方 `train.json` 能在本地或服务器路径上访问
   - 如果本地没有，至少保证服务器上有稳定路径
   - 统一记录数据所在路径，避免脚本默认找不到

2. 总控 verifier 主线
   - 以 `docs/DIAGNOSIS_AND_VERIFIER_PLAN.md` 为主计划
   - 以后所有 verifier 相关实验，都记录配置、命令和结果

3. 开新分支推进 verifier
   - 建议新分支：`codex/verifier-pipeline`
   - 只把 verifier 路线相关代码放到这条分支上，避免和旧 round3 实验混在一起

4. 开卡后执行第一轮 verifier 训练
   - 用 `outputs/sft_train_verifier.jsonl`
   - 用 `outputs/sft_valid_verifier.jsonl`
   - 先训练 `Qwen2.5-7B-Instruct` verifier LoRA

5. 跑 verifier dev 评测
   - 先逐选项预测
   - 再合并成题级 `answers`
   - 报告必须包含：
     - overall ACC
     - per-domain ACC
     - single vs multi ACC
     - over-predict / under-predict

## Teammate Work

1. 数据侧验证
   - 检查 `scripts/build_verifier_data.py` 生成的数据是否符合预期
   - 随机抽查：
     - 单选题 10 条
     - 多选题 10 条
     - temporal / spatial / hybrid 各至少 5 条
   - 确认 `target_label=yes/no` 没有错位

2. 设计 verifier 推理脚本
   - 新增脚本建议：`scripts/infer_verifier.py`
   - 输入：标准题目 JSONL
   - 过程：每题拆 A/B/C/D 四次判定
   - 输出：
     - option-level raw predictions
     - merged question-level answers

3. 设计合并策略
   - 先做 deterministic baseline：
     - yes 就选，no 就不选
   - 再预留阈值/打分版本：
     - 如果以后有 logit/probability，可加阈值控制
   - 要特别防止空答案

4. 误差分析脚本扩展
   - 在现有分析基础上新增：
     - single vs multi
     - over-predict
     - under-predict
     - empty prediction

5. 准备 verifier 训练配置草案
   - 复制 `configs/train_lora.yaml`
   - 新建 verifier 版本配置
   - 只改：
     - train/valid file
     - output_dir
     - adapter_name

## Joint Check Before GPU

1. 验证 verifier 数据文件存在
2. 验证字段 schema 稳定
3. 验证 option-level 到 question-level 合并逻辑
4. 确认评测脚本输出 single/multi 统计

## GPU Stage

开卡后只做这一条最小闭环：

1. 训练 verifier LoRA
2. 跑 dev
3. 和当前 `0.3625` 主线对比

只有 verifier 不提升，才考虑：

1. `Qwen3-8B`
2. 其他 7B/8B reasoning 底座
