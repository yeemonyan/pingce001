# SCoRE2026 Work Timeline Report

## 1. 总体结论

我们这条线已经从“数据切分 + SFT 构造”推进到“verifier 方案验证”。
当前判断是：

- 继续堆 prompt 或切 MoE，不是主线。
- verifier 方向已经搭好脚手架，下一步应主攻 option-level verifier。
- 已验证：zero-shot dev `0.1819` -> LoRA dev `0.3625`，说明 7B 底座还有空间。
- 但线上 `0.149` 明显低于 dev，说明泛化不稳，不能只靠 prompt。

## 2. 时间顺序

### 2026-05-01

- 新建并完成 `Phase B0: dev split`
- 分支：`codex/phase-b0-dev-split`
- 产物：
  - `data/splits/train_ids.json`
  - `data/splits/dev_ids.json`
  - `outputs/dev_split_report.json`
- 结果：
  - 固定随机种子 `2026`
  - 按 domain / language / 单多选尽量分层

### 2026-05-02

- 完成 `Phase B: SFT 数据构造`
- 分支：`codex/phase-b-sft-data`
- 产物：
  - `outputs/sft_train_answer_only.jsonl`
  - `outputs/sft_valid_answer_only.jsonl`
  - `outputs/sft_train_rationale_json.jsonl`
  - `outputs/sft_valid_rationale_json.jsonl`
  - `outputs/sft_data_report.json`
- 结果：
  - 同时保留 `answer_only` 和 `rationale_json`
  - 官方训练集未引入测试集内容

### 2026-05-02

- 完成 baseline 本地 dev 评测和错误分析
- 产物：
  - `outputs/dev_baseline_predictions.jsonl`
  - `outputs/dev_baseline_eval.json`
  - `outputs/error_report_baseline_dev.json`
  - `outputs/error_cases_baseline_dev.jsonl`
  - `docs/ERROR_ANALYSIS_BASELINE.md`
- 结果：
  - baseline dev 可跑通
  - 错误集中在多选漏选、单选扩多选、时空题

### 2026-05-02 to 2026-05-03

- 完成 prompt v2 和调优记录
- 分支：
  - `codex/phase-d-prompt-tuning`
- 产物：
  - `configs/system_prompts_v2.yaml`
  - `docs/PROMPT_NOTES.md`
  - `docs/ERROR_PRIORITY_TABLE.md`
  - `docs/SFT_AND_PROMPT_RECOMMENDATION.md`
- 结果：
  - temporal / spatial / hybrid 是优先优化方向
  - 后续 prompt 仅做短硬修订，不再做大面积替换

### 2026-05-03

- 完成 round3 LoRA 错题分析
- 分支：`codex/round3-lora-analysis`
- 产物：
  - `outputs/error_report_lora_dev.json`
  - `outputs/error_cases_lora_dev.jsonl`
  - `docs/ERROR_ANALYSIS_LORA.md`
  - `configs/system_prompts_round3_short.yaml`
- 关键结论：
  - LoRA dev `0.3625`，`261/720`
  - `spatial_reference_error = 249`
  - `single_to_multi = 47`
  - `multi_missing = 43`
  - `temporal_calculation_error = 35`
  - hybrid dev 仅 `0.25`
  - hybrid prompt 有拖后腿迹象

### 2026-05-04

- 完成 vote 规则与 rationale 下一轮建议
- 产物：
  - `docs/VOTE_RULES_LORA.md`
  - `docs/VOTE_RULES_LORA_SUMMARY.md`
  - `docs/RATIONALE_JSON_NEXT_TRAINING_NOTES.md`
- 结论：
  - vote 只建议开在 `temporal / spatial / hybrid`
  - 不建议全局开 vote
  - 如果下一轮 rationales 继续做，建议只做短结构化理由，不做长 CoT

### 2026-05-05

- 完成 option-level verifier probe 的基础脚手架
- 分支：`codex/round4-verifier-probe`
- 产物：
  - `scripts/verifier_utils.py`
  - `scripts/build_verifier_data.py`
  - `scripts/infer_verifier.py`
  - `docs/VERIFIER_PROBE_PLAN.md`
  - `tests/test_verifier_utils.py`
- 结果：
  - 每题拆成 4 个 option-level yes/no 样本
  - verifier mock 推理能合并回题级答案

### 2026-05-13

- 完成 verifier 数据审计和训练配置草案
- 产物：
  - `scripts/audit_verifier_data.py`
  - `tests/test_audit_verifier_data.py`
  - `configs/train_lora_verifier.yaml`
  - `outputs/verifier_data_audit.json`
- 结果：
  - `target_label=yes/no` 对齐正确
  - 抽样检查固定覆盖：
    - 单选 10
    - 多选 10
    - temporal/spatial/hybrid 各 5
  - verifier 数据量：
    - train `11520`
    - valid `2880`

### 2026-05-18

- 完成 verifier 数据 QA 和固定 pipeline
- 产物：
  - `outputs/sft_train_verifier.jsonl`
  - `outputs/sft_valid_verifier.jsonl`
  - `scripts/run_verifier_pipeline.ps1`
  - `docs/VERIFIER_EXPERIMENT_LOG.md`
- 结果：
  - 数据 audit 增加：
    - 中文题
    - 长约束题
  - 支持输入变体：
    - all options
    - candidate only
  - 支持 instruction 变体：
    - base
    - domain hint
  - 支持 oversampling temporal / spatial / hybrid
  - mock pipeline 全流程通过

### 2026-05-19

- 重试服务器连接和 Git 上传
- 结果：
  - GitHub push 成功
  - 服务器端口 50890 连接超时，无法非交互确认服务器产物

## 3. 当前已经做完的东西

- dev split
- SFT 数据构造
- baseline dev 评测
- baseline / LoRA 错题分析
- prompt v2 和 prompt 备注
- vote 规则草案
- verifier option-level 数据构造
- verifier 推理脚手架
- verifier 数据审计
- verifier 固定 pipeline
- verifier 训练配置草案

## 4. 当前重点判断

### 不再主攻

- 大面积 prompt 替换
- MoE 切换
- 全局 self-consistency 投票

### 主攻方向

- option-level verifier
- temporal / spatial / hybrid 的局部模板
- multi-answer 的 yes/no 判定稳定性
- single_to_multi / multi_missing 的减少

## 5. 当前关键文件

- `outputs/sft_train_verifier.jsonl`
- `outputs/sft_valid_verifier.jsonl`
- `outputs/verifier_data_report.json`
- `outputs/verifier_data_audit.json`
- `configs/train_lora_verifier.yaml`
- `scripts/run_verifier_pipeline.ps1`
- `scripts/build_verifier_data.py`
- `scripts/audit_verifier_data.py`
- `scripts/infer_verifier.py`

## 6. 还没最终确认的部分

- 服务器侧 verifier 训练是否已经真正跑起来，当前未能非交互核实
- 后续 verifier LoRA 的真实 dev 分数还没出
- 线上结果还没有新一轮 verifier 对照

## 7. 下一步建议

1. 先在服务器上跑 `scripts/run_verifier_pipeline.ps1` 对应的 verifier 训练/评测命令。
2. 先看 verifier 对 `temporal`、`spatial`、`multi-answer` 的改进幅度。
3. 如果 verifier 确实优于当前 0.3625 主线，再考虑：
   - `Qwen2.5-7B + verifier LoRA`
   - `Qwen3-8B verifier probe`
4. 如果 verifier 仍然不稳，再做领域专项输入模板，而不是回去大改 prompt。
