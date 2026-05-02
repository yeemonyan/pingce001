# Error Type Priority Table

基于 `outputs/error_report_baseline_dev.json` 的 Qwen2.5-7B-Instruct dev baseline。

## Overall Result

- Dev total: 720
- Correct: 131
- ACC: 0.181944

## Priority Table

| Priority | Error / Area | Evidence | Why It Matters | Next Action |
| ---: | --- | --- | --- | --- |
| 1 | temporal | Temporal ACC is lowest: `21/200 = 0.105`. Temporal also has `multi_missing=57`, `single_to_multi=32`, `temporal_calculation_error=38`. | 时间题是当前最明显短板，且同时影响单选、多选和计算链。 | 先跑 `system_prompts_v2.yaml`，重点验证 temporal 是否提升。 |
| 2 | spatial_reference_error | Total `231`; spatial domain alone has `128` such errors. | 空间参考系错误量最大，说明左右、上下、朝向、观察者视角仍不稳。 | v2 暂不改 spatial，下一轮单独做 `system_prompts_v3.yaml`。 |
| 3 | single_to_multi | Total `132`; social domain has `58` over-selection errors. | 单选题过选会直接丢分，说明模型不够克制。 | SFT 先跑 `answer_only`，强化精确答案边界。 |
| 4 | multi_missing | Total `76`; temporal domain has `57`. | 多选漏选是多选题主要损失来源。 | SFT 加强多选样本，比较 `answer_only` vs `rationale_json`。 |
| 5 | social | Social ACC `19/100 = 0.19`; over-selection heavy. | 社会关系题需要处理别名、角色转写和逆关系。 | 用 v2 social prompt 验证关系图和逐项核对是否减少过选。 |
| 6 | hybrid | Hybrid ACC `5/20 = 0.25`; sample small but depends on multi-domain merging. | 融合题数量少但难度高，容易只凭单线索作答。 | 用 v2 hybrid prompt 验证“先拆分再联立”的收益。 |
| 7 | output_format_error | Total `49`; natural has `33`, spatial has `15`. | 空输出或无法抽取答案会白白丢分。 | 继续保留格式清洗；SFT 优先 `answer_only` 稳定 JSON 输出。 |

## Short Ranking

1. 最优先：`temporal`
2. 第二优先：`spatial`
3. 第三优先：`social / hybrid`
4. 全局贯穿：多选边界，包括 `single_to_multi` 和 `multi_missing`

## Practical Conclusion

不要一次性大改所有 prompt。下一轮先验证 `system_prompts_v2.yaml` 对 temporal/social/hybrid 的影响；如果 temporal 有明显提升，再单独开 v3 优化 spatial。
