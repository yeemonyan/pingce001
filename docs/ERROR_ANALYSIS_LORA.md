# LoRA Dev Error Analysis

## Summary

- Total: 720
- Correct: 261
- Accuracy: 0.3625

## Focus Error Counts

- `single_to_multi`: 47
- `multi_missing`: 43
- `temporal_calculation_error`: 35
- `spatial_reference_error`: 249

## Hybrid Subset

- Total: 20
- Correct: 5
- Accuracy: 0.25
- Likely prompt drag: True
- Note: Hybrid dev accuracy is low and failures cluster around joint/spatial constraints; test a shorter hybrid prompt with explicit fallback to general.

## Recommendation

- Use `configs/system_prompts_round3_short.yaml` for the next dev prompt experiment.
- Prioritize temporal and spatial first; watch whether hybrid improves or regresses with fallback-to-general wording.
- Keep answer_only LoRA as the current training baseline; use rationale_json next only if prompt-only gains are limited.

## Sample Cases

- `SCoRE2026-train-1003` domain=`spatial` type=`spatial_reference_error` gold=['C'] pred=['D'] question='___ occupies the third position to the left of David.'
- `SCoRE2026-train-101` domain=`temporal` type=`multi_missing` gold=['A', 'C', 'D'] pred=['A', 'D'] question='Select the incorrect statement(s): ____'
- `SCoRE2026-train-1012` domain=`temporal` type=`multi_missing` gold=['A', 'D'] pred=['A'] question='在他学日语之后4天，____。'
- `SCoRE2026-train-1014` domain=`temporal` type=`temporal_calculation_error` gold=['B'] pred=['C'] question='3 days after Jack goes jogging, ____.'
- `SCoRE2026-train-1016` domain=`natural` type=`spatial_reference_error` gold=['D'] pred=['A'] question='糍粑在____号照片上。'
- `SCoRE2026-train-1017` domain=`natural` type=`natural_property_error` gold=['A'] pred=['C'] question='4号场馆中养的是____。'
- `SCoRE2026-train-1021` domain=`spatial` type=`spatial_reference_error` gold=['A', 'C'] pred=['A', 'B'] question='尹志平不在___旁边。'
- `SCoRE2026-train-1024` domain=`natural` type=`natural_property_error` gold=['D'] pred=['C'] question='Spider is kept in enclosure No.____.'
- `SCoRE2026-train-1028` domain=`spatial` type=`spatial_reference_error` gold=['A'] pred=['B'] question='君子兰的右邻在___的正下方。'
- `SCoRE2026-train-1033` domain=`spatial` type=`spatial_reference_error` gold=['D'] pred=['B'] question='___在君子兰右边。'
- `SCoRE2026-train-1034` domain=`temporal` type=`single_to_multi` gold=['A'] pred=['A', 'C'] question='Jack goes jogging 2 days before ____.'
- `SCoRE2026-train-1038` domain=`spatial` type=`spatial_reference_error` gold=['C'] pred=['B'] question='百合的右邻在___的正下方。'
- `SCoRE2026-train-1040` domain=`social` type=`spatial_reference_error` gold=['B'] pred=['C'] question='以下选项不正确的是___'
- `SCoRE2026-train-1057` domain=`temporal` type=`temporal_calculation_error` gold=['B'] pred=['A'] question='The gap between the time ____ and the time Jack retired is 17 years.'
- `SCoRE2026-train-1061` domain=`natural` type=`spatial_reference_error` gold=['B'] pred=['D'] question='君子兰在____号照片上。'
- `SCoRE2026-train-1062` domain=`temporal` type=`single_to_multi` gold=['C'] pred=['A', 'B', 'C'] question='以下选项中正确的是____'
- `SCoRE2026-train-1081` domain=`social` type=`spatial_reference_error` gold=['A'] pred=['B'] question='以下选项不正确的是___'
- `SCoRE2026-train-1105` domain=`temporal` type=`spatial_reference_error` gold=['B', 'C', 'D'] pred=['A', 'B', 'C'] question='以下选项中正确的是____'
- `SCoRE2026-train-1133` domain=`spatial` type=`single_to_multi` gold=['A'] pred=['A', 'B'] question='The tier where ___ is located is next to the tier where Geranium is located.'
- `SCoRE2026-train-115` domain=`temporal` type=`multi_missing` gold=['A', 'B', 'C', 'D'] pred=['A', 'B', 'C'] question='Select the correct statement(s): ____'
- `SCoRE2026-train-1160` domain=`spatial` type=`multi_missing` gold=['A', 'B', 'D'] pred=['A', 'B'] question='君子兰与___不同侧。'
- `SCoRE2026-train-1164` domain=`temporal` type=`temporal_calculation_error` gold=['A', 'D'] pred=['A', 'C'] question='Select the correct statement(s): ____'
- `SCoRE2026-train-1173` domain=`natural` type=`natural_property_error` gold=['A'] pred=['B'] question='A flower or grass is planted in field No.____.'
- `SCoRE2026-train-118` domain=`temporal` type=`temporal_calculation_error` gold=['C', 'D'] pred=['A', 'C'] question='The gap between the time Jack goes jogging and the time ____ is 1 day.'
