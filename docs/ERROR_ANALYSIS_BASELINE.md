# Baseline Dev Error Analysis

## Summary

- Total: 720
- Correct: 131
- Accuracy: 0.181944
- Missing predictions: 0

## Domain Results

| Domain | Total | Correct | Accuracy | Errors |
| --- | ---: | ---: | ---: | ---: |
| hybrid | 20 | 5 | 0.25 | 15 |
| natural | 200 | 50 | 0.25 | 150 |
| social | 100 | 19 | 0.19 | 81 |
| spatial | 200 | 36 | 0.18 | 164 |
| temporal | 200 | 21 | 0.105 | 179 |

## Failure Types

- `multi_constraint_failure`: 14
- `multi_missing`: 76
- `natural_property_error`: 78
- `output_format_error`: 49
- `single_to_multi`: 132
- `social_relation_error`: 23
- `spatial_reference_error`: 128
- `temporal_calculation_error`: 89

## Recommendations

- Prioritize temporal because it has the lowest dev accuracy (0.105).
- Strengthen multi-answer training and decoding; many errors miss one or more gold labels.
- Add constraints that discourage over-selecting labels on single-answer questions.
- Improve temporal prompts with explicit timeline tables and weekday/year arithmetic.
- Improve spatial prompts with explicit reference-frame tracking for left/right/up/down.
- For hybrid questions, force separate domain notes before merging constraints.

## Sample Error Cases

- `SCoRE2026-train-1002` domain=`natural` type=`output_format_error` gold=['A', 'C'] pred=[] question='A plant is on photo No.____.'
- `SCoRE2026-train-101` domain=`temporal` type=`multi_missing` gold=['A', 'C', 'D'] pred=['A', 'C'] question='Select the incorrect statement(s): ____'
- `SCoRE2026-train-1010` domain=`social` type=`single_to_multi` gold=['D'] pred=['A', 'C'] question='以下选项正确的是___'
- `SCoRE2026-train-1012` domain=`temporal` type=`temporal_calculation_error` gold=['A', 'D'] pred=['A', 'C'] question='在他学日语之后4天，____。'
- `SCoRE2026-train-1013` domain=`spatial` type=`spatial_reference_error` gold=['B'] pred=['C'] question='Monthly Rose is to the left of ___.'
- `SCoRE2026-train-1014` domain=`temporal` type=`temporal_calculation_error` gold=['B'] pred=['C'] question='3 days after Jack goes jogging, ____.'
- `SCoRE2026-train-1015` domain=`natural` type=`natural_property_error` gold=['C'] pred=['B'] question='葡萄在____号照片上。'
- `SCoRE2026-train-1016` domain=`natural` type=`natural_property_error` gold=['D'] pred=['B'] question='糍粑在____号照片上。'
- `SCoRE2026-train-1017` domain=`natural` type=`natural_property_error` gold=['A'] pred=['C'] question='4号场馆中养的是____。'
- `SCoRE2026-train-102` domain=`natural` type=`natural_property_error` gold=['B'] pred=['D'] question='____ is planted in field No.4.'
- `SCoRE2026-train-1022` domain=`hybrid` type=`multi_constraint_failure` gold=['D'] pred=['A'] question='海螺和___在同一层'
- `SCoRE2026-train-1024` domain=`natural` type=`output_format_error` gold=['D'] pred=[] question='Spider is kept in enclosure No.____.'
- `SCoRE2026-train-1027` domain=`spatial` type=`spatial_reference_error` gold=['A'] pred=['C'] question='___在郁金香正上方。'
- `SCoRE2026-train-1028` domain=`spatial` type=`spatial_reference_error` gold=['A'] pred=['B'] question='君子兰的右邻在___的正下方。'
- `SCoRE2026-train-1033` domain=`spatial` type=`spatial_reference_error` gold=['D'] pred=['C'] question='___在君子兰右边。'
- `SCoRE2026-train-1034` domain=`temporal` type=`temporal_calculation_error` gold=['A'] pred=['B'] question='Jack goes jogging 2 days before ____.'
- `SCoRE2026-train-1038` domain=`spatial` type=`spatial_reference_error` gold=['C'] pred=['D'] question='百合的右邻在___的正下方。'
- `SCoRE2026-train-1040` domain=`social` type=`social_relation_error` gold=['B'] pred=['C'] question='以下选项不正确的是___'
- `SCoRE2026-train-1056` domain=`temporal` type=`single_to_multi` gold=['A'] pred=['B', 'D'] question='____ on Friday.'
- `SCoRE2026-train-1057` domain=`temporal` type=`single_to_multi` gold=['B'] pred=['B', 'C'] question='The gap between the time ____ and the time Jack retired is 17 years.'
