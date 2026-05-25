# V4 Improvement Plan: Breaking the 14.9% Barrier

## Root Cause Analysis

### Why is online score (14.9%) so much lower than dev (38.5%)?

**Primary cause: Cardinality hint dependency**

The SFT training data ALWAYS includes `answer_count_instruction`:
```
Choose all correct option labels. Output only JSON.

这是一道单选题。你只能输出一个最符合题意的选项。
```

This creates a crutch: the model learns to rely on the explicit single/multi hint.

At inference WITHOUT the hint:
1. The prompt says "Choose **ALL** correct option labels" → biases toward over-selection
2. Model doesn't know if question is single or multi → defaults to multi (over-selection)
3. Single-answer questions: model outputs 2+ options → 0 points
4. Multi-answer questions: model sometimes gets correct (~23.4%)

**Math matches observations:**
- Test: ~508 single, ~492 multi (estimated)
- Single Qs without hint: ~6.7% correct (vs 43.2% with hint)
- Multi Qs without hint: ~23.4% correct  
- Expected: 508×0.067 + 492×0.234 ≈ 149/1000 = 14.9% ✓

**Secondary cause: Test distribution shift**
- Dev: 2.8% hybrid, 97.2% single-domain
- Test: 67.5% hybrid, 32.5% single-domain
- Hybrid questions are harder (35% dev accuracy even WITH hint)

## V4 Solutions

### Solution 1: Fix Prompt Bias (immediate, no retraining)

Change `"Choose all correct option labels"` → `"Select every correct option and output only JSON."`

The old prompt explicitly says "ALL", biasing toward plural. The new prompt is neutral.

**Expected impact**: Reduces over-selection on single-answer questions.

### Solution 2: Cardinality Classifier (immediate, no retraining)

Rule-based classifier trained on dev patterns:
- `statement(s)` → 87.5% multi
- `以下选项中` (with 中) → 81.4% multi  
- `Select the correct/incorrect statement(s)` → 80-96% multi
- Natural/social domain → 96%+ single
- Temporal domain → 58% multi

**Accuracy on dev**: 81.1% (584/720)
- Multi precision: 59.2%
- Multi recall: 66.1%

**Integration**: `--auto-count-hint` flag in `infer_score.py` injects predicted hints.

### Solution 3: Self-Determination System Prompts (immediate)

New `system_prompts_v4_self_determine.yaml`:
- Each domain prompt now includes: "First, read the question to determine if it asks for a single answer or multiple answers"
- Shows both JSON formats: `{"answer":["A"]}` for single, `{"answer":["A","B"]}` for multi
- Domain-specific hints about cardinality patterns

### Solution 4: Neutral SFT Retraining (requires GPU time)

New `build_sft_data_v2.py`:
- Removes `answer_count_instruction` from user prompts
- Instead: "请阅读题目，判断是单选题还是多选题，然后选出所有正确选项"
- Assistant output includes reasoning: `{"reasoning": "这是一道单选题。", "answers": ["A"]}`
- This teaches the model to self-determine cardinality

**Training config**: `configs/train_qwen_v2_neutral.yaml`

## Expected Improvements

| Approach | Est. Dev | Est. Test | Training Needed |
|----------|----------|-----------|-----------------|
| Current (old prompt, no hint) | ~5-15% | 14.9% | No |
| Fix prompt only (v4 prompts) | ~20-27% | ~18-22% | No |
| + Classifier hints | ~28-32% | ~20-25% | No |
| + Neutral SFT retrain | ~32-38% | ~25-35% | Yes (~3h on 3090) |

## Files Created/Modified

### New files:
- `scripts/predict_cardinality.py` - Single/multi classifier
- `scripts/build_sft_data_v2.py` - Neutral SFT data builder
- `scripts/run_submission_v4.sh` - V4 submission pipeline
- `configs/system_prompts_v4_self_determine.yaml` - Self-determination prompts
- `configs/train_qwen_v2_neutral.yaml` - V2 training config
- `docs/V4_IMPROVEMENT_PLAN.md` - This document

### Modified files:
- `scripts/infer_score.py` - Added `--auto-count-hint`, fixed prompt bias

## Quick Start

### Option A: Fastest (no retraining)
```bash
# 1. Predict cardinality for test data
python scripts/predict_cardinality.py \
  --input outputs/test_prompts.jsonl \
  --output outputs/test_prompts_with_cardinality.jsonl

# 2. Run inference with auto hints + v4 prompts
python scripts/infer_score.py \
  --backend transformers \
  --model-path models/Qwen2.5-7B-Instruct \
  --adapter-path checkpoints/qwen2p5_7b_lora_mixed_reasoning_seed2026 \
  --prompts configs/system_prompts_v4_self_determine.yaml \
  --input outputs/test_prompts_with_cardinality.jsonl \
  --output outputs/test_v4_predictions.jsonl \
  --dtype bfloat16 --device-map auto \
  --auto-count-hint

# 3. Format & submit
python scripts/format_submission.py \
  --input outputs/test_v4_predictions.jsonl \
  --output outputs/submissions/v4_submission.json
```

### Option B: With retraining (better)
```bash
# 1. Build v2 SFT data (neutral, no crutch)
python scripts/build_sft_data_v2.py

# 2. Train new adapter
python scripts/train_lora.py --config configs/train_qwen_v2_neutral.yaml

# 3. Run submission pipeline
bash scripts/run_submission_v4.sh
```

### Option C: Dev evaluation
```bash
# Compare approaches on dev
# 1. With gold hint (upper bound)
python scripts/infer_score.py ... --answer-count-hint

# 2. With auto hint (classifier)
python scripts/infer_score.py ... --auto-count-hint

# 3. No hint (baseline)
python scripts/infer_score.py ... 
```
