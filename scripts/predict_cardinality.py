"""Predict single/multi cardinality for SCoRE2026 questions.

Uses rule-based classifier derived from dev data analysis.
Apply to test records to add cardinality hints before inference.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _infer_domain(rec: dict[str, Any]) -> str:
    """Infer domain from record fields, similar to infer_score.infer_domain."""
    domain = str(rec.get('domain', '')).strip().lower()
    if domain:
        # Standardize known domain names
        if 'time' in domain or 'temporal' in domain:
            return 'temporal'
        if 'space' in domain or 'spatial' in domain:
            return 'spatial'
        if 'social' in domain:
            return 'social'
        if 'nature' in domain or 'natural' in domain:
            return 'natural'
        if 'hybrid' in domain:
            return 'hybrid'

    # Fallback: infer from question + text content
    text = ' '.join(str(rec.get(k, '')) for k in ('text', 'question', 'prompt'))
    text_lower = text.lower()

    has_spatial = any(w in text_lower for w in ('left', 'right', 'clockwise', 'adjacent', '层', '左', '右', '顺时针'))
    has_temporal = any(w in text_lower for w in ('year', 'day', 'weekday', 'monday', 'before', 'after', '周', '年', '之前', '之后'))
    has_social = any(w in text_lower for w in ('friend', 'father', 'teacher', 'subordinate', 'neighbor', 'boyfriend', 'girlfriend', '朋友', '父', '师', '下属', '邻居', '前妻', '前女友'))
    has_natural = any(w in text_lower for w in ('item', 'food', 'drink', 'tool', 'animal', 'photo', '物品', '食品', '饮品', '工具', '动物'))

    count = sum((has_spatial, has_temporal, has_social, has_natural))
    if count >= 2:
        return 'temporal' if has_temporal else 'hybrid'
    if has_spatial:
        return 'spatial'
    if has_temporal:
        return 'temporal'
    if has_social:
        return 'social'
    if has_natural:
        return 'natural'
    return 'general'


def classify_record(rec: dict[str, Any]) -> str:
    """Determine if a question is single-answer or multi-answer.

    Returns 'single' or 'multi'.

    Key signals (from dev data analysis, 720 samples):
    - statement(s) in question → 87.5% multi
    - Select the correct statement(s) → 96% multi
    - Select the incorrect statement(s) → 80.6% multi
    - 以下选项中 (with 中) → 81.4% multi
    - natural/social domain → 96%+ single
    - spatial domain → 77% single
    - temporal domain → 58% multi (most ambiguous)
    """
    q = rec.get('question', '')
    domain = _infer_domain(rec)

    has_temporal = domain == 'temporal'
    has_natural = domain == 'natural'
    has_social = domain == 'social'
    has_spatial = domain == 'spatial'
    has_hybrid = domain == 'hybrid'

    # ---- Strong multi indicators ----
    multi_score = 0.0

    if 'statement(s)' in q.lower():
        multi_score += 4.0
    if '以下选项中' in q:
        multi_score += 3.5
    if 'select the correct' in q.lower():
        multi_score += 4.0
    if 'select the incorrect' in q.lower():
        multi_score += 3.0
    # Also check Chinese equivalent patterns
    if re.search(r'以下.*正确的', q):
        multi_score += 1.0  # weaker but still indicates possible multi

    # ---- Strong single indicators ----
    single_score = 0.0

    if re.search(r'_在', q):
        single_score += 3.0
    if re.search(r'在_', q) and '以下选项中' not in q:
        single_score += 1.5
    if has_natural and not has_temporal and not has_spatial:
        single_score += 2.5
    if has_social and not has_temporal:
        single_score += 3.0
    if has_natural and has_spatial and not has_temporal:
        single_score += 2.0

    net = multi_score - single_score

    if net >= 3.0:
        return 'multi'
    if net <= -2.0:
        return 'single'

    # For temporal domains:
    #   True distribution: ~58% multi, ~42% single
    #   But: wrong multi hint on single Q costs 43.2% acc
    #        wrong single hint on multi Q costs 23.4% acc
    #   Expected loss of multi hint: 0.42 × 0.432 = 0.181
    #   Expected loss of single hint: 0.58 × 0.234 = 0.136
    #   → single hint is safer when uncertain (net ≈ 0)
    #   → only predict multi when net >= 1.0 (moderate signal)
    if has_temporal:
        return 'multi' if net >= 1.0 else 'single'

    # Default: lean toward single (over-selection costs more expected accuracy)
    return 'single'


def cardinality_hint(kind: str, language: str = 'en') -> str:
    """Build the cardinality hint string for a given kind."""
    if kind == 'single':
        if language == 'zh':
            return "这是一道单选题。你只能选择一个最符合题意的选项。"
        return "This is a single-answer question. You must select exactly one best-supported option."
    if language == 'zh':
        return "这是一道多选题。请选出所有正确选项，不要漏选，也不要多选。"
    return "This is a multi-answer question. Select all supported options without missing any and without adding extras."


def detect_language(rec: dict[str, Any]) -> str:
    """Detect whether a record is Chinese or English."""
    text = f"{rec.get('text', '')} {rec.get('question', '')}"
    return 'zh' if any('一' <= ch <= '鿿' for ch in text) else 'en'


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def dump_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Predict single/multi cardinality for SCoRE2026 questions.')
    parser.add_argument('--input', required=True, help='Input JSONL (dev or test prompts).')
    parser.add_argument('--output', required=True, help='Output JSONL with cardinality_hint field added.')
    parser.add_argument('--stats', action='store_true', help='Print prediction statistics.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.input))

    single_count = 0
    multi_count = 0
    for rec in records:
        kind = classify_record(rec)
        language = detect_language(rec)
        rec['predicted_cardinality'] = kind
        rec['cardinality_hint'] = cardinality_hint(kind, language)
        if kind == 'single':
            single_count += 1
        else:
            multi_count += 1

    dump_jsonl(records, Path(args.output))

    if args.stats:
        print(json.dumps({
            'total': len(records),
            'predicted_single': single_count,
            'predicted_multi': multi_count,
            'output': str(args.output),
        }, ensure_ascii=False))


if __name__ == '__main__':
    main()
