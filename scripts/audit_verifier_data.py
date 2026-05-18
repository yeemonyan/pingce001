"""Audit option-level verifier data with fixed random samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verifier_utils import load_jsonl, write_json


def group_by_question(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["question_id"])].append(item)
    return dict(grouped)


def validate_question_items(question_id: str, items: list[dict[str, Any]]) -> list[str]:
    errors = []
    labels = {item["option_label"] for item in items}
    if labels != {"A", "B", "C", "D"}:
        errors.append(f"{question_id}: expected A/B/C/D, got {sorted(labels)}")
    for item in items:
        expected = "yes" if item["option_label"] in set(item.get("gold_answer") or []) else "no"
        if item.get("target_label") != expected:
            errors.append(
                f"{item['id']}: target_label={item.get('target_label')} expected={expected}"
            )
        if item.get("label") not in {None, item.get("target_label")}:
            errors.append(f"{item['id']}: label alias mismatch")
    return errors


def question_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    first = items[0]
    user_prompt = first.get("messages", [{}, {"content": ""}])[1].get("content", "")
    return {
        "question_id": first["question_id"],
        "domain": first.get("domain"),
        "language": first.get("language"),
        "answer_kind": first.get("answer_kind"),
        "gold_answer": first.get("gold_answer"),
        "prompt_length": len(user_prompt),
        "input_variant": first.get("input_variant"),
        "instruction_variant": first.get("instruction_variant"),
        "targets": {
            item["option_label"]: item["target_label"]
            for item in sorted(items, key=lambda item: item["option_label"])
        },
        "option_text": {
            item["option_label"]: item.get("option_text")
            for item in sorted(items, key=lambda item: item["option_label"])
        },
    }


def sample_questions(
    grouped: dict[str, list[dict[str, Any]]],
    predicate,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = [items for items in grouped.values() if predicate(items[0])]
    rng.shuffle(candidates)
    return [question_summary(items) for items in candidates[:count]]


def audit(
    items: list[dict[str, Any]],
    seed: int,
    sample_count: int,
    domain_count: int,
    language_count: int = 10,
    long_count: int = 10,
) -> dict[str, Any]:
    grouped = group_by_question(items)
    errors = []
    for question_id, question_items in grouped.items():
        errors.extend(validate_question_items(question_id, question_items))

    rng = random.Random(seed)
    samples = {
        "single": sample_questions(
            grouped,
            lambda item: item.get("answer_kind") == "single",
            sample_count,
            rng,
        ),
        "multi": sample_questions(
            grouped,
            lambda item: item.get("answer_kind") == "multi",
            sample_count,
            rng,
        ),
        "temporal": sample_questions(
            grouped,
            lambda item: item.get("domain") == "temporal",
            domain_count,
            rng,
        ),
        "spatial": sample_questions(
            grouped,
            lambda item: item.get("domain") == "spatial",
            domain_count,
            rng,
        ),
        "hybrid": sample_questions(
            grouped,
            lambda item: item.get("domain") == "hybrid",
            domain_count,
            rng,
        ),
        "chinese": sample_questions(
            grouped,
            lambda item: item.get("language") == "zh",
            language_count,
            rng,
        ),
        "long_constraints": [
            question_summary(items)
            for items in sorted(
                grouped.values(),
                key=lambda question_items: len(question_items[0].get("messages", [{}, {"content": ""}])[1].get("content", "")),
                reverse=True,
            )[:long_count]
        ],
    }
    return {
        "source_count": len(items),
        "question_count": len(grouped),
        "seed": seed,
        "validation": {
            "passed": not errors,
            "error_count": len(errors),
            "errors": errors[:20],
        },
        "sample_requirements": {
            "single_questions": sample_count,
            "multi_questions": sample_count,
            "temporal_questions": domain_count,
            "spatial_questions": domain_count,
            "hybrid_questions": domain_count,
            "chinese_questions": language_count,
            "long_constraint_questions": long_count,
        },
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit verifier option-level data.")
    parser.add_argument("--input", default="outputs/verifier_valid.jsonl")
    parser.add_argument("--output", default="outputs/verifier_data_audit.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--domain-count", type=int, default=5)
    parser.add_argument("--language-count", type=int, default=10)
    parser.add_argument("--long-count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(
        load_jsonl(Path(args.input)),
        args.seed,
        args.sample_count,
        args.domain_count,
        language_count=args.language_count,
        long_count=args.long_count,
    )
    write_json(report, Path(args.output))
    print(json.dumps(report["validation"], ensure_ascii=False))
    if not report["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
