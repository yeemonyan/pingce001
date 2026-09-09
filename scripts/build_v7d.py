"""Build the test-safe V7d submission from two prediction files.

V7d routes temporal/hybrid questions predicted as multi-answer to the
mixed-reasoning predictions and keeps answer-only predictions everywhere else.
The cardinality file must contain ``predicted_cardinality``; gold answers are
never read by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_score import load_json_or_jsonl, normalize_answer


def record_id(record: dict[str, Any]) -> str:
    value = record.get("id")
    if not isinstance(value, str) or not value:
        raise ValueError("every prediction record must contain a non-empty string id")
    return value


def domain_of(record: dict[str, Any]) -> str:
    return str(record.get("domain") or record.get("category") or "").lower()


def prediction_by_id(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_id(record)
        if key in result:
            raise ValueError(f"duplicate id in {label}: {key}")
        result[key] = record
    return result


def use_mixed(domain: str, predicted_cardinality: str) -> bool:
    is_temporal = "time" in domain or "temporal" in domain
    is_hybrid = "hybrid" in domain
    return (is_temporal or is_hybrid) and predicted_cardinality == "multi"


def build_v7d(
    *,
    input_records: list[dict[str, Any]],
    answer_only_records: list[dict[str, Any]],
    mixed_records: list[dict[str, Any]],
    cardinality_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    answer_only = prediction_by_id(answer_only_records, "answer-only predictions")
    mixed = prediction_by_id(mixed_records, "mixed-reasoning predictions")
    cardinality = prediction_by_id(cardinality_records, "cardinality predictions")

    result: list[dict[str, Any]] = []
    counts = {"answer_only": 0, "mixed_reasoning": 0}

    for input_record in input_records:
        key = record_id(input_record)
        if key not in answer_only or key not in mixed or key not in cardinality:
            raise ValueError(f"missing aligned prediction for {key}")

        predicted_kind = cardinality[key].get("predicted_cardinality")
        if predicted_kind not in {"single", "multi"}:
            raise ValueError(
                f"{key} must have predicted_cardinality=single|multi; "
                "gold answer fields are not accepted"
            )

        source = "mixed_reasoning" if use_mixed(domain_of(input_record), predicted_kind) else "answer_only"
        selected = mixed[key] if source == "mixed_reasoning" else answer_only[key]
        answers = normalize_answer(selected.get("answers"))
        if not answers:
            answers = normalize_answer(selected.get("answer"))
        if not answers:
            raise ValueError(f"{key} has empty answers in {source} predictions")

        result.append({"id": key, "answers": answers})
        counts[source] += 1

    return result, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a test-safe SCoRE2026 V7d submission.")
    parser.add_argument("--input", required=True, help="Normalized test prompts JSON/JSONL.")
    parser.add_argument("--answer-only", required=True, help="Answer-only predictions JSON/JSONL.")
    parser.add_argument("--mixed-reasoning", required=True, help="Mixed-reasoning predictions JSON/JSONL.")
    parser.add_argument("--cardinality", required=True, help="Predictions with predicted_cardinality fields.")
    parser.add_argument("--output", required=True, help="Official submission JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_records = load_json_or_jsonl(Path(args.input))
    answer_only_records = load_json_or_jsonl(Path(args.answer_only))
    mixed_records = load_json_or_jsonl(Path(args.mixed_reasoning))
    cardinality_records = load_json_or_jsonl(Path(args.cardinality))

    submission, counts = build_v7d(
        input_records=input_records,
        answer_only_records=answer_only_records,
        mixed_records=mixed_records,
        cardinality_records=cardinality_records,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(submission), "sources": counts, "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
