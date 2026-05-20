"""Validate an official-format SCoRE2026 submission JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_score import VALID_LABELS, load_json_or_jsonl, normalize_answer


def expected_ids(prefix: str, start: int, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(start, start + count)]


def validate_submission_records(
    records: list[dict[str, Any]],
    *,
    expected_count: int | None = 1000,
    id_prefix: str = "SCoRE2026-test-",
    id_start: int = 1,
) -> dict[str, Any]:
    errors: list[str] = []
    normalized_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index} is not an object")
            continue

        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            errors.append(f"record {index} has invalid id")
            continue
        if record_id in seen_ids:
            errors.append(f"duplicate id: {record_id}")
        seen_ids.add(record_id)

        answers = normalize_answer(record.get("answers"))
        if not answers:
            errors.append(f"{record_id} has empty or invalid answers")
        normalized_records.append({"id": record_id, "answers": answers})

    if expected_count is not None and len(records) != expected_count:
        errors.append(f"expected {expected_count} records, got {len(records)}")

    missing_ids: list[str] = []
    unexpected_ids: list[str] = []
    if expected_count is not None:
        expected = expected_ids(id_prefix, id_start, expected_count)
        expected_set = set(expected)
        actual_set = {record["id"] for record in normalized_records}
        missing_ids = [record_id for record_id in expected if record_id not in actual_set]
        unexpected_ids = sorted(actual_set - expected_set)
        if missing_ids:
            errors.append(f"missing ids: {', '.join(missing_ids[:10])}")
        if unexpected_ids:
            errors.append(f"unexpected ids: {', '.join(unexpected_ids[:10])}")

    invalid_answers = [
        record["id"]
        for record in normalized_records
        if any(answer not in VALID_LABELS for answer in record["answers"])
    ]
    if invalid_answers:
        errors.append(f"invalid answer labels found: {', '.join(invalid_answers[:10])}")

    return {
        "ok": not errors,
        "count": len(records),
        "errors": errors,
        "missing_ids": missing_ids,
        "unexpected_ids": unexpected_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a SCoRE2026 submission JSON file.")
    parser.add_argument("--input", required=True, help="Submission JSON/JSONL path.")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    parser.add_argument("--expected-count", type=int, default=1000, help="Expected submission size.")
    parser.add_argument("--id-prefix", default="SCoRE2026-test-", help="Expected id prefix.")
    parser.add_argument("--id-start", type=int, default=1, help="Expected id starting index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_or_jsonl(Path(args.input))
    report = validate_submission_records(
        records,
        expected_count=args.expected_count,
        id_prefix=args.id_prefix,
        id_start=args.id_start,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")

    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
