"""Format prediction JSON/JSONL into the official SCoRE2026 submission JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_score import load_json_or_jsonl, normalize_answer


def normalize_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    record_id = record.get("id")
    if record_id is None:
        raise ValueError("prediction record missing id")

    answers = normalize_answer(record.get("answers"))
    if not answers:
        answers = normalize_answer(record.get("answer"))
    if not answers:
        raise ValueError(f"prediction record {record_id} has empty answers")

    return {"id": record_id, "answers": answers}


def format_submission(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_prediction_record(record) for record in records]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format SCoRE2026 predictions into official submission JSON.")
    parser.add_argument("--input", required=True, help="Prediction JSON/JSONL path.")
    parser.add_argument("--output", required=True, help="Output submission JSON path.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation for output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_json_or_jsonl(Path(args.input))
    submission = format_submission(records)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission, ensure_ascii=False, indent=args.indent) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(submission), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
