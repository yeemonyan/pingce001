"""Convert inference predictions into the SCoRE2026 submission JSON format.

Default output is a JSON array of objects:
[
  {"answer": ["A"]},
  {"answer": ["A", "B"]}
]

Use --include-id when you want an audit-friendly file that keeps record ids.
Use the default no-id format for official submission unless the platform asks
for ids explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VALID_LABELS = ("A", "B", "C", "D")


def load_predictions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and isinstance(data.get("predictions"), list):
            records = data["predictions"]
        else:
            raise ValueError("Prediction JSON must be a list or contain a predictions list")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every prediction must be a JSON object")
    return records


def normalize_answer(value: Any, fallback: str) -> list[str]:
    if value is None:
        labels: list[str] = []
    elif isinstance(value, list):
        labels = [str(item).strip().upper() for item in value]
    elif isinstance(value, str):
        labels = re.findall(r"\b[A-D]\b", value.upper())
    else:
        labels = []

    normalized = []
    for label in labels:
        if label in VALID_LABELS and label not in normalized:
            normalized.append(label)

    return normalized or [fallback]


def convert_records(
    predictions: list[dict[str, Any]],
    include_id: bool,
    fallback: str,
) -> list[dict[str, Any]]:
    submission = []
    for index, prediction in enumerate(predictions):
        answer = normalize_answer(prediction.get("answer"), fallback)
        item: dict[str, Any] = {"answer": answer}
        if include_id:
            item["id"] = prediction.get("id", index)
        submission.append(item)
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCoRE2026 submission JSON.")
    parser.add_argument("--input", required=True, help="Prediction JSONL/JSON from infer_score.py.")
    parser.add_argument("--output", required=True, help="Submission .json path.")
    parser.add_argument(
        "--include-id",
        action="store_true",
        help="Keep ids in the submission objects for audit/debug files.",
    )
    parser.add_argument(
        "--fallback",
        default="D",
        choices=VALID_LABELS,
        help="Answer to use if a prediction has no valid labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(Path(args.input))
    submission = convert_records(predictions, args.include_id, args.fallback)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote={len(submission)} output={output_path}")


if __name__ == "__main__":
    main()
