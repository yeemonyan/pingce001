"""Convert inference predictions into SCoRE2026 submission files.

The safer default is ``jsonl_with_id`` because the official repository notes
that result files can be JSONL with one object per line containing ``id`` and
``answer``. ``system_json`` is kept for compatibility with systems that accept
a plain JSON array.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal


VALID_LABELS = ("A", "B", "C", "D")
OFFICIAL_FORMATS = ("jsonl_with_id", "system_json")
OfficialFormat = Literal["jsonl_with_id", "system_json"]


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
    official_format: OfficialFormat,
    fallback: str,
) -> list[dict[str, Any]]:
    submission = []
    for index, prediction in enumerate(predictions):
        answer = normalize_answer(prediction.get("answer"), fallback)
        if official_format == "jsonl_with_id":
            item: dict[str, Any] = {"id": prediction.get("id", index), "answer": answer}
        else:
            item = {"answer": answer}
        submission.append(item)
    return submission


def write_submission(records: list[dict[str, Any]], path: Path, official_format: OfficialFormat) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if official_format == "jsonl_with_id":
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCoRE2026 submission files.")
    parser.add_argument("--input", required=True, help="Prediction JSONL/JSON from infer_score.py.")
    parser.add_argument("--output", required=True, help="Submission output path.")
    parser.add_argument(
        "--official-format",
        default="jsonl_with_id",
        choices=OFFICIAL_FORMATS,
        help="Output format. Default keeps id and writes JSONL, matching the stricter official note.",
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
    submission = convert_records(predictions, args.official_format, args.fallback)
    output_path = Path(args.output)
    write_submission(submission, output_path, args.official_format)
    print(f"wrote={len(submission)} format={args.official_format} output={output_path}")


if __name__ == "__main__":
    main()
