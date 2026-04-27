"""Parse and normalize SCoRE2026 JSON/JSONL records.

Expected input fields:
- text: scenario text
- question: question sentence with blank
- options: dict mapping A/B/C/D to option text
- answer: optional list of labels, absent in test data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


VALID_LABELS = ("A", "B", "C", "D")


class ScoreFormatError(ValueError):
    """Raised when a SCoRE record is malformed."""


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "examples", "records"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise ScoreFormatError(f"Unsupported JSON top-level type: {type(data).__name__}")


def normalize_options(options: Any) -> dict[str, str]:
    if not isinstance(options, dict):
        raise ScoreFormatError("options must be an object mapping labels to strings")

    normalized: dict[str, str] = {}
    for label, value in options.items():
        key = str(label).strip().upper()
        if key not in VALID_LABELS:
            raise ScoreFormatError(f"invalid option label: {label!r}")
        if not isinstance(value, str) or not value.strip():
            raise ScoreFormatError(f"option {key} must be a non-empty string")
        normalized[key] = value.strip()

    if len(normalized) < 2:
        raise ScoreFormatError("at least two options are required")
    return {label: normalized[label] for label in VALID_LABELS if label in normalized}


def normalize_answer(answer: Any) -> list[str] | None:
    if answer is None:
        return None
    if not isinstance(answer, list) or not answer:
        raise ScoreFormatError("answer must be a non-empty list when present")

    normalized = []
    for item in answer:
        label = str(item).strip().upper()
        if label not in VALID_LABELS:
            raise ScoreFormatError(f"invalid answer label: {item!r}")
        if label not in normalized:
            normalized.append(label)
    return normalized


def build_prompt(record: dict[str, Any], options: dict[str, str]) -> str:
    option_lines = "\n".join(f"{label}. {value}" for label, value in options.items())
    return (
        "请阅读下面的 SCoRE2026 常识推理题，只输出最终答案字母列表。\n\n"
        f"材料：{record['text'].strip()}\n\n"
        f"问题：{record['question'].strip()}\n\n"
        f"选项：\n{option_lines}\n\n"
        '答案格式示例：["A"] 或 ["A","B"]'
    )


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    for field in ("text", "question", "options"):
        if field not in record:
            raise ScoreFormatError(f"record {index} missing required field: {field}")

    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ScoreFormatError(f"record {index} text must be a non-empty string")
    if not isinstance(record["question"], str) or not record["question"].strip():
        raise ScoreFormatError(f"record {index} question must be a non-empty string")

    options = normalize_options(record["options"])
    answer = normalize_answer(record.get("answer"))
    record_id = record.get("id", index)

    normalized = {
        "id": record_id,
        "text": record["text"].strip(),
        "question": record["question"].strip(),
        "options": options,
        "answer": answer,
        "has_answer": answer is not None,
        "prompt": build_prompt(record, options),
    }
    for metadata_field in ("domain", "category", "type", "task_type"):
        if metadata_field in record:
            normalized[metadata_field] = record[metadata_field]
    return normalized


def dump_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize SCoRE2026 JSON/JSONL data.")
    parser.add_argument("--input", required=True, help="Input .json or .jsonl file.")
    parser.add_argument("--output", required=True, help="Output normalized .jsonl file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_records(input_path)
    normalized = [normalize_record(record, index) for index, record in enumerate(records)]
    dump_jsonl(normalized, output_path)
    with_answer = sum(1 for record in normalized if record["has_answer"])
    print(f"parsed={len(normalized)} with_answer={with_answer} output={output_path}")


if __name__ == "__main__":
    main()
