"""Build option-level yes/no verifier data for SCoRE2026."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_sft_data import normalize_records_by_id, select_records
from scripts.make_dev_split import detect_language
from scripts.verifier_utils import build_verifier_item, option_labels, write_json, write_jsonl


DEFAULT_INPUT = Path("data/raw/train.json")
DEFAULT_TRAIN_IDS = Path("data/splits/train_ids.json")
DEFAULT_DEV_IDS = Path("data/splits/dev_ids.json")
DEFAULT_OUTPUT_DIR = Path("outputs")


def load_ids(path: Path) -> list[Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ensure_language(records: list[dict[str, Any]]) -> None:
    for record in records:
        if not record.get("language"):
            record["language"] = detect_language(record)


def build_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for record in records:
        for label in option_labels(record["options"]):
            items.append(build_verifier_item(record, label))
    return items


def validate_items(items: list[dict[str, Any]]) -> None:
    for item in items:
        if str(item["question_id"]).startswith("SCoRE2026-test-"):
            raise ValueError(f"test sample leaked into verifier data: {item['question_id']}")
        if item["target_label"] not in {"yes", "no"}:
            raise ValueError(f"bad verifier label: {item['id']}")
        if item["label"] != item["target_label"]:
            raise ValueError(f"label alias mismatch: {item['id']}")
        if item["option_label"] not in {"A", "B", "C", "D"}:
            raise ValueError(f"bad option label: {item['id']}")
        if [message["role"] for message in item["messages"]] != ["system", "user", "assistant"]:
            raise ValueError(f"bad messages: {item['id']}")
        payload = json.loads(item["messages"][2]["content"])
        if payload != {"target_label": item["target_label"]}:
            raise ValueError(f"assistant label mismatch: {item['id']}")


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(items),
        "questions": len({item["question_id"] for item in items}),
        "target_label": dict(sorted(Counter(item["target_label"] for item in items).items())),
        "domain": dict(sorted(Counter(item["domain"] for item in items).items())),
        "answer_kind": dict(sorted(Counter(item["answer_kind"] for item in items).items())),
        "positive_rate": round(
            sum(1 for item in items if item["target_label"] == "yes") / len(items),
            6,
        ) if items else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build option-level SCoRE2026 verifier JSONL files.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Official training JSON file.")
    parser.add_argument("--train-ids", default=str(DEFAULT_TRAIN_IDS), help="Train id split JSON.")
    parser.add_argument("--dev-ids", default=str(DEFAULT_DEV_IDS), help="Dev id split JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_DIR / "verifier_data_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    paths = {
        "train": output_dir / "verifier_train.jsonl",
        "valid": output_dir / "verifier_valid.jsonl",
    }

    by_id = normalize_records_by_id(Path(args.input))
    train_records = select_records(by_id, load_ids(Path(args.train_ids)))
    valid_records = select_records(by_id, load_ids(Path(args.dev_ids)))
    ensure_language(train_records)
    ensure_language(valid_records)

    train_items = build_items(train_records)
    valid_items = build_items(valid_records)
    validate_items(train_items)
    validate_items(valid_items)

    write_jsonl(train_items, paths["train"])
    write_jsonl(valid_items, paths["valid"])
    report = {
        "files": {
            str(paths["train"]): {"split": "train", **summarize(train_items)},
            str(paths["valid"]): {"split": "valid", **summarize(valid_items)},
        },
        "source": {
            "input": str(Path(args.input)),
            "train_ids": str(Path(args.train_ids)),
            "dev_ids": str(Path(args.dev_ids)),
            "test_data_used": False,
        },
        "format": {
            "unit": "one training row per option",
            "target_label": "yes iff option label is in the official answer set, else no",
        },
    }
    write_json(report, Path(args.report))
    print(json.dumps({"train": len(train_items), "valid": len(valid_items), "report": args.report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
