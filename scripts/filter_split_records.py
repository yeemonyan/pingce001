"""Filter normalized SCoRE records by a split id list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer_score import infer_domain
from scripts.make_dev_split import detect_language
from scripts.parse_score_json import dump_jsonl, load_records, normalize_record


DEFAULT_INPUT = Path("data/raw/train.json")
DEFAULT_IDS = Path("data/splits/dev_ids.json")
DEFAULT_OUTPUT = Path("outputs/dev_prompts.jsonl")


def load_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("ids file must be a JSON list")
    return [str(item) for item in data]


def normalize_records(input_path: Path) -> dict[str, dict[str, Any]]:
    records = [normalize_record(record, index) for index, record in enumerate(load_records(input_path))]
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record["id"])
        if record_id in by_id:
            raise ValueError(f"duplicate record id: {record_id}")
        record["domain"] = infer_domain(record)
        record["language"] = detect_language(record)
        by_id[record_id] = record
    return by_id


def filter_records(by_id: dict[str, dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    missing = [record_id for record_id in ids if record_id not in by_id]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} ids not found in input: {preview}")
    return [by_id[record_id] for record_id in ids]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter SCoRE records by split ids.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Official train JSON or normalized JSONL.")
    parser.add_argument("--ids-file", default=str(DEFAULT_IDS), help="JSON list of ids to keep.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Filtered normalized JSONL output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = load_ids(Path(args.ids_file))
    records = filter_records(normalize_records(Path(args.input)), ids)
    dump_jsonl(records, Path(args.output))
    print(json.dumps({"input": args.input, "ids": len(ids), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
