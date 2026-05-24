"""Merge base teacher responses with retry responses by record id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge teacher retry responses into a base response file.")
    parser.add_argument("--base", required=True, help="Original teacher response JSONL.")
    parser.add_argument("--retry", required=True, help="Retry response JSONL. Duplicate ids overwrite base.")
    parser.add_argument("--output", required=True, help="Merged output JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_items = load_jsonl(Path(args.base))
    retry_items = load_jsonl(Path(args.retry))

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for item in base_items:
        record_id = str(item.get("id", "")).strip()
        if not record_id:
            continue
        if record_id not in merged:
            order.append(record_id)
        merged[record_id] = item

    replaced = 0
    added = 0
    for item in retry_items:
        record_id = str(item.get("id", "")).strip()
        if not record_id:
            continue
        if record_id in merged:
            replaced += 1
        else:
            added += 1
            order.append(record_id)
        merged[record_id] = item

    output_items = [merged[record_id] for record_id in order]
    write_jsonl(output_items, Path(args.output))

    report = {
        "base_count": len(base_items),
        "retry_count": len(retry_items),
        "merged_count": len(output_items),
        "replaced_count": replaced,
        "added_count": added,
        "output": args.output,
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
