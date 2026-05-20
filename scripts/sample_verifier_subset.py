"""Build a small question-level subset from verifier train/valid JSONL."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_by_question(records: list[dict[str, Any]], questions: int, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["question_id"])].append(record)
    question_ids = list(grouped.keys())
    random.shuffle(question_ids)
    selected_ids = set(question_ids[:questions])
    sampled = [record for record in records if str(record["question_id"]) in selected_ids]
    sampled.sort(key=lambda record: (str(record["question_id"]), str(record["option_label"])))
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a small verifier question subset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--questions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.input))
    sampled = sample_by_question(records, args.questions, args.seed)
    write_jsonl(sampled, Path(args.output))
    print(json.dumps({"input": args.input, "output": args.output, "rows": len(sampled), "questions": args.questions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
