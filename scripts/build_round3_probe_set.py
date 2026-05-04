"""Build a small dev probe set for low-cost round3 prompt / vote validation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer_score import infer_domain, load_jsonl


DEFAULT_INPUT = "outputs/dev_prompts.jsonl"
DEFAULT_OUTPUT = "outputs/dev_probe_round3.jsonl"

TARGETS = {
    "spatial": 30,
    "temporal": 30,
    "natural": 20,
    "hybrid": 10,
}


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_probe(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[infer_domain(record)].append(record)

    probe: list[dict[str, Any]] = []
    for domain, target in TARGETS.items():
        candidates = list(buckets.get(domain, []))
        random.shuffle(candidates)
        probe.extend(candidates[:target])

    probe.sort(key=lambda record: str(record.get("id")))
    return probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small round3 dev probe set.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.input))
    probe = build_probe(records, args.seed)
    write_jsonl(probe, Path(args.output))
    print(json.dumps({"count": len(probe), "output": args.output, "targets": TARGETS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
