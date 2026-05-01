"""Create a stable stratified train/dev split for SCoRE2026 training data."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.infer_score import infer_domain
from scripts.parse_score_json import load_records, normalize_record


DEFAULT_INPUT = Path("data/raw/train.json")
DEFAULT_TRAIN_IDS = Path("data/splits/train_ids.json")
DEFAULT_DEV_IDS = Path("data/splits/dev_ids.json")
DEFAULT_REPORT = Path("outputs/dev_split_report.json")
DEFAULT_SEED = 2026
DEFAULT_DEV_RATIO = 0.2


def detect_language(record: dict[str, Any]) -> str:
    text = f"{record.get('text', '')}\n{record.get('question', '')}"
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def answer_kind(record: dict[str, Any]) -> str:
    answer = record.get("answer") or record.get("answers") or []
    return "multi" if len(answer) > 1 else "single"


def stratify_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (infer_domain(record), detect_language(record), answer_kind(record))


def split_records(
    records: list[dict[str, Any]],
    dev_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[stratify_key(record)].append(record)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []

    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        dev_size = round(len(group) * dev_ratio)
        if len(group) > 1:
            dev_size = max(1, min(len(group) - 1, dev_size))
        else:
            dev_size = 0
        dev.extend(group[:dev_size])
        train.extend(group[dev_size:])

    train.sort(key=lambda item: str(item["id"]))
    dev.sort(key=lambda item: str(item["id"]))
    return train, dev


def count_distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    if field == "domain":
        counter = Counter(infer_domain(record) for record in records)
    elif field == "language":
        counter = Counter(detect_language(record) for record in records)
    elif field == "answer_kind":
        counter = Counter(answer_kind(record) for record in records)
    else:
        raise ValueError(f"unsupported distribution field: {field}")
    return dict(sorted(counter.items()))


def build_report(
    records: list[dict[str, Any]],
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    seed: int,
    dev_ratio: float,
) -> dict[str, Any]:
    strata = Counter("|".join(stratify_key(record)) for record in records)
    return {
        "seed": seed,
        "dev_ratio": dev_ratio,
        "total_count": len(records),
        "train_count": len(train),
        "dev_count": len(dev),
        "train": {
            "domain": count_distribution(train, "domain"),
            "language": count_distribution(train, "language"),
            "answer_kind": count_distribution(train, "answer_kind"),
        },
        "dev": {
            "domain": count_distribution(dev, "domain"),
            "language": count_distribution(dev, "language"),
            "answer_kind": count_distribution(dev, "answer_kind"),
        },
        "all": {
            "domain": count_distribution(records, "domain"),
            "language": count_distribution(records, "language"),
            "answer_kind": count_distribution(records, "answer_kind"),
            "strata": dict(sorted(strata.items())),
        },
    }


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified SCoRE2026 train/dev split.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Official training JSON file.")
    parser.add_argument("--train-ids", default=str(DEFAULT_TRAIN_IDS), help="Output train id list.")
    parser.add_argument("--dev-ids", default=str(DEFAULT_DEV_IDS), help="Output dev id list.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Output split report JSON.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed random seed.")
    parser.add_argument("--dev-ratio", type=float, default=DEFAULT_DEV_RATIO, help="Dev split ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_records = load_records(Path(args.input))
    records = [normalize_record(record, index) for index, record in enumerate(raw_records)]
    records = [record for record in records if record.get("has_answer")]

    train, dev = split_records(records, dev_ratio=args.dev_ratio, seed=args.seed)
    write_json([record["id"] for record in train], Path(args.train_ids))
    write_json([record["id"] for record in dev], Path(args.dev_ids))
    write_json(build_report(records, train, dev, seed=args.seed, dev_ratio=args.dev_ratio), Path(args.report))

    print(
        json.dumps(
            {
                "input": str(args.input),
                "total": len(records),
                "train": len(train),
                "dev": len(dev),
                "seed": args.seed,
                "dev_ratio": args.dev_ratio,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
