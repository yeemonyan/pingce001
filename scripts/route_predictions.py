"""Route between two SCoRE2026 prediction files.

The usual use case is to keep a dense model as the default prediction source
and selectively replace hard buckets with a MoE prediction file.  The script is
CPU-only and can be used for both dev analysis and test-time submission
assembly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_score import (  # noqa: E402
    align_predictions,
    evaluate,
    load_json_or_jsonl,
    normalize_answer,
)


def domain_of(record: dict[str, Any]) -> str:
    return str(record.get("domain") or record.get("category") or record.get("type") or "unknown")


def answer_kind(record: dict[str, Any] | None) -> str:
    if record is None:
        return "unknown"
    answers = normalize_answer(record.get("answer"))
    if not answers:
        return "unknown"
    return "multi" if len(answers) > 1 else "single"


def parse_bucket(value: str) -> tuple[str, str]:
    parts = value.strip().lower().split(":")
    if len(parts) != 2 or parts[1] not in {"single", "multi", "*"}:
        raise argparse.ArgumentTypeError(
            "Buckets must be DOMAIN:KIND, for example spatial:single, temporal:multi, or hybrid:*"
        )
    return parts[0], parts[1]


def should_use_candidate(
    *,
    input_record: dict[str, Any],
    base_pred: dict[str, Any] | None,
    candidate_pred: dict[str, Any] | None,
    buckets: set[tuple[str, str]],
    kind_source: str,
) -> bool:
    domain = domain_of(input_record).lower()
    if kind_source == "gold":
        kind = answer_kind(input_record)
    elif kind_source == "base":
        kind = answer_kind(base_pred)
    elif kind_source == "candidate":
        kind = answer_kind(candidate_pred)
    else:  # pragma: no cover - argparse choices prevent this
        raise ValueError(f"Unsupported kind source: {kind_source}")
    return (domain, kind) in buckets or (domain, "*") in buckets


def routed_predictions(
    input_records: list[dict[str, Any]],
    base_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    buckets: set[tuple[str, str]],
    kind_source: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_aligned = align_predictions(input_records, base_records)
    candidate_aligned = align_predictions(input_records, candidate_records)

    routed: list[dict[str, Any]] = []
    counts = {"base": 0, "candidate": 0, "missing_candidate": 0}

    for (input_record, base_pred), (_, candidate_pred) in zip(base_aligned, candidate_aligned):
        use_candidate = should_use_candidate(
            input_record=input_record,
            base_pred=base_pred,
            candidate_pred=candidate_pred,
            buckets=buckets,
            kind_source=kind_source,
        )
        source = "candidate" if use_candidate and candidate_pred is not None else "base"
        if use_candidate and candidate_pred is None:
            counts["missing_candidate"] += 1

        selected = candidate_pred if source == "candidate" else base_pred
        answer = normalize_answer(selected.get("answer")) if selected else []
        routed.append(
            {
                "id": input_record.get("id"),
                "domain": domain_of(input_record),
                "answer": answer,
                "source": source,
            }
        )
        counts[source] += 1

    return routed, counts


def dump_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route between dense and candidate SCoRE predictions.")
    parser.add_argument("--input", required=True, help="Input/gold JSONL used for ids and domains.")
    parser.add_argument("--base", required=True, help="Default prediction JSON/JSONL.")
    parser.add_argument("--candidate", required=True, help="Prediction JSON/JSONL used for selected buckets.")
    parser.add_argument("--output", required=True, help="Routed prediction JSONL.")
    parser.add_argument(
        "--use-candidate-bucket",
        action="append",
        type=parse_bucket,
        default=[],
        help="Bucket to replace, e.g. spatial:single. Repeatable. Use domain:* for all kinds.",
    )
    parser.add_argument(
        "--kind-source",
        choices=("gold", "base", "candidate"),
        default="base",
        help=(
            "Where to infer single/multi from. Use base or candidate for test-time-safe routing; "
            "gold is dev-only diagnostics because test data has no answers."
        ),
    )
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    parser.add_argument("--max-mistakes", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_records = load_json_or_jsonl(Path(args.input))
    base_records = load_json_or_jsonl(Path(args.base))
    candidate_records = load_json_or_jsonl(Path(args.candidate))
    buckets = set(args.use_candidate_bucket)

    predictions, source_counts = routed_predictions(
        input_records,
        base_records,
        candidate_records,
        buckets=buckets,
        kind_source=args.kind_source,
    )
    dump_jsonl(predictions, Path(args.output))

    report: dict[str, Any] = {
        "kind_source": args.kind_source,
        "candidate_buckets": [f"{domain}:{kind}" for domain, kind in sorted(buckets)],
        "source_counts": source_counts,
    }
    if any(normalize_answer(record.get("answer")) for record in input_records):
        eval_report = evaluate(input_records, predictions)
        eval_report["mistakes"] = eval_report["mistakes"][: args.max_mistakes]
        report["evaluation"] = eval_report

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
