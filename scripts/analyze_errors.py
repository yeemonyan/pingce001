"""Analyze baseline/dev prediction errors by domain and failure type."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_score import align_predictions, load_json_or_jsonl, normalize_answer
from scripts.infer_score import infer_domain


DEFAULT_GOLD = Path("outputs/dev_prompts.jsonl")
DEFAULT_PRED = Path("outputs/dev_baseline_predictions.jsonl")
DEFAULT_REPORT = Path("outputs/error_report_baseline_dev.json")
DEFAULT_CASES = Path("outputs/error_cases_baseline_dev.jsonl")
DEFAULT_DOC = Path("docs/ERROR_ANALYSIS_BASELINE.md")

FAILURE_TYPES = (
    "single_to_multi",
    "multi_missing",
    "spatial_reference_error",
    "temporal_calculation_error",
    "social_relation_error",
    "natural_property_error",
    "multi_constraint_failure",
    "output_format_error",
    "wrong_single_choice",
)

SPATIAL_CUES = ("left", "right", "above", "below", "clockwise", "adjacent", "左", "右", "上", "下", "顺时针")
TEMPORAL_CUES = ("year", "day", "before", "after", "weekday", "week", "month", "年", "天", "之前", "之后", "周", "星期")
SOCIAL_CUES = ("friend", "teacher", "subordinate", "neighbor", "father", "mother", "buddy", "disciple", "朋友", "师傅", "下属", "邻居", "父", "母", "弟子")
NATURAL_CUES = ("food", "drink", "tool", "animal", "color", "plant", "flower", "fruit", "食品", "饮品", "工具", "动物", "颜色", "植物", "花", "水果")


def text_blob(record: dict[str, Any]) -> str:
    options = record.get("options") or {}
    option_text = " ".join(str(value) for value in options.values()) if isinstance(options, dict) else ""
    return " ".join(str(record.get(key, "")) for key in ("text", "question")) + " " + option_text


def classify_failure(gold: dict[str, Any], pred: dict[str, Any] | None, gold_answer: list[str], pred_answer: list[str]) -> str:
    if pred is None or not pred_answer:
        return "output_format_error"
    if len(gold_answer) == 1 and len(pred_answer) > 1:
        return "single_to_multi"
    if len(gold_answer) > 1 and set(pred_answer) < set(gold_answer):
        return "multi_missing"

    domain = infer_domain(gold)
    blob = text_blob(gold).lower()

    # Prefer the declared/inferred domain first. Cue words are only a fallback
    # for records that land in "general" after routing.
    if domain == "hybrid":
        return "multi_constraint_failure"
    if domain == "temporal":
        return "temporal_calculation_error"
    if domain == "social":
        return "social_relation_error"
    if domain == "natural":
        return "natural_property_error"
    if domain == "spatial":
        return "spatial_reference_error"
    if any(word in blob for word in TEMPORAL_CUES):
        return "temporal_calculation_error"
    if any(word in blob for word in SOCIAL_CUES):
        return "social_relation_error"
    if any(word in blob for word in NATURAL_CUES):
        return "natural_property_error"
    if any(word in blob for word in SPATIAL_CUES):
        return "spatial_reference_error"
    return "wrong_single_choice"


def analyze(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]], max_cases_per_type: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aligned = align_predictions(gold_records, pred_records)
    domain_total: Counter[str] = Counter()
    domain_correct: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    failure_by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    cases: list[dict[str, Any]] = []
    kept_by_type: Counter[str] = Counter()
    total = correct = missing = invalid_gold = 0

    for index, (gold, pred) in enumerate(aligned):
        gold_answer = normalize_answer(gold.get("answer"))
        if not gold_answer:
            invalid_gold += 1
            continue
        pred_answer = normalize_answer(pred.get("answer")) if pred else []
        domain = infer_domain(gold)
        matched = set(gold_answer) == set(pred_answer)
        total += 1
        correct += int(matched)
        missing += int(pred is None)
        domain_total[domain] += 1
        domain_correct[domain] += int(matched)

        if matched:
            continue

        failure_type = classify_failure(gold, pred, gold_answer, pred_answer)
        failure_counts[failure_type] += 1
        failure_by_domain[domain][failure_type] += 1
        case = {
            "index": index,
            "id": gold.get("id"),
            "domain": domain,
            "language": gold.get("language"),
            "failure_type": failure_type,
            "gold": gold_answer,
            "prediction": pred_answer,
            "question": gold.get("question"),
            "options": gold.get("options"),
            "text_preview": str(gold.get("text", ""))[:320],
            "raw_output": (pred or {}).get("raw_output"),
        }
        if kept_by_type[failure_type] < max_cases_per_type:
            cases.append(case)
            kept_by_type[failure_type] += 1

    per_domain = {}
    for domain in sorted(domain_total):
        count = domain_total[domain]
        domain_ok = domain_correct[domain]
        per_domain[domain] = {
            "total": count,
            "correct": domain_ok,
            "accuracy": round(domain_ok / count, 6) if count else None,
            "errors": count - domain_ok,
            "failure_types": dict(sorted(failure_by_domain[domain].items())),
        }

    report = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else None,
        "missing_predictions": missing,
        "invalid_gold_records": invalid_gold,
        "per_domain": per_domain,
        "failure_counts": dict(sorted(failure_counts.items())),
        "case_count": len(cases),
        "recommendations": build_recommendations(per_domain, failure_counts),
    }
    return report, cases


def build_recommendations(per_domain: dict[str, Any], failure_counts: Counter[str]) -> list[str]:
    recommendations = []
    if per_domain:
        worst_domain, stats = min(
            per_domain.items(),
            key=lambda item: item[1]["accuracy"] if item[1]["accuracy"] is not None else 1,
        )
        recommendations.append(
            f"Prioritize {worst_domain} because it has the lowest dev accuracy ({stats['accuracy']})."
        )
    if failure_counts.get("multi_missing", 0):
        recommendations.append("Strengthen multi-answer training and decoding; many errors miss one or more gold labels.")
    if failure_counts.get("single_to_multi", 0):
        recommendations.append("Add constraints that discourage over-selecting labels on single-answer questions.")
    if failure_counts.get("temporal_calculation_error", 0):
        recommendations.append("Improve temporal prompts with explicit timeline tables and weekday/year arithmetic.")
    if failure_counts.get("spatial_reference_error", 0):
        recommendations.append("Improve spatial prompts with explicit reference-frame tracking for left/right/up/down.")
    if failure_counts.get("multi_constraint_failure", 0):
        recommendations.append("For hybrid questions, force separate domain notes before merging constraints.")
    return recommendations


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_markdown(report: dict[str, Any], cases: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Baseline Dev Error Analysis",
        "",
        "## Summary",
        "",
        f"- Total: {report['total']}",
        f"- Correct: {report['correct']}",
        f"- Accuracy: {report['accuracy']}",
        f"- Missing predictions: {report['missing_predictions']}",
        "",
        "## Domain Results",
        "",
        "| Domain | Total | Correct | Accuracy | Errors |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for domain, stats in report["per_domain"].items():
        lines.append(f"| {domain} | {stats['total']} | {stats['correct']} | {stats['accuracy']} | {stats['errors']} |")
    lines.extend(["", "## Failure Types", ""])
    for failure_type, count in report["failure_counts"].items():
        lines.append(f"- `{failure_type}`: {count}")
    lines.extend(["", "## Recommendations", ""])
    for item in report["recommendations"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Sample Error Cases", ""])
    for case in cases[:20]:
        lines.append(
            f"- `{case['id']}` domain=`{case['domain']}` type=`{case['failure_type']}` "
            f"gold={case['gold']} pred={case['prediction']} question={case.get('question')!r}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SCoRE2026 dev prediction errors.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="Dev gold JSONL.")
    parser.add_argument("--pred", default=str(DEFAULT_PRED), help="Prediction JSONL.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Report JSON output.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="Error cases JSONL output.")
    parser.add_argument("--doc", default=str(DEFAULT_DOC), help="Markdown summary output.")
    parser.add_argument("--max-cases-per-type", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, cases = analyze(
        load_json_or_jsonl(Path(args.gold)),
        load_json_or_jsonl(Path(args.pred)),
        max_cases_per_type=args.max_cases_per_type,
    )
    write_json(report, Path(args.report))
    write_jsonl(cases, Path(args.cases))
    write_markdown(report, cases, Path(args.doc))
    print(json.dumps({"report": args.report, "cases": len(cases), "accuracy": report["accuracy"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
