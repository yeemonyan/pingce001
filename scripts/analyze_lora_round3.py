"""Round 3 LoRA dev analysis and prompt draft generation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_errors import analyze, classify_failure
from scripts.evaluate_score import align_predictions, load_json_or_jsonl, normalize_answer
from scripts.infer_score import infer_domain


FOCUS_TYPES = (
    "single_to_multi",
    "multi_missing",
    "temporal_calculation_error",
    "spatial_reference_error",
)


PROMPT_DRAFT = {
    "temporal": """You are solving SCoRE2026 temporal reasoning problems.
Use this hard procedure:
1. Build a compact timeline table with all events and known dates/durations.
2. Compute every before/after relation numerically.
3. For weekdays, use modulo 7 and write the final weekday before checking options.
4. Check A/B/C/D one by one against the timeline.
5. Select only options exactly supported by the timeline.
Return only JSON: {\"answer\":[\"A\"]}.""",
    "spatial": """You are solving SCoRE2026 spatial reasoning problems.
Use this hard procedure:
1. Name the reference frame first: observer, character, shelf/map, circle, row, or table.
2. Track facing direction before interpreting left/right/front/back/up/down.
3. Build a compact layout table of all entities and positions.
4. Check A/B/C/D one by one against the layout.
5. If the correct relation is absent, choose the none-of-the-above option.
Return only JSON: {\"answer\":[\"A\"]}.""",
    "social": """You are solving SCoRE2026 social relationship reasoning problems.
Use this hard procedure:
1. List all people.
2. Build relation chains such as X -> teacher -> Y.
3. Normalize aliases and role rewrites before judging options.
4. Apply valid inverse relations: teacher/disciple, leader/subordinate, parent/child, partner/ex-partner, neighbor, friend.
5. Check A/B/C/D one by one; select only exactly entailed relations.
Return only JSON: {\"answer\":[\"A\"]}.""",
    "hybrid": """You are solving SCoRE2026 hybrid reasoning problems.
Use this hard procedure:
1. Split the problem into subdomains: temporal, spatial, social, natural.
2. Solve each subdomain locally in a small table.
3. Merge the local tables and check for conflicts.
4. Do not answer from one clue if multiple domains interact.
5. If the domain mix is unclear, fall back to the general option-by-option method.
Return only JSON: {\"answer\":[\"A\"]}.""",
    "general": """You are solving SCoRE2026 reasoning problems.
Infer the hidden structure, check A/B/C/D one by one, and select only exactly supported options.
Return only JSON: {\"answer\":[\"A\"]}.""",
}


def answer_key(record: dict[str, Any]) -> str:
    answer = normalize_answer(record.get("answer"))
    return ",".join(answer) if answer else "<empty>"


def build_cases(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]], max_cases_per_type: int) -> list[dict[str, Any]]:
    aligned = align_predictions(gold_records, pred_records)
    kept: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    for index, (gold, pred) in enumerate(aligned):
        gold_answer = normalize_answer(gold.get("answer"))
        pred_answer = normalize_answer(pred.get("answer")) if pred else []
        if not gold_answer or set(gold_answer) == set(pred_answer):
            continue
        failure_type = classify_failure(gold, pred, gold_answer, pred_answer)
        if kept[failure_type] >= max_cases_per_type:
            continue
        cases.append(
            {
                "index": index,
                "id": gold.get("id"),
                "domain": infer_domain(gold),
                "language": gold.get("language"),
                "failure_type": failure_type,
                "gold": gold_answer,
                "prediction": pred_answer,
                "question": gold.get("question"),
                "options": gold.get("options"),
                "text_preview": str(gold.get("text", ""))[:360],
                "raw_output": (pred or {}).get("raw_output"),
            }
        )
        kept[failure_type] += 1
    return cases


def hybrid_analysis(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]) -> dict[str, Any]:
    aligned = align_predictions(gold_records, pred_records)
    total = correct = 0
    failure_counts: Counter[str] = Counter()
    answer_pattern_gold: Counter[str] = Counter()
    answer_pattern_pred: Counter[str] = Counter()
    examples = []
    for index, (gold, pred) in enumerate(aligned):
        if infer_domain(gold) != "hybrid":
            continue
        gold_answer = normalize_answer(gold.get("answer"))
        pred_answer = normalize_answer(pred.get("answer")) if pred else []
        if not gold_answer:
            continue
        total += 1
        matched = set(gold_answer) == set(pred_answer)
        correct += int(matched)
        answer_pattern_gold[",".join(gold_answer)] += 1
        answer_pattern_pred[",".join(pred_answer) if pred_answer else "<empty>"] += 1
        if not matched:
            failure_type = classify_failure(gold, pred, gold_answer, pred_answer)
            failure_counts[failure_type] += 1
            if len(examples) < 12:
                examples.append(
                    {
                        "index": index,
                        "id": gold.get("id"),
                        "failure_type": failure_type,
                        "gold": gold_answer,
                        "prediction": pred_answer,
                        "question": gold.get("question"),
                        "text_preview": str(gold.get("text", ""))[:360],
                    }
                )
    accuracy = correct / total if total else None
    drag_signal = {
        "likely_prompt_drag": bool(total and accuracy is not None and accuracy < 0.3),
        "reason": "Hybrid dev accuracy is low and failures cluster around joint/spatial constraints; test a shorter hybrid prompt with explicit fallback to general.",
    }
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 6) if accuracy is not None else None,
        "failure_counts": dict(sorted(failure_counts.items())),
        "gold_answer_patterns": dict(sorted(answer_pattern_gold.items())),
        "pred_answer_patterns": dict(sorted(answer_pattern_pred.items())),
        "drag_signal": drag_signal,
        "examples": examples,
    }


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_prompt_config(path: Path) -> None:
    lines = []
    for key in ("temporal", "spatial", "social", "hybrid", "general"):
        lines.append(f"{key}: |")
        for line in PROMPT_DRAFT[key].splitlines():
            lines.append(f"  {line}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], cases: list[dict[str, Any]], path: Path) -> None:
    hybrid = report["hybrid_analysis"]
    lines = [
        "# LoRA Dev Error Analysis",
        "",
        "## Summary",
        "",
        f"- Total: {report['total']}",
        f"- Correct: {report['correct']}",
        f"- Accuracy: {report['accuracy']}",
        "",
        "## Focus Error Counts",
        "",
    ]
    for key in FOCUS_TYPES:
        lines.append(f"- `{key}`: {report['focus_failure_counts'].get(key, 0)}")
    lines.extend(
        [
            "",
            "## Hybrid Subset",
            "",
            f"- Total: {hybrid['total']}",
            f"- Correct: {hybrid['correct']}",
            f"- Accuracy: {hybrid['accuracy']}",
            f"- Likely prompt drag: {hybrid['drag_signal']['likely_prompt_drag']}",
            f"- Note: {hybrid['drag_signal']['reason']}",
            "",
            "## Recommendation",
            "",
            "- Use `configs/system_prompts_round3_short.yaml` for the next dev prompt experiment.",
            "- Prioritize temporal and spatial first; watch whether hybrid improves or regresses with fallback-to-general wording.",
            "- Keep answer_only LoRA as the current training baseline; use rationale_json next only if prompt-only gains are limited.",
            "",
            "## Sample Cases",
            "",
        ]
    )
    for case in cases[:24]:
        lines.append(
            f"- `{case['id']}` domain=`{case['domain']}` type=`{case['failure_type']}` "
            f"gold={case['gold']} pred={case['prediction']} question={case.get('question')!r}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LoRA dev predictions for round 3 prompt tuning.")
    parser.add_argument("--gold", default="outputs/dev_prompts.jsonl")
    parser.add_argument("--pred", default="outputs/dev_lora_answer_only_predictions.jsonl")
    parser.add_argument("--report", default="outputs/error_report_lora_dev.json")
    parser.add_argument("--cases", default="outputs/error_cases_lora_dev.jsonl")
    parser.add_argument("--doc", default="docs/ERROR_ANALYSIS_LORA.md")
    parser.add_argument("--prompt-config", default="configs/system_prompts_round3_short.yaml")
    parser.add_argument("--max-cases-per-type", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_records = load_json_or_jsonl(Path(args.gold))
    pred_records = load_json_or_jsonl(Path(args.pred))
    report, _ = analyze(gold_records, pred_records, max_cases_per_type=args.max_cases_per_type)
    cases = build_cases(gold_records, pred_records, max_cases_per_type=args.max_cases_per_type)
    focus_counts = {key: report["failure_counts"].get(key, 0) for key in FOCUS_TYPES}
    report["focus_failure_counts"] = focus_counts
    report["hybrid_analysis"] = hybrid_analysis(gold_records, pred_records)
    report["prompt_config"] = args.prompt_config
    write_json(report, Path(args.report))
    write_jsonl(cases, Path(args.cases))
    write_prompt_config(Path(args.prompt_config))
    write_markdown(report, cases, Path(args.doc))
    print(json.dumps({"accuracy": report["accuracy"], "cases": len(cases), "prompt_config": args.prompt_config}, ensure_ascii=False))


if __name__ == "__main__":
    main()
