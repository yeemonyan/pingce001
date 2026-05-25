"""Run Day 2 SCoRE2026 zero-shot inference.

The script reads normalized JSONL records produced by parse_score_json.py,
selects a domain-specific system prompt, generates model predictions, extracts
answer labels, and optionally computes validation accuracy.

Use --backend mock for local smoke tests without downloading a 7B model.
Use --backend transformers on the GPU server after downloading the model.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import re
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

try:
    import yaml
except ImportError:  # pragma: no cover - handled by runtime message
    yaml = None


VALID_LABELS = ("A", "B", "C", "D")
DEFAULT_PROMPTS_PATH = Path("configs") / "system_prompts_v2.yaml"


DOMAIN_ALIASES = {
    "space": "spatial",
    "spatial": "spatial",
    "空间": "spatial",
    "temporal": "temporal",
    "time": "temporal",
    "时间": "temporal",
    "social": "social",
    "society": "social",
    "社会": "social",
    "natural": "natural",
    "nature": "natural",
    "自然": "natural",
    "hybrid": "hybrid",
    "fusion": "hybrid",
    "mixed": "hybrid",
    "space+nature": "hybrid",
    "space_nature": "hybrid",
    "space-nature": "hybrid",
    "spatial+natural": "hybrid",
    "spatial_natural": "hybrid",
    "spatial-natural": "hybrid",
    "space+social": "hybrid",
    "spatial+social": "hybrid",
    "time+nature": "hybrid",
    "temporal+natural": "hybrid",
    "time+social": "hybrid",
    "temporal+social": "hybrid",
    "融合": "hybrid",
}


class GenerationBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate model output for a single record."""


class MockBackend:
    """Deterministic backend for smoke tests and CI.

    It returns the gold answer when present, otherwise A. This does not measure
    model quality; it verifies data flow, prompt selection, answer extraction,
    and output formatting.
    """

    def __init__(self, current_record: dict[str, Any] | None = None) -> None:
        self.current_record = current_record

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        answer = (self.current_record or {}).get("answer") or ["A"]
        return json.dumps({"answer": answer}, ensure_ascii=False)


def ensure_sklearn_runtime() -> None:
    """Provide a tiny sklearn.metrics fallback for broken inference envs."""

    try:
        from sklearn.metrics import roc_curve  # noqa: F401
        return
    except Exception:
        metrics_module = types.ModuleType("sklearn.metrics")
        metrics_module.__spec__ = importlib.machinery.ModuleSpec(
            "sklearn.metrics",
            loader=None,
        )

        def roc_curve(*args: object, **kwargs: object) -> None:
            raise RuntimeError("sklearn.metrics.roc_curve is unavailable in this runtime.")

        metrics_module.roc_curve = roc_curve

        sklearn_module = sys.modules.get("sklearn") or types.ModuleType("sklearn")
        sklearn_module.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None)
        sklearn_module.metrics = metrics_module
        sys.modules["sklearn"] = sklearn_module
        sys.modules["sklearn.metrics"] = metrics_module


class TransformersBackend:
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        dtype: str,
        device_map: str,
    ) -> None:
        ensure_sklearn_runtime()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if os.environ.get("FORCE_GROUPED_MM_FALLBACK") == "1":
            import transformers.integrations.moe as moe_integration

            moe_integration._can_use_grouped_mm = lambda input, weight, offs: False

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        moe_experts_implementation = os.environ.get("MOE_EXPERTS_IMPLEMENTATION")
        if moe_experts_implementation and hasattr(model.config, "_experts_implementation"):
            model.config._experts_implementation = moe_experts_implementation
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=self.top_p if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_prompts(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        prompts = parse_simple_block_yaml(text)
    else:
        prompts = yaml.safe_load(text)
    if not isinstance(prompts, dict):
        raise ValueError(f"Prompt config must be a mapping: {path}")
    return {str(key): str(value).strip() for key, value in prompts.items()}


def parse_simple_block_yaml(text: str) -> dict[str, str]:
    """Parse the simple `key: |` prompt YAML used by this project."""
    prompts: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.rstrip().endswith(": |"):
            current_key = line.split(":", 1)[0].strip()
            prompts[current_key] = []
            continue
        if current_key is not None:
            prompts[current_key].append(line[2:] if line.startswith("  ") else line)
    return {key: "\n".join(lines).strip() for key, lines in prompts.items()}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_no} is not a JSON object")
        records.append(record)
    return records


def normalize_domain(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return DOMAIN_ALIASES.get(text)


def infer_domain(record: dict[str, Any]) -> str:
    for key in ("domain", "category", "type", "task_type"):
        domain = normalize_domain(record.get(key))
        if domain:
            return domain

    text = " ".join(
        str(record.get(key, ""))
        for key in ("text", "question")
    ).lower()
    has_spatial = any(word in text for word in ("left", "right", "clockwise", "adjacent", "层", "左", "右", "顺时针", "位置"))
    has_temporal = any(word in text for word in ("year", "day", "before", "after", "周", "年", "之前", "之后", "开始", "结束"))
    has_social = any(word in text for word in ("friend", "father", "teacher", "subordinate", "neighbor", "朋友", "父", "师", "下属", "邻居"))
    has_natural = any(word in text for word in ("item", "food", "drink", "tool", "animal", "物品", "食品", "饮品", "工具", "动物"))

    count = sum((has_spatial, has_temporal, has_social, has_natural))
    if count >= 2:
        return "hybrid"
    if has_spatial:
        return "spatial"
    if has_temporal:
        return "temporal"
    if has_social:
        return "social"
    if has_natural:
        return "natural"
    return "general"


def build_user_prompt(record: dict[str, Any]) -> str:
    return build_user_prompt_with_count_hint(record, include_answer_count_hint=False)


def auto_count_hint(record: dict[str, Any]) -> str:
    """Use predicted_cardinality or cardinality_hint field if available."""
    # Option 1: pre-computed hint text
    hint = record.get('cardinality_hint')
    if hint:
        return str(hint)
    # Option 2: predicted_cardinality field
    kind = record.get('predicted_cardinality')
    if kind in ('single', 'multi'):
        language = str(record.get('language') or '').strip().lower()
        if not language:
            text = f"{record.get('text', '')} {record.get('question', '')}"
            language = 'zh' if any('一' <= ch <= '鿿' for ch in text) else 'en'
        if kind == 'single':
            return (
                "这是一道单选题。你只能选择一个最符合题意的选项。"
                if language == 'zh'
                else "This is a single-answer question. You must select exactly one best-supported option."
            )
        else:
            return (
                "这是一道多选题。请选出所有正确选项，不要漏选，也不要多选。"
                if language == 'zh'
                else "This is a multi-answer question. Select all supported options without missing any and without adding extras."
            )
    return ''


def infer_answer_kind(record: dict[str, Any]) -> str:
    answers = record.get("answer")
    if isinstance(answers, list) and len(answers) > 1:
        return "multi"
    return "single"


def answer_count_hint(record: dict[str, Any]) -> str:
    language = str(record.get("language") or "").strip().lower()
    if not language:
        text = f"{record.get('text', '')} {record.get('question', '')}"
        language = "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "en"
    kind = infer_answer_kind(record)
    if kind == "single":
        return (
            "这是一道单选题。你只能选择一个最符合题意的选项。"
            if language == "zh"
            else "This is a single-answer question. You must select exactly one best-supported option."
        )
    return (
        "这是一道多选题。请选出所有正确选项，不要漏选，也不要多选。"
        if language == "zh"
        else "This is a multi-answer question. Select all supported options without missing any and without adding extras."
    )


def build_user_prompt_with_count_hint(
    record: dict[str, Any],
    *,
    include_answer_count_hint: bool = False,
    include_auto_count_hint: bool = False,
) -> str:
    option_lines = "\n".join(
        f"{label}. {value}"
        for label, value in record["options"].items()
    )
    # Use neutral phrasing that does NOT bias toward plural.
    # "Choose ALL correct" caused over-selection when single/multi hint was missing.
    prompt = (
        f"Text:\n{record['text']}\n\n"
        f"Question:\n{record['question']}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Select every correct option and output only JSON."
    )
    if include_answer_count_hint:
        prompt += "\n\n" + answer_count_hint(record)
    elif include_auto_count_hint:
        hint = auto_count_hint(record)
        if hint:
            prompt += "\n\n" + hint
    return prompt


def extract_answer(output: str, allowed_labels: set[str]) -> list[str]:
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            for key in ("answer", "answers"):
                if isinstance(parsed.get(key), list):
                    labels = [str(item).strip().upper() for item in parsed[key]]
                    return dedupe_valid_labels(labels, allowed_labels)
        if isinstance(parsed, list):
            labels = [str(item).strip().upper() for item in parsed]
            return dedupe_valid_labels(labels, allowed_labels)
    except json.JSONDecodeError:
        pass

    for key in ("answer", "answers"):
        match = re.search(rf'"{key}"\s*:\s*\[([^\]]+)\]', output, flags=re.IGNORECASE)
        if match:
            labels = re.findall(r"[A-D]", match.group(1).upper())
            result = dedupe_valid_labels(labels, allowed_labels)
            if result:
                return result

    labels = re.findall(r"\b[A-D]\b", output.upper())
    return dedupe_valid_labels(labels, allowed_labels)


def dedupe_valid_labels(labels: list[str], allowed_labels: set[str]) -> list[str]:
    result = []
    for label in labels:
        if label in allowed_labels and label not in result:
            result.append(label)
    return result


def is_correct(prediction: list[str], gold: Any) -> bool | None:
    if gold is None:
        return None
    return set(prediction) == set(str(item).strip().upper() for item in gold)


def run_inference(
    records: list[dict[str, Any]],
    prompts: dict[str, str],
    backend: GenerationBackend,
    *,
    include_answer_count_hint: bool = False,
    include_auto_count_hint: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs = []
    correct = 0
    scored = 0
    domain_counts: Counter[str] = Counter()

    for record in records:
        if isinstance(backend, MockBackend):
            backend.current_record = record
        domain = infer_domain(record)
        domain_counts[domain] += 1
        system_prompt = prompts.get(domain) or prompts["general"]
        allowed_labels = set(record["options"].keys())
        raw_output = backend.generate(
            system_prompt,
            build_user_prompt_with_count_hint(
                record,
                include_answer_count_hint=include_answer_count_hint,
                include_auto_count_hint=include_auto_count_hint,
            ),
        )
        prediction = extract_answer(raw_output, allowed_labels)
        matched = is_correct(prediction, record.get("answer"))
        if matched is not None:
            scored += 1
            correct += int(matched)
        outputs.append(
            {
                "id": record.get("id"),
                "domain": domain,
                "answer": prediction,
                "raw_output": raw_output,
                "gold": record.get("answer"),
                "correct": matched,
            }
        )

    metrics: dict[str, Any] = {
        "total": len(records),
        "scored": scored,
        "correct": correct,
        "accuracy": (correct / scored) if scored else None,
        "domain_counts": dict(domain_counts),
    }
    return outputs, metrics


def dump_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCoRE2026 Day 2 inference.")
    parser.add_argument("--input", required=True, help="Normalized input JSONL.")
    parser.add_argument("--output", required=True, help="Prediction output JSONL.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH), help="System prompts YAML.")
    parser.add_argument("--backend", choices=("mock", "transformers"), default="transformers")
    parser.add_argument("--model-path", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--answer-count-hint",
        action="store_true",
        help="Append gold single/multi hint (dev-only, uses gold answer).",
    )
    parser.add_argument(
        "--auto-count-hint",
        action="store_true",
        help="Append predicted single/multi hint from cardinality_hint or predicted_cardinality field.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(Path(args.prompts))
    records = load_jsonl(Path(args.input))

    if args.backend == "mock":
        backend: GenerationBackend = MockBackend()
    else:
        backend = TransformersBackend(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            dtype=args.dtype,
            device_map=args.device_map,
        )

    outputs, metrics = run_inference(
        records, prompts, backend,
        include_answer_count_hint=bool(args.answer_count_hint),
        include_auto_count_hint=bool(args.auto_count_hint),
    )
    dump_jsonl(outputs, Path(args.output))
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
