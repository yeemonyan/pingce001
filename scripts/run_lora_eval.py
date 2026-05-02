"""Run dev inference/evaluation for a trained LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_score import evaluate, load_json_or_jsonl
from scripts.infer_score import (
    DEFAULT_PROMPTS_PATH,
    TransformersBackend,
    dump_jsonl,
    load_jsonl,
    load_prompts,
    run_inference,
)


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter on SCoRE2026 dev data.")
    parser.add_argument("--config", default="configs/train_lora.yaml", help="Training YAML config.")
    parser.add_argument("--adapter-path", default=None, help="Optional adapter/checkpoint path override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    model_cfg = config["model"]
    training_cfg = config["training"]
    eval_cfg = config["evaluation"]

    adapter_path = args.adapter_path or training_cfg["output_dir"]
    prompts = load_prompts(Path(config["data"].get("prompt_config", str(DEFAULT_PROMPTS_PATH))))
    records = load_jsonl(Path(eval_cfg["dev_prompts"]))

    backend = TransformersBackend(
        model_path=str(Path(model_cfg["local_model_path"]) if Path(model_cfg["local_model_path"]).exists() else model_cfg["base_model"]),
        adapter_path=adapter_path,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        dtype="bfloat16",
        device_map="auto",
    )
    predictions, _ = run_inference(records, prompts, backend)
    prediction_path = Path(eval_cfg["prediction_output"])
    dump_jsonl(predictions, prediction_path)

    report = evaluate(load_json_or_jsonl(Path(eval_cfg["dev_prompts"])), predictions)
    report_path = Path(eval_cfg["eval_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prediction_output": str(prediction_path), "eval_report": str(report_path), "accuracy": report["accuracy"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
