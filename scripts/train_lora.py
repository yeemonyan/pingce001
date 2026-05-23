"""Train a LoRA adapter for SCoRE2026 using PEFT + Transformers."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def build_chat_text(messages: list[dict[str, str]], tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def make_dataset(path: Path, tokenizer: Any, max_seq_length: int) -> Any:
    from datasets import Dataset

    rows = []
    for record in load_jsonl(path):
        text = build_chat_text(record["messages"], tokenizer)
        tokenized = tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)
        tokenized["labels"] = list(tokenized["input_ids"])
        rows.append(tokenized)
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for SCoRE2026.")
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and data without training.")
    return parser.parse_args()


def is_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1 or "LOCAL_RANK" in os.environ


def build_training_arguments_kwargs(training_cfg: dict[str, Any], training_arguments_cls: Any) -> dict[str, Any]:
    signature = inspect.signature(training_arguments_cls.__init__)
    params = signature.parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(training_cfg["output_dir"]),
        "seed": int(training_cfg["seed"]),
        "num_train_epochs": float(training_cfg["num_train_epochs"]),
        "learning_rate": float(training_cfg["learning_rate"]),
        "weight_decay": float(training_cfg["weight_decay"]),
        "warmup_ratio": float(training_cfg["warmup_ratio"]),
        "lr_scheduler_type": str(training_cfg["lr_scheduler_type"]),
        "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training_cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
        "logging_steps": int(training_cfg["logging_steps"]),
        "save_strategy": str(training_cfg["save_strategy"]),
        "save_total_limit": int(training_cfg["save_total_limit"]),
        "bf16": bool(training_cfg["bf16"]),
        "fp16": bool(training_cfg["fp16"]),
        "report_to": [],
        "remove_unused_columns": False,
    }
    if is_distributed():
        kwargs["ddp_find_unused_parameters"] = False
    eval_value = str(training_cfg["eval_strategy"])
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_value
    return kwargs


def attach_trainer_processing_kwargs(trainer_kwargs: dict[str, Any], tokenizer: Any, trainer_cls: Any) -> dict[str, Any]:
    signature = inspect.signature(trainer_cls.__init__)
    params = signature.parameters
    if "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    return trainer_kwargs


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]
    lora_cfg = config["lora"]

    set_seed(int(training_cfg["seed"]))

    model_path = Path(model_cfg["local_model_path"])
    resolved_model_path = str(model_path if model_path.exists() else model_cfg["base_model"])

    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = make_dataset(Path(data_cfg["train_file"]), tokenizer, int(training_cfg["max_seq_length"]))
    valid_dataset = make_dataset(Path(data_cfg["valid_file"]), tokenizer, int(training_cfg["max_seq_length"]))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model_path": resolved_model_path,
                    "train_size": len(train_dataset),
                    "valid_size": len(valid_dataset),
                    "max_seq_length": int(training_cfg["max_seq_length"]),
                    "output_dir": training_cfg["output_dir"],
                },
                ensure_ascii=False,
            )
        )
        return

    import torch
    from peft import LoraConfig, get_peft_model

    torch_dtype = torch.bfloat16 if bool(training_cfg.get("bf16", False)) else torch.float16
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "low_cpu_mem_usage": True,
    }
    if not is_distributed():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        **model_kwargs,
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=str(lora_cfg["bias"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    if bool(training_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(**build_training_arguments_kwargs(training_cfg, TrainingArguments))
    trainer_kwargs = attach_trainer_processing_kwargs(
        {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": valid_dataset,
            "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        },
        tokenizer,
        Trainer,
    )
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    (output_dir / "train_summary.json").write_text(
        json.dumps({"train_size": len(train_dataset), "valid_size": len(valid_dataset), "config": config}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
