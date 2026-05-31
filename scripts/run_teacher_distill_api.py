"""Run API teacher distillation requests against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_TIMEOUT = 180


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher distillation requests via an OpenAI-compatible API.")
    parser.add_argument("--input", required=True, help="Teacher request JSONL file.")
    parser.add_argument("--output", required=True, help="Teacher response JSONL file.")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("DASHSCOPE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-requests", type=int, default=0, help="0 means all requests.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--flush-every", type=int, default=1)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for item in load_jsonl(path):
        record_id = str(item.get("id", "")).strip()
        if record_id:
            ids.add(record_id)
    return ids


def build_request_payload(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    reasoning_effort: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    return payload


def request_once(
    *,
    session: requests.Session,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    url = base_url.rstrip("/") + "/chat/completions"
    response = session.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.text
    body = response.json()
    choices = body.get("choices") or []
    content = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = str(message.get("content", "")).strip()
    parsed = extract_json_object(content)
    return body, content, parsed


def run_single(
    item: dict[str, Any],
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
    max_retries: int,
    temperature: float,
    reasoning_effort: str,
    max_tokens: int,
    sleep_between: float,
) -> dict[str, Any]:
    session = requests.Session()
    payload = build_request_payload(model, list(item["messages"]), temperature, reasoning_effort, max_tokens)
    last_error = ""
    try:
        for attempt in range(1, max_retries + 1):
            try:
                body, content, parsed = request_once(
                    session=session,
                    base_url=base_url,
                    api_key=api_key,
                    payload=payload,
                    timeout=timeout,
                )
                result = {
                    "id": item["id"],
                    "split": item.get("split"),
                    "domain": item.get("domain"),
                    "language": item.get("language"),
                    "answer_kind": item.get("answer_kind"),
                    "request_version": item.get("request_version"),
                    "teacher_model": model,
                    "teacher_base_url": base_url,
                    "attempt": attempt,
                    "usage": body.get("usage"),
                    "raw_response": body,
                    "raw_output": content,
                    "response_json": parsed,
                }
                if parsed is None:
                    result["parse_error"] = "json_not_found"
                if sleep_between > 0:
                    time.sleep(sleep_between)
                return result
            except Exception as exc:  # pragma: no cover - network/runtime
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    time.sleep(min(2**attempt + random.random(), 10))
        result = {
            "id": item["id"],
            "split": item.get("split"),
            "domain": item.get("domain"),
            "language": item.get("language"),
            "answer_kind": item.get("answer_kind"),
            "request_version": item.get("request_version"),
            "teacher_model": model,
            "teacher_base_url": base_url,
            "attempt": max_retries,
            "response_json": None,
            "raw_output": "",
            "error": last_error or "unknown_error",
        }
        if sleep_between > 0:
            time.sleep(sleep_between)
        return result
    finally:
        session.close()


def append_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    if not items:
        return
    ensure_parent(path)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing API key env: {args.api_key_env}")

    random.seed(args.seed)
    input_path = Path(args.input)
    output_path = Path(args.output)
    requests_data = load_jsonl(input_path)
    if args.max_requests > 0:
        requests_data = requests_data[: args.max_requests]

    completed_ids = load_completed_ids(output_path) if args.resume else set()
    pending = [item for item in requests_data if str(item["id"]) not in completed_ids]

    lock = threading.Lock()
    completed = 0
    written_batch: list[dict[str, Any]] = []

    def flush(force: bool = False) -> None:
        nonlocal written_batch
        if not written_batch:
            return
        if force or len(written_batch) >= max(1, args.flush_every):
            append_jsonl(output_path, written_batch)
            written_batch = []

    def handle_result(result: dict[str, Any]) -> None:
        nonlocal completed, written_batch
        with lock:
            completed += 1
            written_batch.append(result)
            flush()
            if completed % max(1, args.progress_every) == 0 or completed == len(pending):
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": len(pending),
                            "last_id": result.get("id"),
                            "parsed": result.get("response_json") is not None,
                            "error": result.get("error"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if not pending:
        print(json.dumps({"completed": 0, "total": 0, "message": "nothing_to_do"}, ensure_ascii=False))
        return

    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                run_single,
                item,
                base_url=args.base_url,
                model=args.model,
                api_key=api_key,
                timeout=args.timeout,
                max_retries=args.max_retries,
                temperature=args.temperature,
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
                sleep_between=args.sleep_between,
            )
            for item in pending
        ]
        for future in cf.as_completed(futures):
            handle_result(future.result())
    flush(force=True)
    print(json.dumps({"completed": completed, "total": len(pending), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
