#!/usr/bin/env python3
"""Evaluate Kimi K3 on the complete official GSM8K test split."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

from kimi_k3_prompt import render_prompt

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.replace(",", "").rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else str(number)


def extract_gold(answer: str) -> str | None:
    match = re.search(r"####\s*([^\n]+)", answer)
    numbers = NUMBER_RE.findall(match.group(1)) if match else []
    return normalize_number(numbers[-1] if numbers else None)


def extract_prediction(text: str) -> str | None:
    marked = re.findall(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    numbers = marked or NUMBER_RE.findall(text)
    return normalize_number(numbers[-1] if numbers else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="moonshotai/Kimi-K3")
    parser.add_argument("--limit", type=int, default=1319)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output",
                        type=Path,
                        default=Path("artifacts/kimi-k3-gsm8k.jsonl"))
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    dataset_path = hf_hub_download("openai/gsm8k",
                                   "main/test-00000-of-00001.parquet",
                                   repo_type="dataset")
    rows = pd.read_parquet(dataset_path).iloc[args.offset:args.offset +
                                              args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_lock = threading.Lock()
    correct = completed = 0
    started = time.monotonic()

    def evaluate_row(item):
        index, row = item
        instruction = (
            "Solve this math problem. Show concise work and end with "
            f"'#### ' followed by only the numeric answer.\n\n{row['question']}"
        )
        payload = {
            "model": args.model,
            "prompt": render_prompt(instruction),
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        record = {
            "index": int(index),
            "question": row["question"],
            "gold": extract_gold(row["answer"]),
        }
        request_started = time.monotonic()
        response = None
        try:
            response = requests.post(
                f"{args.base_url.rstrip('/')}/completions",
                json=payload,
                timeout=args.timeout)
            record["status_code"] = response.status_code
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0].get("text") or ""
            record.update({
                "content":
                content,
                "finish_reason":
                body["choices"][0].get("finish_reason"),
                "usage":
                body.get("usage"),
                "prediction":
                extract_prediction(content),
            })
            record["correct"] = record["prediction"] == record["gold"]
        except Exception as error:
            record["error"] = repr(error)
            if response is not None:
                record["response_text"] = response.text[:4000]
        record["latency_seconds"] = time.monotonic() - request_started
        return record

    with args.output.open("w", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(evaluate_row, item)
                for item in rows.iterrows()
            ]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                with output_lock:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                if record.get("status_code") == 200:
                    completed += 1
                    correct += int(record.get("correct", False))
                accuracy = correct / completed if completed else 0.0
                print(
                    f"{completed}/{len(rows)} completed; {correct} correct; "
                    f"accuracy={accuracy:.4f}; index={record['index']}; "
                    f"latency={record['latency_seconds']:.2f}s",
                    flush=True)

    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "requested": len(rows),
                "completed": completed,
                "correct": correct,
                "accuracy": correct / completed if completed else None,
                "elapsed_seconds": elapsed,
            },
            indent=2))


if __name__ == "__main__":
    main()
