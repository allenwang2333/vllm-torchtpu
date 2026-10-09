#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic greedy needle-in-a-haystack and reasoning probes for Qwen3.5-397B."""

import argparse
import json
import sys
import urllib.request
from typing import Any


def build_needle_prompt(target_words: int, needle_code: str) -> str:
    filler = (
        "The sky above the port was the color of television, tuned to a dead channel. "
        "Clouds drifted slowly across the horizon while engineers inspected the "
        "distributed pipeline stages and verified tensor-parallel reductions. "
    )
    filler_words = len(filler.split())
    repeats = max(2, target_words // filler_words)
    half = repeats // 2
    prefix = filler * half
    suffix = filler * (repeats - half)
    needle = (
        f"\n[CRITICAL RECORD: The secret verification passkey for this stage is "
        f"'{needle_code}'. Remember this exact passkey.]\n"
    )
    question = (
        "\nBased on the text above, what is the exact secret verification passkey? "
        "Reply with the passkey."
    )
    return prefix + needle + suffix + question


def format_qwen35_prompt(user_prompt: str) -> str:
    return (
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def query_completion(
    host: str,
    port: int,
    model: str,
    prompt: str,
    max_tokens: int = 64,
) -> str:
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": format_qwen35_prompt(prompt),
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["text"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--output", default="/tmp/qwen35_probes.json")
    args = parser.parse_args()

    probes: list[dict[str, Any]] = [
        {
            "name": "math_reasoning",
            "prompt": (
                "Solve concisely in 2-3 sentences: A train travels 120 km at "
                "60 km/h, then 180 km at 90 km/h. What is the average speed "
                "for the entire journey in km/h? State the final numeric value."
            ),
            "expected": "75",
            "max_tokens": 512,
        },
        {
            "name": "needle_4k",
            "prompt": build_needle_prompt(2800, "OMEGA-7429-TPU"),
            "expected": "OMEGA-7429-TPU",
            "max_tokens": 48,
        },
        {
            "name": "needle_8k",
            "prompt": build_needle_prompt(5600, "SIGMA-9183-VLLM"),
            "expected": "SIGMA-9183-VLLM",
            "max_tokens": 48,
        },
        {
            "name": "needle_16k",
            "prompt": build_needle_prompt(11200, "DELTA-4402-PALLAS"),
            "expected": "DELTA-4402-PALLAS",
            "max_tokens": 48,
        },
        {
            "name": "needle_32k",
            "prompt": build_needle_prompt(22400, "KAPPA-6017-STAGE"),
            "expected": "KAPPA-6017-STAGE",
            "max_tokens": 48,
        },
    ]

    results = []
    all_passed = True
    for probe in probes:
        text = query_completion(
            args.host,
            args.port,
            args.model,
            probe["prompt"],
            max_tokens=probe["max_tokens"],
        )
        passed = probe["expected"] in text
        all_passed = all_passed and passed
        record = {
            "name": probe["name"],
            "expected": probe["expected"],
            "passed": passed,
            "output": text,
        }
        results.append(record)
        print(
            f"[PROBE] {probe['name']}: passed={passed} expected={probe['expected']!r} "
            f"output={text!r}",
            flush=True,
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"all_passed": all_passed, "results": results}, f, indent=2)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
