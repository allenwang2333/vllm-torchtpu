#!/usr/bin/env python3
"""Reproduce the `dsv4.state.csa.g4` trim crash against a running DSv4 P/D pair.

Bring the pair up with `launch_dsv4_dp4_dp4.sh`, then run this against the
proxy. It sends a length sweep first and a short prompt second, because the
crash needs both: on an idle replica the short prompt completes normally.

Expected failure, on the prefill replica, which then dies with
`EngineDeadError` and takes every other in-flight request with it:

    Stage-3 sliding-window block table still repeats a page after trimming:
    group=dsv4.state.csa.g4, kept=[0, 0], num_tokens=113, page_tokens=16,
    window=8

Exit code 0 means the crash reproduced, 1 means it did not.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from dsv4_pd_oracle import build_prompt, post


def send(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = post(f"{base}/v1/completions", {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
    }, 900.0)
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    args = parser.parse_args()

    # Churn the allocator. These all succeed; the point is the state they
    # leave behind, not their output.
    for length in (127, 128, 129, 512, 2048, 4096, 8192):
        for attempt in range(2):
            rng = random.Random(1000 * length + attempt)
            prompt = build_prompt(rng, length, f"{rng.randrange(10**7, 10**8)}",
                                  f"{rng.randrange(10**7, 10**8)}", 128)
            body = send(args.base, args.model, prompt, 24)
            print(json.dumps({
                "phase": "warmup",
                "len": length,
                "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
            }), flush=True)

    # The short one. `num_tokens` reaches the connector as 113.
    rng = random.Random(129)
    prompt = build_prompt(rng, 129, f"{rng.randrange(10**7, 10**8)}",
                          f"{rng.randrange(10**7, 10**8)}", 128)
    try:
        body = send(args.base, args.model, prompt, 48)
    except Exception as exc:  # noqa: BLE001 - any failure is the repro
        print(f"REPRODUCED: {str(exc)[:200]}", flush=True)
        print("Check the prefill log for 'still repeats a page after trimming'.",
              flush=True)
        return 0
    print("NOT REPRODUCED:", json.dumps({
        "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
        "text": body["choices"][0]["text"].strip()[:70],
    }), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
