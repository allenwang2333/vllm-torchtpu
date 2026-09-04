#!/usr/bin/env python3
"""Throughput and latency for DeepSeek-V4 P/D disaggregation (PR #684).

`vllm bench serve` wants a tokenizer, and the bench pod has no copy of the
checkpoint and no HF path to DeepSeek-V4-Flash, so this measures the same
quantities without one. Prompt length is set in filler sentences and read back
from the server's own `usage.prompt_tokens`, which is the number that matters
anyway.

What comes out is client-side and includes the toy proxy, which serializes the
prefill leg and adds a hop. Treat the absolute numbers as a floor for the
16-chip pair rather than as the model's speed.

TTFT is measured with `stream: true`. The proxy forwards the client's stream
flag to the decode replica and passes the bytes through, so the first SSE chunk
really is the decode replica's first token, one proxy hop later.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

FILLER = [
    "The harbor master logged every vessel that entered the bay before dawn.",
    "A slow rain settled over the terraced fields and softened the clay paths.",
    "Engineers rerouted the aqueduct after the eastern span showed cracks.",
    "The archivist catalogued letters from the expedition of last winter.",
    "Merchants argued about grain tariffs under the arcades of the exchange.",
    "A lighthouse keeper recorded wind speeds in a leather-bound ledger.",
    "The observatory's brass dome creaked as it rotated toward the meridian.",
    "Cartographers disputed the elevation of the northern pass for a decade.",
    "The mill wheel turned unevenly whenever the sluice gate silted up.",
    "Botanists pressed alpine flowers between sheets of absorbent paper.",
    "The night train carried mail sacks, timber samples, and two violins.",
    "Masons squared the granite blocks with chalk lines and iron mallets.",
]


def filler_words(rng: random.Random, approx_tokens: int) -> str:
    # The filler sentences run about 13 tokens each. Landing in the right
    # neighbourhood is enough, since the served length is read back per request.
    return " ".join(
        rng.choice(FILLER) for _ in range(max(1, approx_tokens // 13)))


def one_request(base: str, model: str, prompt: str, max_tokens: int,
                timeout: float) -> dict:
    """Send one streamed completion and time its first and last chunk."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
        # The proxy strips stream_options off the prefill leg only, so the
        # usage chunk still comes back from the decode replica.
        "stream_options": {
            "include_usage": True
        },
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        f"{base}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at = None
    output_tokens = 0
    prompt_tokens = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    continue
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                text = "".join(
                    choice.get("text", "")
                    for choice in chunk.get("choices", []))
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output_tokens += 1
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        return {"ok": False, "error": repr(exc)[:200]}

    finished = time.perf_counter()
    if first_token_at is None or output_tokens == 0:
        return {"ok": False, "error": "no tokens returned"}
    return {
        "ok":
        True,
        "ttft_ms": (first_token_at - started) * 1000.0,
        "e2e_ms": (finished - started) * 1000.0,
        # Time per output token after the first, which is the decode rate.
        "tpot_ms":
        ((finished - first_token_at) * 1000.0 / max(1, output_tokens - 1)),
        "output_tokens":
        output_tokens,
        "prompt_tokens":
        prompt_tokens,
    }


def run_level(base: str, model: str, prompts: list[str], max_tokens: int,
              concurrency: int, timeout: float) -> dict:
    pending: queue.Queue = queue.Queue()
    for index, prompt in enumerate(prompts):
        pending.put((index, prompt))
    results: list[dict] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                _, prompt = pending.get_nowait()
            except queue.Empty:
                return
            outcome = one_request(base, model, prompt, max_tokens, timeout)
            with lock:
                results.append(outcome)

    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall_s = time.perf_counter() - started

    good = [r for r in results if r["ok"]]
    if not good:
        return {
            "concurrency": concurrency,
            "requests": len(results),
            "failures": len(results),
            "errors": sorted({r.get("error", "")
                              for r in results})[:3],
        }

    ttfts = sorted(r["ttft_ms"] for r in good)
    out_tokens = sum(r["output_tokens"] for r in good)
    in_tokens = sum(r["prompt_tokens"] or 0 for r in good)
    return {
        "concurrency":
        concurrency,
        "requests":
        len(results),
        "failures":
        len(results) - len(good),
        "wall_s":
        round(wall_s, 2),
        "prompt_tokens_median":
        statistics.median(r["prompt_tokens"] or 0 for r in good),
        "output_tokens_total":
        out_tokens,
        "request_throughput_per_s":
        round(len(good) / wall_s, 3),
        "output_throughput_tok_s":
        round(out_tokens / wall_s, 1),
        "total_throughput_tok_s":
        round((in_tokens + out_tokens) / wall_s, 1),
        "ttft_ms_p50":
        round(statistics.median(ttfts), 1),
        "ttft_ms_p99":
        round(ttfts[min(len(ttfts) - 1, int(0.99 * len(ttfts)))], 1),
        "tpot_ms_p50":
        round(statistics.median(r["tpot_ms"] for r in good), 2),
        "e2e_ms_p50":
        round(statistics.median(r["e2e_ms"] for r in good), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",
                        default="http://127.0.0.1:10000",
                        help="proxy base URL (disagg) or a single server")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--requests-per-level", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    # Distinct prompts, so nothing is answered out of the prefix cache. That
    # would measure the cache rather than the transfer.
    def prompts_for(tag: int, count: int) -> list[str]:
        return [
            filler_words(random.Random(tag * 100_003 + i), args.prompt_tokens)
            for i in range(count)
        ]

    for prompt in prompts_for(0, args.warmup):
        one_request(args.base, args.model, prompt, args.max_tokens,
                    args.timeout)

    report = []
    for level in levels:
        summary = run_level(args.base, args.model,
                            prompts_for(level, args.requests_per_level),
                            args.max_tokens, level, args.timeout)
        report.append(summary)
        print(json.dumps(summary), flush=True)

    print(json.dumps({
        "config": vars(args),
        "levels": report
    }, indent=2),
          flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({
                "config": vars(args),
                "levels": report
            },
                      handle,
                      indent=2)
    return 1 if any(level.get("failures") for level in report) else 0


if __name__ == "__main__":
    sys.exit(main())
