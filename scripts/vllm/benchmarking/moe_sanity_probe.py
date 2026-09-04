#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ask a running server eight questions it cannot get wrong if it loaded.

BK #299 served a Kimi-K3 whose routed experts were half uninitialized -- the
Run:AI streaming filter fetched a rank's block while the layer had built a
chip's -- and reported a healthy median TPOT for it. Nothing raised, no NaN
reached the log, and `--save-result` without `--save-detailed` persists no
generated text, so the benchmark could not have noticed. A latency number from
an arm that changes what the model *is* proves nothing without this.

Stdlib only: it runs inside the benchmark container next to the server.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.error
import urllib.request

# Short, unambiguous, and answerable by any model worth benchmarking. The
# point is not to measure quality -- it is to separate "this model works" from
# "half its experts are `torch.empty`".
PROMPTS: list[tuple[str, tuple[str, ...]]] = [
    ("The capital city of France is", ("paris", )),
    ("2 + 2 =", ("4", "four")),
    ("The chemical symbol for water is", ("h2o", "h₂o")),
    ("Monday, Tuesday, Wednesday,", ("thursday", )),
    ("Complete the sequence: 1, 2, 3, 4,", ("5", "five")),
    ("The largest planet in the solar system is", ("jupiter", )),
    ("Translate to French: good morning ->", ("bonjour", )),
    ("The opposite of hot is", ("cold", )),
]


def _repeat_fraction(text: str, n: int = 4) -> float:
    """Share of the text covered by its most repeated n-gram of words.

    A model whose expert weights are garbage typically locks into a short
    cycle; a healthy one does not repeat any 4-gram more than incidentally.
    """
    words = text.split()
    if len(words) < 2 * n:
        return 0.0
    grams = collections.Counter(
        tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(grams.values()) * n / len(words)


def score(records: list[dict], min_hits: int,
          max_repeat: float) -> tuple[bool, list[str], dict]:
    """``(passed, reasons, metrics)`` for a probe's completions.

    Pure, so the gate itself is testable on recorded output without a server:
    see ``tests/tools/test_moe_sanity_probe.py``, which drives it with both a
    healthy and a corrupted transcript.
    """
    hits, empties, worst_repeat = 0, 0, 0.0
    for record in records:
        text = (record.get("text") or "").strip().lower()
        if not text:
            empties += 1
            continue
        if any(expect in text for expect in record["expect"]):
            hits += 1
        worst_repeat = max(worst_repeat, _repeat_fraction(text))

    reasons = []
    if hits < min_hits:
        reasons.append(f"only {hits}/{len(records)} prompts answered "
                       f"correctly (need {min_hits})")
    if empties:
        reasons.append(f"{empties} empty completions")
    if worst_repeat > max_repeat:
        reasons.append(f"degenerate output: a 4-gram covers "
                       f"{worst_repeat:.0%} of one completion "
                       f"(limit {max_repeat:.0%})")
    metrics = {
        "hits": hits,
        "asked": len(records),
        "empty": empties,
        "worst_repeat_fraction": round(worst_repeat, 3),
    }
    return not reasons, reasons, metrics


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _served_model(base: str, fallback: str, timeout: float) -> str:
    try:
        with urllib.request.urlopen(f"{base}/v1/models",
                                    timeout=timeout) as response:
            models = json.loads(response.read()).get("data") or []
        return models[0]["id"] if models else fallback
    except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
        return fallback


def collect(base: str, model: str, max_tokens: int,
            timeout: float) -> list[dict]:
    records = []
    for prompt, expect in PROMPTS:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        try:
            body = _post(f"{base}/v1/completions", payload, timeout)
            text = body["choices"][0]["text"]
        except urllib.error.HTTPError:
            body = _post(
                f"{base}/v1/chat/completions", {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                }, timeout)
            text = body["choices"][0]["message"]["content"]
        records.append({
            "prompt": prompt,
            "expect": list(expect),
            "text": text
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--label", default="", help="arm name, for the report")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--min-hits", type=int, default=6)
    parser.add_argument("--max-repeat", type=float, default=0.6)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    model = args.model or _served_model(base, "default", args.timeout)
    records = collect(base, model, args.max_tokens, args.timeout)
    passed, reasons, metrics = score(records, args.min_hits, args.max_repeat)

    report = {
        "label": args.label,
        "model": model,
        "passed": passed,
        "reasons": reasons,
        "metrics": metrics,
        "records": records,
    }
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
    print(
        json.dumps(
            {k: report[k]
             for k in ("label", "passed", "reasons", "metrics")},
            indent=2))
    for record in records:
        print(f"  {record['prompt']!r} -> {record['text'].strip()[:80]!r}")
    if not passed:
        print("SANITY PROBE FAILED: this arm's number is void.",
              file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
