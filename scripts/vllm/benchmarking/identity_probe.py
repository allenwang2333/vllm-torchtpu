# SPDX-License-Identifier: Apache-2.0
"""Tier-0 semantics gate: fixed prompts, greedy decode, exact output comparison.

For a change that should preserve semantics -- a scheduler flag, a batching
change, a fused kernel epilogue -- the right test is not "does GSM8K still
score 0.97". It is "is the output identical". Any differing token is a bug,
and an accuracy benchmark would not catch it.

This does NOT reuse `vllm bench serve --save-result`: that strips the
per-request list fields, so the saved JSON carries no generated text. Verified
on a real run -- the result keys contain every summary statistic and no
`generated_texts`. Owning the probe also means owning the prompts, which is
what makes the comparison meaningful.

Deterministic by construction: fixed prompt set, temperature 0, fixed
max_tokens, one request at a time.

  python3 identity_probe.py --record  --out baselines/identity/<config>.json
  python3 identity_probe.py --compare --out baselines/identity/<config>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue.",
    "def fibonacci(n):",
    "List the first five prime numbers:",
    "Q: If a train travels 60 km in 45 minutes, what is its speed in km/h?\nA:",
    "Translate to French: The weather is pleasant today.",
    "Summarize: TPUs are accelerators designed for machine learning workloads.",
    "Continue the sequence: 2, 4, 8, 16,",
]


def generate(host, port, model, prompt, max_tokens):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "seed": 42, "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://{host}:{port}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", action="store_true")
    g.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    outs = []
    for i, p in enumerate(PROMPTS):
        try:
            outs.append(generate(a.host, a.port, a.model, p, a.max_tokens))
        except Exception as e:                       # a dead server is a failure,
            print(f"IDENTITY ERROR request {i}: {e}")  # not a silent pass
            sys.exit(2)

    path = Path(a.out)
    if a.record:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"model": a.model, "max_tokens": a.max_tokens,
             "prompts": PROMPTS, "outputs": outs}, indent=1))
        print(f"IDENTITY BASELINE RECORDED: {len(outs)} outputs -> {path}")
        # a model emitting the same token forever is the degenerate case this
        # lane has already hit once; say so at record time rather than later
        degenerate = sum(1 for o in outs if len(set(o.split())) <= 2)
        if degenerate:
            print(f"IDENTITY WARN: {degenerate}/{len(outs)} outputs look degenerate")
        return

    if not path.exists():
        print(f"IDENTITY SKIP: no baseline at {path}")
        return
    want = json.loads(path.read_text())["outputs"]
    if len(want) != len(outs):
        print(f"IDENTITY FAIL: {len(outs)} outputs, baseline has {len(want)}")
        sys.exit(1)
    diffs = [i for i, (a_, b_) in enumerate(zip(outs, want)) if a_ != b_]
    if not diffs:
        print(f"IDENTITY PASS: {len(outs)}/{len(outs)} outputs identical")
        return
    print(f"IDENTITY FAIL: {len(diffs)}/{len(outs)} outputs differ")
    for i in diffs[:3]:
        print(f"  [{i}] prompt   : {PROMPTS[i]!r}")
        print(f"      baseline : {want[i][:120]!r}")
        print(f"      candidate: {outs[i][:120]!r}")
    sys.exit(1)


if __name__ == "__main__":
    main()
