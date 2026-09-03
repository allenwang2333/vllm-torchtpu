#!/usr/bin/env python3
"""Accuracy oracle for DeepSeek-V4 P/D disaggregation (PR #684).

The transfer failures this PR can produce are quiet. A tag that reads the
wrong cache group's block table still moves real bytes into real pages, and a
sliding-window trim that is off by one still produces a page list of plausible
length. Nothing raises; the decode replica just attends over the wrong KV.
So the check has to be recall of specific content, not fluency.

Two needles, because DSv4's groups cover different spans:

  deep   -- placed early in the prompt, recoverable only through the CSA,
            indexer and HCA caches, which are cache group 0.
  recent -- placed inside the last `sliding_window` tokens, which is the only
            span the SWA caches (groups 1 and 2) hold. These are the caches
            overlaid on the CSA NoPE arrays, and the ones whose block tables
            get trimmed, so this needle is the sharp probe of the PR.

The length sweep straddles the window boundary on purpose. At
`sliding_window=128`, `_stage3_skipped_window_pages` computes
`max(0, num_tokens - 128 + 1) // page_tokens`, which is zero at and below 128
and turns on just above it. Lengths 127/128/129 sit on that edge; the larger
ones make the trimmed prefix dominate the block table.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
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


def post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"{url} -> HTTP {exc.code}: {exc.read()[:2000]!r}") from exc


def filler_words(rng: random.Random, approx_tokens: int) -> str:
    # The filler sentences run about 13 tokens each; this only has to land in
    # the right neighbourhood, since the actual length is measured after.
    return " ".join(rng.choice(FILLER) for _ in range(max(1, approx_tokens // 13)))


def build_prompt(rng: random.Random, target_tokens: int, deep_code: str,
                 recent_code: str, window: int) -> str:
    """A prompt with one needle near the front and one inside the window."""
    deep = (f" The archive vault code is {deep_code}. Remember it. ")
    recent = (f" The duty officer code is {recent_code}. Remember it. ")
    # Keep the recent needle plus the question inside the last `window`
    # tokens, so the SWA caches are the only place it can be read from.
    tail_budget = max(24, window // 3)
    head = filler_words(rng, max(13, target_tokens - tail_budget - 60))
    tail = filler_words(rng, tail_budget)
    return (deep + head + recent + tail +
            "\n\nQuestion: What is the duty officer code?\nAnswer: "
            "The duty officer code is")


def build_deep_prompt(rng: random.Random, target_tokens: int,
                      deep_code: str) -> str:
    head = filler_words(rng, max(13, target_tokens - 40))
    return (f" The archive vault code is {deep_code}. Remember it. " + head +
            "\n\nQuestion: What is the archive vault code?\nAnswer: "
            "The archive vault code is")


def run_case(base: str, model: str, prompt: str, expect: str, timeout: float,
             label: str) -> dict:
    started = time.time()
    body = post(
        f"{base}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 24,
            "temperature": 0.0,
            "seed": 42,
        },
        timeout,
    )
    text = body["choices"][0]["text"]
    usage = body.get("usage", {})
    ok = expect in text
    return {
        "label": label,
        "ok": ok,
        "expected": expect,
        "got": text.strip()[:80],
        "prompt_tokens": usage.get("prompt_tokens"),
        "elapsed_s": round(time.time() - started, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000",
                        help="proxy base URL (disagg) or a single server")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--window", type=int, default=128,
                        help="model sliding_window, for needle placement")
    parser.add_argument("--lengths", default="127,128,129,512,2048,4096",
                        help="approximate prompt lengths to sweep")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    results: list[dict] = []
    failures = 0

    for length in lengths:
        for attempt in range(args.repeats):
            rng = random.Random(1000 * length + attempt)
            deep_code = f"{rng.randrange(10**7, 10**8)}"
            recent_code = f"{rng.randrange(10**7, 10**8)}"

            recent_prompt = build_prompt(rng, length, deep_code, recent_code,
                                         args.window)
            case = run_case(args.base, args.model, recent_prompt, recent_code,
                            args.timeout, f"recent/swa len~{length} #{attempt}")
            results.append(case)
            failures += not case["ok"]
            print(json.dumps(case), flush=True)

            # The deep needle is only meaningful once the prompt is long
            # enough that it has fallen out of the sliding window.
            if length > args.window * 2:
                deep_prompt = build_deep_prompt(rng, length, deep_code)
                case = run_case(args.base, args.model, deep_prompt, deep_code,
                                args.timeout,
                                f"deep/csa len~{length} #{attempt}")
                results.append(case)
                failures += not case["ok"]
                print(json.dumps(case), flush=True)

    summary = {
        "total": len(results),
        "failures": failures,
        "passed": len(results) - failures,
    }
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"summary": summary, "results": results}, handle,
                      indent=2)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
