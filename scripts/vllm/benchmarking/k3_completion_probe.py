#!/usr/bin/env python3
"""Capture greedy completions from a live server for paired numerics comparison.

Emits one JSON file of {prompt_id: {text, tokens}} for a fixed, deterministic
prompt set. Two captures can then be diffed to answer "how far apart are these
two server configurations?" -- and, crucially, two captures of the *same*
configuration across a restart calibrate how far apart "identical" already is
(TP+EP decode is not bit-reproducible across restarts).

The prompts are long on purpose: a short prompt barely touches the KV cache, so
it cannot see a KV-cache dtype change.
"""
import argparse
import json
import random
import sys
import urllib.error
import urllib.request

_WORDS = (
    "harbor lantern gravel meadow copper thistle beacon marrow cobalt "
    "quarry pelican ribbon sandstone velvet juniper anchor tundra ember "
    "prairie granite willow cinder falcon lagoon mantle orchid pewter "
    "quartz saffron trellis walnut zephyr almond bracken citrine"
).split()


def build_prompts(n: int, words_per_prompt: int) -> list[str]:
    """Deterministic filler passages, each ending in the same instruction."""
    out = []
    for i in range(n):
        rng = random.Random(1000 + i)
        body = " ".join(rng.choice(_WORDS) for _ in range(words_per_prompt))
        out.append(
            f"Log fragment {i}. The following is a record of observations.\n"
            f"{body}\n\n"
            "Summarize the record above in a single sentence, then list three "
            "words that appeared in it.\nAnswer:")
    return out


def complete(base_url: str, model: str, prompt: str, max_tokens: int,
             timeout: float) -> dict:
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,
            "logprobs": 1,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    choice = body["choices"][0]
    lp = choice.get("logprobs") or {}
    return {
        "text": choice["text"],
        "tokens": lp.get("tokens") or [],
        "token_logprobs": lp.get("token_logprobs") or [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--words-per-prompt", type=int, default=2600)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    results = {}
    for idx, prompt in enumerate(
            build_prompts(args.num_prompts, args.words_per_prompt)):
        try:
            results[str(idx)] = complete(args.base_url, args.model, prompt,
                                         args.max_tokens, args.timeout)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            results[str(idx)] = {"error": f"{type(exc).__name__}: {exc}"}
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    ok = sum(1 for v in results.values() if "error" not in v)
    print(f"k3_completion_probe: {ok}/{len(results)} completions -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
