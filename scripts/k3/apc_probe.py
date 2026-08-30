#!/usr/bin/env python3
"""Greedy determinism probe for prefix caching on the unified KV pool.

The concurrency sweep runs with ``--ignore-eos`` and never inspects a token, so
a server whose cache restores attention KV without the matching KDA recurrent
state still reports a healthy tok/s. This probe is the part that looks.

It issues a fixed set of prompts as **token ID arrays** rather than text, so the
prompt length is exact and reproducible: block-boundary behaviour is the thing
being tested, and a prompt whose length depends on how the server tokenised a
string cannot test it. The same prompts run in three phases:

  cold         after POST /reset_prefix_cache -- nothing can hit
  warm         the same prompts in the same order -- everything should hit
  interleaved  the same prompts in reverse order, so each request resumes into
               a slot whose previous occupant had a *different* prefix

Greedy decoding is deterministic, so within an arm every phase must produce
byte-identical text. Across arms (pool off vs pool on) the same holds up to the
numerics of the pooled path. Compare two runs with ``apc_diff.py``.

  ENDPOINT=http://10.13.232.43:8000 python3 scripts/k3/apc_probe.py \
      --out results/<arm>/apc-probe.json
"""
import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request

MODEL = os.environ.get("MODEL", "moonshotai/Kimi-K3")

# Deterministic across arms and across invocations: the whole point is that two
# servers see the identical byte sequence. Kept clear of the low IDs where the
# specials live (K3: bos 163584, eos 163586) and well inside the 163840 vocab.
_RNG = random.Random(20260819)
_STREAM = [_RNG.randrange(1000, 20000) for _ in range(4096)]


def _ids(n, salt=None):
    """First `n` ids of the shared stream, optionally with a distinct tail."""
    if salt is None:
        return list(_STREAM[:n])
    rng = random.Random(salt)
    return list(_STREAM[:n]) + [rng.randrange(20000, 40000) for _ in range(120)]


def build_prompts(block_size):
    """Prompts chosen to land on and just past pool block boundaries.

    A pooled prefix-cache hit restores whole blocks, and the KDA state is seeded
    from the previous block on a boundary crossing, so `block_size` and
    `block_size + 1` exercise different code (`_collect_mamba_state_seed_copies`
    fires for the second and not the first). They are prefixes of one another,
    which also maximises cross-prompt hits.
    """
    prompts = [("distinct", _ids(200, salt="no-shared-prefix"))]
    for blocks in (1, 2, 3):
        n = block_size * blocks
        prompts.append((f"boundary-{n}", _ids(n)))
        prompts.append((f"boundary-{n + 1}", _ids(n + 1)))
    for k in range(4):
        prompts.append((f"shared500-tail{k}", _ids(500, salt=f"tail{k}")))
    return prompts


def post(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def run_phase(endpoint, phase, prompts, max_tokens, timeout):
    rows = []
    for name, ids in prompts:
        payload = {
            "model": MODEL,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            # Fixed length: a divergence that only shortens the answer is still
            # a divergence, and a stop token would hide the tail where a
            # corrupted recurrent state shows up first.
            "ignore_eos": True,
        }
        try:
            body = post(f"{endpoint}/v1/completions", payload, timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            rows.append({"name": name, "phase": phase, "prompt_len": len(ids),
                         "error": repr(exc)})
            print(f"  {phase:12s} {name:22s} ERROR {exc}", flush=True)
            continue
        usage = body.get("usage", {})
        details = usage.get("prompt_tokens_details") or {}
        row = {
            "name": name,
            "phase": phase,
            "prompt_len": len(ids),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "text": body["choices"][0]["text"],
        }
        rows.append(row)
        print(f"  {phase:12s} {name:22s} prompt={row['prompt_tokens']} "
              f"cached={row['cached_tokens']} "
              f"text={row['text'][:48]!r}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-size", type=int, default=162,
                    help="pool block size; 162 is Kimi-K3 at TP=32")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    if not args.endpoint:
        ap.error("set --endpoint or ENDPOINT")
    endpoint = args.endpoint.rstrip("/")

    prompts = build_prompts(args.block_size)
    # Only meaningful where prefix caching is on; a 404 on the no-APC arms is
    # expected and not a failure.
    try:
        post(f"{endpoint}/reset_prefix_cache", {}, 60)
        print("reset_prefix_cache: ok")
    except Exception as exc:  # noqa: BLE001 - informational only
        print(f"reset_prefix_cache: {exc}")

    rows = []
    rows += run_phase(endpoint, "cold", prompts, args.max_tokens, args.timeout)
    rows += run_phase(endpoint, "warm", prompts, args.max_tokens, args.timeout)
    rows += run_phase(endpoint, "interleaved", list(reversed(prompts)),
                      args.max_tokens, args.timeout)

    by = {}
    for r in rows:
        by.setdefault(r["name"], {})[r["phase"]] = r
    diverged = []
    for name, phases in sorted(by.items()):
        cold = phases.get("cold", {}).get("text")
        for phase in ("warm", "interleaved"):
            got = phases.get(phase, {}).get("text")
            if cold is None or got is None or got != cold:
                diverged.append((name, phase))
    cached = sum(r.get("cached_tokens") or 0 for r in rows)

    out = {
        "endpoint": endpoint,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "rows": rows,
        "diverged": diverged,
        "total_cached_tokens": cached,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"\nwithin-arm divergences (cold vs warm/interleaved): "
          f"{len(diverged)}/{2 * len(by)}")
    for name, phase in diverged:
        print(f"  DIVERGED {name} @ {phase}")
    print(f"total cached_tokens across all phases: {cached}")
    print(f"-> {args.out}")
    return 1 if diverged else 0


if __name__ == "__main__":
    sys.exit(main())
