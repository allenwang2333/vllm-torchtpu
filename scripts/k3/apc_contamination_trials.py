#!/usr/bin/env python3
"""Repeat the Mamba-state contamination check with a cold cache each trial.

`smoke_mamba_state_contamination.py` compares a cached round's 64 token ids
against an *uncached* round-1 reference. On K3 it failed once (3 of 6 cached
rounds) and then passed three re-runs -- because the re-runs were no longer
cold: the ledger's leading blocks were already cached, so round 1 was itself a
cache hit and the comparison had become warm-vs-warm. This build exposes no
/reset_prefix_cache, so a genuinely cold trial needs fresh content.

Two things this adds, both aimed at telling a real bug from the numeric noise
already measured on this model (a cache-free server reproduced only 11 of 33
completions byte-for-byte across two identical runs):

* **A salt**, so every trial starts cold and trials are independent samples.
* **A warm-vs-warm control inside each trial.** Round 2 and round 3 do the same
  amount of prefill against the same cache state, so any difference between
  them is nondeterminism and nothing else. Round 1 (cold) against round 2
  (warm) is the comparison under test. A cache bug moves the second and not the
  first; noise moves both.

It also records *where* the token streams part. A lost or contaminated
recurrent state derails the continuation immediately; noise flips a token deep
into a low-confidence tail.

  ENDPOINT=http://... python3 scripts/k3/apc_contamination_trials.py --trials 6
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL = os.environ.get("MODEL", "moonshotai/Kimi-K3")


def build_prefix(tag, lines, salt, filler=0):
    """A ledger that differs from token 0 for every salt.

    The salt has to appear in every record, not just the header: two ledgers
    sharing a header prefix share cached blocks, and a trial whose leading
    blocks are already cached is not a cold trial -- which is exactly how the
    first attempt at this reproduction fooled itself. All salts are four
    digits so the token count does not move between them.
    """
    assert 1000 <= salt <= 9999, "salts must be the same width"
    body = "\n".join(
        f"{tag}{salt} record {i:04d}: quantity {(i * 7 + len(tag) + salt) % 97}"
        for i in range(lines))
    return (f"Inventory ledger {tag}{salt}. "
            f"Read every record before answering.\n{body}\n"
            "\nSummarise the ledger above in one sentence, then state the "
            "final record's quantity.\n" + (" x" * filler))


def complete(endpoint, prompt, max_tokens, timeout):
    req = urllib.request.Request(
        f"{endpoint}/v1/completions",
        data=json.dumps({
            "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "return_token_ids": True,
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {"tokens": choice.get("token_ids") or [],
            "text": choice.get("text", ""),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens") or 0}


def compare(a, b):
    """(identical, index of first difference, number of differing positions)."""
    ta, tb = a["tokens"], b["tokens"]
    n = min(len(ta), len(tb))
    first = next((i for i in range(n) if ta[i] != tb[i]), None)
    if first is None and len(ta) == len(tb):
        return True, None, 0
    diff = sum(1 for i in range(n) if ta[i] != tb[i]) + abs(len(ta) - len(tb))
    return False, first, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("ENDPOINT"))
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--lines", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--salt-base", type=int, default=1000,
                    help="first salt; advance it between invocations or the "
                         "trials re-use content this server already cached "
                         "and stop being cold")
    ap.add_argument("--align-block", type=int, default=0,
                    help="pad prompts to an exact multiple of this many "
                         "tokens (162 = the K3 TP32 pool block)")
    ap.add_argument("--out")
    args = ap.parse_args()
    if not args.endpoint:
        ap.error("set --endpoint or ENDPOINT")
    endpoint = args.endpoint.rstrip("/")

    tags = ["ALPHA", "BRAVO"]

    # The original failure had BRAVO at 2430 prompt tokens = exactly 15 pool
    # blocks, and an exact multiple is a special case for the cache: vLLM
    # cannot serve the final block from cache because it needs at least one
    # token to run the model on. Reproducing it needs the prompt length pinned
    # to a multiple of the block size, so measure the unpadded length once and
    # pad with single-token filler. The probe uses a throwaway salt so the
    # trials themselves still start cold.
    filler = {t: 0 for t in tags}
    if args.align_block:
        for tag in tags:
            base = complete(endpoint, build_prefix(tag, args.lines, 9999), 1,
                            args.timeout)["prompt_tokens"]
            need = (-base) % args.align_block
            got = complete(endpoint,
                           build_prefix(tag, args.lines, 9999, need), 1,
                           args.timeout)["prompt_tokens"]
            # " x" may not be one token; correct once and re-check.
            if got % args.align_block:
                per = max(1, (got - base) // need) if need else 1
                need = ((-base) % args.align_block) // per
                got = complete(endpoint,
                               build_prefix(tag, args.lines, 9999, need), 1,
                               args.timeout)["prompt_tokens"]
            filler[tag] = need
            print(f"align {tag}: {base} + {need} filler -> {got} tokens "
                  f"({got / args.align_block:.2f} blocks)"
                  f"{'' if got % args.align_block == 0 else '  NOT ALIGNED'}")
        print()

    rows = []
    print(f"{'trial':>5} {'tag':<6} {'prompt':>7} {'r1$':>6} {'r2$':>6} "
          f"{'r3$':>6}  {'cold-vs-warm':<22} {'warm-vs-warm':<22}")
    for trial in range(args.trials):
        salt = args.salt_base + trial * 37
        prompts = {t: build_prefix(t, args.lines, salt, filler[t])
                   for t in tags}
        # Interleaved: each request resumes into a slot the other tag last
        # wrote, which is the condition the whole check exists for.
        rounds = []
        for _ in range(4):
            rounds.append({t: complete(endpoint, prompts[t], args.max_tokens,
                                       args.timeout) for t in tags})
        for tag in tags:
            r1, r2, r3, r4 = (rounds[0][tag], rounds[1][tag],
                              rounds[2][tag], rounds[3][tag])
            # Cold reference against the first cached round: the comparison
            # the contamination smoke makes.
            cw_same, cw_at, cw_n = compare(r1, r2)
            # The true noise control. By round 3 the hit count has stopped
            # growing (everything but the final block is cached), so rounds 3
            # and 4 do identical prefill against an identical cache state and
            # any difference between them is nondeterminism alone. Rounds 2
            # and 3 are NOT a control -- they still differ in how much they
            # recompute.
            ww_same, ww_at, ww_n = compare(r3, r4)
            def fmt(same, at, n):
                return "identical" if same else f"differs @{at} ({n} tokens)"
            print(f"{trial:>5} {tag:<6} {r1['prompt_tokens']:>7} "
                  f"{r1['cached_tokens']:>6} {r2['cached_tokens']:>6} "
                  f"{r3['cached_tokens']:>6}/{r4['cached_tokens']:<6} "
                  f"{fmt(cw_same, cw_at, cw_n):<22} "
                  f"{fmt(ww_same, ww_at, ww_n):<22}")
            rows.append({"trial": trial, "tag": tag, "salt": salt,
                         "prompt_tokens": r1["prompt_tokens"],
                         "cached": [r["cached_tokens"] for r in
                                    (r1, r2, r3, r4)],
                         "cold_vs_warm": [cw_same, cw_at, cw_n],
                         "warm_vs_warm": [ww_same, ww_at, ww_n],
                         "texts": [r["text"] for r in (r1, r2, r3, r4)]})

    cw_bad = sum(1 for r in rows if not r["cold_vs_warm"][0])
    ww_bad = sum(1 for r in rows if not r["warm_vs_warm"][0])
    print(f"\ncold-vs-warm  differing: {cw_bad}/{len(rows)}   (comparison under test)")
    print(f"warm-vs-warm  differing: {ww_bad}/{len(rows)}   "
          f"(noise control: rounds 3 and 4, identical cache state)")
    print("a cache bug raises the first and not the second; "
          "noise raises both")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "cold_vs_warm_bad": cw_bad,
                       "warm_vs_warm_bad": ww_bad, "n": len(rows)}, fh, indent=1)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
