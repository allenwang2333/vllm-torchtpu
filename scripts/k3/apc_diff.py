#!/usr/bin/env python3
"""Compare two apc_probe.py runs prompt-by-prompt.

The within-arm check in apc_probe only proves a server is self-consistent: a
pooled path that corrupts every request identically would pass it. This is the
cross-arm check -- the pooled/APC arm against the per-layer arm that shares the
model, the flags and the prompts and differs only in where the KDA state lives.

Greedy decoding is deterministic, so a difference here is either a real
divergence or the known numeric one: the pooled path reassociates the SSM
update, which moves the f32 state by ~1 ulp and can flip a token whose top-2
logits are within that. Report the position of the first differing character so
the two can be told apart -- a late divergence in an otherwise identical
sentence is numerics, a completely different answer is not.

  python3 scripts/k3/apc_diff.py results/<control>/apc-probe.json \
                                 results/<treatment>/apc-probe.json
"""
import json
import sys


def load(path):
    data = json.load(open(path))
    out = {}
    for row in data["rows"]:
        out[(row["name"], row["phase"])] = row
    return data, out


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    (da, a), (db, b) = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(a) | set(b))
    same = diff = missing = 0
    print(f"control   {sys.argv[1]}  (cached_tokens={da['total_cached_tokens']})")
    print(f"treatment {sys.argv[2]}  (cached_tokens={db['total_cached_tokens']})")
    print()
    for key in keys:
        ra, rb = a.get(key), b.get(key)
        if ra is None or rb is None or "text" not in ra or "text" not in rb:
            missing += 1
            print(f"  MISSING  {key[0]:22s} {key[1]:12s}")
            continue
        if ra["text"] == rb["text"]:
            same += 1
            continue
        diff += 1
        n = common_prefix_len(ra["text"], rb["text"])
        print(f"  DIFFER   {key[0]:22s} {key[1]:12s} "
              f"first difference at char {n}/{len(ra['text'])}")
        print(f"      control  ...{ra['text'][max(0, n - 20):n + 40]!r}")
        print(f"      treatment...{rb['text'][max(0, n - 20):n + 40]!r}")
    total = same + diff + missing
    print(f"\nidentical {same}/{total}   differing {diff}   missing {missing}")
    return 1 if diff or missing else 0


if __name__ == "__main__":
    sys.exit(main())
