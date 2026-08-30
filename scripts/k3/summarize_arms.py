#!/usr/bin/env python3
"""Collect the arms of the pooled-prefix-caching comparison into one table.

  python3 scripts/k3/summarize_arms.py \
      mainref=results/k3-mainref pooloff=results/k3-pooloff poolapc=results/k3-poolapc

Reads what apc_arm_check.sh leaves behind: the two vllm-bench result files, the
lm-eval GSM8K log and the greedy probe. Prints one row per arm per benchmark and
the deltas against the first arm named, which is the control by construction.
"""
import json
import os
import re
import sys

BENCHES = [("bench-nocache", "independent prompts"),
           ("bench-sharedprefix", "800-token shared prefix")]
FIELDS = [("median_ttft_ms", "TTFT ms", 1),
          ("median_tpot_ms", "TPOT ms", 2),
          ("output_throughput", "out tok/s", 1),
          ("request_throughput", "req/s", 3),
          ("total_token_throughput", "tot tok/s", 1)]


def read_bench(arm_dir, name):
    path = os.path.join(arm_dir, name, "c1.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def read_gsm8k(arm_dir):
    path = os.path.join(arm_dir, "gsm8k.log")
    if not os.path.exists(path):
        return None
    best = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("|gsm8k"):
                m = re.findall(r"\|\s*([0-9.]+)\s*\|", line)
                if m:
                    best = line.strip()
    return best


def read_probe(arm_dir):
    path = os.path.join(arm_dir, "apc-probe.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    arms = []
    for spec in args:
        name, _, path = spec.partition("=")
        arms.append((name, path or name))

    for bench, label in BENCHES:
        print(f"\n### {label}  ({bench}, concurrency 1)")
        rows = [(n, read_bench(p, bench)) for n, p in arms]
        base = next((d for _, d in rows if d), None)
        head = f"{'arm':<12}" + "".join(f"{lbl:>13}" for _, lbl, _ in FIELDS)
        print(head)
        print("-" * len(head))
        for name, data in rows:
            if not data:
                print(f"{name:<12}{'(missing)':>13}")
                continue
            cells = ""
            for key, _, prec in FIELDS:
                v = data.get(key)
                if v is None:
                    cells += f"{'-':>13}"
                    continue
                cell = f"{v:.{prec}f}"
                if data is not base and base and base.get(key):
                    cell += f" ({100 * (v / base[key] - 1):+.1f}%)"
                cells += f"{cell:>13}"
            print(f"{name:<12}{cells}")

    print("\n### GSM8K")
    for name, path in arms:
        print(f"{name:<12}{read_gsm8k(path) or '(missing)'}")

    print("\n### greedy prefix probe")
    print(f"{'arm':<12}{'within-arm divergences':>24}{'cached tokens':>16}")
    for name, path in arms:
        pr = read_probe(path)
        if not pr:
            print(f"{name:<12}{'(missing)':>24}")
            continue
        print(f"{name:<12}{len(pr['diverged']):>24}{pr['total_cached_tokens']:>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
