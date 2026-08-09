#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Find the first dispatch-sequence divergence between ranks.

Reads the ``rank<N>.tsv`` files written by
``vllm_torchtpu.runner.dispatch_probe`` and reports the first sequence
position where the ranks stop agreeing. That position is the divergence
that deadlocks the step.

Usage:
    python tools/diff_dispatch_trace.py [trace_dir] [--context N]

``trace_dir`` defaults to ``/tmp/dispatch_trace``.
"""
import argparse
import os
import sys
from collections import OrderedDict


def load(path):
    """Ordered (event, detail) rows, skipping the header."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            seq, step, event = parts[0], parts[1], parts[2]
            detail = parts[3] if len(parts) > 3 else ""
            rows.append((int(seq), int(step), event, detail))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", nargs="?", default="/tmp/dispatch_trace")
    ap.add_argument("--context", type=int, default=6)
    args = ap.parse_args()

    if not os.path.isdir(args.trace_dir):
        sys.exit(f"no such directory: {args.trace_dir}")

    files = sorted(
        (f for f in os.listdir(args.trace_dir) if f.endswith(".tsv")),
        key=lambda f: int("".join(c for c in f if c.isdigit()) or 0))
    if len(files) < 2:
        sys.exit(f"need at least 2 rank traces in {args.trace_dir}, "
                 f"found {len(files)}")

    traces = OrderedDict()
    for f in files:
        traces[f[:-4]] = load(os.path.join(args.trace_dir, f))

    for name, rows in traces.items():
        print(f"{name}: {len(rows)} events, last step {rows[-1][1] if rows else '-'}")
    print()

    base_name, base = next(iter(traces.items()))
    shortest = min(len(r) for r in traces.values())
    diverged_at = None

    for i in range(shortest):
        # Compare the event identity and its detail; both matter, since the
        # same site can launch different shapes (a recompile) as well as be
        # skipped entirely.
        key = (base[i][2], base[i][3])
        for name, rows in traces.items():
            if (rows[i][2], rows[i][3]) != key:
                diverged_at = i
                break
        if diverged_at is not None:
            break

    if diverged_at is None:
        print(f"No divergence in the first {shortest} events "
              "(traces agree up to the shortest rank).")
        lengths = {n: len(r) for n, r in traces.items()}
        if len(set(lengths.values())) > 1:
            print("\nBut the traces are DIFFERENT LENGTHS — the short ranks "
                  "stopped dispatching first:")
            for n, ln in sorted(lengths.items(), key=lambda kv: kv[1]):
                print(f"  {n}: {ln}")
            print(f"\nLast events after position {shortest - 1}:")
            for n, rows in traces.items():
                if len(rows) > shortest:
                    nxt = rows[shortest]
                    print(f"  {n}: {nxt[2]}  {nxt[3]}")
        return

    lo = max(0, diverged_at - args.context)
    print(f"FIRST DIVERGENCE at event index {diverged_at} "
          f"(step {base[diverged_at][1]})\n")
    for i in range(lo, min(shortest, diverged_at + 2)):
        marker = ">>>" if i == diverged_at else "   "
        print(f"{marker} [{i}]")
        for name, rows in traces.items():
            print(f"      {name:<10} {rows[i][2]:<28} {rows[i][3]}")
        print()


if __name__ == "__main__":
    main()
