#!/usr/bin/env python3
"""Aggregate [DPWASTE] log lines (from TORCHTPU_DP_WASTE_LOG=1) to quantify DP
lockstep device waste: padded tokens the ranks actually forward vs real tokens.

Line format (one per step per worker):
  [DPWASTE] real=R fwd_padded=F chunks=C dummy=D bucket=B numreqs=N

Waste fraction = 1 - sum(real)/sum(fwd_padded), aggregated across all ranks and
steps (so per-step/per-rank grouping isn't needed). Split into prefill-ish
(real>numreqs) vs decode (real==numreqs). Also reports dummy-chunk share.

Usage: analyze_dpwaste.py <server.log> [more.log ...]
"""
import re
import sys

PAT = re.compile(
    r"\[DPWASTE\] real=(\d+) fwd_padded=(\d+) chunks=(\d+) dummy=(\d+) "
    r"bucket=(\S+) numreqs=(\d+)")


def bucketize(rows):
    real = sum(r[0] for r in rows)
    fwd = sum(r[1] for r in rows)
    chunks = sum(r[2] for r in rows)
    dummy = sum(r[3] for r in rows)
    n = len(rows)
    waste = (1 - real / fwd) if fwd else 0.0
    return n, real, fwd, chunks, dummy, waste


def report(label, rows):
    if not rows:
        print(f"{label:<16} (no steps)")
        return
    n, real, fwd, chunks, dummy, waste = bucketize(rows)
    dummy_share = dummy / chunks if chunks else 0.0
    print(f"{label:<16} steps={n:<6} real_tok={real:<12} fwd_padded={fwd:<12} "
          f"WASTE={100*waste:5.1f}%  dummy_chunks={100*dummy_share:4.1f}% "
          f"({dummy}/{chunks})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rows = []
    for fp in sys.argv[1:]:
        with open(fp, errors="ignore") as f:
            for ln in f:
                m = PAT.search(ln)
                if m:
                    real, fwd, chunks, dummy = (int(m.group(i)) for i in range(1, 5))
                    numreqs = int(m.group(6))
                    rows.append((real, fwd, chunks, dummy, numreqs))
    if not rows:
        print("no [DPWASTE] lines found")
        sys.exit(2)

    allrows = [(r[0], r[1], r[2], r[3]) for r in rows]
    prefill = [(r[0], r[1], r[2], r[3]) for r in rows if r[0] > r[4]]
    decode = [(r[0], r[1], r[2], r[3]) for r in rows if r[0] <= r[4]]

    print(f"parsed {len(rows)} [DPWASTE] step-samples (all ranks)\n")
    report("ALL", allrows)
    report("PREFILL(real>N)", prefill)
    report("DECODE(real<=N)", decode)
    print("\nWASTE = 1 - real_tokens/forwarded_padded_tokens across all ranks*steps.")
    print("High prefill waste => cross-rank token/chunk imbalance (LB lever).")
    print("dummy_chunks% = share of forward passes that were pure lockstep dummies.")


if __name__ == "__main__":
    main()
