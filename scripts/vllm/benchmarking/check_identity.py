# SPDX-License-Identifier: Apache-2.0
"""Tier-0 semantics gate: compare generated output against a baseline run.

For a change that is supposed to preserve semantics -- a scheduler flag, a
batching change, a collective rewrite -- the correct test is not "does GSM8K
still score 0.97". It is "is the output identical". With greedy decode, a
fixed seed and fixed prompts, any differing token is a bug, and an accuracy
benchmark would not catch it.

`vllm bench serve --save-result` already writes `generated_texts` into its
result JSON, so nothing extra has to be captured. This only compares.

Works with dummy weights, which gives the layer-trimmed proxy a semantics
gate it otherwise cannot have. Note the limit: on dummy weights this proves
*invariance*, not correctness. Identical garbage is still identical.

Usage
-----
  # record a reference
  python3 check_identity.py --record run/isl8192_osl1024_c4.json \
      --baseline baselines/identity/<config>.c4.json

  # gate a candidate
  python3 check_identity.py --result run/isl8192_osl1024_c4.json \
      --baseline baselines/identity/<config>.c4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_texts(path: Path) -> list[str]:
    d = json.loads(path.read_text())
    texts = d.get("generated_texts")
    if texts is None:
        sys.exit(f"ERROR: {path} has no 'generated_texts'. "
                 "Did the run pass --save-result?")
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", help="candidate result JSON to gate")
    ap.add_argument("--record", help="result JSON to store as the reference")
    ap.add_argument("--baseline", required=True, help="reference JSON path")
    ap.add_argument("--max-report", type=int, default=3,
                    help="how many divergences to print")
    args = ap.parse_args()

    base_path = Path(args.baseline)

    if args.record:
        texts = load_texts(Path(args.record))
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps({"generated_texts": texts}, indent=1))
        print(f"IDENTITY BASELINE RECORDED: {len(texts)} outputs -> {base_path}")
        return

    if not args.result:
        sys.exit("ERROR: pass --result to gate, or --record to store a reference")
    if not base_path.exists():
        sys.exit(f"ERROR: no identity baseline at {base_path}. Record one first.")

    got = load_texts(Path(args.result))
    want = load_texts(base_path)

    if len(got) != len(want):
        print(f"IDENTITY FAIL: {len(got)} outputs, baseline has {len(want)}")
        sys.exit(1)

    diffs = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
    if not diffs:
        print(f"IDENTITY PASS: {len(got)}/{len(got)} outputs byte-identical")
        return

    print(f"IDENTITY FAIL: {len(diffs)}/{len(got)} outputs differ")
    for i in diffs[:args.max_report]:
        a, b = got[i], want[i]
        # first differing character, so the report points at the divergence
        j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]),
                 min(len(a), len(b)))
        lo = max(0, j - 40)
        print(f"\n  request {i}, first difference at char {j}:")
        print(f"    baseline : ...{b[lo:j+40]!r}")
        print(f"    candidate: ...{a[lo:j+40]!r}")
    sys.exit(1)


if __name__ == "__main__":
    main()
