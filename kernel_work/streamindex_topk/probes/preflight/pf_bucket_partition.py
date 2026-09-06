# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PREFLIGHT: does the shipping bucket land on P = 1 at the A/B shape?

The A/B deployment is GLM-5.2 at `max_model_len = 66560`, so the shipping
bucket is `min(66560, MAX_SLICE_WORDS) = min(66560, 65536) = 65536` -- EXACTLY
the threshold.  `sparsecore_topk._pick_partition` computes
`p_min = ceil(n / MAX_SLICE_WORDS)`, so 65536 gives 1 and 65537 gives 2.  The
whole of Option B's prize is claimed to come from being on the P = 1 side.
One word either way flips it, and nothing has confirmed which side the real
scores width lands on at the real `max_model_len`.

This probe does not argue it.  It:

  1. recomputes both scores widths from the composed kernel's own sizing code
     (`num_sublanes_total`), at the production page_size and the deployment's
     `max_model_len`, for the incumbent tuple (2,2,2) AND Option E's (4,2,4)
     AND the second call site's (3,2,2);
  2. calls the real `sparsecore_topk._pick_partition` on each;
  3. sweeps `b` (rows) because `_pick_partition` also raises `p` for small
     batches to fill subcores -- a decode batch is not the mixed batch;
  4. reports the margin: how many words of slack the bucket has before P = 2.

Pure arithmetic against the shipped constants.  No TPU work, so it is tagged
`measured` only in the sense that the constants are read out of the installed
package rather than quoted from a document.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir,
                                     os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as SC  # noqa


def cdiv(a, b):
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def incumbent_sublanes(pages_per_seq, page_size, tup):
    """composed file, `kv_len_bucket is None` branch (== the incumbent)."""
    return max(align_to(pages_per_seq, p) * page_size // 128 for p in tup)


def composed_sublanes(bucket, pages_per_seq, page_size, k):
    """composed file, RIGHT-SIZE branch."""
    live = min(bucket, pages_per_seq * page_size)
    return max(cdiv(live, 128), cdiv(k, 128))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-model-len", type=int, default=66560)
    ap.add_argument("--page-size", type=int, default=1024)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    mml, ps, k = a.max_model_len, a.page_size, a.k
    pps = cdiv(mml, ps)
    bucket = min(mml, SC.MAX_SLICE_WORDS)

    print("=== shipped constants, read from the installed package ===")
    print(f"  sparsecore_topk.__file__   = {SC.__file__}")
    print(f"  MAX_SLICE_WORDS            = {SC.MAX_SLICE_WORDS}")
    print(f"  LANES                      = {SC.LANES}")
    print()
    print(f"=== A/B deployment: max_model_len={mml} page_size={ps} k={k} ===")
    print(f"  pages_per_seq              = {pps}")
    print(f"  shipping bucket            = min({mml}, {SC.MAX_SLICE_WORDS})"
          f" = {bucket}")
    print()

    rows = []
    tuples = {"(2,2,2) mla_attention_op.py:119 -- PRODUCTION": (2, 2, 2),
              "(4,2,4) Option E": (4, 2, 4),
              "(3,2,2) deepseek_v4_indexer.py:94": (3, 2, 2)}

    print(f"{'arm':<46} {'sublanes':>9} {'n words':>9} {'p_min':>6}")
    for label, tup in tuples.items():
        sl = incumbent_sublanes(pps, ps, tup)
        n = sl * 128
        rows.append(("incumbent " + label, tup, sl, n, cdiv(n,
                                                            SC.MAX_SLICE_WORDS)))
        print(f"{'incumbent ' + label:<46} {sl:>9} {n:>9} "
              f"{cdiv(n, SC.MAX_SLICE_WORDS):>6}")
    sl = composed_sublanes(bucket, pps, ps, k)
    n_comp = sl * 128
    rows.append(("composed (bucket, tuple-independent)", None, sl, n_comp,
                 cdiv(n_comp, SC.MAX_SLICE_WORDS)))
    print(f"{'composed (bucket, tuple-independent)':<46} {sl:>9} "
          f"{n_comp:>9} {cdiv(n_comp, SC.MAX_SLICE_WORDS):>6}")

    print()
    print("=== real _pick_partition(b, n, k), swept over batch rows b ===")
    print(f"{'b':>5} " + "".join(f"{lbl.split()[0]:>12}"
                                 for lbl, *_ in rows))
    sweep = []
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        ps_row = [SC._pick_partition(b, n, k) for *_, n, _ in rows]
        sweep.append({"b": b, "p": ps_row})
        print(f"{b:>5} " + "".join(f"{p:>12}" for p in ps_row))

    n_inc_prod = rows[0][3]
    p_inc = SC._pick_partition(1024, n_inc_prod, k)
    p_comp = SC._pick_partition(1024, n_comp, k)

    print()
    print("=== the sentence the prize rests on ===")
    margin = SC.MAX_SLICE_WORDS - n_comp
    print(f"  composed n = {n_comp}; MAX_SLICE_WORDS = {SC.MAX_SLICE_WORDS}; "
          f"slack before P=2 is {margin} words "
          f"({margin // 128} sublanes)")
    print(f"  p_min(composed)  = {cdiv(n_comp, SC.MAX_SLICE_WORDS)}   <-- "
          f"{'P=1 SIDE' if cdiv(n_comp, SC.MAX_SLICE_WORDS) == 1 else 'P=2 SIDE'}")
    print(f"  p_min(incumbent (2,2,2)) = {cdiv(n_inc_prod, SC.MAX_SLICE_WORDS)}")
    print(f"  at a large mixed batch (b=1024): incumbent p={p_inc}, "
          f"composed p={p_comp}")
    ok = (cdiv(n_comp, SC.MAX_SLICE_WORDS) == 1
          and cdiv(n_inc_prod, SC.MAX_SLICE_WORDS) == 2)
    print(f"  VERDICT: {'CONFIRMED -- composed P=1, incumbent P=2' if ok else 'NOT CONFIRMED'}")

    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"max_model_len": mml, "page_size": ps, "k": k,
                       "pages_per_seq": pps, "bucket": bucket,
                       "MAX_SLICE_WORDS": SC.MAX_SLICE_WORDS,
                       "rows": [{"arm": r[0], "tuple": r[1], "sublanes": r[2],
                                 "n": r[3], "p_min": r[4]} for r in rows],
                       "sweep": sweep, "confirmed": ok}, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
