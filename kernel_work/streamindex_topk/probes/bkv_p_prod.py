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
"""Follow-up to `step_cost.py`: does the `bkv_p` win survive at PRODUCTION
shapes, and is the kernel still correct when it is taken?

`step_cost.py` [measured: j-e8eaacd3-6be4-4cdf-95a2] found a ~500 ns fixed cost
per inner `(bq, bkv)` iteration and, as a by-product, that raising `bkv_p` from
the shipped 2 to 4 made a decode call 8.62% faster. That measurement was taken
at `max_model_len = 8192`, deliberately: it pins `num_sublanes_total` at 64
across the whole ladder so the step-count slope is not confounded by a growing
scores buffer.

Production is `max_model_len = 9216`, and there the confound is the question.
`pages_per_seq(9216) = 9`, so `align_to(9, bkv_p)` moves:

    bkv_p                 2     4     8
    align_to(9, bkv_p)   10    12    16
    num_sublanes_total   80    96   128       <- +20%, +60%
    scores bytes         x1.0  x1.2  x1.6

So at production the two effects pull in opposite directions: raising `bkv_p`
removes inner-step overhead (worth ~500 ns each) and simultaneously widens the
scores buffer that is written and then read again by the SparseCore. Which one
wins is not derivable from the 8192 measurement and is not derivable from the
roofline either, because the roofline has no term for the per-step cost. It is
this script.

Three things are measured here, in one job:

  1. `--mode sweep`  -- `bkv_p` in {2, 4, 8} at the REAL regimes from
     `instruments/shapes.py`, through the read-only `microbench.run_regime`, so
     the numbers are directly comparable to step 1's table.
  2. `--mode correct` -- the float64-oracle check at `bkv_p` 4 and 8. A faster
     kernel that selects different indices is not a faster kernel. `bkv_p` is a
     blocking factor and *should* be value-neutral, which is exactly the kind
     of belief that is cheap to check and embarrassing to assume.
  3. `--mode control` -- the negative control at the winning `bkv_p`, so the
     correctness run above is not a check that has never failed.

How `bkv_p` is varied without touching a read-only instrument
-------------------------------------------------------------
`microbench.run_regime` and `test_correctness.run` both read
`shapes.NUM_KV_PAGES_PER_BLOCK` internally rather than taking it as an
argument. Both files are READ-ONLY after step 1 and are not forked here -- a
yardstick that changes between candidates is not a yardstick.

Instead this script rebinds the module attribute `shapes.NUM_KV_PAGES_PER_BLOCK`
around each call. The instruments are imported and executed exactly as written;
only the constant they read is different, which is precisely the variable under
test. The rebinding is undone in a `finally` so an exception cannot leak a
changed constant into a later arm.

Sign convention: percentages are changes in TIME, negative is faster.
"""

import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])

from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
from kernel_work.streamindex_topk.instruments import microbench as MB  # noqa: E402
from kernel_work.streamindex_topk.instruments import test_correctness as TC  # noqa: E402

LADDER = [2, 4, 8]  # 2 is the shipped value (mla_attention_op.py:119)


class bkv_p_set:
    """Rebind `shapes.NUM_KV_PAGES_PER_BLOCK` for the duration of a block.

    Only entry [0] -- the decode band -- is varied when `decode_only`, because
    that is the band `step_cost.py` measured and the band the mixed regimes do
    not use. Set `decode_only=False` to move all three.
    """

    def __init__(self, bkv_p: int, decode_only: bool = True):
        old = S.NUM_KV_PAGES_PER_BLOCK
        self.old = old
        self.new = ((bkv_p, old[1], old[2]) if decode_only
                    else (bkv_p, bkv_p, bkv_p))

    def __enter__(self):
        S.NUM_KV_PAGES_PER_BLOCK = self.new
        return self.new

    def __exit__(self, *exc):
        S.NUM_KV_PAGES_PER_BLOCK = self.old
        return False


def sweep(impl, regime_names, reps, decode_only):
    rows = []
    for name in regime_names:
        regime = S.REGIMES_BY_NAME[name]
        for bkv_p in LADDER:
            try:
                with bkv_p_set(bkv_p, decode_only) as tup:
                    r = MB.run_regime(impl, regime, early_exit=False,
                                      reps=reps, measure_topk=False)
                r["bkv_p"] = bkv_p
                r["bkv_tuple"] = list(tup)
                # The confound, reported alongside every row so no reader has
                # to recompute it: how wide the scores buffer became.
                #
                # These two MUST be recomputed here and must overwrite what
                # run_regime reported. `num_sublanes_total` and `scores_bytes`
                # take `bkv_ps` as a DEFAULT argument, which Python binds at
                # function-definition time -- so the values in the row are the
                # shipped bkv_p=2 ones regardless of the rebinding above, and
                # would understate the widened buffer at every rung. The timing
                # itself is unaffected: the kernel call reads the module
                # attribute at call time.
                subl = S.num_sublanes_total(regime.max_model_len, bkv_ps=tup)
                r["num_sublanes_total"] = subl
                r["sublanes_at_this_bkv_p"] = subl
                r["scores_MB"] = regime.num_tokens * subl * 128 * 4 / 1e6
            except Exception as exc:  # noqa: BLE001
                r = dict(regime=name, bkv_p=bkv_p, error=repr(exc)[:400])
            rows.append(r)
            print(json.dumps(r), flush=True)

    print("\n=== bkv_p at PRODUCTION shapes (median ms/call, 1 TensorCore) ===")
    hdr = (f"{'regime':<14}{'bkv_p':>6}{'subl':>6}{'scoresMB':>10}"
           f"{'inner':>7}{'full_ms':>10}{'spread%':>9}{'d time %':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name in regime_names:
        base = next((r["full_ms"] for r in rows
                     if r.get("regime") == name and r.get("bkv_p") == 2
                     and "error" not in r), None)
        for r in [x for x in rows if x.get("regime") == name]:
            if "error" in r:
                print(f"{name:<14}{r['bkv_p']:>6}  ERROR {r['error'][:60]}")
                continue
            dt = ((r["full_ms"] - base) / base * 100.0) if base else float("nan")
            print(f"{name:<14}{r['bkv_p']:>6}"
                  f"{r['sublanes_at_this_bkv_p']:>6}{r['scores_MB']:>10.1f}"
                  f"{r['inner']:>7}{r['full_ms']:>10.4f}"
                  f"{r['full_spread_pct']:>9.2f}{dt:>10.2f}")
    print("\n`d time %` is against bkv_p=2, the shipped value. Negative is "
          "faster. `subl` is num_sublanes_total -- it MOVES with bkv_p here "
          "(80/96/128 at max_model_len=9216), which is the confound this run "
          "exists to price, not a defect.")
    return rows


def correctness(impl, regime_names, ladder, negative_control):
    rc = 0
    for name in regime_names:
        regime = S.REGIMES_BY_NAME[name]
        for bkv_p in ladder:
            with bkv_p_set(bkv_p) as tup:
                try:
                    n, failures, st = TC.run(
                        impl, regime, negative_control=negative_control)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{name} bkv_p={bkv_p}] ERROR {exc!r}")
                    rc = 1
                    continue
            margin = (f"worst_rank={st['worst_rank_seen']} of allowed "
                      f"{S.INDEX_TOPK + TC.RANK_SLACK}, min_recall="
                      f"{st['min_recall']:.5f} of required {TC.MIN_RECALL}")
            tag = f"[{name} bkv_p={bkv_p} tuple={tup}]"
            if negative_control:
                if failures:
                    print(f"{tag} NEGATIVE CONTROL OK -- fired on "
                          f"{len(failures)} of {n} rows. {margin}")
                else:
                    print(f"{tag} NEGATIVE CONTROL DID NOT FIRE over {n} rows "
                          f"({margin}). The check is vacuous here and must "
                          f"not be quoted as evidence.")
                    rc = 1
            elif failures:
                print(f"{tag} FAIL -- {len(failures)} of {n} rows bad "
                      f"({margin})")
                for f in failures[:3]:
                    print("   ", f)
                rc = 1
            else:
                print(f"{tag} PASS -- all {n} rows a valid top-"
                      f"{S.INDEX_TOPK} vs the float64 oracle. {margin}")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl",
                    default="vllm_torchtpu.kernels.deepseek_v4.streamindex_topk")
    ap.add_argument("--mode", choices=["sweep", "correct", "control"],
                    default="sweep")
    ap.add_argument("--regimes", default="decode_prod")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--ladder", default="4,8",
                    help="bkv_p values for the correctness modes")
    ap.add_argument("--all-bands", action="store_true",
                    help="move all three entries of num_kv_pages_per_block, "
                         "not just the decode band")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    import importlib
    impl = importlib.import_module(a.impl)
    names = [n for n in a.regimes.split(",") if n]
    print(f"shipped NUM_KV_PAGES_PER_BLOCK = {S.NUM_KV_PAGES_PER_BLOCK}; "
          f"mode={a.mode}; regimes={names}", flush=True)

    if a.mode == "sweep":
        rows = sweep(impl, names, a.reps, not a.all_bands)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(rows, f, indent=2)
        return 0

    ladder = [int(x) for x in a.ladder.split(",") if x]
    return correctness(impl, names, ladder,
                       negative_control=(a.mode == "control"))


if __name__ == "__main__":
    sys.exit(main())
