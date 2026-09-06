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
"""Variant E -- Retune. Incumbent vs variant, in ONE job, same instruments.

What the variant is
-------------------
`num_kv_pages_per_block` goes from the shipped `(2, 2, 2)`
(`mla_attention_op.py:119`) to `(4, 2, 4)`. One compile-time tuple. No
arithmetic change, no new code, no kernel edit. `retune.patch` is the whole of
it.

Why this script exists when `probes/bkv_p_prod.py` already swept the tuple
--------------------------------------------------------------------------
The prize was measured across TWO jobs -- decode in `j-98399433-7763-4385-81ca`
and mixed in `j-3aa755dd-900b-4f31-9fa3` -- and the `(4,2,4)` spelling was
never run at all. A step-3 variant is judged against the incumbent measured
alongside it, so every regime here carries its own `(2,2,2)` arm taken in the
same process, minutes apart, on the same chip.

The four arms, and why the fourth is not redundant
--------------------------------------------------
    (2,2,2)   the incumbent -- the denominator of every delta in this run
    (4,2,4)   the variant, as `retune.patch` spells it
    (4,4,4)   the variant as the probe record spells it
    (2,2,2)   the incumbent AGAIN, last

`num_sublanes_total` is a `max` over the whole tuple, so `(4,2,4)` and
`(4,4,4)` are the same 4 in every quantity that reaches the hardware -- same
scores width, same decode band, same mixed band. The two arms are therefore a
built-in replicate: whatever separates them is drift, not design, and it puts a
floor under how small a difference this instrument can resolve.

The repeated incumbent is the drift control proper. `bkv_p_RESULT.md`'s mixed
rows land 1.6-1.7% on the slow side against 1.3-1.8% spreads, i.e. inside the
dispersion of a single arm. Spread within an arm does not bound drift ACROSS
arms in one process -- compile order, allocator state and host temperature all
move between them -- so the repeat is what says whether a 1.6% gap between two
arms means anything at all. If the two `(2,2,2)` arms differ by as much as the
variant does, the mixed rows are not readable as a cost.

How the tuple is varied without touching a read-only instrument
---------------------------------------------------------------
`instruments/` is READ-ONLY after step 1: a yardstick that changes between
candidates is not a yardstick. `microbench.run_regime` reads
`shapes.NUM_KV_PAGES_PER_BLOCK` internally, so this script rebinds that module
attribute around each call and restores it in a `finally`. `microbench.py` and
`test_correctness.py` are imported and executed exactly as written. This is the
same mechanism `probes/bkv_p_prod.py` used, which is what makes the numbers
here comparable to that file's.

`early_exit=False`, matching step 1's baseline table and every `bkv_p` arm
measured so far. Production sets it True (`mla_attention_op.py:57`); at these
shapes the gate is false, so it adds a `lax.cond` and nothing else, and it is
held constant across both arms either way.

Sign convention: every percentage is a change in TIME, `(new - old)/old`.
Negative is faster.
"""

import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])

from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
from kernel_work.streamindex_topk.instruments import microbench as MB  # noqa: E402
from kernel_work.streamindex_topk.instruments import test_correctness as TC  # noqa: E402

INCUMBENT = (2, 2, 2)

# label, tuple. Order matters: the first arm is the denominator, the last is
# the drift control, and the two middle arms are the replicate pair.
DEFAULT_ARMS = [
    ("incumbent", (2, 2, 2)),
    ("variant", (4, 2, 4)),
    ("variant-alt", (4, 4, 4)),
    ("incumbent-repeat", (2, 2, 2)),
]


class tuple_set:
    """Rebind `shapes.NUM_KV_PAGES_PER_BLOCK` for the duration of a block."""

    def __init__(self, tup):
        self.old = S.NUM_KV_PAGES_PER_BLOCK
        self.new = tuple(int(x) for x in tup)

    def __enter__(self):
        S.NUM_KV_PAGES_PER_BLOCK = self.new
        return self.new

    def __exit__(self, *exc):
        S.NUM_KV_PAGES_PER_BLOCK = self.old
        return False


def _annotate(row, regime, tup):
    """Overwrite the width columns `run_regime` reported.

    `num_sublanes_total` and `scores_bytes` take `bkv_ps` as a DEFAULT
    argument, which Python binds at function-definition time -- so the values
    in the row are the shipped `(2,2,2)` ones no matter what the rebinding did,
    and would understate every raised arm. The timing itself is unaffected: the
    kernel call reads the module attribute at call time. Same trap
    `probes/bkv_p_prod.py` documents; recorded again here because the row is
    wrong in a way that looks right.
    """
    subl = S.num_sublanes_total(regime.max_model_len, bkv_ps=tup)
    row["num_sublanes_total"] = subl
    row["scores_MB"] = regime.num_tokens * subl * 128 * 4 / 1e6
    row["bkv_tuple"] = list(tup)
    return row


def sweep(impl, regime_names, reps, arms):
    rows = []
    for name in regime_names:
        regime = S.REGIMES_BY_NAME[name]
        for label, tup in arms:
            try:
                with tuple_set(tup) as t:
                    r = MB.run_regime(impl, regime, early_exit=False,
                                      reps=reps, measure_topk=False)
                r = _annotate(r, regime, t)
            except Exception as exc:  # noqa: BLE001
                r = dict(regime=name, bkv_tuple=list(tup),
                         error=repr(exc)[:400])
            r["arm"] = label
            rows.append(r)
            print(json.dumps(r), flush=True)

    print("\n=== E -- Retune: incumbent and variant, same job, same "
          "instrument (median ms/call, 1 TensorCore, early_exit=False) ===")
    hdr = (f"{'regime':<14}{'arm':<18}{'tuple':>10}{'subl':>6}{'scoresMB':>10}"
           f"{'num_bkv':>9}{'full_ms':>12}{'spread%':>9}"
           f"{'d time %':>10}{'speedup x':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name in regime_names:
        mine = [x for x in rows if x.get("regime") == name]
        base = next((r["full_ms"] for r in mine
                     if r.get("arm") == "incumbent" and "error" not in r), None)
        for r in mine:
            tstr = ",".join(str(x) for x in r["bkv_tuple"])
            if "error" in r:
                print(f"{name:<14}{r['arm']:<18}{tstr:>10}  ERROR "
                      f"{r['error'][:70]}")
                continue
            dt = ((r["full_ms"] - base) / base * 100.0) if base else float("nan")
            sp = (base / r["full_ms"]) if base else float("nan")
            print(f"{name:<14}{r['arm']:<18}{tstr:>10}"
                  f"{r['num_sublanes_total']:>6}{r['scores_MB']:>10.1f}"
                  f"{r['num_bkv']:>9}{r['full_ms']:>12.6f}"
                  f"{r['full_spread_pct']:>9.2f}{dt:>10.2f}{sp:>11.4f}")

    print("\n`d time %` is (arm - incumbent)/incumbent in TIME; negative is "
          "faster. `speedup x` is incumbent/arm; >1 is faster. Both are "
          "against the FIRST incumbent arm of the same regime, in this "
          "process. `subl` is num_sublanes_total, a max over the whole tuple, "
          "so it moves for every band. `num_bkv` is the inner iteration count "
          "for THIS regime's own band.")

    print("\n=== drift control: the two incumbent arms against each other ===")
    print("Same tuple, same regime, same process, measured first and last. "
          "This bounds what a difference between two arms can mean; a variant "
          "delta smaller than this is not readable.")
    for name in regime_names:
        mine = [x for x in rows if x.get("regime") == name
                and "error" not in x]
        a = next((r for r in mine if r.get("arm") == "incumbent"), None)
        b = next((r for r in mine if r.get("arm") == "incumbent-repeat"), None)
        if not (a and b):
            print(f"{name:<14} drift control not available")
            continue
        d = (b["full_ms"] - a["full_ms"]) / a["full_ms"] * 100.0
        print(f"{name:<14} first={a['full_ms']:.6f} ms  "
              f"last={b['full_ms']:.6f} ms  drift={d:+.2f} %")

    print("\n=== replicate: (4,2,4) against (4,4,4) ===")
    print("Identical in everything that executes -- max(tuple) is 4 for both, "
          "the decode band reads slot 0 = 4 for both, the mixed band reads "
          "slot 2 = 4 for both. Any gap here is instrument noise.")
    for name in regime_names:
        mine = [x for x in rows if x.get("regime") == name
                and "error" not in x]
        a = next((r for r in mine if r.get("arm") == "variant"), None)
        b = next((r for r in mine if r.get("arm") == "variant-alt"), None)
        if not (a and b):
            print(f"{name:<14} replicate not available")
            continue
        d = (b["full_ms"] - a["full_ms"]) / a["full_ms"] * 100.0
        print(f"{name:<14} (4,2,4)={a['full_ms']:.6f} ms  "
              f"(4,4,4)={b['full_ms']:.6f} ms  gap={d:+.2f} %  "
              f"subl {a['num_sublanes_total']} vs {b['num_sublanes_total']}")
    return rows


def correctness(impl, regime_names, tuples, negative_control):
    rc = 0
    for name in regime_names:
        regime = S.REGIMES_BY_NAME[name]
        for tup in tuples:
            with tuple_set(tup) as t:
                try:
                    n, failures, st = TC.run(
                        impl, regime, negative_control=negative_control)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{name} tuple={t}] ERROR {exc!r}")
                    rc = 1
                    continue
            margin = (f"worst_rank={st['worst_rank_seen']} of allowed "
                      f"{S.INDEX_TOPK + TC.RANK_SLACK}, min_recall="
                      f"{st['min_recall']:.5f} of required {TC.MIN_RECALL}")
            tag = f"[{name} tuple={t}]"
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
    ap.add_argument("--tuples", default="",
                    help="override the arm list, e.g. '2,2,2;4,2,4'. In "
                         "sweep mode the first entry is the denominator.")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    import importlib
    impl = importlib.import_module(a.impl)
    names = [n for n in a.regimes.split(",") if n]
    print(f"shipped NUM_KV_PAGES_PER_BLOCK = {S.NUM_KV_PAGES_PER_BLOCK} "
          f"(incumbent); mode={a.mode}; regimes={names}", flush=True)
    if S.NUM_KV_PAGES_PER_BLOCK != INCUMBENT:
        print(f"WARNING: instruments/shapes.py no longer ships {INCUMBENT}. "
              "The 'incumbent' arm below is this script's constant, not the "
              "instrument's -- check before quoting.", flush=True)

    tuples = ([tuple(int(v) for v in t.split(",")) for t in a.tuples.split(";")]
              if a.tuples else None)

    if a.mode == "sweep":
        arms = ([(f"arm{i}", t) for i, t in enumerate(tuples)] if tuples
                else DEFAULT_ARMS)
        rows = sweep(impl, names, a.reps, arms)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(rows, f, indent=2)
        return 0

    return correctness(impl, names, tuples or [(4, 2, 4)],
                       negative_control=(a.mode == "control"))


if __name__ == "__main__":
    sys.exit(main())
