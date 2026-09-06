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
"""Accuracy gate for area-6 row **A2 -- clamp the per-head weight, `w <- max(w, 0)`**.

A2 is NOT A1. A1 dropped the ReLU and replaced the function outright:

    A1:  score[t,s] = (sum_h w[t,h] * q[t,h]) . kv[s]

A2 keeps the ReLU and changes only the weight:

    true: score[t,s] = sum_h relu(q[t,h] . kv[s]) *          w[t,h]
    A2:   score[t,s] = sum_h relu(q[t,h] . kv[s]) * max(w[t,h], 0)

Why the gate is run at all
--------------------------
A2 makes the precondition of option E1 true **by construction**: with `w >= 0`
every per-head term is non-negative, so the head sum is monotone in each term
and E1's transformation is admissible. E1 is a measured-but-disqualified 13.70%
win; if A2 passes, E1 stops being disqualified. So the only question here is
whether A2 changes the selected top-k, and that is decided by arithmetic and by
the distribution of `w` -- not by Pallas, tiling, dtype or the MXU.

As in `variants/drop-relu/a1_gate.py`, this evaluates the **idealised** A2:
exact float64 arithmetic, an exact top-k, and the identical causal visibility
mask, taken from the float64 ReLU oracle itself. Any real A2 kernel is strictly
worse than this. So:

  * idealised A2 FAILS  =>  every A2 kernel fails, and E1 stays disqualified.
  * idealised A2 PASSES =>  E1's precondition is establishable; the real kernel
    still has to be built and re-gated.

How the surrogate is computed, and why it cannot drift
------------------------------------------------------
The A2 scorer is **not** a re-implementation. `instruments/test_correctness.py`
already contains the exact ReLU form (`oracle_scores`), and it reads the weight
out of `d["_weights"]`. A2 is therefore obtained by handing that same read-only
function a shallow copy of the input dict whose `_weights` entry is clamped.
Identical code path, identical masking, one changed array. Nothing in
`instruments/` is edited; `TOKEN_CHUNK` is rebound as a module attribute purely
to bound host memory at kv_len=65536, exactly as `a1_gate.py` does.

`select_topk`, `build_ranks` and `check_fast` are imported from `a1_gate.py`
(commit `35ea3bcc`) by path rather than copied, so A2's verdict is produced by
the same selection and the same criterion that produced A1's.

Negative control
----------------
The same mutation the instrument and `a1_gate.py` use: zero the q rows of the
second half of the heads, score with the mutated q *through the A2 surrogate*,
compare against the UNMUTATED oracle, and require the check to fire.

Self-reference arm
------------------
The oracle scored and selected exactly. Must be a perfect PASS; if it is not,
the harness is broken and no other row on that regime means anything.

Sign convention: this file reports no timings. Verdicts only.
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/kernel_work/", 1)[0])

from kernel_work.streamindex_topk.instruments import inputs as I  # noqa: E402
from kernel_work.streamindex_topk.instruments import shapes as S  # noqa: E402
from kernel_work.streamindex_topk.instruments import (  # noqa: E402
    test_correctness as TC)

_HERE = os.path.dirname(os.path.abspath(__file__))
_A1_GATE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "variants", "drop-relu", "a1_gate.py"))


def _load_a1_gate():
    """Import `variants/drop-relu/a1_gate.py` by path.

    `variants/drop-relu` is not an importable package name (the hyphen) and
    must not become one. Reusing its `select_topk` / `build_ranks` /
    `check_fast` is what makes A2's verdict comparable to A1's rather than
    merely similar to it.
    """
    spec = importlib.util.spec_from_file_location("a1_gate", _A1_GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A1G = _load_a1_gate()


def clamp_weights(d):
    """Shallow copy of the input dict with `w <- max(w, 0)`.

    Returns (d2, stats) where `stats` records how much of `w` the clamp
    actually moved -- a gate on an input distribution where the clamp is a
    no-op would be vacuous, so the fraction clamped is reported next to every
    verdict.
    """
    w = np.asarray(d["_weights"])
    w_clamped = np.maximum(w, 0.0)
    n = w.size
    n_neg = int((w < 0.0).sum())
    stats = dict(
        w_elems=n,
        w_negative=n_neg,
        w_negative_frac=n_neg / n if n else float("nan"),
        w_min=float(w.min()),
        w_max=float(w.max()),
        rows_all_negative=int((w < 0.0).all(axis=1).sum()),
        mean_heads_zeroed_per_row=float((w < 0.0).sum(axis=1).mean()),
    )
    d2 = dict(d)
    d2["_weights"] = w_clamped.astype(w.dtype)
    return d2, stats


def run_regime(regime: S.Regime, seed=0, token_chunk=None):
    k = S.INDEX_TOPK
    if token_chunk:
        TC.TOKEN_CHUNK = token_chunk   # module attribute rebind, not an edit
    allowed = k + TC.RANK_SLACK

    t0 = time.time()
    d = I.build(regime, seed=seed, with_dequant=True)
    oracle = TC.oracle_scores(d, regime)           # float64 ReLU reference
    visible = np.isfinite(oracle)
    vis_idx = [np.flatnonzero(visible[t]).astype(np.int32)
               for t in range(oracle.shape[0])]
    vis_counts = np.array([len(v) for v in vis_idx])
    print(f"[{regime.name}] oracle built in {time.time()-t0:.1f}s; "
          f"T={oracle.shape[0]} kv_len={oracle.shape[1]} "
          f"n_visible min/max={vis_counts.min()}/{vis_counts.max()} k={k} "
          f"TOKEN_CHUNK={TC.TOKEN_CHUNK}", flush=True)

    d_clamped, wstats = clamp_weights(d)
    print(f"[{regime.name}] clamp bites: {wstats['w_negative']} of "
          f"{wstats['w_elems']} weights are negative "
          f"({wstats['w_negative_frac']*100:.2f}%), w in "
          f"[{wstats['w_min']:.4f}, {wstats['w_max']:.4f}], mean heads zeroed "
          f"per token = {wstats['mean_heads_zeroed_per_row']:.2f} of "
          f"{S.INDEX_N_HEADS}, rows with every head zeroed = "
          f"{wstats['rows_all_negative']}", flush=True)

    ranks = A1G.build_ranks(oracle)

    # ---- A2 gate -----------------------------------------------------------
    t1 = time.time()
    surr = TC.oracle_scores(d_clamped, regime)     # identical code path
    idxs = A1G.select_topk(surr, vis_idx, k)
    n, failures, st = TC.check(idxs, oracle, k)
    margin = (f"worst_rank={st['worst_rank_seen']} of allowed {allowed}, "
              f"min_recall={st['min_recall']:.5f} of required {TC.MIN_RECALL}")
    if failures:
        print(f"[{regime.name} A2] FAIL -- {len(failures)} of {n} rows bad "
              f"({margin})", flush=True)
        for f in failures[:3]:
            print("   ", f, flush=True)
    else:
        print(f"[{regime.name} A2] PASS -- all {n} rows are a valid top-{k} "
              f"against the float64 oracle. {margin}", flush=True)

    # `check_fast` validated against the instrument on this arm, so the
    # mean_recall reported below is not from an unvalidated re-implementation.
    n2, f2, first2, st2 = A1G.check_fast(idxs, oracle, ranks, vis_counts, k)
    assert (n2, len(failures)) == (n, f2) and \
        st2["worst_rank_seen"] == st["worst_rank_seen"] and \
        abs(st2["min_recall"] - st["min_recall"]) < 1e-12, \
        f"check_fast disagrees with instrument check: {st2} vs {st}"
    print(f"[{regime.name}] check_fast validated against "
          f"instruments/test_correctness.check on the A2 arm; "
          f"A2 mean_recall={st2['mean_recall']:.5f} "
          f"({time.time()-t1:.1f}s)", flush=True)

    a2 = dict(regime=regime.name, arm="A2", rows=n, bad_rows=len(failures),
              worst_rank=st["worst_rank_seen"], min_recall=st["min_recall"],
              mean_recall=st2["mean_recall"],
              allowed_rank=allowed, min_recall_required=TC.MIN_RECALL,
              verdict="FAIL" if failures else "PASS",
              first_failure=(failures[0] if failures else None))
    del surr, idxs

    # ---- negative control on the A2 scorer ---------------------------------
    qm = np.array(d["_q_f32"], copy=True)
    qm[:, S.INDEX_N_HEADS // 2:, :] = 0.0
    d_ctl = dict(d_clamped)
    d_ctl["_q_f32"] = qm
    surr_c = TC.oracle_scores(d_ctl, regime)
    idxs_c = A1G.select_topk(surr_c, vis_idx, k)
    nc, fc, stc = TC.check(idxs_c, oracle, k)
    mc = (f"worst_rank={stc['worst_rank_seen']} of allowed {allowed}, "
          f"min_recall={stc['min_recall']:.5f} of required {TC.MIN_RECALL}")
    if fc:
        print(f"[{regime.name} A2-control] NEGATIVE CONTROL OK -- fired on "
              f"{len(fc)} of {nc} rows. {mc}", flush=True)
    else:
        print(f"[{regime.name} A2-control] NEGATIVE CONTROL DID NOT FIRE over "
              f"{nc} rows ({mc}). The check is vacuous here.", flush=True)
    ctl = dict(regime=regime.name, arm="A2-negative-control", rows=nc,
               bad_rows=len(fc), worst_rank=stc["worst_rank_seen"],
               min_recall=stc["min_recall"],
               verdict="FIRED" if fc else "DID-NOT-FIRE")
    del surr_c, idxs_c

    # ---- exact-arithmetic reference arm ------------------------------------
    idxs_ref = A1G.select_topk(oracle, vis_idx, k)
    nr, fr, str_ = TC.check(idxs_ref, oracle, k)
    print(f"[{regime.name} self-ref] {'PASS' if not fr else 'BROKEN'} -- "
          f"worst_rank={str_['worst_rank_seen']}, "
          f"min_recall={str_['min_recall']:.5f}", flush=True)
    selfref = dict(regime=regime.name, arm="self-reference", rows=nr,
                   bad_rows=len(fr), worst_rank=str_["worst_rank_seen"],
                   min_recall=str_["min_recall"],
                   verdict="OK" if not fr else "HARNESS-BROKEN")
    del idxs_ref

    return dict(regime=regime.name, kv_len=regime.kv_len,
                num_tokens=regime.num_tokens, k=k,
                rank_slack=TC.RANK_SLACK, min_recall_required=TC.MIN_RECALL,
                token_chunk=TC.TOKEN_CHUNK, weight_stats=wstats,
                a2=a2, control=ctl, self_ref=selfref)


def tiny_regime():
    return S.Regime(name="tiny", phase="mixed", max_num_seqs=2, num_tokens=64,
                    kv_len=4096, max_model_len=4096, note="CPU dry run only")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default=",".join(r.name for r in S.REGIMES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-chunk", default="",
                    help="comma-separated, one per regime, or a single value "
                         "used for all; blank leaves the instrument default")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true",
                    help="tiny CPU-only shape; validates the code path only, "
                         "produces no verdict")
    a = ap.parse_args(argv)

    regimes = ([tiny_regime()] if a.dry
               else [S.REGIMES_BY_NAME[n] for n in a.regimes.split(",") if n])
    chunks = [int(x) for x in a.token_chunk.split(",") if x.strip()]
    if chunks and len(chunks) == 1:
        chunks = chunks * len(regimes)
    if chunks and len(chunks) != len(regimes):
        raise SystemExit("--token-chunk must be blank, one value, or one per "
                         "regime")

    results = []
    for i, r in enumerate(regimes):
        results.append(run_regime(r, seed=a.seed,
                                  token_chunk=(chunks[i] if chunks else None)))
        print("", flush=True)

    # ---- the table, in the same columns A1's gate reported -----------------
    print("=== A2 (clamp w <- max(w, 0), keep the ReLU) -- gate table ===")
    hdr = (f"{'regime':<14}{'rows':>7}{'bad':>7}{'worst_rank':>12}"
           f"{'allowed':>9}{'min_recall':>12}{'required':>10}"
           f"{'control':>14}{'self-ref':>11}{'verdict':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['regime']:<14}{r['a2']['rows']:>7}{r['a2']['bad_rows']:>7}"
              f"{r['a2']['worst_rank']:>12}{r['a2']['allowed_rank']:>9}"
              f"{r['a2']['min_recall']:>12.5f}"
              f"{r['a2']['min_recall_required']:>10}"
              f"{r['control']['verdict']:>14}"
              f"{r['self_ref']['verdict']:>11}"
              f"{r['a2']['verdict']:>9}")
    n_fail = sum(1 for r in results if r["a2"]["verdict"] == "FAIL")
    print(f"\n{len(results) - n_fail} PASS / {n_fail} FAIL of "
          f"{len(results)} regimes.")
    print("A marginal regime is a FAIL. min_recall = 0.989 is a FAIL.")
    print("Consequence for E1: A2 PASSES on every regime => E1's "
          "non-negativity precondition is establishable and E1 is no longer "
          "disqualified. Any FAIL => E1 stays disqualified.")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"wrote {a.out}", flush=True)
    print("=== JSON ===", flush=True)
    print(json.dumps(results), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
