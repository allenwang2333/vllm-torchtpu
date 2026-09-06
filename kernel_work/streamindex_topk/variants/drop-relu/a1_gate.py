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
"""Accuracy gate for area-6 row **A1 -- drop the ReLU**, and its A3 fallback.

`area6-algorithm.md` row A1: "Must be evaluated **before** any perf number."
This script does that literally, and it does it WITHOUT writing a kernel.

Why no kernel
-------------
A1 replaces

    score[t,s] = sum_h relu(q[t,h] . kv[s]) * w[t,h]          (the true function)

with

    score[t,s] = (sum_h w[t,h] * q[t,h]) . kv[s]              (A1's function)

That is a different function of the same inputs, not a different way of
evaluating the same one, so its accuracy is decided by arithmetic and by the
input distribution -- not by Pallas, tiling, dtype or the MXU. This script
therefore evaluates the **idealised** A1: exact float64 arithmetic, an exact
top-k, and the identical causal visibility mask taken from the float64 ReLU
oracle itself. Any real A1 kernel (fp8 operands, f32 accumulate, a re-quantised
collapsed query vector, an approximate SparseCore selection) is strictly worse
than this. So:

  * idealised A1 FAILS  =>  every A1 kernel fails. The candidate is dead and
    no kernel is worth building.
  * idealised A1 PASSES =>  nothing is proven about a real kernel; the real one
    still has to be built and re-gated.

The reference, the criterion, RANK_SLACK and MIN_RECALL all come from the
read-only instrument `instruments/test_correctness.py`; nothing here is a
re-derivation of them. `TOKEN_CHUNK` is rebound (module attribute, from this
script, per the read-only rule) purely to bound host memory at kv_len=65536.

A3 (the fallback)
-----------------
Row A3: score all S with A1's cheap form, keep a shortlist of the top M, then
exact-rescore the shortlist with the real ReLU form and take the top k of that.
Here the "exact rescore" reads the float64 oracle at the shortlist positions,
which is again the idealised best case. Recall of the true top-k after that
rescore is exactly `|shortlist ∩ true top-k| / k`, because the rescore orders
the shortlist by the true score.

Negative control
----------------
Same mutation as the instrument: zero the q rows of the second half of the
heads, score with the mutated q, compare against the UNMUTATED oracle, and
require the check to fire. Under A1 as under the ReLU form this is exactly
"drop 16 of 32 heads" (row A6), so it is a live discriminator of the same
family and magnitude as the approximation under test.
"""

import argparse
import json
import sys
import time

import numpy as np

from kernel_work.streamindex_topk.instruments import inputs as I
from kernel_work.streamindex_topk.instruments import shapes as S
from kernel_work.streamindex_topk.instruments import test_correctness as TC


# ---------------------------------------------------------------------------
# scorers
# ---------------------------------------------------------------------------

def a1_scores(d, regime: S.Regime, q_f32: np.ndarray,
              visible: np.ndarray) -> np.ndarray:
    """float64 A1 scores[num_tokens, kv_len], -inf where masked.

        qw[t]      = sum_h w[t,h] * q[t,h]        # [T, D], the collapse
        score[t,s] = qw[t] . kv[s]

    `visible` is the boolean visibility mask taken from the oracle, so the
    causal masking is bit-identical to the reference's by construction rather
    than by a re-implementation that could drift.
    """
    w = d["_weights"].astype(np.float64)          # [T, H]
    dequant = d["_dequant"]                       # f32[num_pages, PAGE, D]
    block_table = d["_block_table"]
    q_lens = d["_q_lens"]
    cu = np.asarray(d["cu_q_lens"])

    q = q_f32.astype(np.float64)                  # [T, H, D]
    qw = np.einsum("thd,th->td", q, w, optimize=True)   # [T, D]

    T = q.shape[0]
    kv_len = regime.kv_len
    out = np.full((T, kv_len), -np.inf, dtype=np.float64)
    for b in range(regime.max_num_seqs):
        if q_lens[b] == 0:
            continue
        pages = block_table[b]
        seq_kv = dequant[pages].reshape(-1, S.INDEX_HEAD_DIM)[:kv_len]
        seq_kv = seq_kv.astype(np.float64)
        t0, t1 = int(cu[b]), int(cu[b + 1])
        out[t0:t1] = qw[t0:t1] @ seq_kv.T
    return np.where(visible, out, -np.inf)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def select_topk(surrogate: np.ndarray, vis_idx, k: int) -> np.ndarray:
    """Exact top-k of `surrogate`, in the kernel's output format.

    Returns int64[T, k], -1 padded, matching `impl.streamindex_topk`'s contract
    (`test_correctness.check` reads `got[got >= 0]`).
    """
    T = surrogate.shape[0]
    out = np.full((T, k), -1, dtype=np.int64)
    for t in range(T):
        vis = vis_idx[t]
        n_visible = len(vis)
        if n_visible == 0:
            continue
        expected_n = min(k, n_visible)
        if expected_n >= n_visible:
            out[t, :expected_n] = vis
            continue
        s = surrogate[t, vis]
        part = np.argpartition(-s, expected_n - 1)[:expected_n]
        out[t, :expected_n] = vis[part]
    return out


def select_a3(surrogate: np.ndarray, oracle: np.ndarray, vis_idx, k: int,
              M: int) -> np.ndarray:
    """Two-stage A3: top-M by the cheap form, then exact rescore, then top-k."""
    T = surrogate.shape[0]
    out = np.full((T, k), -1, dtype=np.int64)
    for t in range(T):
        vis = vis_idx[t]
        n_visible = len(vis)
        if n_visible == 0:
            continue
        expected_n = min(k, n_visible)
        m_eff = min(M, n_visible)
        if m_eff >= n_visible:
            shortlist = vis
        else:
            s = surrogate[t, vis]
            shortlist = vis[np.argpartition(-s, m_eff - 1)[:m_eff]]
        if expected_n >= len(shortlist):
            out[t, :len(shortlist)] = shortlist
            continue
        o = oracle[t, shortlist]
        part = np.argpartition(-o, expected_n - 1)[:expected_n]
        out[t, :expected_n] = shortlist[part]
    return out


# ---------------------------------------------------------------------------
# a fast re-implementation of the instrument's check, for the M sweep only
# ---------------------------------------------------------------------------

def build_ranks(oracle: np.ndarray) -> np.ndarray:
    """rank_of[t, i] = 0-based position of index i in the oracle's descending
    order for row t. Same construction as test_correctness.check, hoisted out
    of the per-arm loop so an 11-point M sweep does not re-sort 11 times."""
    T, W = oracle.shape
    ranks = np.empty((T, W), dtype=np.int32)
    arange = np.arange(W, dtype=np.int32)
    for t in range(T):
        order = np.argsort(-oracle[t], kind="stable")
        ranks[t, order] = arange
    return ranks


def check_fast(idxs, oracle, ranks, vis_counts, k):
    """Equivalent of test_correctness.check using a precomputed rank matrix.

    Asserted equal to the instrument's own `check` on the A1 arm of every
    regime before any sweep number is reported.
    """
    failures = 0
    first = None
    worst_rank_seen = 0
    min_recall = 1.0
    recall_sum = 0.0
    n_checked = 0
    T = oracle.shape[0]
    for t in range(T):
        n_visible = int(vis_counts[t])
        if n_visible == 0:
            continue
        n_checked += 1
        expected_n = min(k, n_visible)
        got = idxs[t]
        got_valid = got[got >= 0]
        if len(got_valid) != expected_n or \
                len(np.unique(got_valid)) != len(got_valid):
            failures += 1
            if first is None:
                first = f"row {t}: bad cardinality/uniqueness"
            continue
        r = ranks[t, got_valid]
        worst_rank = int(r.max()) + 1
        recall = float((r < expected_n).mean())
        worst_rank_seen = max(worst_rank_seen, worst_rank)
        min_recall = min(min_recall, recall)
        recall_sum += recall
        if worst_rank > expected_n + TC.RANK_SLACK or recall < TC.MIN_RECALL:
            failures += 1
            if first is None:
                first = (f"row {t}: worst_rank={worst_rank} (allowed "
                         f"{expected_n + TC.RANK_SLACK}), recall={recall:.5f}")
    return n_checked, failures, first, dict(
        worst_rank_seen=worst_rank_seen, min_recall=min_recall,
        mean_recall=(recall_sum / n_checked) if n_checked else float("nan"))


# ---------------------------------------------------------------------------

def tiny_regime():
    return S.Regime(name="tiny", phase="mixed", max_num_seqs=2, num_tokens=64,
                    kv_len=4096, max_model_len=4096, note="CPU dry run only")


def run_regime(regime: S.Regime, m_sweep, seed=0, token_chunk=None):
    k = S.INDEX_TOPK
    if token_chunk:
        TC.TOKEN_CHUNK = token_chunk   # module attribute rebind, not an edit
    t0 = time.time()
    d = I.build(regime, seed=seed, with_dequant=True)
    oracle = TC.oracle_scores(d, regime)           # float64 ReLU reference
    visible = np.isfinite(oracle)
    vis_idx = [np.flatnonzero(visible[t]).astype(np.int32)
               for t in range(oracle.shape[0])]
    vis_counts = np.array([len(v) for v in vis_idx])
    print(f"[{regime.name}] oracle built in {time.time()-t0:.1f}s; "
          f"T={oracle.shape[0]} kv_len={oracle.shape[1]} "
          f"n_visible min/max={vis_counts.min()}/{vis_counts.max()} k={k}",
          flush=True)

    q_f32 = d["_q_f32"]

    # ---- A1 gate -----------------------------------------------------------
    surr = a1_scores(d, regime, q_f32, visible)
    idxs = select_topk(surr, vis_idx, k)
    n, failures, st = TC.check(idxs, oracle, k)
    allowed = k + TC.RANK_SLACK
    margin = (f"worst_rank={st['worst_rank_seen']} of allowed {allowed}, "
              f"min_recall={st['min_recall']:.5f} of required {TC.MIN_RECALL}")
    if failures:
        print(f"[{regime.name} A1] FAIL -- {len(failures)} of {n} rows bad "
              f"({margin})", flush=True)
        for f in failures[:3]:
            print("   ", f, flush=True)
    else:
        print(f"[{regime.name} A1] PASS -- all {n} rows are a valid top-{k} "
              f"against the float64 oracle. {margin}", flush=True)
    a1 = dict(regime=regime.name, arm="A1", rows=n, bad_rows=len(failures),
              worst_rank=st["worst_rank_seen"], min_recall=st["min_recall"],
              allowed_rank=allowed, min_recall_required=TC.MIN_RECALL,
              verdict="FAIL" if failures else "PASS")

    # ---- negative control on the A1 scorer ---------------------------------
    qm = np.array(q_f32, copy=True)
    qm[:, S.INDEX_N_HEADS // 2:, :] = 0.0
    surr_c = a1_scores(d, regime, qm, visible)
    idxs_c = select_topk(surr_c, vis_idx, k)
    nc, fc, stc = TC.check(idxs_c, oracle, k)
    mc = (f"worst_rank={stc['worst_rank_seen']} of allowed {allowed}, "
          f"min_recall={stc['min_recall']:.5f} of required {TC.MIN_RECALL}")
    if fc:
        print(f"[{regime.name} A1-control] NEGATIVE CONTROL OK -- fired on "
              f"{len(fc)} of {nc} rows. {mc}", flush=True)
    else:
        print(f"[{regime.name} A1-control] NEGATIVE CONTROL DID NOT FIRE over "
              f"{nc} rows ({mc}). The check is vacuous here.", flush=True)
    ctl = dict(regime=regime.name, arm="A1-negative-control", rows=nc,
               bad_rows=len(fc), worst_rank=stc["worst_rank_seen"],
               min_recall=stc["min_recall"],
               verdict="FIRED" if fc else "DID-NOT-FIRE")
    del surr_c, idxs_c

    # ---- exact-arithmetic reference arm ------------------------------------
    # The oracle scoring itself, selected exactly. Must be a perfect PASS; if it
    # is not, the harness is broken and no other row on this regime means
    # anything.
    idxs_ref = select_topk(oracle, vis_idx, k)
    nr, fr, str_ = TC.check(idxs_ref, oracle, k)
    print(f"[{regime.name} self-ref] {'PASS' if not fr else 'BROKEN'} -- "
          f"worst_rank={str_['worst_rank_seen']}, "
          f"min_recall={str_['min_recall']:.5f}", flush=True)
    selfref = dict(regime=regime.name, arm="self-reference", rows=nr,
                   bad_rows=len(fr), worst_rank=str_["worst_rank_seen"],
                   min_recall=str_["min_recall"],
                   verdict="OK" if not fr else "HARNESS-BROKEN")
    del idxs_ref

    # ---- A3 sweep ----------------------------------------------------------
    ranks = build_ranks(oracle)
    # validate check_fast against the instrument on the A1 arm
    n2, f2, first2, st2 = check_fast(idxs, oracle, ranks, vis_counts, k)
    assert (n2, len(failures)) == (n, f2) and \
        st2["worst_rank_seen"] == st["worst_rank_seen"] and \
        abs(st2["min_recall"] - st["min_recall"]) < 1e-12, \
        f"check_fast disagrees with instrument check: {st2} vs {st}"
    print(f"[{regime.name}] check_fast validated against "
          f"instruments/test_correctness.check on the A1 arm; "
          f"A1 mean_recall={st2['mean_recall']:.5f}", flush=True)
    a1["mean_recall"] = st2["mean_recall"]
    del idxs

    sweep = []
    for M in m_sweep:
        ia = select_a3(surr, oracle, vis_idx, k, M)
        na, fa, firsta, sta = check_fast(ia, oracle, ranks, vis_counts, k)
        saturated = bool(M >= int(vis_counts.max()))
        frac_S = M / float(regime.kv_len)
        print(f"[{regime.name} A3 M={M}] "
              f"{'PASS' if fa == 0 else 'FAIL'} -- {fa} of {na} rows bad, "
              f"worst_rank={sta['worst_rank_seen']} of allowed {allowed}, "
              f"min_recall={sta['min_recall']:.5f} of required "
              f"{TC.MIN_RECALL}, mean_recall={sta['mean_recall']:.5f}, "
              f"M/S={frac_S:.4f}"
              f"{' [M >= n_visible: exact by construction]' if saturated else ''}",
              flush=True)
        sweep.append(dict(regime=regime.name, arm="A3", M=M, rows=na,
                          bad_rows=fa, worst_rank=sta["worst_rank_seen"],
                          min_recall=sta["min_recall"],
                          mean_recall=sta["mean_recall"], M_over_S=frac_S,
                          saturated=saturated,
                          verdict="PASS" if fa == 0 else "FAIL"))
        del ia

    return dict(regime=regime.name, kv_len=regime.kv_len,
                num_tokens=regime.num_tokens, k=k,
                rank_slack=TC.RANK_SLACK, min_recall_required=TC.MIN_RECALL,
                a1=a1, control=ctl, self_ref=selfref, a3=sweep)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", default="decode_prod")
    ap.add_argument("--m-sweep", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-chunk", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true",
                    help="tiny CPU-only shape; validates the code path only, "
                         "produces no verdict")
    a = ap.parse_args(argv)

    m_sweep = [int(x) for x in a.m_sweep.split(",") if x.strip()]
    regimes = ([tiny_regime()] if a.dry
               else [S.REGIMES_BY_NAME[n] for n in a.regimes.split(",") if n])

    results = []
    for r in regimes:
        results.append(run_regime(r, m_sweep, seed=a.seed,
                                  token_chunk=a.token_chunk or None))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"wrote {a.out}", flush=True)
    print("=== JSON ===", flush=True)
    print(json.dumps(results), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
