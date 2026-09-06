"""Correctness gate for streamindex_topk at GLM-5.2 geometry.

Exits non-zero on any failure, so it can be the last command in a cdk RUN_CMD
and the job status means what it says.

The reference recomputes
    score[t, j] = scale[j] * sum_h w[t,h] * relu(q[t,h,:] . k8[j,:])
in f32 in the kernel's own operation order, then takes an exact top-k.

WHAT IS AND IS NOT ASSERTED, and why. Four properties are exact and carry no
tolerance: the selected count, uniqueness, index range, and that padded token
rows are -1-filled. The selection itself cannot be asserted exactly. Measured
on the incumbent (j-284a8842-3db6-44e0-ba49) every case disagreed with an
exact-match reference, always at rank 2014-2047 of 2048 and always by
<= 0.005 on scores spanning ~2.1: the kernel and an f32 reference order
near-ties at the cut line differently. An exact gate therefore red-lights the
baseline, and a gate that fails on the baseline cannot judge a candidate.

So the selection is judged on the two properties a *wrong* kernel violates and
a near-tie does not:

  deficit -- how far below the true k-th score the worst selected element sits,
             as a fraction of the top-to-cut score range. Near-ties reach a
             hair below the cut. A kernel scoring garbage, or one that dropped
             a KV block containing winners, reaches far below it.
  churn   -- what fraction of the true top-k is missing. Same discrimination
             as a count: dropping a block loses roughly that block's share of
             the winners, which for one of 1024 blocks at 1M context is still
             orders of magnitude above a handful of boundary swaps.

Both are printed on every run whether or not they trip, so a regression shows
up as a number moving rather than only as a pass/fail flip.

WHAT THE CASE LIST HAS TO COVER, and why it is not just "a few shapes". The
selector is TWO-STAGE above one row-slice: it partitions each row into P column
slices, selects within each, then selects across the P*kp survivors. P is not a
constant -- a candidate dispatches over a ladder (1, 2, 4, 8, 16) picked from
`max(row_lengths)` -- and the P=1 path shares almost nothing with the P=16 one:
no stage-2 gather, no cross-slice merge, no emitted keys, no per-row candidate
crop. A case list whose every entry resolves to P=1 prints `5/5 cases passed`
with all of that arbitrarily broken, which is what this file did until the
long-context cases below were added. `print_geometry` now prints the resolved
rung per case and the set of rungs exercised, so coverage is in the log rather
than inferred. See `case_geometry`.

  python3 correctness.py                     # all cases, 3 seeds
  python3 correctness.py --case dec-2x8k
  python3 correctness.py --seeds 1           # one draw: faster, weaker
  python3 correctness.py --negative-control
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapes import (BKV_P, BQ_SZ, CR, DECODE_REQ_BATCH_SIZE, DH, H, K, S,  # noqa: E402
                    SOFTMAX_SCALE, NHEAD_SCALE, VMEM_LIMIT, cdiv, padded_n,
                    quantize_fp8_ue8m0)

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk  # noqa: E402
# The SHIPPING partition rule, imported rather than restated, so the geometry
# this file prints cannot drift from the geometry the kernel runs. It is a
# private name on purpose: if it is renamed the gate should break loudly here
# rather than quietly print a stale rung.
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
    _pick_partition as _inc_pick_partition  # noqa: E402

# Thresholds for the two tolerance-carrying properties. Both are set from
# MEASURED endpoints, not guessed, and both are re-derived and printed on every
# --negative-control run, so a future recalibration is also a measurement.
#
# Each sits at the geometric centre of the window between the weakest damaged
# case (must FAIL) and the worst acceptable case (must PASS):
#
#   deficit  window (0.00486, 0.04605)  9.5x   -> 0.015, 3.1x either side
#   churn    window (0.00439, 0.15869)  36.1x  -> 0.03,  6.8x / 5.3x
#
# Both windows are floored by dec-B-2x8k and ceilinged by pre-1x8k undamaged.
# Confirmed load-bearing on hardware: j-ae538ec1-d606-48f4-8c11 (Succeeded,
# exit 0) prints `tol=... is INSIDE that window -- load-bearing` for BOTH,
# with drop-block failing 5/5 and tail-swap passing 5/5.
#
# `--negative-control` deliberately corrupts the selection and asserts these
# thresholds FAIL, so the gate is a discriminator rather than an assertion.
# Two things it caught that a passing gate did not, both worth keeping in mind
# before anyone retunes these by feel:
#
# 1. TOL_DEFICIT was 0.05, and that was OUTSIDE its window. drop-block reaches
#    only 0.04605 at dec-B-2x8k, so a kernel skipping a KV block slid under the
#    threshold there and ONLY churn caught it (j-b0c4901b-5df9-4718-8c3c). The
#    gate passed while running on one of its two metrics -- exactly the state a
#    negative control exists to expose, since a damage mode that perturbed
#    deficit but not churn would have gone straight through.
# 2. The ceiling is the max over everything that must pass INCLUDING the
#    undamaged incumbent, not over the tail-swap arm alone. For deficit the
#    undamaged ceiling (0.00486) is ABOVE tail-swap's (0.00249), because
#    exchanging the 8 worst selected for the true next-best RAISES the minimum
#    score. Reading the window off tail-swap would have opened it below the
#    incumbent's own error and red-lighted a correct kernel.
#
# Retune from the separation block the control prints, never by feel.
TOL_DEFICIT = 0.015   # fraction of the top-to-cut score range
TOL_CHURN = 0.03      # fraction of the true top-k missing (~61 of 2048)

NSWAP = 8             # near-ties exchanged by the tail-swap arm
SLICE_DEN = 8         # drop-slice blanks 1/SLICE_DEN of the live row

# A damage arm is only evidence where it is big enough to be caught. Both arms
# blank a band of columns and re-select; with scores i.i.d. across positions the
# share of true winners inside the band is the band's share of the live row, so
# each arm has a churn PREDICTABLE FROM GEOMETRY ALONE (`predicted_churn`). An
# arm is REQUIRED to fail a case only where that prediction clears the tolerance
# by this factor; elsewhere it is run and reported but carries no verdict.
#
# This is what forced `drop-slice` into existence. drop-block blanks ONE KV
# block, so its churn is ~bkv_sz/L and decays with context. On the LONG row of
# each case -- the row the case exists to exercise:
#
#     dec-2x8k      2048/8192   = 25.00%  >> 3%   catches
#     dec-B-2x8k    1024/6144   = 16.67%  >> 3%   catches  (measured 15.87%)
#     dec-B-2x96k   1024/98304  =  1.04%   < 3%   BLIND
#     dec-B-2x160k  1024/163840 =  0.62%   < 3%   BLIND
#     dec-B-2x320k  1024/327680 =  0.31%   < 3%   BLIND
#     dec-B-2x520k  1024/532480 =  0.19%   < 3%   BLIND
#
# so on the long row of every long-context case -- exactly the rows the ladder
# rungs above P=1 live on -- a gate whose only must-fail arm is drop-block
# asserts nothing. Two of those cases are still REQUIRED to fail drop-block,
# because their SHORT companion row does see 20-26%; `predicted_churn` returns
# both figures and the control prints both, so an arm that passes a case while
# protecting the wrong row of it is visible rather than hidden behind the
# case's verdict.
#
# `drop-slice` blanks a fixed FRACTION of the row instead of a fixed number of
# columns, so its prediction is 1/SLICE_DEN = 12.5% at every length. Neither
# tolerance was touched to make this work; the arm was added because the
# tolerances are right and the old arm had stopped reaching them.
REQUIRE_FACTOR = 2.0

# Production geometry, shrunk only in the number of sequences/tokens so a f32
# reference is tractable. page_size, head dim, head count and k are the real
# ones -- shrinking those would test a kernel configuration nobody runs.
#
# The four `dec-B-2x*` rows exist to reach the ladder rungs above P=1 and for no
# other reason. At (B) geometry n = 1048576 and P_MAX = 16, so tn = 65536 and a
# rung covers p*tn columns; `live_max` is therefore the ONLY dial that moves the
# rung, and one case can exercise exactly one rung. The lengths are picked just
# above each coverage boundary (65536, 131072, 262144, 524288) so the case sits
# on the rung it names with room to spare, and every one carries a SECOND,
# SHORTER sequence so the rows differ in how many slices they fill -- a uniform
# batch leaves the per-row candidate crop inert and tests nothing.
#
# They are the expensive rows: the f32 reference is O(L * H * DH) per token row,
# so each is decode (qlen=1, two live rows) rather than prefill. Nothing else is
# shrunk -- page_size, head dim, head count and k are the production ones.
CASES = {
    #                 psz   pps  lens                 qlen  T
    "dec-2x8k":   dict(psz=1024, pps=9, lens=[8192, 8192], qlen=1, T=16),
    "dec-ragged": dict(psz=1024, pps=9, lens=[8193, 3000, 9216, 2049], qlen=1, T=16),
    "pre-1x8k":   dict(psz=1024, pps=9, lens=[8192], qlen=128, T=128),
    "dec-B-2x8k": dict(psz=512, pps=2048, lens=[8192, 6144], qlen=1, T=16),
    "pre-B-1x4k": dict(psz=512, pps=2048, lens=[4096], qlen=128, T=128),
    # -- long context: one case per ladder rung above P=1 ---------------------
    #                                          live_max     rung  live_slices
    "dec-B-2x96k":  dict(psz=512, pps=2048,  # 98304  > 65536   P=2   [2, 1]
                         lens=[98304, 5000], qlen=1, T=16),
    "dec-B-2x160k": dict(psz=512, pps=2048,  # 163840 >131072   P=4   [3, 2]
                         lens=[163840, 70000], qlen=1, T=16),
    "dec-B-2x320k": dict(psz=512, pps=2048,  # 327680 >262144   P=8   [5, 1]
                         lens=[327680, 4096], qlen=1, T=16),
    "dec-B-2x520k": dict(psz=512, pps=2048,  # 532480 >524288   P=16  [9, 4]
                         lens=[532480, 200000], qlen=1, T=16),
}


def case_geometry(c):
    """Selector geometry for one case: the rung it lands on, and why.

    `p_max` is the SHIPPING `_pick_partition`'s answer, imported rather than
    restated. A candidate may legally choose a lower ceiling -- the growth guard
    collapses P 2 -> 1 at (A) geometry, where raising P lowers no wave count --
    so at (A) this prints the incumbent's rung and a candidate's own resolved
    rung can be one step below it. At (B) the capacity floor `cdiv(n, 64K)`
    binds and every arm agrees on 16, which is where the rung coverage that
    matters comes from.

    `p_run` buckets UPWARD, matching the device rule: the smallest ladder entry
    whose coverage `p * tn` reaches `live_max`. Rounding down would leave live
    columns outside every slice, and the selector cannot notice -- it still
    returns k unique in-range indices, drawn from a truncated candidate set.
    """
    n = padded_n(c["psz"], c["pps"])
    p_max = _inc_pick_partition(c["T"], n, K)
    tn = n // p_max
    ladder = tuple(1 << i for i in range(p_max.bit_length()))
    live_max = max(c["lens"])
    p_run = ladder[sum(1 for p in ladder[:-1] if p * tn < live_max)]
    return dict(n=n, p_max=p_max, tn=tn, ladder=ladder, live_max=live_max,
                p_run=p_run,
                live_slices=[min(cdiv(L, tn), p_run) for L in c["lens"]])


def print_geometry(names):
    """The coverage table. Printed before any case runs, so a gate that is blind
    to four of the five rungs says so at the top of its own log."""
    print("=== case geometry: which selector rung each case executes ===",
          flush=True)
    rungs = {}
    for n in names:
        g = case_geometry(CASES[n])
        rungs.setdefault(g["p_run"], []).append(n)
        print(f"  {n:14} n={g['n']:8}  p_max={g['p_max']:2}  tn={g['tn']:6}"
              f"  live_max={g['live_max']:7}  ->  p_run={g['p_run']:2}"
              f"  live_slices={g['live_slices']}", flush=True)
    missing = [p for p in (1, 2, 4, 8, 16) if p not in rungs]
    print(f"  rungs exercised: {sorted(rungs)}"
          + (f"   NOT exercised: {missing} -- the gate is blind to these"
             if missing else "   (all five)"), flush=True)
    return rungs


def predicted_churn(c, mode):
    """Churn a damage arm should produce, from geometry alone, before it runs.

    Returns `(worst_row, longest_row)`, and the pair is the point. The band each
    arm blanks is a known number of columns; scores are i.i.d. across positions,
    so the expected share of the true top-k inside it is the band's share of the
    live row. A case fails if ANY row fails, so `worst_row` -- the shortest row,
    where a fixed band is the largest fraction -- is what decides whether an arm
    is required to fail. But `longest_row` is what says whether the arm reaches
    the row the case exists for: at dec-B-2x320k drop-block is required (its
    4096-token companion row sees 25%) while on the 327680-token row it is
    0.31%, so the arm is protecting the wrong row. Both are printed.
    """
    per = []
    for L in c["lens"]:
        n_valid = max(1, L - c["qlen"] + 1)
        want = min(K, n_valid)
        if mode == "drop-block":
            w = c["psz"] * BKV_P[0]
        elif mode == "drop-slice":
            w = min(n_valid // SLICE_DEN, n_valid - want)
        else:
            return 0.0, 0.0
        per.append(min(max(w, 0), n_valid) / n_valid)
    return max(per), per[max(range(len(per)), key=lambda i: c["lens"][i])]


def build_case(c, seed=0):
    psz, pps, lens, qlen, T = c["psz"], c["pps"], c["lens"], c["qlen"], c["T"]
    R = len(lens)
    pages_per = [cdiv(L, psz) for L in lens]
    starts = np.concatenate([[0], np.cumsum(pages_per)]).astype(np.int64)
    nb = int(starts[-1])

    kc, kq, kw = jax.random.split(jax.random.key(seed), 3)
    keys = jax.random.normal(kc, (nb, psz // 4, 4, DH), jnp.float32)
    q8, sc = quantize_fp8_ue8m0(keys, DH)
    vb = jax.lax.bitcast_convert_type(q8, jnp.uint8)
    sb = jax.lax.bitcast_convert_type(sc, jnp.uint8)  # squeezed scale -> trailing dim 1
    pad = jnp.zeros((nb, psz // 4, 4, 256 - DH - 1), jnp.uint8)
    cache_kv = jnp.concatenate([vb, sb, pad], axis=-1)
    # Keys kept UNSCALED, with the e8m0 scale alongside, so the reference can
    # apply it in the kernel's order. `sc` already carries a trailing dim of 1
    # (one 128-wide quant block per row).
    k8 = np.asarray(q8.astype(jnp.float32))
    scf = np.asarray(sc.astype(jnp.float32))

    pi = np.zeros((S, pps), dtype=np.int32)
    for i in range(R):
        pi[i, :pages_per[i]] = np.arange(starts[i], starts[i + 1])

    sl = np.zeros(S, np.int32)
    sl[:R] = lens
    ql = np.zeros(S, np.int32)
    ql[:R] = qlen
    cu = np.zeros(S + 1, np.int32)
    cu[1:] = np.cumsum(ql)

    D = R if qlen == 1 else 0
    q = jax.random.normal(kq, (T, H, DH), jnp.float32).astype(jnp.float8_e4m3fn)
    w = (jax.random.uniform(kw, (T, H), jnp.float32, -1.5, 1.5)
         * SOFTMAX_SCALE * NHEAD_SCALE)

    args = (q, w, cache_kv, jnp.asarray(sl), jnp.asarray(pi.reshape(-1)),
            jnp.asarray(cu), jnp.asarray([D, D, R], jnp.int32))
    return args, dict(k8=k8, sc=scf, pi=pi, lens=lens, qlen=qlen, psz=psz,
                      R=R, T=T, cu=cu,
                      q=np.asarray(q.astype(jnp.float32)), w=np.asarray(w))


def reference_scores(m, seq_id, tok):
    """f32 reference for one token row: scores over its sequence's positions.

    Mirrors the kernel's arithmetic order exactly (compute_scores, :322-338):
    the dot product is taken on the *unscaled* fp8 values, ReLU'd, weighted,
    summed over heads, and only then multiplied by the per-key e8m0 scale.
    Folding the scale into the key first is mathematically identical (the
    scale is a positive power of two) but leaves room to wonder, and this
    reference exists precisely to remove that room.
    """
    L, psz, qlen = m["lens"][seq_id], m["psz"], m["qlen"]
    pos = L - qlen + (tok - m["cu"][seq_id])
    j = np.arange(L)
    page = m["pi"][seq_id, j // psz]
    slot = j % psz
    kk = m["k8"][page, slot // 4, slot % 4]               # [L, DH] unscaled
    sc = m["sc"][page, slot // 4, slot % 4, 0]            # [L]
    dots = np.maximum(m["q"][tok] @ kk.T, 0.0)            # [H, L]
    s = (m["w"][tok][:, None] * dots).sum(0) * sc         # [L]
    return np.where(j <= pos, s, -np.inf)


def _corrupt(idx, m, mode):
    """Damage a correct result the way a wrong kernel would, so the thresholds
    can be shown to fire. Used only by --negative-control.

    `drop-block` is the failure this gate exists to catch: a kernel whose KV
    walk misses one block returns k indices, all unique, all in range -- every
    structural property still holds. Only deficit and churn can see it.
    `drop-slice` is the same idea one level up, for the failure the column
    PARTITION produces, and it is the arm that still bites at 256k-1M where
    drop-block's one block is 0.2% of the row. `tail-swap` is the benign arm
    that must pass. Which arms are required to fail which cases is decided by
    `predicted_churn` and REQUIRE_FACTOR, from geometry, before anything runs.

    All three arms damage the row *by true score*, not by index. An earlier version
    refilled from `pool[:n]` -- the lowest-numbered unselected positions -- and
    that is not either failure: it draws uniformly from the rejected set, so it
    made `tail-swap` (which must pass) look like a catastrophe. A skipped block
    and a boundary tie are both defined in score order, so both need `ref`.
    """
    out = idx.copy()
    for seq_id in range(m["R"]):
        for tok in range(int(m["cu"][seq_id]), int(m["cu"][seq_id + 1])):
            row = out[tok]
            sel = row[row >= 0]
            if not sel.size:
                continue
            ref = reference_scores(m, seq_id, tok)
            n_valid = int(np.isfinite(ref).sum())
            order = np.argsort(-ref, kind="stable")       # best first
            if mode == "drop-slice":
                # The failure the PARTITION produces, as opposed to the one the
                # KV walk produces. Two mechanisms reach it and both are live in
                # this kernel: a ladder rung resolved one step too low (which is
                # why `ladder_index` buckets upward), and a per-row stage-2
                # candidate list cropped one slice short. Either leaves the
                # selector blind to a contiguous TAIL of the row's columns while
                # every structural property still holds.
                #
                # It exists because drop-block stops discriminating at long
                # context -- see REQUIRE_FACTOR. Blanking a FRACTION of the row
                # rather than a fixed block keeps the damage at ~1/SLICE_DEN of
                # the winners at 8k and at 520k alike.
                w = min(n_valid // SLICE_DEN, n_valid - len(sel))
                if w <= 0:
                    # Nothing to blank without making the row structurally
                    # short, which would be a different (and easier) failure
                    # than the one being modelled. Leave the row alone; the case
                    # still fails on its other rows.
                    continue
                cand = order[order < n_valid - w]
                take = cand[: len(sel)]
            elif mode == "drop-block":
                # A kernel whose KV walk misses one block still returns the
                # best k it saw -- so blank a bkv-sized band and re-select the
                # top-k over what remains, in score order.
                bkv_sz = m["psz"] * BKV_P[0]
                lo = (len(sel) // 2 // bkv_sz) * bkv_sz
                hi = lo + bkv_sz
                cand = order[(order < lo) | (order >= hi)]
                cand = cand[np.isfinite(ref[cand])]
                take = cand[: len(sel)]
            elif mode == "tail-swap":
                # The benign failure: exchange the worst few selected for the
                # *next-best* rejected -- the near-ties straddling the cut.
                # Must NOT trip the gate.
                # `sel` is in the kernel's arbitrary emission order, so the
                # worst few have to be found by score, not by position.
                #
                # The rejected pool is taken as a set difference, not as
                # `live[len(sel):]`. The incumbent's own selection is not
                # exactly the true top-k (0.44% churn at pre-1x8k), so the
                # rank-based slice overlaps `sel` and the swap emitted
                # DUPLICATE indices -- j-470d481e failed pre-1x8k on exactly
                # that, with deficit and churn both far inside tolerance.
                live = order[np.isfinite(ref[order])]
                rej = live[~np.isin(live, sel)]
                # A row with almost nothing rejected has no boundary for ties
                # to straddle. At dec-ragged tok3, L=2049 against k=2048, the
                # single reject is the row's WORST score, and swapping it in
                # models no kernel that could exist -- it drove deficit to
                # 0.05061 in j-470d481e and red-lighted an arm that must pass.
                # A correct-but-different kernel returns the same set here.
                if len(rej) < 4 * NSWAP:
                    continue
                nxt = rej[:NSWAP]
                keep = sel[np.argsort(-ref[sel], kind="stable")][
                    : len(sel) - NSWAP]
                take = np.concatenate([keep, nxt])
            row[:] = -1
            row[: len(take)] = take
    return out


def run_case(name, verbose=True, corrupt="", stats=None, seed=0):
    c = CASES[name]
    label = f"{name}@s{seed}"
    args, m = build_case(c, seed=seed)
    idx = np.asarray(streamindex_topk(
        *args, k=K, compression_ratio=CR, num_kv_pages_per_block=BKV_P,
        num_queries_per_block=BQ_SZ, vmem_limit_bytes=VMEM_LIMIT,
        decode_req_batch_size=DECODE_REQ_BATCH_SIZE, enable_early_exit=True))
    if corrupt:
        idx = _corrupt(idx, m, corrupt)

    fails = []
    n_live = int(m["cu"][m["R"]])
    worst_def, worst_churn = 0.0, 0.0
    for seq_id in range(m["R"]):
        for tok in range(int(m["cu"][seq_id]), int(m["cu"][seq_id + 1])):
            ref = reference_scores(m, seq_id, tok)
            n_valid = int(np.isfinite(ref).sum())
            want_n = min(K, n_valid)
            got = idx[tok]
            sel = got[got >= 0]

            # -- structural: these are exact, no tolerance is defensible ------
            if len(sel) != want_n:
                fails.append(f"{label} tok{tok}: {len(sel)} selected, want {want_n}")
                continue
            if len(np.unique(sel)) != len(sel):
                fails.append(f"{label} tok{tok}: duplicate indices")
                continue
            if sel.size and (sel.max() >= m["lens"][seq_id] or sel.min() < 0):
                fails.append(f"{label} tok{tok}: index out of range")
                continue

            # -- selection quality --------------------------------------------
            order = np.sort(ref[np.isfinite(ref)])[::-1]
            kth, top = order[want_n - 1], order[0]
            scale = max(top - kth, 1e-6)
            # How far below the true cut did the kernel reach? A kernel that
            # scores garbage, or skips a KV block whose keys were winners,
            # reaches far below it. Boundary near-ties reach barely below it.
            deficit = (kth - ref[sel].min()) / scale
            # How much of the true top-k did it miss? Same discrimination,
            # measured as a count rather than a magnitude -- a kernel that
            # drops a whole block misses ~ that block's share of the winners.
            true_set = set(np.argsort(-ref)[:want_n].tolist())
            churn = len(true_set - set(sel.tolist())) / want_n
            worst_def, worst_churn = max(worst_def, deficit), max(worst_churn, churn)
            if deficit > TOL_DEFICIT:
                fails.append(f"{label} tok{tok}: reached {deficit:.4f} of the "
                             f"top-to-cut range below the true k-th "
                             f"(kth {kth:.6g}, worst selected {ref[sel].min():.6g})")
            if churn > TOL_CHURN:
                fails.append(f"{label} tok{tok}: missed {churn:.2%} of the true "
                             f"top-{want_n}")

    # Padded token rows must be entirely -1.
    for tok in range(n_live, m["T"]):
        if not (idx[tok] == -1).all():
            fails.append(f"{label} tok{tok}: padded row not -1-filled")

    if verbose:
        st = "PASS" if not fails else "FAIL"
        print(f"[{st}] {label:18} R={m['R']} qlen={m['qlen']} "
              f"live_rows={n_live}  worst_deficit={worst_def:.5f} "
              f"(tol {TOL_DEFICIT})  worst_churn={worst_churn:.3%} "
              f"(tol {TOL_CHURN:.1%})  lens={m['lens']}", flush=True)
        for f in fails[:6]:
            print(f"        {f}", flush=True)
    if stats is not None:
        stats.append((label, worst_def, worst_churn))
    return fails


def _window(label, tol, i, floor_rows, ceil_rows, note=""):
    """Print the admissible-tolerance window for one metric and one damage arm.

    Separation, computed rather than eyeballed. A threshold is only a
    discriminator if it fits strictly between the arms; report the window so the
    next recalibration is arithmetic on measured endpoints. j-470d481e is why
    this is printed: the deficit arms OVERLAPPED there (0.04605 vs 0.05061) and
    reading it off the table was the only way to find out.

    The ceiling is the max over EVERYTHING that must pass, which includes the
    undamaged incumbent -- not just tail-swap. j-b0c4901b showed why: tail-swap's
    deficit ceiling (0.00249) came out BELOW the undamaged one (0.00486), because
    exchanging the 8 worst selected for the true next-best raises the minimum
    score. Reading the window off tail-swap alone would have put the floor of the
    admissible range under the incumbent's own error and red-lighted a correct
    kernel.
    """
    if not floor_rows:
        print(f"  {label:8} no case where this arm is required to fail{note}",
              flush=True)
        return
    lo = min(s[i] for s in floor_rows)      # weakest damaged case
    hi = max(s[i] for s in ceil_rows)       # worst case that must still pass
    lo_n = min(floor_rows, key=lambda s: s[i])[0]
    hi_n = max(ceil_rows, key=lambda s: s[i])[0]
    if hi < lo:
        inside = hi < tol < lo
        print(f"  {label:8} SEPARATES: any tol in ({hi:.5f}, {lo:.5f}) works, "
              f"ratio {lo / max(hi, 1e-9):.1f}x  [floor {lo_n}, ceiling {hi_n}]"
              f"{note}", flush=True)
        print(f"           tol={tol} is {'INSIDE' if inside else 'OUTSIDE'} "
              f"that window -- "
              f"{'load-bearing' if inside else 'NOT a discriminator at this tol'}",
              flush=True)
    else:
        print(f"  {label:8} OVERLAPS: damaged floor {lo:.5f} ({lo_n}) is below "
              f"acceptable ceiling {hi:.5f} ({hi_n}) -- no threshold on this "
              f"metric separates{note}", flush=True)


def negative_control(names, clean=(), seeds=(0,)):
    """Show the gate is a discriminator, not an assertion.

    Runs the candidate, then damages its output three ways and checks that the
    thresholds separate them. Exits non-zero unless all of these hold:

      drop-block  -> must FAIL every case where its predicted churn clears
                     REQUIRE_FACTOR * TOL_CHURN. That qualifier is new and it is
                     not a loosening: drop-block's damage is one KV block, so at
                     96k-520k it is 0.2-1.0% churn against a 3% tolerance and
                     asserting it must fail there would be asserting something
                     false about the arm, not something true about the kernel.
                     It still must fail all five short cases, exactly as before.
      drop-slice  -> must FAIL every case. This is the arm that replaces
                     drop-block's duty at long context; it blanks 1/SLICE_DEN of
                     the row, so its damage does not decay with L.
      tail-swap   -> must PASS every case (the gate tolerates near-ties).

    A gate that cannot do all three is not evidence about a candidate, and this
    is the run that establishes it. See CLAUDE.md: "would this check have failed
    if the change were wrong?"
    """
    bad, swap, floors = [], [], {}
    print("=== negative control ===", flush=True)
    for mode, why in (
            ("drop-block", "a kernel whose KV walk misses one block"),
            ("drop-slice", f"a kernel blind to the last 1/{SLICE_DEN} of each "
                           f"row -- a rung resolved one step low, or a "
                           f"candidate list cropped one slice short")):
        print(f"-- {mode}: {why} --", flush=True)
        rows, skipped = [], []
        for n in names:
            pred, pred_long = predicted_churn(CASES[n], mode)
            required = pred > REQUIRE_FACTOR * TOL_CHURN
            for s in seeds:
                st = []
                f = run_case(n, corrupt=mode, stats=st, seed=s)
                print(f"        predicted churn: worst row {pred:7.3%}, "
                      f"LONGEST row {pred_long:7.3%} -> "
                      f"{'MUST FAIL' if required else 'not discriminating here'}"
                      f", {'caught' if f else 'NOT CAUGHT'}", flush=True)
                (rows if required else skipped).append(st[0])
                if required and not f:
                    bad.append(f"{n}@s{s}: {mode} was NOT caught, though its "
                               f"predicted churn {pred:.3%} is over "
                               f"{REQUIRE_FACTOR}x the {TOL_CHURN:.1%} "
                               f"tolerance -- the gate is blind to a failure it "
                               f"should see")
        floors[mode] = (rows, skipped)
    print(f"-- tail-swap: {NSWAP} boundary near-ties exchanged. MUST PASS --",
          flush=True)
    for n in names:
        for s in seeds:
            f = run_case(n, corrupt="tail-swap", stats=swap, seed=s)
            if f:
                bad.append(f"{n}@s{s}: tail-swap was rejected -- the gate is too "
                           f"tight and will red-light correct candidates")

    print("\n-- separation (must-fail floor vs must-pass ceiling) --", flush=True)
    ok = list(swap) + [(f"{n}/undamaged", d, c) for n, d, c in clean]
    for mode in ("drop-block", "drop-slice"):
        rows, skipped = floors[mode]
        print(f"  [{mode}] required on {len(rows)} of {len(rows) + len(skipped)} "
              f"(case, seed) pairs", flush=True)
        for label, tol, i in (("deficit", TOL_DEFICIT, 1),
                              ("churn", TOL_CHURN, 2)):
            _window(label, tol, i, rows, ok)
        if skipped:
            worst = max(skipped, key=lambda s: s[2])
            print(f"           where NOT required, the arm reached at most "
                  f"churn {worst[2]:.3%} ({worst[0]}) -- below the {TOL_CHURN:.1%} "
                  f"tolerance, which is why it carries no verdict there",
                  flush=True)

    print(f"\nnegative control: {'PASS' if not bad else 'FAIL'}", flush=True)
    for b in bad:
        print(f"  {b}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of input draws (seeds 0..N-1). One draw is one "
                         "arrangement of near-ties at the cut; the properties "
                         "asserted here are distributional, so a single draw "
                         "understates the gate")
    ap.add_argument("--negative-control", action="store_true",
                    help="prove the thresholds separate a damaged selection "
                         "from boundary near-ties, then exit")
    a = ap.parse_args()
    names = [a.case] if a.case else list(CASES)
    seeds = tuple(range(max(1, a.seeds)))
    print_geometry(names)
    print(f"=== {len(names)} cases x {len(seeds)} seeds ===", flush=True)
    allf, clean = [], []
    for s in seeds:
        for n in names:
            allf += run_case(n, stats=clean, seed=s)
    ncase = len(names) * len(seeds)
    print(f"\n{ncase - len(set(f.split()[0] for f in allf))}/{ncase} "
          f"(case, seed) pairs passed, {len(allf)} assertion failures")
    if a.negative_control:
        allf += negative_control(names, clean, seeds)
    sys.exit(1 if allf else 0)


if __name__ == "__main__":
    main()
