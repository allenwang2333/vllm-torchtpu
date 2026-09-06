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

  python3 correctness.py            # all cases
  python3 correctness.py --case dec-2x8k
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapes import (BKV_P, BQ_SZ, CR, DECODE_REQ_BATCH_SIZE, DH, H, K, S,  # noqa: E402
                    SOFTMAX_SCALE, NHEAD_SCALE, VMEM_LIMIT, cdiv,
                    quantize_fp8_ue8m0)

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    streamindex_topk  # noqa: E402

# Thresholds for the two tolerance-carrying properties. CALIBRATED, not guessed:
# both numbers below sit between two measured quantities, and both are printed
# on every run so a future recalibration is also a measurement.
#
#   incumbent, all 5 cases  (j-31ca6c31-a430-4463-8dbf):
#       worst deficit 0.00486   worst churn 0.439%
#   a kernel that drops one KV block at (A):  bkv_sz = 2048, n = 9216, so one
#       block is ~22% of the live range; at a 22-25% selection rate that costs
#       ~20% of the true top-k.  [modelled -- see the negative control below]
#
# So the discriminating gap is 0.44% against ~20%, a factor of 45. Both
# thresholds are placed near the geometric centre of their gap rather than
# hard against the incumbent: TOL_CHURN=0.03 is 6.8x above the incumbent's
# worst and 6.7x below a dropped block. The earlier 0.005 left pre-1x8k with
# 12% margin, which would have red-lighted candidates for reordering ties.
#
# `--negative-control` deliberately corrupts the selection and asserts these
# thresholds FAIL, so the gate is a discriminator rather than an assertion.
TOL_DEFICIT = 0.05    # fraction of the top-to-cut score range
TOL_CHURN = 0.03      # fraction of the true top-k missing (~61 of 2048)

# Production geometry, shrunk only in the number of sequences/tokens so a f32
# reference is tractable. page_size, head dim, head count and k are the real
# ones -- shrinking those would test a kernel configuration nobody runs.
CASES = {
    #                 psz   pps  lens                 qlen  T
    "dec-2x8k":   dict(psz=1024, pps=9, lens=[8192, 8192], qlen=1, T=16),
    "dec-ragged": dict(psz=1024, pps=9, lens=[8193, 3000, 9216, 2049], qlen=1, T=16),
    "pre-1x8k":   dict(psz=1024, pps=9, lens=[8192], qlen=128, T=128),
    "dec-B-2x8k": dict(psz=512, pps=2048, lens=[8192, 6144], qlen=1, T=16),
    "pre-B-1x4k": dict(psz=512, pps=2048, lens=[4096], qlen=128, T=128),
}


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

    Both arms damage the row *by true score*, not by index. An earlier version
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
            order = np.argsort(-ref, kind="stable")       # best first
            if mode == "drop-block":
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
                live = order[np.isfinite(ref[order])]
                nxt = live[len(sel): len(sel) + 8]
                keep = sel[np.argsort(-ref[sel], kind="stable")][
                    : max(0, len(sel) - len(nxt))]
                take = np.concatenate([keep, nxt])
            row[:] = -1
            row[: len(take)] = take
    return out


def run_case(name, verbose=True, corrupt=""):
    c = CASES[name]
    args, m = build_case(c)
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
                fails.append(f"{name} tok{tok}: {len(sel)} selected, want {want_n}")
                continue
            if len(np.unique(sel)) != len(sel):
                fails.append(f"{name} tok{tok}: duplicate indices")
                continue
            if sel.size and (sel.max() >= m["lens"][seq_id] or sel.min() < 0):
                fails.append(f"{name} tok{tok}: index out of range")
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
                fails.append(f"{name} tok{tok}: reached {deficit:.4f} of the "
                             f"top-to-cut range below the true k-th "
                             f"(kth {kth:.6g}, worst selected {ref[sel].min():.6g})")
            if churn > TOL_CHURN:
                fails.append(f"{name} tok{tok}: missed {churn:.2%} of the true "
                             f"top-{want_n}")

    # Padded token rows must be entirely -1.
    for tok in range(n_live, m["T"]):
        if not (idx[tok] == -1).all():
            fails.append(f"{name} tok{tok}: padded row not -1-filled")

    if verbose:
        st = "PASS" if not fails else "FAIL"
        print(f"[{st}] {name:14} R={m['R']} qlen={m['qlen']} lens={m['lens']} "
              f"live_rows={n_live}  worst_deficit={worst_def:.5f} "
              f"(tol {TOL_DEFICIT})  worst_churn={worst_churn:.3%} "
              f"(tol {TOL_CHURN:.1%})", flush=True)
        for f in fails[:6]:
            print(f"        {f}", flush=True)
    return fails


def negative_control(names):
    """Show the gate is a discriminator, not an assertion.

    Runs the incumbent, then damages its output two ways and checks that the
    thresholds separate them. Exits non-zero unless BOTH hold:
      drop-block  -> must FAIL every case (the gate catches a wrong kernel)
      tail-swap   -> must PASS every case (the gate tolerates near-ties)
    A gate that cannot do both is not evidence about a candidate, and this is
    the run that establishes it. See CLAUDE.md: "would this check have failed
    if the change were wrong?"
    """
    bad = []
    print("=== negative control ===", flush=True)
    print("-- drop-block: a kernel whose KV walk misses one block. MUST FAIL --",
          flush=True)
    for n in names:
        if not run_case(n, corrupt="drop-block"):
            bad.append(f"{n}: drop-block was NOT caught -- the gate is blind "
                       f"to the failure it exists to catch")
    print("-- tail-swap: 8 boundary near-ties exchanged. MUST PASS --",
          flush=True)
    for n in names:
        f = run_case(n, corrupt="tail-swap")
        if f:
            bad.append(f"{n}: tail-swap was rejected -- the gate is too tight "
                       f"and will red-light correct candidates")
    print(f"\nnegative control: {'PASS' if not bad else 'FAIL'}", flush=True)
    for b in bad:
        print(f"  {b}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    ap.add_argument("--negative-control", action="store_true",
                    help="prove the thresholds separate a dropped KV block "
                         "from boundary near-ties, then exit")
    a = ap.parse_args()
    names = [a.case] if a.case else list(CASES)
    allf = []
    for n in names:
        allf += run_case(n)
    print(f"\n{len(names) - len(set(f.split()[0] for f in allf))}/{len(names)} "
          f"cases passed, {len(allf)} assertion failures")
    if a.negative_control:
        allf += negative_control(names)
    sys.exit(1 if allf else 0)


if __name__ == "__main__":
    main()
