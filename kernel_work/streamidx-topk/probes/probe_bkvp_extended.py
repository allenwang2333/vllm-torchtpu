#!/usr/bin/env python3
"""Candidate 0, extended: `bkv_p` swept to 9 and past it, plus an f64 oracle.

Why
---

`probes/baseline-device-time.md` swept `bkv_p in {2,3,4}` and found `p = 3`
beating the incumbent `p = 2` by 9.5 % on isolated stage A at decode.  It also
fitted stage A decode to

    t_us  =  6.24  +  7.41 * num_bkv  +  6.65 * pages_fetched_per_seq

(r^2 0.9967, max residual 1.3 %, **ctx absent from the model entirely**).  That
fit is the whole reason this file exists, because of what it extrapolates to.
At ctx 8704, `pages_per_seq = 9` and `num_bkv = ceil(8704 / (1024 p))`:

    p    num_bkv  pages   modelled t_us      exact?
    1      9        9       132.8            yes (9 = 9x1)
    2      5       10       109.8            no  (overfetch 1 page)
    3      3        9        88.4            yes (9 = 3x3)
    4      3       12       108.3            no  (overfetch 3)
    5      2       10        87.6            no  (overfetch 1)
    6      2       12       100.9            no
    7      2       14       114.2            no
    8      2       16       127.5            no
    9      1        9        73.5            yes (9 = 1x9)
   10      1       10        80.2            no
   12      1       12        93.5            no
   16      1       16       120.1            no

Every one of those is **modelled**, not measured.  The point of the sweep is
that the model makes two sharp, falsifiable predictions:

  1. `p = 9` is the minimum, at 73.5 us `modelled` -- 33 % under `p = 2`, and
     4x the size of the 9.5 % that was actually measured at `p = 3`.
  2. `p = 5` (87.6) should tie `p = 3` (88.4) despite `p = 5` being *inexact*.
     If the win were purely an "exactness" effect, `p = 5` would lose.

Prediction 2 is the one that matters for the shape of any fix.  The exactness
story says the win belongs to the **ctx bucket** (pick `p` so that
`p | cdiv(ctx, page_size)`), which makes the fix a function.  The two-term cost
model says the win belongs to **block count** -- fewer, larger blocks are
cheaper almost regardless of divisibility -- which makes the fix a constant,
`p = pages_per_seq`, chosen once from `max_model_len`.  The two agree at
`p = 9` and disagree at `p = 5`, `p = 10` and across the ctx sweep in section D.

Sections
--------

A. Stage A at ctx 8704, **one common `num_sublanes_total` of 256** for every
   `p`, so score-matrix width cannot be the confound.  `p` from 1 to 32.
A2. Past the wall: `p in {40, 48, 64}` at `nst = 8p`, purely to find where the
   compile stops and to record the message it stops with.  The bkv double
   buffer is `2 * seq_batch_size(4) * p*256 * 4 * 256` bytes = **p * 2 MiB**,
   so `p = 32` is 64 MiB against a 64 MiB TensorCore VMEM and a
   `DEFAULT_VMEM_LIMIT_BYTES` of 100 MiB.  Where it actually stops, and with
   which of those two limits, is the answer to "how far past 9".
B. Correctness: an **f64 numpy top-k oracle** at `prod_decode_b64_ctx8704`,
   covering every `p` this file times in the full op -- not just 2 and 3.
   `tests/kernels/test_dsv4_streamindex_topk.py` parametrises `bkv_p` only
   where `page_size * bkv_p == 128`, so it never runs production `p` at page
   size 1024, and the previous run's `index_sets_match_bkvp2` was
   self-consistency against the incumbent, which would pass if both were wrong.
   Two things are reported that a recall figure alone would hide:
     * every disagreement's **score gap** `s_oracle[missed] - s_oracle[taken]`,
       against **sigma** = `max |s_stageA_f32 - s_oracle_f64|`, the numerical
       noise floor.  Gaps at or under a few sigma are benign near-ties; a gap
       far above sigma is a real defect.
     * whether the **incumbent `p = 2` misses the same positions**.  Option C's
       run found gate 2's 0.999 recall criterion failing on the incumbent
       (`min_recall 0.996094`, 10 of 2560 indices).  A disagreement the
       incumbent also has is pre-existing and is not candidate 0's.
C. The full op at ctx in {8192, 8704, 9216}.  `num_sublanes_total` for the
   tuple `(p,2,2)` is `max(align_to(9,p), 10) * 8`, which is **80 for
   p in {1,2,3,5,9,10}** and larger otherwise -- so those six `p` are
   like-for-like in the full op with no width confound, and 4/8/16 are run too
   but flagged.
D. The exactness test proper: stage A at ctx in {4096, 6144, 8192} (i.e.
   `pages_per_seq` demand of 4, 6, 8) x `p in {2,3,4,6,9}`, at common width.
   If exactness rules, the winner moves with the ctx bucket.  If block count
   rules, the largest `p` wins at every ctx.
E. Width controls: `p in {2,3,9}` at nst 80 and nst 128, to re-establish in
   this job that nst is not a confound (the previous run found 96 vs 80 agreed
   to 0.4 %).
F. Where the VMEM wall is, and **whether raising `vmem_limit_bytes` moves it**.
   "`p = 9` does not fit" and "`p = 9` does not fit at the default scoped
   limit" are different results with different remedies.  Both
   `make_stage_a_fn` and `streamindex_topk` take `vmem_limit_bytes` as a
   call-site static, so the limit is swept without touching `src/`; the
   incumbent's own `DEFAULT_VMEM_LIMIT_BYTES` is 100 MiB
   (`streamindex_topk.py:28`), which is already above the 32 MiB a
   *default-scoped* Mosaic kernel would get.  This section **compiles without
   running**, which is both cheap and the only safe way to probe large `p`:
   `lower().compile()` is enough for Mosaic to accept or reject the
   allocation, and not running means the page-walk over-read below cannot
   fault.  A second, timed sub-part asks whether a raised limit changes the
   *speed* of a `p` that already fits -- Mosaic uses the scoped limit when it
   decides how much to pipeline, so "fits" and "fits well" are not the same
   question.

The pipelining question
-----------------------

`num_bkv` falls to **1** at `p >= 9`, and a single kv block has nothing to
double-buffer against: the bkv double buffer is 2 * 4 * (p*256) * 4 * 256 =
**p * 2 MiB** and at `num_bkv == 1` half of it is dead weight.  So `p = 9`
could fit and still lose, and the cost model above -- fitted only on
`num_bkv in {2,3,5}` -- has no term that would know.  Section D is the
discriminator, because it contains **page-count-matched pairs that differ only
in block count**:

    ctx 4096:  p=2 -> 2 blocks / 4 pages   vs   p=4 -> 1 block / 4 pages
    ctx 6144:  p=3 -> 2 blocks / 6 pages   vs   p=6 -> 1 block / 6 pages

Same bytes moved, same buffer width, `num_bkv` 2 against 1.  If the one-block
arm is *slower* than the two-block arm at matched pages, pipelining collapse is
real and it caps how far `p` can usefully go -- independently of VMEM.

Method is section 4's, unchanged: chain slope with a real data dependency,
r^2 >= 0.999 required, `hlo_stats` confirming independently from
`probe_bkvx_trace.py`.  Nothing here is an end-to-end model benchmark, and no
`src/` or `instruments/` file is modified.

Run:  python probe_bkvp_extended.py [--quick] [--skip A,A2,B,C,D,E,F,G]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
INSTRUMENTS = os.environ.get("BDT_INSTRUMENTS", "/root/instruments")
if INSTRUMENTS not in sys.path:
    sys.path.insert(0, INSTRUMENTS)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--quick", action="store_true")
AP.add_argument("--skip", default="")
AP.add_argument("--name", default="bkvp-extended")
# `p > pages_per_seq` walks the block table past the end of the sequence's own
# row: `_fetch_bkv` indexes `page_indices[batch_idx * pages_per_seq + kv_p]`
# for `kv_p` up to `p`, and `page_indices` is exactly `max_num_seqs *
# pages_per_seq` long, so the last sequence reads past the array.  The kernel
# runs with `disable_bounds_checks=True`, the positions are masked to -inf so
# the OUTPUT stays correct, but the DMA source page number is garbage and the
# fetch itself could fault.  `bkvx_shard.sh` therefore runs the `p <= 9` region
# and the `p > 9` region as two separate processes, so a fault in the second
# cannot cost the first -- these two flags are how it splits them.
AP.add_argument("--p-min", type=int, default=0)
AP.add_argument("--p-max", type=int, default=10**6)
# Section B's oracle coverage is derived from the p values the *full op* is
# timed at, which is the right default but leaves a gap once section G shows
# the recommendation is `p = P` for every ctx bucket: p = 6 and p = 7 then
# become deployable values that the full-op sweep never ran and so the oracle
# never checked.  This flag adds p values to B's coverage without touching the
# timing plan, so the correctness answer can span the whole 1..9 range.
AP.add_argument("--oracle-p", default="",
                help="extra bkv_p values for section B, comma separated")
ARGS = AP.parse_args()


def _pfilter(seq):
    return tuple(x for x in seq if ARGS.p_min <= x <= ARGS.p_max)

TRIALS = 3 if ARGS.quick else 9
WARMUP = 2 if ARGS.quick else 3
TARGET_S = 0.02 if ARGS.quick else 0.05
CHAIN_R = (1, 2) if ARGS.quick else (1, 2, 3)
R2_REJECT = 0.999
SKIP = {s for s in ARGS.skip.split(",") if s}

# Section A: the primary sweep, at one common width.
NST_COMMON = 256
P_SWEEP = _pfilter((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 32))
# Section A2: past the wall. nst = 8p, since num_bkv == 1 for all of these.
P_BEYOND = _pfilter((40, 48, 64))
# Section C: the full op.
FULL_CTX = (8192, 8704, 9216)
# nst == 80, like-for-like with the incumbent:
FULL_P_LFL = _pfilter((1, 2, 3, 5, 9, 10))
# nst > 80, width-confounded, run anyway and flagged:
FULL_P_WIDE = _pfilter((4, 8, 16))
# Section D: the exactness test.
EXACT_CTX = (4096, 6144, 8192)
EXACT_P = _pfilter((2, 3, 4, 6, 9))
# Section E.
CTRL = tuple((nst, _pfilter(ps)) for nst, ps in
             ((80, (2, 3, 9)), (128, (2, 3, 9, 16))))
# Section F: where the VMEM wall actually is, and whether it moves.
# `_stage_a.make_stage_a_fn` and `streamindex_topk` both take
# `vmem_limit_bytes` as a call-site static, so the scoped limit can be varied
# without touching `src/`. 100 MiB is `DEFAULT_VMEM_LIMIT_BYTES`
# (`streamindex_topk.py:28`) and is what production runs at today.
F_P = _pfilter((6, 8, 9, 10, 12, 16, 24, 32, 48))
F_LIMITS_MIB = (16, 32, 64, 100, 128, 256)
F_TIMED_P = _pfilter((9, 12, 16))
F_TIMED_LIMITS_MIB = (100, 256)
# Section G: is the optimum p exactly `P = cdiv(ctx, page_size)`?
#
# Job j-6a35b078 measured, at ctx 8704 and common width, that `p = 9 = P` is
# the global minimum (82.38 us against the incumbent's 109.49) and that at
# **matched page count** fewer blocks is always cheaper. Both point at
# `argmin_p t = P`. Two things that claim needs and did not get:
#   * the argmin located directly, by sweeping p around P at many ctx -- job 1
#     only had ctx 4096/6144/8192 at p in {2,3,4,6,9}, which never ran
#     `p = 8` at ctx 8192, i.e. never ran `p = P` at the one bucket where
#     P = 8. The predicted winner there was untested.
#   * the **full op** at short ctx. `p = 9` is a compile-time static, so if it
#     were adopted as a constant it would also run on short sequences, where
#     job 1's stage A says it costs 82.3 us against the incumbent's 48.9. Only
#     the full op says what that is worth end to end, and only `p in {2,9}`
#     can be compared there without a score-matrix width confound (both 80).
G_CTX = (2048, 3072, 4096, 5120, 6144, 7168, 8192)
G_FULL_CTX = (2048, 4096, 6144, 8192)
G_FULL_P = _pfilter((2, 9))

CTX_MAIN = 8704

RESULTS: dict = {
    "purpose": ("bkv_p swept to 9 and past it, at common score-matrix width, "
                "plus an f64 numpy top-k oracle over every p timed in the "
                "full op."),
    "methodology": {
        "trials": TRIALS, "warmup": WARMUP, "target_s": TARGET_S,
        "chain_r": list(CHAIN_R), "quick": ARGS.quick,
        "r2_reject_below": R2_REJECT,
        "device_statistic": "chain slope: d t / d R, intercept discarded",
        "nst_common_section_a": NST_COMMON,
        "p_min": ARGS.p_min, "p_max": ARGS.p_max,
        "resolved_arms": {
            "A_p_sweep": list(P_SWEEP), "A2_p_beyond": list(P_BEYOND),
            "C_full_p_like_for_like": list(FULL_P_LFL),
            "C_full_p_width_confounded": list(FULL_P_WIDE),
            "C_ctx": list(FULL_CTX),
            "D_exact_ctx": list(EXACT_CTX), "D_exact_p": list(EXACT_P),
            "E_controls": [[n, list(ps)] for n, ps in CTRL],
            "F_p": list(F_P), "F_limits_mib": list(F_LIMITS_MIB),
            "F_timed_p": list(F_TIMED_P),
            "F_timed_limits_mib": list(F_TIMED_LIMITS_MIB),
        },
        "note": ("wall clock is NOT reported as device time anywhere; the "
                 "dispatch floor is a clamp, not an addend, and is never "
                 "subtracted."),
    },
    "modelled_prediction": {
        "model": "t_us = 6.24 + 7.41*num_bkv + 6.65*pages_fetched_per_seq",
        "source": ("probes/baseline-device-time.md sec.3, fitted on "
                   "bkv_p in {2,3,4} x ctx in {8192,8704,9216}, r2 0.9967"),
        "status": "MODELLED, not measured. Every t_us below is a prediction.",
    },
}


def flush(tag: str) -> None:
    emit_json(ARGS.name, RESULTS)
    p("emitted", tag)


def cdiv(a, b):
    return -(-a // b)


def align_to(x, a):
    return cdiv(x, a) * a


def model_us(num_bkv, pages):
    return 6.24 + 7.41 * num_bkv + 6.65 * pages


def zero_dep(idx_scalar):
    return jnp.minimum(idx_scalar, jnp.int32(0))


def fit(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    a = np.vstack([np.ones_like(xs), xs]).T
    (intercept, slope), *_ = np.linalg.lstsq(a, ys, rcond=None)
    pred = intercept + slope * xs
    ss_res = float(((ys - pred)**2).sum())
    ss_tot = float(((ys - ys.mean())**2).sum())
    return {
        "intercept_us": float(intercept),
        "slope_us_per_call": float(slope),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "points_us": {int(x): float(y) for x, y in zip(xs, ys)},
    }


def short(e) -> str:
    s = repr(e)
    return s if len(s) <= 900 else s[:900] + " ...[truncated]"


def chain_slope(build, args, label, rs=None, marker=None):
    rs = rs or CHAIN_R
    marker_rs = {min(rs), max(rs)} if marker else set()
    pts, counts = {}, {}
    for r in rs:
        fn = build(r)
        t = time_arm(fn, args, WARMUP, TRIALS, TARGET_S)
        pts[r] = t["best_s"] * 1e6
        if r in marker_rs:
            try:
                counts[r] = fn.lower(*args).compile().as_text().count(marker)
            except Exception as e:
                counts[r] = f"ERR {short(e)}"
        del fn
    out = fit(list(pts), [pts[r] for r in pts])
    out["marker"] = marker
    out["marker_counts"] = {str(k): v for k, v in counts.items()}
    ok = None
    if marker and counts and all(isinstance(v, int) for v in counts.values()):
        vals = [counts[r] for r in sorted(counts)]
        ok = vals[0] > 0 and all(b > a for a, b in zip(vals, vals[1:]))
    out["chain_intact"] = ok
    out["r2_ok"] = bool(out["r2"] >= R2_REJECT)
    p(label, f"slope={out['slope_us_per_call']:.2f}us/call "
             f"r2={out['r2']:.6f} intercept={out['intercept_us']:.2f}us "
             f"pts={ {k: round(v, 2) for k, v in pts.items()} } "
             f"intact={ok}")
    return out


# ---------------------------------------------------------------------------
# f64 oracle
# ---------------------------------------------------------------------------

def build_f64_oracle(inputs, geo, B, row_lengths, k):
    """Exact top-k index sets and scores, in numpy float64.

    The score is the kernel's own formula, transcribed from `compute_scores`
    (`streamindex_topk.py:305-355`):

        s[t, m] = sum_h max(0, dot(q[t,h], K[m])) * w[t,h] * scale[m]

    with `scale[m]` the e8m0 byte at lane `head_dim`.  `_make_cache` writes 127
    there, and e8m0 127 is exactly 2**0, so the dequantized key **is** the fp8
    value byte and the scale is exactly 1.0 -- asserted below rather than
    assumed.  Positions at or past the row's `_effective_row_lengths` entry are
    masked out, which is the same `valid_mask AND causal_mask` the kernel
    applies (both reduce to `position < row_length` at decode, `q_len == 1`).

    f64 throughout: the kernel accumulates the head dot products in f32, so an
    f32 reference could not distinguish the kernel's error from its own.
    """
    import ml_dtypes

    cache = np.asarray(inputs["cache_kv"])            # u8 [pages,256,4,256]
    q = np.asarray(inputs["q"]).astype(np.float64)    # [T,32,128]
    w = np.asarray(inputs["indexer_weights"]).astype(np.float64)  # [T,32]
    page_idx = np.asarray(inputs["page_indices"]).reshape(
        -1, geo.pages_per_seq)
    cu = np.asarray(inputs["cu_q_lens"])
    rl = np.asarray(row_lengths)
    num_tokens = q.shape[0]

    # scale bytes must all be 127 (== 2**0) or the transcription above is wrong
    scale_bytes = np.unique(cache[:, :, :, B.D_IDX])
    assert scale_bytes.tolist() == [127], f"unexpected e8m0 bytes {scale_bytes}"

    # token -> sequence, the same mapping `_effective_row_lengths` computes
    seq_of_token = np.searchsorted(cu[1:], np.arange(num_tokens), side="right")

    max_rl = int(rl.max())
    scores = np.full((num_tokens, max_rl), -np.inf, np.float64)
    top_idx = np.zeros((num_tokens, k), np.int64)
    kth_gap = np.zeros(num_tokens, np.float64)

    cur_seq, kmat = -1, None
    for t in range(num_tokens):
        s_id = int(seq_of_token[t])
        n = int(rl[t])
        if n <= 0:
            continue
        if s_id != cur_seq:
            blocks = []
            for pg in page_idx[s_id]:
                b = cache[int(pg)].reshape(B.PAGE_SIZE, B.LKV_DIM)[:, :B.D_IDX]
                blocks.append(np.ascontiguousarray(b).view(
                    ml_dtypes.float8_e4m3fn).astype(np.float64))
            kmat = np.concatenate(blocks, axis=0)      # [pages_per_seq*1024,D]
            cur_seq = s_id
        kk = kmat[:n]                                   # [n, D]
        sh = np.maximum(q[t] @ kk.T, 0.0)               # [32, n]
        row = (sh * w[t][:, None]).sum(axis=0)          # [n]
        scores[t, :n] = row
        order = np.argsort(-row, kind="stable")
        top_idx[t] = order[:k]
        # margin between the last kept and the first dropped: how near a tie
        # the k-th place actually is, independent of any kernel.
        kth_gap[t] = (row[order[k - 1]] - row[order[k]]
                      if n > k else np.inf)
    return scores, top_idx, kth_gap


def compare_to_oracle(out_np, scores, top_idx, k, sigma):
    """Per-row recall plus the score gap at every disagreement."""
    n_rows, width = out_np.shape[0], scores.shape[1]
    recalls, missed_rows, disagreements = [], 0, []
    all_missed, oor = {}, 0

    def s_at(t, i):
        # The score buffer is `num_positions` wide but only `row_length`
        # positions are real, and `num_positions` grows with bkv_p. An emitted
        # index at or past the row length is masked (-inf) territory: the
        # kernel should never select one, so it is counted rather than
        # silently indexed out of the oracle.
        return scores[t, i] if i < width else -np.inf

    for t in range(n_rows):
        got = out_np[t]
        got = got[got >= 0]
        oor += int((got >= width).sum())
        gs = set(int(v) for v in got)
        os_ = set(int(v) for v in top_idx[t])
        inter = len(gs & os_)
        recalls.append(inter / max(1, len(os_)))
        miss = sorted(os_ - gs)
        extra = sorted(gs - os_)
        if miss:
            missed_rows += 1
            all_missed[t] = miss
            # pair each missed index with the extra that displaced it, worst
            # first, and report the score gap the oracle sees between them.
            ms = sorted(miss, key=lambda i: -s_at(t, i))
            xs = sorted(extra, key=lambda i: -s_at(t, i))
            for mi, xi in zip(ms, xs):
                if len(disagreements) < 200:
                    gap = float(s_at(t, mi) - s_at(t, xi))
                    disagreements.append({
                        "row": t, "missed_idx": int(mi), "taken_idx": int(xi),
                        "s_missed": float(s_at(t, mi)),
                        "s_taken": float(s_at(t, xi)),
                        "gap": gap,
                        "gap_over_sigma": (gap / sigma
                                           if sigma and sigma > 0 else None),
                    })
    return {
        "k": k, "rows": n_rows,
        "min_recall": float(min(recalls)),
        "mean_recall": float(sum(recalls) / len(recalls)),
        "rows_with_a_miss": missed_rows,
        "total_missed_indices": int(sum(len(v) for v in all_missed.values())),
        "worst_gap": (max(d["gap"] for d in disagreements)
                      if disagreements else 0.0),
        "emitted_indices_past_row_length": oor,
        "sigma_used": sigma,
        "disagreements": disagreements,
        "missed_by_row": {str(t): v for t, v in list(all_missed.items())[:64]},
    }


# ---------------------------------------------------------------------------

def main():  # noqa: C901
    tpu_banner()
    import bench_streamidx_topk as B
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sxmod
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        MlaCase, _effective_row_lengths, prepare_index_weights,
        prepare_q_inputs, streamindex_topk)
    import _stage_a as SA

    mods = {}
    for m in (sxmod, scmod, B, SA):
        h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
        mods[os.path.basename(m.__file__)] = {"path": m.__file__, "md5": h}
        p("MODULE", f"{m.__file__} md5={h}")
    RESULTS["modules"] = mods
    RESULTS["jax"] = jax.__version__
    RESULTS["device_kind"] = jax.devices()[0].device_kind
    RESULTS["num_devices"] = len(jax.devices())

    shapes = {s.name: s for s in B.SHAPES}
    static_common = dict(k=B.K_IDX, compression_ratio=B.COMPRESSION_RATIO,
                         num_queries_per_block=B.NUM_QUERIES_PER_BLOCK,
                         enable_early_exit=B.ENABLE_EARLY_EXIT)
    dec = shapes["prod_decode_b64_ctx8704"]
    gdec = B.geometry(dec)
    idec = B.build_inputs(dec, gdec, seed=0)
    RESULTS["geometry_decode"] = dataclasses.asdict(gdec)
    RESULTS["shape"] = dec.name
    p("geometry", RESULTS["geometry_decode"])

    ARGNAMES = ("q", "indexer_weights", "cache_kv", "seq_lens",
                "page_indices", "cu_q_lens", "distribution")

    def op_args(ctx=None):
        args = [idec[n] for n in ARGNAMES]
        if ctx is not None:
            args[3] = jnp.full_like(idec["seq_lens"], ctx)
        return tuple(args)

    def full_chain(tup):
        def build(r):
            def f(q, w, ckv, sl, pi, cql, dist):
                cur, out = sl, None
                for _ in range(r):
                    out = streamindex_topk(
                        q, w, ckv, cur, pi, cql, dist,
                        num_kv_pages_per_block=tup, **static_common)
                    cur = sl + zero_dep(out[0, 0])
                return out
            return jax.jit(f)
        return build

    def stage_a(bkv_p, nst, ctx, vmem_limit_bytes=None):
        """Isolated stage A, the grouped DECODE call, chained through its own
        aliased score buffer -- the incumbent's own data dependency.

        `vmem_limit_bytes` defaults to the incumbent's own
        `DEFAULT_VMEM_LIMIT_BYTES` (100 MiB, `streamindex_topk.py:28`), which
        is what production runs at; section F varies it.
        """
        kw = ({} if vmem_limit_bytes is None
              else {"vmem_limit_bytes": vmem_limit_bytes})
        fn = SA.make_stage_a_fn(
            num_q_heads=B.H_IDX, head_dim=B.D_IDX,
            page_size_per_kv_packing=B.PAGE_SIZE // B.KV_PACKING,
            kv_packing=B.KV_PACKING, lkv_dim=B.LKV_DIM,
            num_sublanes_total=nst, num_kv_pages_per_block=bkv_p,
            num_queries_per_block=B.NUM_QUERIES_PER_BLOCK[0],
            static_q_len=1, seq_batch_size=4, case=MlaCase.DECODE,
            compression_ratio=B.COMPRESSION_RATIO, **kw)
        ntok = gdec.num_tokens

        def build(r):
            def f(q, w, ckv, sl, pi, cql):
                cur = jnp.full((ntok, nst, 128), -jnp.inf, jnp.float32)
                for _ in range(r):
                    cur = fn(q, w, ckv, cur, sl, pi, cql,
                             jnp.int32(0), jnp.int32(ntok))
                return cur
            return jax.jit(f)

        args = (prepare_q_inputs(idec["q"]),
                prepare_index_weights(idec["indexer_weights"],
                                      idec["q"].dtype),
                idec["cache_kv"],
                jnp.full_like(idec["seq_lens"], ctx),
                idec["page_indices"], idec["cu_q_lens"])
        return build, args, fn

    def geom_row(bkv_p, ctx):
        nbkv = max(1, cdiv(ctx, bkv_p * B.PAGE_SIZE))
        pages = nbkv * bkv_p
        tup = (bkv_p, 2, 2)
        nst_full = max(align_to(gdec.pages_per_seq, x) * B.PAGE_SIZE // 128
                       for x in tup)
        return {
            "bkv_p": bkv_p, "ctx": ctx, "num_bkv": nbkv,
            "pages_fetched_per_seq": pages,
            "pages_needed": cdiv(ctx, B.PAGE_SIZE),
            "exact_cover": bool(pages == cdiv(ctx, B.PAGE_SIZE)),
            "sublanes_written": nbkv * (bkv_p * B.PAGE_SIZE // 128),
            "full_op_num_sublanes_total": nst_full,
            "full_op_like_for_like_with_bkvp2":
                nst_full == gdec.num_sublanes_total,
            "bkv_double_buffer_MiB": bkv_p * 2.0,
            "modelled_stage_a_us": round(model_us(nbkv, pages), 2),
        }

    # =====================================================================
    # 0. shape preflight. `jax.eval_shape` traces every arm shape without
    #    compiling or running it, so a wrong argument order or a mis-sized
    #    buffer surfaces in seconds rather than after an hour of compiles.
    #    Systematic breakage is the failure mode that would waste the job; the
    #    per-arm try/except below only contains local breakage.
    # =====================================================================
    pf = {}
    try:
        pf["full_op"] = str(jax.eval_shape(
            full_chain(B.NUM_KV_PAGES_PER_BLOCK)(1), *op_args(CTX_MAIN)))
        b_, a_, _ = stage_a(2, gdec.num_sublanes_total, CTX_MAIN)
        pf["stage_a_p2_nst80"] = str(jax.eval_shape(b_(1), *a_))
        b_, a_, _ = stage_a(9, NST_COMMON, CTX_MAIN)
        pf["stage_a_p9_nst256"] = str(jax.eval_shape(b_(1), *a_))
        pf["ok"] = True
    except Exception as e:
        pf["ok"] = False
        pf["error"] = short(e)
        traceback.print_exc()
    RESULTS["shape_preflight"] = pf
    p("shape_preflight", pf)
    flush("preflight")

    # =====================================================================
    # A. stage A, ctx 8704, common width
    # =====================================================================
    if "A" not in SKIP:
        rows = []
        for bp in P_SWEEP:
            row = geom_row(bp, CTX_MAIN)
            row["num_sublanes_total"] = NST_COMMON
            if row["sublanes_written"] > NST_COMMON:
                row["skipped"] = "would write past the buffer (no bounds chk)"
                rows.append(row)
                continue
            try:
                bld, args, _ = stage_a(bp, NST_COMMON, CTX_MAIN)
                row["chain"] = chain_slope(bld, args,
                                           f"A/stageA/nst{NST_COMMON}/p{bp}",
                                           marker="StreamIdxTC")
                m = row["modelled_stage_a_us"]
                row["measured_over_modelled"] = round(
                    row["chain"]["slope_us_per_call"] / m, 4)
                del bld, args
            except Exception as e:
                row["chain"] = {"error": short(e)}
                p(f"A/p{bp}", f"FAILED {short(e)}")
                traceback.print_exc()
            rows.append(row)
            RESULTS["A_stage_a_common_width"] = rows
            flush(f"A/p{bp}")

    # =====================================================================
    # A2. past the wall
    # =====================================================================
    if "A2" not in SKIP:
        rows = []
        for bp in P_BEYOND:
            row = geom_row(bp, CTX_MAIN)
            nst = row["sublanes_written"]
            row["num_sublanes_total"] = nst
            row["width_is_common"] = False
            try:
                bld, args, _ = stage_a(bp, nst, CTX_MAIN)
                row["chain"] = chain_slope(bld, args,
                                           f"A2/stageA/nst{nst}/p{bp}",
                                           rs=(1, 2), marker=None)
                del bld, args
            except Exception as e:
                row["chain"] = {"error": short(e)}
                p(f"A2/p{bp}", f"STOPPED: {short(e)}")
            rows.append(row)
            RESULTS["A2_past_the_wall"] = rows
            flush(f"A2/p{bp}")

    # =====================================================================
    # B. correctness: the f64 oracle
    # =====================================================================
    if "B" not in SKIP:
        sec = {}
        try:
            rl = _effective_row_lengths(
                seq_lens=jnp.full_like(idec["seq_lens"], CTX_MAIN),
                cu_q_lens=idec["cu_q_lens"],
                distribution=idec["distribution"],
                num_tokens=gdec.num_tokens,
                num_positions=gdec.num_positions,
                compression_ratio=B.COMPRESSION_RATIO)
            rl_np = np.asarray(jax.block_until_ready(rl))
            sec["row_lengths"] = {"min": int(rl_np.min()),
                                  "max": int(rl_np.max()),
                                  "mean": float(rl_np.mean())}
            p("B/row_lengths", sec["row_lengths"])

            scores, top_idx, kth_gap = build_f64_oracle(
                idec, gdec, B, rl_np, B.K_IDX)
            sec["oracle"] = {
                "dtype": "float64",
                "k": B.K_IDX,
                "kth_place_margin_min": float(np.min(kth_gap)),
                "kth_place_margin_median": float(np.median(kth_gap)),
                "score_min": float(np.min(scores[np.isfinite(scores)])),
                "score_max": float(np.max(scores[np.isfinite(scores)])),
            }
            p("B/oracle", sec["oracle"])
            flush("B/oracle")

            # sigma: the f32 stage A's own deviation from the f64 oracle.
            # Measured on the incumbent p=2 at its production width.
            sigma = None
            try:
                _, a_args, fn2 = stage_a(2, gdec.num_sublanes_total, CTX_MAIN)
                sa = np.asarray(jax.block_until_ready(jax.jit(
                    lambda q, w, c, s, pi, cq: fn2(
                        q, w, c,
                        jnp.full((gdec.num_tokens,
                                  gdec.num_sublanes_total, 128), -jnp.inf,
                                 jnp.float32),
                        s, pi, cq, jnp.int32(0), jnp.int32(gdec.num_tokens))
                )(*a_args)))
                flat = sa.reshape(gdec.num_tokens, -1)
                w_ = scores.shape[1]
                d = np.abs(flat[:, :w_].astype(np.float64) - scores)
                d = d[np.isfinite(scores)]
                sigma = float(d.max())
                sec["sigma"] = {
                    "definition": ("max |stage_A_f32 - oracle_f64| over "
                                   "unmasked positions, incumbent bkv_p=2"),
                    "max_abs": sigma,
                    "mean_abs": float(d.mean()),
                    "rel_to_score_max": sigma / sec["oracle"]["score_max"],
                }
                p("B/sigma", sec["sigma"])
                del sa, flat, d
            except Exception as e:
                sec["sigma"] = {"error": short(e)}
                p("B/sigma", f"FAILED {short(e)}")
                traceback.print_exc()
            flush("B/sigma")

            # every p the full op is timed at, plus the incumbent. p=2 is
            # forced in even when --p-min excludes it: it is the reference
            # both `sigma` and `vs_incumbent` are defined against, and without
            # it a disagreement cannot be attributed.
            p_cov = sorted({2} | set(FULL_P_LFL) | set(FULL_P_WIDE)
                           | {int(x) for x in ARGS.oracle_p.split(",") if x})
            sec["covered_bkv_p"] = p_cov
            arms, per_p_missed = {}, {}
            for bp in p_cov:
                try:
                    tup = (bp, 2, 2)

                    def call(*a, _t=tup):
                        return streamindex_topk(
                            *a, num_kv_pages_per_block=_t, **static_common)

                    out = np.asarray(jax.block_until_ready(
                        jax.jit(call)(*op_args(CTX_MAIN))))
                    r = compare_to_oracle(out, scores, top_idx, B.K_IDX,
                                          sigma)
                    per_p_missed[bp] = {t: set(v) for t, v in
                                        [(int(k_), v) for k_, v in
                                         r["missed_by_row"].items()]}
                    arms[str(bp)] = r
                    p(f"B/p{bp}",
                      f"min_recall={r['min_recall']:.6f} "
                      f"rows_with_a_miss={r['rows_with_a_miss']} "
                      f"missed={r['total_missed_indices']} "
                      f"worst_gap={r['worst_gap']:.4e}")
                except Exception as e:
                    arms[str(bp)] = {"error": short(e)}
                    p(f"B/p{bp}", f"FAILED {short(e)}")
                    traceback.print_exc()
                sec["per_bkv_p"] = arms
                RESULTS["B_f64_oracle"] = sec
                flush(f"B/p{bp}")

            # Is a disagreement candidate 0's, or the incumbent's already?
            base = per_p_missed.get(2, {})
            base_all = {(t, i) for t, s in base.items() for i in s}
            shared = {}
            for bp, mm in per_p_missed.items():
                allm = {(t, i) for t, s in mm.items() for i in s}
                shared[str(bp)] = {
                    "n_missed": len(allm),
                    "n_also_missed_by_incumbent_bkvp2":
                        len(allm & base_all),
                    "n_unique_to_this_p": len(allm - base_all),
                    "identical_to_incumbent": bool(allm == base_all),
                }
            sec["vs_incumbent"] = shared
            p("B/vs_incumbent", shared)
        except Exception as e:
            sec["error"] = short(e)
            traceback.print_exc()
        RESULTS["B_f64_oracle"] = sec
        flush("B")
        try:
            del scores, top_idx
        except Exception:
            pass

    # =====================================================================
    # C. the full op
    # =====================================================================
    if "C" not in SKIP:
        rows = []
        plan = [(ctx, bp) for ctx in FULL_CTX for bp in FULL_P_LFL]
        plan += [(CTX_MAIN, bp) for bp in FULL_P_WIDE]
        for ctx, bp in plan:
            row = geom_row(bp, ctx)
            try:
                bld = full_chain((bp, 2, 2))
                row["chain"] = chain_slope(bld, op_args(ctx),
                                           f"C/full/ctx{ctx}/p{bp}",
                                           marker="StreamIdxTC")
                del bld
            except Exception as e:
                row["chain"] = {"error": short(e)}
                p(f"C/ctx{ctx}/p{bp}", f"FAILED {short(e)}")
                traceback.print_exc()
            rows.append(row)
            RESULTS["C_full_op"] = rows
            flush(f"C/ctx{ctx}/p{bp}")

    # =====================================================================
    # D. the exactness test across ctx buckets
    # =====================================================================
    if "D" not in SKIP:
        rows = []
        for ctx in EXACT_CTX:
            for bp in EXACT_P:
                row = geom_row(bp, ctx)
                row["num_sublanes_total"] = NST_COMMON
                if row["sublanes_written"] > NST_COMMON:
                    row["skipped"] = "would write past the buffer"
                    rows.append(row)
                    continue
                try:
                    bld, args, _ = stage_a(bp, NST_COMMON, ctx)
                    row["chain"] = chain_slope(
                        bld, args, f"D/stageA/ctx{ctx}/p{bp}", marker=None)
                    del bld, args
                except Exception as e:
                    row["chain"] = {"error": short(e)}
                    traceback.print_exc()
                rows.append(row)
                RESULTS["D_exactness_ctx_sweep"] = rows
                flush(f"D/ctx{ctx}/p{bp}")

    # =====================================================================
    # E. width controls
    # =====================================================================
    if "E" not in SKIP:
        rows = []
        for nst, ps in CTRL:
            for bp in ps:
                row = geom_row(bp, CTX_MAIN)
                row["num_sublanes_total"] = nst
                if row["sublanes_written"] > nst:
                    row["skipped"] = "would write past the buffer"
                    rows.append(row)
                    continue
                try:
                    bld, args, _ = stage_a(bp, nst, CTX_MAIN)
                    row["chain"] = chain_slope(
                        bld, args, f"E/stageA/nst{nst}/p{bp}", marker=None)
                    del bld, args
                except Exception as e:
                    row["chain"] = {"error": short(e)}
                    traceback.print_exc()
                rows.append(row)
                RESULTS["E_width_controls"] = rows
                flush(f"E/nst{nst}/p{bp}")

    # =====================================================================
    # F. Where the VMEM wall is, and whether raising the scoped limit moves
    #    it.  "p = 9 does not fit" and "p = 9 does not fit at the default
    #    scoped limit" are different results with different remedies, and only
    #    a sweep of `vmem_limit_bytes` can tell them apart.
    #
    #    This section **compiles and does not run**: `lower().compile()` is
    #    enough to make Mosaic accept or reject the allocation, and not running
    #    means the `p > pages_per_seq` page-walk over-read cannot fault here.
    #    That is why F may safely include p values section A2 has to isolate.
    #
    #    The bkv double buffer is `2 * seq_batch_size(4) * (p*256) * 4 * 256`
    #    bytes = p * 2 MiB; `bo_scores` adds 32 KiB * p. Both are reported so
    #    the wall can be read against an allocation, not just a p.
    # =====================================================================
    if "F" not in SKIP:
        rows = []
        for bp in F_P:
            nst = max(NST_COMMON, max(1, cdiv(CTX_MAIN, bp * B.PAGE_SIZE)) *
                      (bp * B.PAGE_SIZE // 128))
            row = geom_row(bp, CTX_MAIN)
            row["num_sublanes_total"] = nst
            row["bkv_double_buf_bytes"] = 2 * 4 * (bp * 256) * 4 * 256
            row["bo_scores_double_buf_bytes"] = 2 * 4 * (bp * 8) * 128 * 4
            row["limits"] = {}
            for mib in F_LIMITS_MIB:
                lim = mib * 1024 * 1024
                try:
                    bld, args, _ = stage_a(bp, nst, CTX_MAIN,
                                           vmem_limit_bytes=lim)
                    bld(1).lower(*args).compile()
                    row["limits"][str(mib)] = {"compiles": True}
                    p(f"F/p{bp}/limit{mib}MiB", "compiles")
                    del bld, args
                except Exception as e:
                    msg = short(e)
                    row["limits"][str(mib)] = {"compiles": False,
                                               "error": msg}
                    p(f"F/p{bp}/limit{mib}MiB", f"REJECTED {msg[:220]}")
            ok = [int(m) for m, v in row["limits"].items()
                  if v.get("compiles")]
            row["min_vmem_limit_mib_that_compiles"] = min(ok) if ok else None
            row["compiles_at_production_default_100MiB"] = bool(
                row["limits"].get("100", {}).get("compiles"))
            rows.append(row)
            RESULTS["F_vmem_wall"] = rows
            flush(f"F/p{bp}")

        # Does a raised limit change the *speed* of a p that already fits?
        # Mosaic uses the scoped limit when it decides how much to pipeline,
        # so "fits" and "fits well" are not the same question.
        timed = []
        for bp in F_TIMED_P:
            nst = max(NST_COMMON, max(1, cdiv(CTX_MAIN, bp * B.PAGE_SIZE)) *
                      (bp * B.PAGE_SIZE // 128))
            for mib in F_TIMED_LIMITS_MIB:
                row = {"bkv_p": bp, "vmem_limit_mib": mib,
                       "num_sublanes_total": nst}
                try:
                    bld, args, _ = stage_a(bp, nst, CTX_MAIN,
                                           vmem_limit_bytes=mib * 1024 * 1024)
                    row["chain"] = chain_slope(
                        bld, args, f"F/timed/p{bp}/limit{mib}MiB",
                        marker=None)
                    del bld, args
                except Exception as e:
                    row["chain"] = {"error": short(e)}
                    p(f"F/timed/p{bp}/limit{mib}MiB", f"FAILED {short(e)}")
                timed.append(row)
                RESULTS["F_vmem_limit_timed"] = timed
                flush(f"F/timed/p{bp}/{mib}")

    # =====================================================================
    # G. Is the optimum p exactly `P = cdiv(ctx, page_size)`, and what does a
    #    *fixed* p = 9 cost on a short sequence?
    #
    #    G1 locates the argmin directly instead of inferring it: at each ctx it
    #    sweeps {2, P-1, P, P+1, 9} at the common width, so "P wins" is a
    #    minimum that was bracketed rather than a hypothesis that was never
    #    contradicted.  Job 1 never ran `p = P` at ctx 8192 (P = 8 was not in
    #    EXACT_P), which is the one bucket where the prediction was untested.
    #
    #    G2 is the decision-relevant one.  `num_kv_pages_per_block` is a
    #    compile-time static: a constant p = 9 also runs on short sequences.
    #    Stage A says that costs +69 % at ctx 4096; only the full op says what
    #    it is worth once stage B and the residual are in the denominator.
    #    Restricted to p in {2, 9} because both give num_sublanes_total = 80,
    #    so the score-matrix width is not a confound.
    # =====================================================================
    if "G" not in SKIP:
        rows = []
        for ctx in G_CTX:
            P_opt = cdiv(ctx, B.PAGE_SIZE)
            cand = sorted({x for x in (2, P_opt - 1, P_opt, P_opt + 1, 9)
                           if 1 <= x <= 9 and x in set(_pfilter(range(1, 10)))})
            for bp in cand:
                row = geom_row(bp, ctx)
                row["P_opt_cdiv_ctx_pagesize"] = P_opt
                row["is_p_equals_P"] = bool(bp == P_opt)
                row["num_sublanes_total"] = NST_COMMON
                if row["sublanes_written"] > NST_COMMON:
                    row["skipped"] = "would write past the buffer"
                    rows.append(row)
                    continue
                try:
                    bld, args, _ = stage_a(bp, NST_COMMON, ctx)
                    row["chain"] = chain_slope(
                        bld, args, f"G1/stageA/ctx{ctx}/p{bp}", marker=None)
                    del bld, args
                except Exception as e:
                    row["chain"] = {"error": short(e)}
                    traceback.print_exc()
                rows.append(row)
                RESULTS["G1_argmin_vs_P"] = rows
                flush(f"G1/ctx{ctx}/p{bp}")

        rows = []
        for ctx in G_FULL_CTX:
            for bp in G_FULL_P:
                row = geom_row(bp, ctx)
                row["P_opt_cdiv_ctx_pagesize"] = cdiv(ctx, B.PAGE_SIZE)
                try:
                    bld = full_chain((bp, 2, 2))
                    row["chain"] = chain_slope(bld, op_args(ctx),
                                               f"G2/full/ctx{ctx}/p{bp}",
                                               marker="StreamIdxTC")
                    del bld
                except Exception as e:
                    row["chain"] = {"error": short(e)}
                    traceback.print_exc()
                rows.append(row)
                RESULTS["G2_full_op_short_ctx"] = rows
                flush(f"G2/ctx{ctx}/p{bp}")

    flush("done")
    p("done", "all sections complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
