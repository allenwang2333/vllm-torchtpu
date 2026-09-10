"""Two-stage `sparsecore_topk` at GLM-5.2 128K shapes: correctness gate + DMA probe.

Steps 0 and 1 of the 128K top-k campaign. Neither is an optimization; both are
prerequisites that decide whether the optimizations have a subject.

**Why this file exists at all.** `kernels/deepseek_v4/sparsecore_topk.py` has a
two-stage path (`sparsecore_topk.py:418-457`) that has never run. Every GLM-5.2
profile, microbench and job log in this tree is at `--max-model-len=9216`, where
`n = 10240` and `_pick_partition` returns **P=1** -- the single-stage fast path
at `:412-416`. At P=1 there is no stage 2, no candidate-score gather at `:439`
and no slot-to-column gather at `:452`. There is also no pytest for this kernel
anywhere in the repo. So the two-stage path is untested, unbenchmarked and
unprofiled, and nothing may be measured on it until it is shown to be correct.

At `--max-model-len=132096` (128K ISL + 1K OSL, the recipe's own convention):
`pages_per_seq = 129`, `n = align_to(129, 2) * 1024 = 133120`,
`p_min = ceil(133120/65536) = 3`, and the doubling loop at `:366-368` gives
**P=4, n_p=33280, P*k_p = 8192 candidates**.

## `--gate`: does the two-stage path produce an exact top-k?

Two traps this test is built around, both paid for by job `j-cf800874`:

1. **Compare scores, not indices.** The kernel returns indices *unsorted*
   (`sparsecore_topk.py:388`) and its docstring concedes that "equal boundary
   scores may be broken differently". Production score rows carry hundreds of
   exactly-0.0 ties, so any assertion on the index set is asserting something
   the kernel never promised. What it does promise is that the selected scores
   match an exact top-k *as a multiset*, and that is what is checked here.
2. **No relative tolerance.** This is an exact selection, not an
   approximation: the assertion is bit-equality, `max |score diff| == 0.0`.
   A relative tolerance is meaningless against rows whose max is exactly 0.0.

And the trap that outranks both: **a test whose assertion cannot fail is worse
than no test**, because it reads as evidence. Every case here is therefore run a
second time with one returned index replaced by the row's *worst eligible*
index, and the run FAILS if that perturbation does not turn the case red.
Replacing with the minimum rather than shifting by -1 is deliberate: under
`heavy_ties` the neighbouring column frequently carries an identical score, so a
-1 shift leaves the score multiset unchanged and the negative control would
report a bug in the checker that is really a property of the data.
`GATE: PASS` / `GATE: FAIL` is printed on its own line -- a `#` comment cannot
carry an `exit 1` through cdk's YAML fold, so the caller greps for that string
rather than trusting the exit code.

## `--bw`: what is SparseCore's HBM read bandwidth?

Undocumented. Not in `tpu/V7X.md`, not in the roofline skill, and
`write-tpu-kernels/references/hardware-facts.md` states plainly that no
SparseCore fact on this stack is measured. The two existing measured points at
n=10240 bound it only at `>= 682 GB/s`, which leaves stage 1 anywhere between
23% and 87% DMA -- and that band is the whole question for the double-buffering
candidate, since perfect overlap saves `min(DMA%, compute%)`.

So: the same `plsc.VectorSubcoreMesh`, the same 33280-word blocking copy per
subcore per wave, and **zero compute passes**. Sweep the wave count and take the
slope; the intercept absorbs program load and mesh entry. Each wave accumulates
one lane vector out of the landed buffer so the copies cannot be dead-code
eliminated -- one vector add against 2080 chunks of DMA is not a perturbation.

Harness rules are inherited from `glm_dsa_block_bench.py`: block **once** per
rep batch (a per-call `block_until_ready` costs ~138 us and would swamp this),
two rep counts with the paired slope removing fixed host cost, interleaved
rounds, every variant warmed before round 0, and a dispatch floor as the bar.
Read `us_median`, not `us`.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import socket
import sys
import time
import traceback

# `vllm_torchtpu` must be imported before jax is used: its `env_override` sets
# LIBTPU_INIT_ARGS and installs `_patch_jax_pallas_fori_lowering()`, which
# changes the Mosaic these kernels lower to.
import vllm_torchtpu  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    KEY_NEG_INF, LANES, MAX_SLICE_WORDS, NUM_BUCKETS, _align_to,
    _pick_partition, _sc_topk_direct, sparsecore_topk)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from glm_dsa_block_bench import dispatch_floor, measure  # noqa: E402

# ----------------------------------------------------------------- shapes ----
PAGE_SIZE = 1024  # PallasMLAttentionBackend.get_page_size
BKV_P = 2  # num_kv_pages_per_block[0], mla_attention_op.py:107-121
CSA_TOPK = 2048  # k

# The two contexts. 9216 is what every existing artifact ran (P=1); 132096 is
# the 128K target (P=4). Both are derived, never hardcoded, so a change to
# `_pick_partition` or `MAX_SLICE_WORDS` shows up here as a changed assertion
# rather than as a silently wrong shape.
CTX_SHORT = 9216
CTX_128K = 132096

# `b`: the padded token bucket. Prefill is `--max-num-batched-tokens`. Decode at
# max-concurrency 16 with DP16 is one request per engine, so the floor bucket of
# 16 -- NOT the 64 that the c1024 reference job ran at 46 running requests.
PREFILL_CHUNK = 1024
DECODE_T = 16

SEED = 20260904
KINDS = ("strict_desc", "random", "heavy_ties", "boundary")


def topk_n(max_model_len: int) -> int:
    """`n` of the sparsecore_topk call: num_sublanes_total * 128.

    `streamindex_topk.py:602-605` -- `align_to(pages_per_seq, bkv_p) *
    page_size // 128 * 128` over the three bkv_p values, all 2 on the GLM path.
    """
    pages_per_seq = -(-max_model_len // PAGE_SIZE)
    return _align_to(pages_per_seq, BKV_P) * PAGE_SIZE // 128 * 128


def describe(b: int, n: int, k: int = CSA_TOPK) -> dict:
    """Everything about the realised partition, so nothing below assumes it."""
    p = _pick_partition(b, n, k)
    n_p = n // p
    return {
        "b": b,
        "n": n,
        "k": k,
        "P": p,
        "n_p": n_p,
        "slice_len": _align_to(n_p, LANES),
        "candidates": p * min(k, n_p),
        "two_stage": p > 1,
        "stage1_rows": b * p,
        "stage1_waves": -(-(b * p) // 32),
        "stage2_waves": -(-b // 32) if p > 1 else 0,
        "max_slice_words": MAX_SLICE_WORDS,
    }


# ------------------------------------------------------------ correctness ----


def ref_topk_scores(scores: np.ndarray, k: int,
                    row_lengths: np.ndarray) -> np.ndarray:
    """Exact top-k scores per row, descending, -inf suffix-padded.

    The reference is on *scores*, not indices, for the reason in the module
    docstring. Positions at or beyond `row_lengths` and -inf scores are never
    eligible (`sparsecore_topk.py:26-28`).
    """
    b, _ = scores.shape
    out = np.full((b, k), -np.inf, np.float32)
    for i in range(b):
        row = scores[i, :int(row_lengths[i])]
        elig = row[row > -np.inf]
        vals = np.sort(elig)[::-1][:k]
        out[i, :vals.size] = vals
    return out


def check_case(scores: np.ndarray, idx: np.ndarray, k: int, ref: np.ndarray
               ) -> dict:
    """Bit-exact multiset comparison of selected scores against `ref`.

    `ref` is passed in rather than recomputed because the negative control runs
    this a second time on the same case and the reference sort over
    `b x 133120` is the expensive part.

    Returns diagnostics, not a bare bool -- a False alone would make the
    negative control unreadable, since "failed" has to be distinguishable from
    "failed for the reason we injected".
    """
    b, n = scores.shape
    got = np.full((b, k), -np.inf, np.float32)

    n_valid_got = np.zeros(b, np.int64)
    n_valid_ref = np.isfinite(ref).sum(axis=1)
    oob = 0
    dup = 0
    for i in range(b):
        sel = idx[i]
        live = sel[sel >= 0]
        n_valid_got[i] = live.size
        if live.size:
            if live.max() >= n:
                oob += 1
                continue
            if np.unique(live).size != live.size:
                dup += 1
            vals = np.sort(scores[i, live])[::-1]
            got[i, :vals.size] = vals

    # -1 padding must be a suffix, not interleaved (`:25-26`).
    pad_ok = all(
        np.all(idx[i, int(n_valid_got[i]):] < 0) and np.all(
            idx[i, :int(n_valid_got[i])] >= 0) for i in range(b))

    finite = np.isfinite(ref) & np.isfinite(got)
    diff = np.abs(ref[finite] - got[finite]) if finite.any() else np.zeros(1)
    max_diff = float(diff.max()) if diff.size else 0.0
    count_ok = bool((n_valid_got == n_valid_ref).all())

    return {
        "ok": bool(max_diff == 0.0 and count_ok and pad_ok and oob == 0
                   and dup == 0),
        "max_score_diff": max_diff,
        "count_ok": count_ok,
        "pad_suffix_ok": bool(pad_ok),
        "rows_out_of_range": int(oob),
        "rows_with_dupes": int(dup),
        "n_valid_got_min": int(n_valid_got.min()),
        "n_valid_ref_min": int(n_valid_ref.min()),
    }


def _score_rows(kind: str, b: int, n: int, rng: np.random.Generator):
    """Four distributions, each probing a different way the kernel can be wrong.

    `strict_desc` is the oracle case -- tie-free and monotone, so the answer is
    known in closed form. `random` is the realistic one. `heavy_ties` is the
    case that makes an index-based assertion meaningless and is the reason this
    file compares scores. `boundary` puts the entire top-k inside the last
    partition, which is the two-stage path's own failure mode: stage 2 has to
    cope with three of its four candidate blocks being nothing but padding.
    """
    if kind == "strict_desc":
        return np.repeat(-np.arange(n, dtype=np.float32)[None, :], b, axis=0)
    if kind == "random":
        return rng.standard_normal((b, n), dtype=np.float32)
    if kind == "heavy_ties":
        # 17 distinct values over 133120 columns: every boundary is a tie, and
        # the k-th and (k+1)-th scores are equal in every row. int8 first --
        # `rng.integers` defaults to int64, which is 1 GiB at this shape.
        return rng.integers(0, 17, size=(b, n),
                            dtype=np.int8).astype(np.float32)
    if kind == "boundary":
        s = np.full((b, n), -1e30, np.float32)
        s[:, -(CSA_TOPK + 64):] = rng.standard_normal(
            (b, CSA_TOPK + 64)).astype(np.float32)
        return s
    raise ValueError(kind)


def _row_lengths(kind: str, b: int, n: int, n_p: int) -> np.ndarray:
    """Per-row visible KV lengths straddling every partition boundary.

    These are the values `_effective_row_lengths` (`streamindex_topk.py:482`)
    produces at chunked prefill, where every row in the chunk sees a different
    amount of context. Cycling them across rows covers every edge in one launch
    instead of one launch per edge.

    `boundary` is the exception and runs at full length: its whole point is
    that the mass sits in the last partition, which is only true if the last
    partition is visible at all.
    """
    if kind == "boundary":
        return np.full(b, n, np.int32)
    edges = sorted({
        min(max(e, 0), n)
        for e in (0, 1, 16, n_p - 1, n_p, n_p + 1, 2 * n_p, 2 * n_p + 17,
                  3 * n_p, n - 1, n)
    })
    return np.array([edges[i % len(edges)] for i in range(b)], np.int32)


def _perturb(idx: np.ndarray, scores: np.ndarray, rl: np.ndarray):
    """Replace one selected index with the row's worst eligible column.

    Returns `(perturbed_idx, victim_row)` or `(None, None)` if no row has both
    a live result and something worse to swap in.

    Not a -1 shift: under `heavy_ties` the neighbouring column carries the same
    score often enough that the multiset would be unchanged, and the control
    would then read as a broken checker rather than as a property of the data.
    The row minimum is guaranteed to differ from a top-k member whenever the
    row is not entirely constant.
    """
    b = idx.shape[0]
    for i in range(b):
        live = np.flatnonzero(idx[i] >= 0)
        L = int(rl[i])
        if live.size == 0 or L < 2:
            continue
        row = scores[i, :L]
        finite = np.flatnonzero(row > -np.inf)
        if finite.size == 0:
            continue
        worst = int(finite[np.argmin(row[finite])])
        if row[worst] == row[int(idx[i, live[0]])]:
            continue  # constant row: nothing to detect with
        bad = idx.copy()
        bad[i, live[0]] = worst
        return bad, i
    return None, None


def gate(ctx: int, batches: tuple[int, ...], verbose: bool,
         variant: str = "base"):
    n = topk_n(ctx)
    rng = np.random.default_rng(SEED)
    flags = VARIANTS[variant]
    print(f"\n=== GATE  max_model_len={ctx}  n={n}  variant={variant} "
          f"{flags or '(incumbent)'} ===", flush=True)

    all_ok = True
    results = []
    for b in batches:
        shape = describe(b, n)
        print(f"\n-- b={b}: P={shape['P']} n_p={shape['n_p']} "
              f"cand={shape['candidates']} two_stage={shape['two_stage']} "
              f"stage1_waves={shape['stage1_waves']}", flush=True)
        if ctx == CTX_128K and not shape["two_stage"]:
            print(f"  !! P={shape['P']} at the 128K shape -- the two-stage "
                  f"path is NOT being exercised. Nothing below is evidence "
                  f"about it.")
            all_ok = False

        for kind in KINDS:
            scores = _score_rows(kind, b, n, rng)
            rl = _row_lengths(kind, b, n, shape["n_p"])
            t0 = time.perf_counter()
            idx = np.asarray(
                jax.block_until_ready(
                    sparsecore_topk(jnp.asarray(scores), CSA_TOPK,
                                    jnp.asarray(rl), **flags)))
            dt = time.perf_counter() - t0

            ref = ref_topk_scores(scores, CSA_TOPK, rl)
            res = check_case(scores, idx, CSA_TOPK, ref)

            bad, victim = _perturb(idx, scores, rl)
            neg_ok = None
            if bad is not None:
                neg_ok = not check_case(scores, bad, CSA_TOPK, ref)["ok"]

            ok = res["ok"] and (neg_ok is True)
            all_ok &= ok
            negtxt = ("red" if neg_ok else
                      "GREEN-BUG" if neg_ok is False else "NO-CONTROL")
            print(f"  [{'ok  ' if ok else 'FAIL'}] {kind:<11} "
                  f"maxdiff={res['max_score_diff']:.6g} "
                  f"count_ok={res['count_ok']} pad={res['pad_suffix_ok']} "
                  f"oob={res['rows_out_of_range']} "
                  f"dup={res['rows_with_dupes']} negctl={negtxt} "
                  f"({dt * 1e3:.0f} ms)", flush=True)
            if verbose and not ok:
                print(f"       {json.dumps(res)}")
            results.append({"b": b, "kind": kind, "variant": variant,
                            "shape": shape,
                            "negative_control_went_red": neg_ok,
                            "negative_control_row": victim, **res})
            del scores, ref

    return all_ok, results


# ------------------------------------------------------- DMA-only probe ------


def _dma_only_kernel(src_hbm, out_hbm, resident_vmem, acc_vmem, sems, *,
                     b: int, slice_len: int, num_waves: int):
    """Stage 1's DMA with all of stage 1's compute removed.

    Deliberately identical to `_topk_kernel` in the two things that set the
    transfer rate -- the same `VectorSubcoreMesh`, and the same blocking
    `make_async_copy` of `slice_len` words per subcore per wave into a VMEM
    scratch of the same size -- and stripped of everything else. What is left
    is one number: how fast 32 subcores can pull from HBM.
    """
    num_subcores = 16
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub

    acc_vmem[pl.ds(0, LANES)] = jnp.zeros((LANES, ), jnp.int32)

    def wave_body(w, _):
        row = w * (2 * num_subcores) + flat
        row_c = jnp.minimum(row, b - 1)
        cp = pltpu.make_async_copy(
            src_hbm.at[pl.ds(row_c * slice_len, slice_len)], resident_vmem,
            sems.at[0])
        cp.start()
        cp.wait()
        # Touch the landed data so the copy cannot be dead-code eliminated: all
        # waves write the same scratch, so without a per-wave use only the last
        # copy is live. One vector add against slice_len/16 = 2080 chunks of
        # transfer is below the noise of the thing being measured.
        acc_vmem[pl.ds(0, LANES)] = (acc_vmem[pl.ds(0, LANES)] +
                                     resident_vmem[pl.ds(0, LANES)])
        return 0

    lax.fori_loop(0, num_waves, wave_body, 0)

    # One writer. All 32 subcores hold a live `acc_vmem` -- `flat` is dynamic,
    # so the accumulation cannot be eliminated on the other 31 -- but only
    # subcore 0 stores it, because 32 concurrent DMAs into the same 16 words is
    # a race whose cost would land inside the number being measured.
    @pl.when(flat == 0)
    def _():
        cp = pltpu.make_async_copy(acc_vmem.at[pl.ds(0, LANES)],
                                   out_hbm.at[pl.ds(0, LANES)], sems.at[1])
        cp.start()
        cp.wait()


def make_dma_thunk(src: jax.Array, slice_len: int, num_waves: int):
    """Build the kernel ONCE and return a 0-arg jitted thunk.

    Constructing `pl.kernel(...)` inside the thunk would re-trace and re-compile
    on every timed call, which is the single easiest way to make this harness
    lie: `measure`'s paired slope would come out as minus the compile time and
    the row would be the fastest thing on the board (job `j-2a092b0d`).
    """
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    b = src.shape[0]
    mesh = plsc.VectorSubcoreMesh(num_cores=sc.num_cores,
                                  num_subcores=sc.num_subcores,
                                  core_axis_name="core",
                                  subcore_axis_name="subcore")
    kern = pl.kernel(
        functools.partial(_dma_only_kernel, b=b, slice_len=slice_len,
                          num_waves=num_waves),
        out_type=jax.ShapeDtypeStruct((LANES, ), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=(
            pltpu.VMEM((slice_len, ), jnp.int32),
            pltpu.VMEM((LANES, ), jnp.int32),
            pltpu.SemaphoreType.DMA((2, )),
        ),
        mesh=mesh,
        name=f"sc_dma_only_sl{slice_len}_w{num_waves}",
    )
    flat = src.reshape(-1)
    f = jax.jit(kern)
    return lambda: f(flat)


def bandwidth(ctx: int, rounds: int) -> dict:
    """Slope of wave count against time, at stage 1's exact slice size.

    Two wave counts would give a slope; four give a residual, which is the only
    way to see that the intercept is real rather than an artifact of the pair
    chosen. The intercept is program load + mesh entry and is NOT bandwidth --
    at n=10240 it is 16-27 us and swamps a 2-wave measurement, which is exactly
    how the two existing measured points came to disagree by 48%.
    """
    n = topk_n(ctx)
    shape = describe(PREFILL_CHUNK, n)
    slice_len = shape["slice_len"]
    print(f"\n=== DMA PROBE  slice_len={slice_len} words "
          f"({slice_len * 4 / 1024:.0f} KiB/subcore) ===", flush=True)

    wave_counts = (4, 8, 16, 32)
    rng = np.random.default_rng(SEED)
    variants = {}
    keep = []
    for w in wave_counts:
        src = jnp.asarray(
            rng.standard_normal((w * 32, slice_len),
                                dtype=np.float32).view(np.int32))
        keep.append(src)
        variants[f"dma/w{w}"] = make_dma_thunk(src, slice_len, w)

    # Three drift nulls: the same w=8 configuration registered three times over
    # three distinct source buffers. Their spread is the noise floor the slope's
    # residual has to be read against. THREE, not one -- the same GLM-5.2 decode
    # null once read +0.075% and -2.567% in two jobs eleven minutes apart, so a
    # single null is a number and not a floor.
    for j in range(3):
        src = jnp.asarray(
            rng.standard_normal((8 * 32, slice_len),
                                dtype=np.float32).view(np.int32))
        keep.append(src)
        variants[f"dma/w8#null{j}"] = make_dma_thunk(src, slice_len, 8)

    floor_us = dispatch_floor()
    print(f"  dispatch floor: {floor_us:.2f} us/call", flush=True)
    res = measure(variants, rounds=rounds)

    nulls = [res[k]["us_median"] for k in res if "#null" in k]
    null_spread = (max(nulls) - min(nulls)) / (sum(nulls) / len(nulls)) * 100
    print(f"  three-null spread at w=8: {null_spread:.2f}% "
          f"({min(nulls):.3f} .. {max(nulls):.3f} us)", flush=True)

    bytes_per_wave = 32 * slice_len * 4
    rows_out = []
    for w in wave_counts:
        r = res[f"dma/w{w}"]
        rows_out.append({"waves": w, **r, "bytes": w * bytes_per_wave})
        print(f"  w={w:<3} us_median={r['us_median']:9.3f}  "
              f"us={r['us']:9.3f}  hostbound={r['hostbound_pct']:5.1f}%  "
              f"suspect={r['suspect']}")

    # Least-squares on us_median: time = intercept + slope * waves.
    xs = np.array([r["waves"] for r in rows_out], np.float64)
    ys = np.array([r["us_median"] for r in rows_out], np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    bw_gbs = bytes_per_wave / (slope * 1e-6) / 1e9

    print(f"\n  slope     = {slope:.4f} us/wave  "
          f"({bytes_per_wave / 2**20:.2f} MiB/wave)")
    print(f"  intercept = {intercept:.3f} us   (program load + mesh entry, "
          f"NOT bandwidth)")
    print(f"  max resid = {np.abs(resid).max():.3f} us "
          f"({np.abs(resid).max() / ys.mean() * 100:.2f}% of mean)")
    print(f"  SC HBM read bandwidth = {bw_gbs:.1f} GB/s   "
          f"[MEASURED, one job -- not a floor]")
    print(f"  TensorCore HBM peak per chiplet = 3364.9 GB/s "
          f"(tpu/V7X.md, MODELLED derate) -> SC is "
          f"{bw_gbs / 3364.9 * 100:.1f}% of it")

    return {"slice_len": slice_len, "bytes_per_wave": bytes_per_wave,
            "rows": rows_out, "slope_us_per_wave": float(slope),
            "intercept_us": float(intercept),
            "max_resid_us": float(np.abs(resid).max()),
            "bandwidth_gb_s": float(bw_gbs),
            "null_spread_pct": float(null_spread),
            "nulls_us_median": nulls,
            "dispatch_floor_us": floor_us,
            "raw": {k: v for k, v in res.items()}}


# --------------------------------------------------------- decomposition ----


def decompose(ctx: int, b: int, rounds: int) -> dict:
    """Price every piece of the two-stage call separately, in one job.

    This is the measurement that decides both candidates, and it is worth more
    than either of them because it replaces the two numbers the whole plan was
    resting on -- neither of which was measured:

    * **Candidate A's prize is `gatherA`, not a model of it.** The plan valued
      it at a modelled 181 us from a granule argument (8192 draws across 1040
      granules of 512 B touch 99.96% of the row, so the gather is a second full
      read of the score matrix). That argument may well be right, but
      `TPU_ROPE_CACHE_TRUNCATE` deleted an op worth 0.246 ms/step in a profile
      and came out 1.26% *worse*, and `TPU_MOE_ARGSORT_REVERT_SCATTER` traded
      O(n log n) for O(n), was bit-identical, and lost 3.0x. An op's modelled
      cost is not the saving from deleting it. Measure the op.
    * **Candidate B's gate is `dma_only / stage1`.** The step-1 probe gives
      2138 GB/s, but a bandwidth is not a DMA fraction: the fraction needs
      stage 1's own time, and running the pure-DMA kernel at stage 1's exact
      wave count makes it a ratio measured inside one job rather than a
      quotient of two extrapolations.

    Every operand is built ONCE, outside the timed thunks, and every thunk is
    jitted once -- a thunk that recompiles inside the loop reports minus its
    compile time and comes out the fastest row on the board.

    The three nulls are three registrations of the identical `full` executable
    over three distinct input buffers. Their spread is the floor that any
    later candidate delta has to clear.
    """
    n = topk_n(ctx)
    shape = describe(b, n)
    p, n_p, k = shape["P"], shape["n_p"], CSA_TOPK
    k_p = min(k, n_p)
    print(f"\n=== DECOMPOSE  b={b} n={n} P={p} n_p={n_p} "
          f"cand={p * k_p} ===", flush=True)

    rng = np.random.default_rng(SEED)
    scores = jnp.asarray(_score_rows("random", b, n, rng))
    rl = jnp.asarray(np.full(b, n, np.int32))

    # Reproduce the wrapper's own intermediates eagerly (sparsecore_topk.py
    # :418-455), so each stage is timed on exactly the operands it sees in
    # production rather than on synthetic stand-ins. Gather cost in particular
    # depends on the index distribution, so these must be the real indices.
    scores_p = scores.reshape(b * p, n_p)
    offsets = jnp.arange(p, dtype=jnp.int32) * n_p
    lengths_p = jnp.clip(rl[:, None] - offsets[None, :], 0, n_p).reshape(b * p)

    f_stage1 = jax.jit(
        lambda s, l: _sc_topk_direct(s, k_p, l, write_empty=True))
    local = f_stage1(scores_p, lengths_p).reshape(b, p, k_p)
    global_cand = jnp.where(local < 0, -1,
                            offsets[None, :, None] + local).reshape(b, p * k_p)
    safe_cand = jnp.maximum(global_cand, 0)

    f_gatherA = jax.jit(lambda s, i: jnp.take_along_axis(s, i, axis=1))
    cand_scores = jnp.where(global_cand >= 0, f_gatherA(scores, safe_cand),
                            -jnp.inf)
    cand_lengths = jnp.where(rl > 0, jnp.int32(p * k_p), jnp.int32(0))

    f_stage2 = jax.jit(lambda s, l: _sc_topk_direct(s, k, l, write_empty=True))
    slots = f_stage2(cand_scores, cand_lengths)
    safe_slots = jnp.maximum(slots, 0)

    f_gatherB = jax.jit(lambda g, s: jnp.take_along_axis(g, s, axis=1))
    f_full = jax.jit(lambda s, l: sparsecore_topk(s, k, l))

    # The pure-DMA floor at stage 1's exact wave count, over a buffer of
    # exactly stage 1's shape -- the same 545 MiB, the same 33280-word copies,
    # the same 32 subcores, and no compute at all.
    s1_waves = -(-(b * p) // 32)
    dma_src = lax.bitcast_convert_type(scores, jnp.int32).reshape(b * p, n_p)

    variants = {
        "full": lambda: f_full(scores, rl),
        "stage1": lambda: f_stage1(scores_p, lengths_p),
        "gatherA": lambda: f_gatherA(scores, safe_cand),
        "stage2": lambda: f_stage2(cand_scores, cand_lengths),
        "gatherB": lambda: f_gatherB(global_cand, safe_slots),
        f"dma_only/w{s1_waves}": make_dma_thunk(dma_src, n_p, s1_waves),
    }
    nulls_src = []
    for j in range(3):
        s2 = jnp.asarray(_score_rows("random", b, n, rng))
        nulls_src.append(s2)
        variants[f"full#null{j}"] = functools.partial(
            lambda x: f_full(x, rl), s2)

    floor_us = dispatch_floor()
    print(f"  dispatch floor: {floor_us:.2f} us/call", flush=True)
    res = measure(variants, rounds=rounds)

    for name in variants:
        r = res[name]
        print(f"  {name:<18} us_median={r['us_median']:10.3f}  "
              f"us={r['us']:10.3f}  hostbound={r['hostbound_pct']:5.1f}%  "
              f"suspect={r['suspect']}")

    nulls = [res[f"full#null{j}"]["us_median"] for j in range(3)] + [
        res["full"]["us_median"]
    ]
    spread = (max(nulls) - min(nulls)) / (sum(nulls) / len(nulls)) * 100

    full_us = res["full"]["us_median"]
    s1 = res["stage1"]["us_median"]
    gA = res["gatherA"]["us_median"]
    s2u = res["stage2"]["us_median"]
    gB = res["gatherB"]["us_median"]
    dma = res[f"dma_only/w{s1_waves}"]["us_median"]
    parts = s1 + gA + s2u + gB

    print(f"\n  four-null spread on `full`: {spread:.2f}% "
          f"({min(nulls):.3f} .. {max(nulls):.3f} us)")
    print(f"  sum(parts) = {parts:.3f} us vs full = {full_us:.3f} us  "
          f"-> parts/full = {parts / full_us:.3f}")
    print("  ^ a ratio far from 1.0 means the pieces do NOT compose: XLA is "
          "fusing or overlapping across the boundaries, and no per-piece "
          "saving can be assumed to reach the whole.")
    print(f"\n  CANDIDATE A prize  = gatherA {gA:.3f} us "
          f"= {gA / full_us * 100:.1f}% of full")
    print(f"  CANDIDATE A2 prize = gatherB {gB:.3f} us "
          f"= {gB / full_us * 100:.1f}% of full")
    print(f"  both gathers       = {gA + gB:.3f} us "
          f"= {(gA + gB) / full_us * 100:.1f}% of full")
    print(f"\n  stage1 = {s1:.3f} us,  pure DMA at w{s1_waves} = {dma:.3f} us")
    dma_frac = dma / s1 * 100 if s1 > 0 else 0.0
    print(f"  CANDIDATE B gate: stage1 is {dma_frac:.1f}% DMA "
          f"-> {'BUILD B' if dma_frac >= 25 else 'DROP B AND C'} "
          f"(gate is 25%)")
    print("  ^ perfect overlap saves min(DMA%, compute%) of stage1, so this "
          f"caps B at {min(dma_frac, 100 - dma_frac):.1f}% of stage1 "
          f"= {min(dma, s1 - dma) / full_us * 100:.1f}% of full.")

    return {"b": b, "n": n, "P": p, "n_p": n_p, "k_p": k_p,
            "stage1_waves": s1_waves, "rows": res,
            "null_spread_pct": float(spread), "nulls_us_median": nulls,
            "sum_parts_us": float(parts), "parts_over_full": float(parts /
                                                                   full_us),
            "gatherA_pct_of_full": float(gA / full_us * 100),
            "gatherB_pct_of_full": float(gB / full_us * 100),
            "stage1_dma_pct": float(dma_frac),
            "candidate_B_gate": "BUILD" if dma_frac >= 25 else "DROP",
            "B_cap_pct_of_full": float(min(dma, s1 - dma) / full_us * 100),
            "dispatch_floor_us": floor_us}


def variant_equality(ctx: int, batches: tuple[int, ...]):
    """Every candidate must be **bit-identical** to the incumbent.

    Stronger than re-running the numpy gate per variant, and far cheaper -- the
    reference sort over `1024 x 133120` is the expensive part of `gate` and
    running it four times buys nothing, because neither candidate is allowed to
    differ from the incumbent even in a tie-break.

    A: stage 1's index emission is untouched, and the score it stores is the
    same fp32 bits the gather would have read back (`monotone_key` is an
    involution), so stage 2 sees an identical input array.
    A2: stage 2 emits `payload[j]` in the same compaction order in which the
    incumbent emits `j` and then gathers `global_cand[j]`.

    So any difference at all is a bug, and the check is `array_equal`. Each
    case also runs a negative control -- one element of the incumbent's output
    is altered and the comparison must go red -- because an equality check
    between two things that are accidentally the same object would pass
    forever.
    """
    n = topk_n(ctx)
    rng = np.random.default_rng(SEED)
    print(f"\n=== VARIANT EQUALITY  n={n} ===", flush=True)

    all_ok = True
    results = []
    for b in batches:
        shape = describe(b, n)
        print(f"\n-- b={b}: P={shape['P']} two_stage={shape['two_stage']}",
              flush=True)
        if not shape["two_stage"]:
            print("  !! P=1: the flags are no-ops here, so equality is "
                  "trivially true and proves nothing about them.")
        for kind in KINDS:
            scores = jnp.asarray(_score_rows(kind, b, n, rng))
            rl = jnp.asarray(_row_lengths(kind, b, n, shape["n_p"]))
            base = np.asarray(
                jax.block_until_ready(sparsecore_topk(scores, CSA_TOPK, rl)))

            # Negative control on the comparison itself.
            bad = base.copy()
            bad[0, 0] = bad[0, 0] + 1
            if np.array_equal(base, bad):
                print(f"  [FAIL] {kind}: the equality check cannot go red.")
                all_ok = False
                continue

            for name, flags in VARIANTS.items():
                if name == "base":
                    continue
                got = np.asarray(
                    jax.block_until_ready(
                        sparsecore_topk(scores, CSA_TOPK, rl, **flags)))
                same = bool(np.array_equal(base, got))
                ndiff = int((base != got).sum()) if not same else 0
                all_ok &= same
                print(f"  [{'ok  ' if same else 'FAIL'}] {kind:<11} "
                      f"{name:<6} identical_to_base={same} "
                      f"n_differing={ndiff}", flush=True)
                results.append({"b": b, "kind": kind, "variant": name,
                                "identical_to_base": same,
                                "n_differing": ndiff})
            del scores

    return all_ok, results


# ------------------------------------------------------- pass-cost probe ----

# The four flag combinations under test. `sparsecore_topk`'s two new static
# args are independent: A deletes the candidate-score gather by having stage 1
# emit the winning score next to its index, A2 deletes the slot-to-column
# gather by having stage 2 emit a payload. Measuring all four rather than just
# base-vs-both is deliberate -- `TPU_STREAMIDX_TRIM_TOPK` was reported as one
# lever whose two halves turned out to contribute 97.4% and 2.6%.
VARIANTS = {
    "base": {},
    "A": {"emit_pairs": True},
    "A2": {"payload_stage2": True},
    "A+A2": {"emit_pairs": True, "payload_stage2": True},
}


def _pass_probe_kernel(src_hbm, out_hbm, resident_vmem, priv_vmem, rowbuf_vmem,
                       keybuf_vmem, sems, *, b: int, slice_len: int,
                       num_waves: int, num_passes: int, kind: str, k: int):
    """One DMA per wave, then `num_passes` walks of the resident slice.

    Stage 1 is 85% of the two-stage call and 5.78% of stage 1 is DMA, so the
    other 94% is walks: cvt (:166), the L0 histogram (:187), three refinements
    (:234) and emit (:291) -- six passes over 33280 words per subcore. This
    probe prices one walk of each kind by sweeping `num_passes` and taking the
    slope, which does two things the full kernel cannot.

    It decomposes the 85%: if 1 x cvt + 4 x scatter + 1 x emit reproduces
    stage 1, the cost is pass count and the lever is removing a pass. If it
    does not, something outside the walks dominates and neither candidate here
    is aimed at it.

    And `emit2 - emit1` is **candidate A's cost side, measured directly**: the
    only thing A adds to the kernel is a second `plsc.store_compressed` under
    the identical mask and offset, which is exactly the difference between
    those two pass kinds. The plan could only bound it at "+15% to +32% of
    stage-1 issue count" from a port model, which left A's sign undetermined.

    Truncating the kernel itself was the other way to build this ladder and is
    rejected: dropping refinement levels leaves `prefix` coarse, `my_take`
    large and `woff` past the end of a `k + LANES` scratch, and bounds checks
    are off. This probe wraps `woff` into range instead, so no pass kind can
    write outside its buffer whatever the data does.
    """
    num_subcores = 16
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * num_subcores + sub
    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    walk_chunks = slice_len // LANES

    rowbuf_vmem[pl.ds(0, LANES)] = jnp.zeros((LANES, ), jnp.int32)

    def wave_body(w, _):
        row = w * (2 * num_subcores) + flat
        row_c = jnp.minimum(row, b - 1)
        cp = pltpu.make_async_copy(
            src_hbm.at[pl.ds(row_c * slice_len, slice_len)], resident_vmem,
            sems.at[0])
        cp.start()
        cp.wait()

        for _p in range(num_passes):
            if kind == "cvt":

                def body(i):
                    bits = resident_vmem[pl.ds(i * LANES, LANES)]
                    m = jnp.bitwise_and(jnp.right_shift(bits, 31),
                                        jnp.int32(0x7FFFFFFF))
                    resident_vmem[pl.ds(i * LANES, LANES)] = jnp.bitwise_xor(
                        bits, m)

                plsc.parallel_loop(0, walk_chunks, unroll=4)(body)
            elif kind.startswith("scatter"):
                # Only the MASK differs across the three scatter kinds; the
                # loop trip count, the address arithmetic and the update are
                # identical, so the slope difference isolates mask density and
                # nothing else.
                #
                # This exists because the first version of this probe had one
                # scatter kind, dense (`mask = key > KEY_NEG_INF`, true for
                # essentially every lane), and was used to price all four of
                # the kernel's scatter passes. Only one of them is dense: the
                # L0 histogram scatters under `valid`, but the three refinement
                # levels scatter under `match = valid & (key >> shift ==
                # prefix)`, which selects ~1/NUM_BUCKETS of lanes at level 1 and
                # less after that. If `addupdate_scatter` skips masked-off
                # lanes rather than issuing them, a refinement pass is far
                # cheaper than the histogram and the uniform-pass model of
                # stage 1 is wrong -- which is exactly what the model's 44%
                # overshoot suggests.
                dens = {"scatter": 0, "scatter_mid": 15,
                        "scatter_sparse": NUM_BUCKETS - 1}[kind]

                def body(i, dens=dens):
                    key = resident_vmem[pl.ds(i * LANES, LANES)]
                    bucket = jnp.bitwise_and(jnp.right_shift(key, 24),
                                             jnp.int32(NUM_BUCKETS - 1))
                    addr = lane_iota * NUM_BUCKETS + bucket
                    m = key > jnp.int32(KEY_NEG_INF)
                    if dens:
                        # Low mantissa bits, so the density is 1/(dens+1)
                        # whatever the score distribution -- unlike the
                        # exponent-derived `key >> 24` the kernel matches on,
                        # whose density under standard normals is anything but
                        # uniform. Data-dependent, so it cannot be folded away.
                        m = m & (jnp.bitwise_and(key, jnp.int32(dens)) == 0)
                    plsc.addupdate_scatter(priv_vmem, (addr, ),
                                           jnp.ones((LANES, ), jnp.int32),
                                           mask=m)

                plsc.parallel_loop(0, walk_chunks, unroll=4)(body)
            else:
                stores = 2 if kind == "emit2" else 1

                def body(i, carry, stores=stores):
                    woff, rank = carry
                    key = resident_vmem[pl.ds(i * LANES, LANES)]
                    j = i * LANES + lane_iota
                    # ~6.25% selected, matching emit's real 2048/33280 density,
                    # and data-dependent so it cannot be folded away.
                    sel = jnp.bitwise_and(key, jnp.int32(15)) == 0
                    tie_rank = plsc.cumsum(sel.astype(jnp.int32)) + rank
                    plsc.store_compressed(rowbuf_vmem.at[pl.ds(woff, LANES)],
                                          j,
                                          mask=sel)
                    if stores == 2:
                        plsc.store_compressed(
                            keybuf_vmem.at[pl.ds(woff, LANES)], key, mask=sel)
                    woff = woff + plsc.all_reduce_population_count(sel)[0]
                    # k is a power of two, so this is one AND and it keeps
                    # every store inside the k + LANES scratch regardless of
                    # how many elements the data happens to select.
                    woff = jnp.bitwise_and(woff, jnp.int32(k - 1))
                    return woff, tie_rank[LANES - 1]

                plsc.parallel_loop(0, walk_chunks, unroll=4,
                                   carry=(jnp.int32(0), jnp.int32(0)))(body)

        # Keep every pass live. `cvt` writes only `resident_vmem` and
        # `scatter` writes only `priv_vmem`; with nothing downstream reading
        # either, the last pass -- and with it the whole slope this probe
        # exists to measure -- is dead code. Two vector adds per wave against
        # 2080 chunks per pass is far below the quantity being measured, and
        # the alternative is a probe that reports a flat line and looks like a
        # finding.
        rowbuf_vmem[pl.ds(0, LANES)] = (rowbuf_vmem[pl.ds(0, LANES)] +
                                        resident_vmem[pl.ds(0, LANES)] +
                                        priv_vmem[pl.ds(0, LANES)])
        return 0

    lax.fori_loop(0, num_waves, wave_body, 0)

    @pl.when(flat == 0)
    def _():
        cp = pltpu.make_async_copy(rowbuf_vmem.at[pl.ds(0, LANES)],
                                   out_hbm.at[pl.ds(0, LANES)], sems.at[1])
        cp.start()
        cp.wait()


def make_pass_thunk(src: jax.Array, slice_len: int, num_waves: int,
                    num_passes: int, kind: str, k: int = CSA_TOPK):
    """Build once, return a 0-arg jitted thunk. See `make_dma_thunk`."""
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    mesh = plsc.VectorSubcoreMesh(num_cores=sc.num_cores,
                                  num_subcores=sc.num_subcores,
                                  core_axis_name="core",
                                  subcore_axis_name="subcore")
    kern = pl.kernel(
        functools.partial(_pass_probe_kernel, b=src.shape[0],
                          slice_len=slice_len, num_waves=num_waves,
                          num_passes=num_passes, kind=kind, k=k),
        out_type=jax.ShapeDtypeStruct((LANES, ), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=(
            pltpu.VMEM((slice_len, ), jnp.int32),
            pltpu.VMEM((LANES * NUM_BUCKETS + LANES, ), jnp.int32),
            pltpu.VMEM((k + LANES, ), jnp.int32),
            pltpu.VMEM((k + LANES, ), jnp.int32),
            pltpu.SemaphoreType.DMA((2, )),
        ),
        mesh=mesh,
        name=f"sc_pass_{kind}_p{num_passes}_sl{slice_len}_w{num_waves}",
    )
    flat = src.reshape(-1)
    f = jax.jit(kern)
    return lambda: f(flat)


def passes(ctx: int, b: int, rounds: int) -> dict:
    """Per-pass cost at stage 1's exact geometry, and A's cost side."""
    n = topk_n(ctx)
    shape = describe(b, n)
    p, n_p = shape["P"], shape["n_p"]
    waves, slice_len = shape["stage1_waves"], shape["slice_len"]
    print(f"\n=== PASSES  b={b} n={n} P={p} n_p={n_p} "
          f"slice_len={slice_len} waves={waves} ===", flush=True)

    rng = np.random.default_rng(SEED)
    scores = jnp.asarray(_score_rows("random", b, n, rng))
    src = lax.bitcast_convert_type(scores, jnp.int32).reshape(b * p, n_p)

    kinds = ("cvt", "scatter", "scatter_mid", "scatter_sparse", "emit1",
             "emit2")
    counts = (0, 1, 2, 4)
    variants = {}
    for kind in kinds:
        for c in counts:
            if c == 0 and kind != "cvt":
                continue  # the 0-pass point is the same kernel for all kinds
            variants[f"{kind}/p{c}"] = make_pass_thunk(src, slice_len, waves,
                                                       c, kind)
    for j in range(3):
        variants[f"cvt/p1#null{j}"] = make_pass_thunk(src, slice_len, waves, 1,
                                                      "cvt")

    floor_us = dispatch_floor()
    print(f"  dispatch floor: {floor_us:.2f} us/call", flush=True)
    res = measure(variants, rounds=rounds)
    for name in variants:
        r = res[name]
        print(f"  {name:<18} us_median={r['us_median']:10.3f}  "
              f"us={r['us']:10.3f}  suspect={r['suspect']}")

    nulls = [res["cvt/p1"]["us_median"]] + [
        res[f"cvt/p1#null{j}"]["us_median"] for j in range(3)
    ]
    spread = (max(nulls) - min(nulls)) / (sum(nulls) / len(nulls)) * 100
    print(f"\n  four-null spread on cvt/p1: {spread:.2f}% "
          f"({min(nulls):.3f} .. {max(nulls):.3f} us)")

    zero = res["cvt/p0"]["us_median"]
    slopes = {}
    for kind in kinds:
        xs = [c for c in counts if c > 0]
        ys = [res[f"{kind}/p{c}"]["us_median"] for c in xs]
        xs_f = [0.0] + [float(c) for c in xs]
        ys_f = [zero] + ys
        mx = sum(xs_f) / len(xs_f)
        my = sum(ys_f) / len(ys_f)
        den = sum((x - mx)**2 for x in xs_f)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs_f, ys_f)) / den
        icpt = my - slope * mx
        resid = max(abs(y - (slope * x + icpt)) for x, y in zip(xs_f, ys_f))
        slopes[kind] = slope
        print(f"  {kind:<8} {slope:9.3f} us/pass  (intercept {icpt:8.3f} us, "
              f"max resid {resid:6.3f} us = {resid / my * 100:.2f}%)")
        # A pass that costs nothing and a pass that was eliminated look
        # identical from the outside, and only one of them is a finding.
        if slope * 4 <= spread / 100 * my:
            print(f"    !! {kind}: the slope is inside the null floor. Read "
                  "this as 'not measured' -- the likeliest cause is that the "
                  "pass was dead-code eliminated, not that it is free.")

    # Two models of stage 1, differing only in what a refinement pass costs.
    #
    # `uniform` prices all four scatter passes at the dense slope. It is the
    # model this probe shipped with, and on the first run it overshot the
    # measured stage 1 by 44%. Inverting that overshoot for a pass count gave
    # "2.595 of 4 passes actually execute", attributed to `active_walk =
    # where(done, 0, walk_chunks)` short-circuiting refinement levels. That
    # reading is retracted: `done = keep_all | (quota == cnt)`, and at
    # total=33280 >> k=2048 `keep_all` is false and the exact-count coincidence
    # should fire on well under 1% of rows -- nowhere near enough to remove 1.4
    # passes. A model's residual is not a measurement of its own missing term.
    #
    # `graded` prices the L0 histogram dense (it scatters under `valid`) and
    # the three refinements at the sparse slope (they scatter under a
    # ~1/NUM_BUCKETS `match`). If that one closes and `uniform` does not, the
    # cost of a scatter tracks mask density and the pass count was never the
    # discrepancy.
    uniform = slopes["cvt"] + 4 * slopes["scatter"] + slopes["emit1"] + zero
    graded = (slopes["cvt"] + slopes["scatter"] +
              3 * slopes["scatter_sparse"] + slopes["emit1"] + zero)
    a_cost = slopes["emit2"] - slopes["emit1"]
    print(f"\n  uniform  1 cvt + 4 x scatter(dense) + 1 emit + {zero:.1f} us "
          f"= {uniform:.3f} us")
    print(f"  graded   1 cvt + 1 scatter(dense) + 3 x scatter(1/{NUM_BUCKETS}) "
          f"+ 1 emit + {zero:.1f} us = {graded:.3f} us")
    print("  ^ compare BOTH against the measured stage1. Whichever closes is "
          "the model; do not invert the loser for a pass count.")
    dense = slopes["scatter"]
    for kind in ("scatter_mid", "scatter_sparse"):
        print(f"  {kind:<14} / scatter(dense) = {slopes[kind] / dense:.3f}")
    print("  ^ ~1.0 means addupdate_scatter issues masked-off lanes anyway "
          "(cost is the walk, not the updates); well below 1.0 means density "
          "is the lever and the refinements are cheap.")
    print(f"\n  CANDIDATE A cost side = emit2 - emit1 = {a_cost:.3f} us "
          "(the second store_compressed, at stage 1's geometry).")
    print("  ^ this is A's cost side ONLY. It excludes the extra k-word output "
          "DMA that emit_keys also adds, so the true cost is a little higher "
          "and the predicted saving a little lower.")

    return {"b": b, "n": n, "P": p, "n_p": n_p, "slice_len": slice_len,
            "waves": waves, "rows": res, "zero_pass_us": float(zero),
            "us_per_pass": {kind: float(s) for kind, s in slopes.items()},
            "modelled_stage1_uniform_us": float(uniform),
            "modelled_stage1_graded_us": float(graded),
            "sparse_over_dense_scatter": {
                kind: float(slopes[kind] / dense)
                for kind in ("scatter_mid", "scatter_sparse")},
            "candidate_A_cost_us": float(a_cost),
            "null_spread_pct": float(spread),
            "dispatch_floor_us": floor_us}


# ------------------------------------------------------------------- A/B ----


def ab(ctx: int, b: int, rounds: int) -> dict:
    """The four flag combinations, interleaved in one job.

    `base` is the unmodified incumbent and is allowed to win. Deltas are
    reported against the three-null interval, not against zero.
    """
    n = topk_n(ctx)
    shape = describe(b, n)
    print(f"\n=== A/B  b={b} n={n} P={shape['P']} n_p={shape['n_p']} "
          f"cand={shape['candidates']} ===", flush=True)
    if not shape["two_stage"]:
        print("  !! P=1: the flags are no-ops on the single-stage path. "
              "Nothing below is evidence about them.")

    rng = np.random.default_rng(SEED)
    scores = jnp.asarray(_score_rows("random", b, n, rng))
    rl = jnp.asarray(np.full(b, n, np.int32))

    # `sparsecore_topk` is already `jax.jit`-wrapped with the flags as static
    # argnames, so binding them here selects a distinct compiled executable per
    # variant without a second jit layer.
    fns = {
        name: functools.partial(sparsecore_topk, k=CSA_TOPK, **flags)
        for name, flags in VARIANTS.items()
    }
    variants = {
        name: functools.partial(lambda f, s, l: f(s, row_lengths=l), fn,
                                scores, rl)
        for name, fn in fns.items()
    }
    for j in range(3):
        s2 = jnp.asarray(_score_rows("random", b, n, rng))
        variants[f"base#null{j}"] = functools.partial(
            lambda f, s, l: f(s, row_lengths=l), fns["base"], s2, rl)

    floor_us = dispatch_floor()
    print(f"  dispatch floor: {floor_us:.2f} us/call", flush=True)
    res = measure(variants, rounds=rounds)
    for name in variants:
        r = res[name]
        print(f"  {name:<14} us_median={r['us_median']:10.3f}  "
              f"us={r['us']:10.3f}  suspect={r['suspect']}")

    nulls = [res["base"]["us_median"]] + [
        res[f"base#null{j}"]["us_median"] for j in range(3)
    ]
    spread = (max(nulls) - min(nulls)) / (sum(nulls) / len(nulls)) * 100
    base = res["base"]["us_median"]
    print(f"\n  four-null spread on base: {spread:.2f}% "
          f"({min(nulls):.3f} .. {max(nulls):.3f} us)")

    deltas = {}
    for name in VARIANTS:
        d = (res[name]["us_median"] - base) / base * 100
        deltas[name] = float(d)
        verdict = ("within null floor" if abs(d) <= spread else
                   "WIN" if d < 0 else "REGRESSION")
        print(f"  {name:<8} {d:+7.2f}%  {verdict}")
    print("  ^ a delta inside the null spread is not a result.")

    return {"b": b, "n": n, "P": shape["P"], "rows": res,
            "null_spread_pct": float(spread), "base_us": float(base),
            "delta_pct": deltas, "dispatch_floor_us": floor_us}


# ------------------------------------------------------------- HLO check ----


def hlo_check(ctx: int, b: int, variant: str = "base") -> dict:
    """Is the candidate-score gather actually in the optimized HLO at P=4?

    The whole case for candidate A is that stage 2 pays for
    `jnp.take_along_axis(scores, safe_cand_indices)` at
    `sparsecore_topk.py:439`. That is a claim about the *optimized* HLO, not
    about the Python: XLA has already been caught in this tree sinking a top-k
    below a collective, and the source-level reading of where a gather happens
    was wrong there. **If the gather is not here, candidate A has no subject.**

    `compile().as_text()` is the post-optimization module, so this needs no
    `--xla_dump_to` and leaves no files behind.

    Run this per VARIANT, not just on `base`. Compiling only the incumbent
    answers "does A have a subject?" and stops there -- it never checks that
    turning the flag on actually removes the gather, which is A's entire
    mechanism. The two are different claims and the second is the one the
    speedup is attributed to. It costs a compile, no device time.
    """
    n = topk_n(ctx)
    shape = describe(b, n)
    flags = VARIANTS[variant]
    print(f"\n=== HLO  b={b} n={n} P={shape['P']} variant={variant} "
          f"flags={flags or '{}'} ===", flush=True)

    f = jax.jit(lambda s, rl: sparsecore_topk(s, CSA_TOPK, rl, **flags))
    txt = f.lower(jax.ShapeDtypeStruct((b, n), jnp.float32),
                  jax.ShapeDtypeStruct((b, ), jnp.int32)).compile().as_text()

    lines = txt.splitlines()
    gathers = [ln.strip() for ln in lines
               if " gather(" in ln or "kGather" in ln]
    customs = [ln.strip() for ln in lines if "custom-call" in ln]
    fusions = [ln.strip() for ln in lines if " fusion(" in ln]

    # Resolve `%name -> shape` across the whole module first.
    #
    # The obvious test -- does the gather's LINE contain the string
    # `f32[b,n]`? -- cannot ever fire, and the first version of this function
    # shipped it. An HLO instruction prints its own OUTPUT shape and refers to
    # its operands by NAME: gather A prints as `f32[1024,8192] gather(
    # %param_0.71, ...)`, so the `f32[1024,133120]` operand shape appears
    # nowhere on that line. The detector reported "no op reads the full score
    # matrix" on a module that contained exactly the gather it was looking for,
    # which is the failure mode this repo keeps paying for: a check that cannot
    # go red reads as evidence of absence.
    defs = {}
    for ln in lines:
        s = ln.strip()
        if not s.startswith("%") or " = " not in s:
            continue
        name, rhs = s.split(" = ", 1)
        defs[name.strip()] = rhs.split(" ", 1)[0]  # the shape token
    def operands_of(line: str) -> list[str]:
        """Operand names of one instruction.

        Second bug in this function, kept documented because it is the same
        mistake as the first in a new costume: `line.find("(")` does NOT find
        the opcode's argument list. The OUTPUT SHAPE contains parentheses of
        its own -- `f32[16,8192]{1,0:T(8,128)}` -- so the naive scan opened at
        the tiling annotation and `rfind(")")` closed somewhere in `metadata`,
        yielding operands like `'%custom-call.6)'` with a stray paren and no
        resolvable name. It printed `reads_full_score_matrix=False`, which is
        the same unearned verdict the string-matching version printed, just
        arrived at differently.

        So: locate the opcode token explicitly, then paren-match forward.
        """
        rhs = line.split(" = ", 1)[1] if " = " in line else line
        m = re.search(r"\s([a-z][a-z0-9-]*)\(", rhs)
        if not m:
            return []
        start, depth, i = m.end(), 1, m.end()
        while i < len(rhs) and depth:
            depth += (rhs[i] == "(") - (rhs[i] == ")")
            i += 1
        return [t.strip() for t in rhs[start:i - 1].split(",")
                if t.strip().startswith("%")]

    def op_name_of(line: str) -> str:
        m = line.find('op_name="')
        return line[m + 9:line.find('"', m + 9)] if m >= 0 else "?"

    full_row = []
    detail = []
    for ln in gathers:
        out_shape = ln.split(" = ", 1)[1].split(" ", 1)[0] if " = " in ln else "?"
        ops = operands_of(ln)
        op_shapes = [defs.get(o, "?") for o in ops]
        # Tri-state, not a bool. An unresolved operand (`%param_0.63` is a
        # fusion-computation parameter and may not be defined in the printed
        # module) means the question was not answered -- reporting `False`
        # there is the same unearned negative twice burned already.
        if not ops:
            reads_full = "no-operands-parsed"
        elif any(f"[{b},{n}]" in sh for sh in op_shapes):
            reads_full = True
        elif any(sh == "?" for sh in op_shapes):
            reads_full = "unresolved"
        else:
            reads_full = False
        detail.append({"out_shape": out_shape, "operands": ops,
                       "operand_shapes": op_shapes,
                       "op_name": op_name_of(ln),
                       "reads_full_score_matrix": reads_full})
        if reads_full is True:
            full_row.append(ln)

    print(f"  gather ops:      {len(gathers)}")
    for d in detail:
        print(f"    out={d['out_shape']:<24} operands="
              f"{list(zip(d['operands'], d['operand_shapes']))}")
        print(f"      op_name={d['op_name']}  "
              f"reads_full_score_matrix={d['reads_full_score_matrix']}")
    print(f"  custom-calls:    {len(customs)}  "
          f"(tpu_custom_call = the SparseCore kernels)")
    print(f"  fusions:         {len(fusions)}")
    print(f"  gathers reading the full f32[{b},{n}] score matrix: "
          f"{len(full_row)}")
    if shape["two_stage"] and not gathers and variant == "base":
        print("  !! no gather at all -- candidate A has NO SUBJECT here.")

    return {"b": b, "n": n, "P": shape["P"], "variant": variant,
            "flags": {k2: bool(v2) for k2, v2 in flags.items()},
            "hlo_lines": len(lines),
            "n_gather": len(gathers), "n_custom_call": len(customs),
            "n_fusion": len(fusions), "n_full_row_gathers": len(full_row),
            "gather_detail": detail, "gathers": gathers[:16]}


# ------------------------------------------------------------------- main ----


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", action="store_true",
                    help="correctness of the two-stage path (step 0)")
    ap.add_argument("--bw", action="store_true",
                    help="SparseCore HBM bandwidth probe (step 1)")
    ap.add_argument("--hlo", action="store_true",
                    help="is the candidate-score gather in the optimized HLO?")
    ap.add_argument("--hlo-variants", type=str, nargs="+",
                    default=sorted(VARIANTS), choices=sorted(VARIANTS),
                    help="which flag combinations to compile. Compile-only, no "
                         "device time. Defaults to all four: checking only "
                         "`base` shows the gather exists but never that a flag "
                         "removes it.")
    ap.add_argument("--decomp", action="store_true",
                    help="price stage1 / gatherA / stage2 / gatherB / pure-DMA "
                         "against the whole call (decides A and B)")
    ap.add_argument("--eq", action="store_true",
                    help="every candidate must be bit-identical to the "
                         "incumbent (flips GATE)")
    ap.add_argument("--passes", action="store_true",
                    help="per-walk-pass cost at stage 1's geometry, and "
                         "candidate A's cost side (emit2 - emit1)")
    ap.add_argument("--ab", action="store_true",
                    help="A/B base vs A vs A2 vs A+A2 on the whole call")
    ap.add_argument("--gate-variants", type=str, nargs="+",
                    default=["base"], choices=sorted(VARIANTS),
                    help="which flag combinations the correctness gate runs. "
                         "A perf number for a variant this did not cover is "
                         "unverified, whatever the A/B says.")
    ap.add_argument("--ctx", type=int, default=CTX_128K,
                    help=f"max_model_len; {CTX_128K} gives P=4, "
                         f"{CTX_SHORT} gives P=1")
    ap.add_argument("--also-short", action="store_true",
                    help="also gate at 9216 (P=1) as a control: the "
                         "single-stage path is what production runs today, so "
                         "a failure there means the harness is wrong, not the "
                         "two-stage path")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[DECODE_T, PREFILL_CHUNK])
    ap.add_argument("--rounds", type=int, default=9,
                    help="round 0 runs 334-613x the warm rounds; 9 is what the "
                         "prior campaigns settled on")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not (args.gate or args.bw or args.hlo or args.decomp or args.passes
            or args.ab or args.eq):
        args.gate = args.bw = args.hlo = args.decomp = True
    # `--eq` is a correctness check, so it participates in GATE. Without this,
    # a job that ran only `--eq --ab` would print `GATE: NOT-RUN` while in fact
    # having checked the one thing that matters for the candidates.
    gate_ran = args.gate or args.eq

    info = pltpu.get_tpu_info()
    print(f"host={socket.gethostname()} devices={jax.device_count()} "
          f"kind={jax.devices()[0].device_kind}")
    print(f"sparse_core={info.sparse_core}")
    print(f"shapes: short ctx {CTX_SHORT} -> n={topk_n(CTX_SHORT)}, "
          f"128K ctx {CTX_128K} -> n={topk_n(CTX_128K)}")
    for b in args.batches:
        print(f"  b={b}: {describe(b, topk_n(args.ctx))}")

    out = {"ctx": args.ctx, "n": topk_n(args.ctx),
           "shapes": {str(b): describe(b, topk_n(args.ctx))
                      for b in args.batches}}
    ok = True

    if args.gate:
        if args.also_short:
            short_ok, out["gate_short"] = gate(CTX_SHORT, tuple(args.batches),
                                               args.verbose)
            ok &= short_ok
        out["gate"] = {}
        for v in args.gate_variants:
            g_ok, out["gate"][v] = gate(args.ctx, tuple(args.batches),
                                        args.verbose, variant=v)
            print(f"  variant {v}: {'ok' if g_ok else 'FAIL'}", flush=True)
            ok &= g_ok

    if args.eq:
        try:
            eq_ok, out["equality"] = variant_equality(args.ctx,
                                                      tuple(args.batches))
        except Exception as e:  # noqa: BLE001
            # A candidate that will not compile is the finding -- a SparseCore
            # output-geometry change is the declared risk of this step -- but
            # it must not be allowed to look like a pass.
            traceback.print_exc()
            out["equality"] = {"error": repr(e)}
            eq_ok = False
        ok &= eq_ok

    if args.hlo:
        # Deliberately does NOT flip GATE. An absent gather says candidate A
        # has no subject; it says nothing about whether the two-stage path is
        # correct, and conflating the two would make a design finding look like
        # a correctness failure.
        try:
            out["hlo"] = {
                str(b): {v: hlo_check(args.ctx, b, variant=v)
                         for v in args.hlo_variants}
                for b in args.batches
            }
            # The point of compiling every variant: a flag whose whole claim is
            # "this deletes a gather" is checkable directly, and cheaply. Print
            # the comparison rather than leaving four dicts for a reader to
            # diff by eye.
            for b in args.batches:
                row = out["hlo"][str(b)]
                if "base" not in row:
                    continue
                nb = row["base"]["n_full_row_gathers"]
                tb = row["base"]["n_gather"]
                print(f"\n  gather delta vs base at b={b} "
                      f"(base: {tb} gathers, {nb} reading f32[{b},{topk_n(args.ctx)}]):")
                for v, d in row.items():
                    if v == "base":
                        continue
                    print(f"    {v:<6} {d['n_gather']:2d} gathers "
                          f"({d['n_gather'] - tb:+d}), "
                          f"{d['n_full_row_gathers']} full-row "
                          f"({d['n_full_row_gathers'] - nb:+d})")
                if "A" in row and row["A"]["n_full_row_gathers"] >= nb:
                    print("    !! emit_pairs did NOT remove the full-row "
                          "gather. Whatever the A/B measured, it is not the "
                          "mechanism A is credited with.")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            out["hlo"] = {"error": repr(e)}

    if args.decomp:
        # Also does not flip GATE: this prices the candidates, it does not
        # check correctness.
        out["decompose"] = {}
        for b in args.batches:
            try:
                out["decompose"][str(b)] = decompose(args.ctx, b, args.rounds)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                out["decompose"][str(b)] = {"error": repr(e)}

    if args.passes:
        out["passes"] = {}
        for b in args.batches:
            try:
                out["passes"][str(b)] = passes(args.ctx, b, args.rounds)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                out["passes"][str(b)] = {"error": repr(e)}

    if args.ab:
        out["ab"] = {}
        for b in args.batches:
            try:
                out["ab"][str(b)] = ab(args.ctx, b, args.rounds)
            except Exception as e:  # noqa: BLE001
                # A candidate that will not compile is the finding -- the
                # SparseCore output-geometry change is the declared risk here.
                traceback.print_exc()
                out["ab"][str(b)] = {"error": repr(e)}

    if args.bw:
        try:
            out["bandwidth"] = bandwidth(args.ctx, args.rounds)
        except Exception as e:  # noqa: BLE001
            # A probe that cannot compile is a finding, not a crash -- but it
            # must not be allowed to look like a pass.
            traceback.print_exc()
            out["bandwidth"] = {"error": repr(e)}
            ok = False

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json}")

    # Own line, greppable, and not a comment: cdk folds RUN_CMD to a single
    # line, so a `#`-guarded `exit 1` silently vanishes (job j-3f04af6b returned
    # Succeeded with three deadlocked combos that way).
    #
    # NOT-RUN is a distinct state from PASS, and conflating them is how this
    # line was wrong: with `--decomp --hlo` and no `--gate`, `ok` was never
    # touched and the script printed a confident `GATE: PASS` for a job in
    # which no gate had executed (job `j-2a01fc53`). A greppable success string
    # that prints when the check did not run is worse than no string at all --
    # it is exactly the unfalsifiable pass this file was written to avoid.
    status = ("PASS" if ok else "FAIL") if gate_ran else "NOT-RUN"
    print(f"\nGATE: {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
