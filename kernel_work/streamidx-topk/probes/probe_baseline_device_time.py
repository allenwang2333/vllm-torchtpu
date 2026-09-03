#!/usr/bin/env python3
"""The corrected device-time baseline of record, plus candidate 0's `bkv_p`.

Why this exists
---------------

`measurement_and_design.md` §1.1 pins the baseline of record at **221.4 us
decode / 857.3 us prefill**, read out of `bench_prod.json` in job
`j-8b1e7acc-ebdc-45d2-b46c`.  Both are **wall clocks**, and
`probes/stage-b-ksweep.md` §0 established that wall clock on this harness is
not device time: `instruments/bench_streamidx_topk.py`'s `time_arm` enqueues
`iters` calls asynchronously and blocks **once**, so the per-call figure is
`max(device, host enqueue)` plus whatever share of the first call's round trip
`iters` failed to amortise.  A host dispatch floor under a batched async
measurement is therefore a **clamp, not an addend** -- subtracting it is
invalid, and any op whose device time falls below it reads the floor instead
of itself.

Two components have already been re-measured as device time, and both came in
far below their recorded wall clock:

  * stage B at prefill: 481.0 us wall -> **335.5 us** device (chain slope,
    `probes/stage-b-ksweep.md` §1);
  * the whole decode call: 221.4 us wall -> **170.5 us** device (`hlo_stats`,
    `probes/R2.md` §1, job `j-7cd67df2`).

Nobody has measured the **prefill call's** device time.  That is the single
largest open uncertainty in this exercise, because every "achieved fraction of
the modelled floor" divides by these two numbers, and the prefill gate
currently sits anywhere in 43 %-63 %.

What this probe measures
------------------------

Device time for the full `streamidx_topk` op, and for stage A and stage B
separately, at both production shapes -- by **two independent methods**:

  1. **Chain slope** (this file).  R calls chained inside one `jax.jit` with a
     real data dependency, so R kernel executions cost one dispatch; the slope
     of `t(R)` is marginal device time per call with the host term differenced
     out, and the intercept is what that host term was.  Every fit reports
     r^2 -- the k-sweep probe's fits returned r^2 >= 0.9997, and anything
     below 0.999 is rejected -- and a **chain-integrity check**: the count of
     `StreamIdxTC` / `sc_topk` custom calls in the optimized HLO must grow
     with R, or the compiler collapsed the chain and the slope is meaningless.
     r^2 alone would not notice a collapse.
  2. **`hlo_stats`** per-instruction device self time, produced by
     `probe_bdt_trace.py` in a separate process (it needs the jax 0.11 side
     venv) and read off the pod afterwards.  It carries no dispatch floor at
     all, and R2's two arms on separate chips agreed to 0.02 %.

The two are reported side by side in `baseline-device-time.md`.  If they
disagree by more than a few percent, that disagreement is the result.

The host dispatch floor is also measured in-process, in both the arities
`bench_streamidx_topk.py` uses, purely so the reader can see what it was.
**It is used for nothing and is never subtracted.**

Candidate 0
-----------

`num_kv_pages_per_block` (`bkv_p`) is 2 at decode.  P7a measured 2 -> 3 at
-11.9 % of the decode call, but that was a contended four-shard fan-out and
ratios only, and P7a §A4.3 flags that `bkv_p = 3` may be exploiting
`align_to(9, 3) == 9` at `pages_per_seq = 9`.  Here it is swept over
`bkv_p in {2,3,4}` x `ctx in {8192, 8704, 9216}` in **device time**, in the
same process as the baseline above, against its own controls.  The incumbent
`bkv_p = 2` is one of the arms and is allowed to win.

Note what does and does not vary across that ctx sweep, because it decides
what the sweep can prove.  `pages_per_seq = cdiv(max_model_len 9216, 1024) = 9`
at all three, so `align_to(9, bkv_p)` -- the alignment P7a suspected -- is
**constant**.  What moves is `num_bkv = cdiv(ctx, bkv_p*1024)` and hence the
pages actually fetched per sequence:

    ctx    bkv_p=2         bkv_p=3        bkv_p=4
    8192   4 blk /  8 pg   3 blk / 9 pg   2 blk /  8 pg
    8704   5 blk / 10 pg   3 blk / 9 pg   3 blk / 12 pg
    9216   5 blk / 10 pg   3 blk / 9 pg   3 blk / 12 pg

So at 8704/9216 `bkv_p = 3` both issues 40 % fewer blocks *and* fetches fewer
pages, while at 8192 it issues only 25 % fewer and fetches *more*.  If the
-11.9 % is a `9 = 3x3` effect it should shrink or vanish at ctx 8192.

Two confounds are handled explicitly:

  * **The score matrix width moves with `bkv_p`.**
    `num_sublanes_total = max(align_to(pages_per_seq, p) * page_size // 128
    for p in num_kv_pages_per_block)`, so with the tuple `(bkv_p, 2, 2)` it is
    80 sublanes (10240 positions) for `bkv_p in {2,3}` but **96 (12288)** for
    `bkv_p = 4`.  P7a had to exclude `bkv_p = 4` for exactly this reason.
    Here the **isolated stage-A launcher** (`_stage_a.make_stage_a_fn`) takes
    `num_sublanes_total` as an explicit argument, so all three `bkv_p` are
    also measured at a common width -- the first like-for-like `bkv_p = 4`
    number in this exercise.  96 is the smallest width that holds every arm:
    the kernel runs with `disable_bounds_checks=True`, so a width below
    `num_bkv * bkv_sz // 128` would write past the buffer silently.
  * **Correctness.**  The three `bkv_p` must produce the same top-k index
    set.  §2 checks that directly at every (bkv_p, ctx) point, before any
    timing is quoted.  It is a supplement to, not a substitute for,
    `tests/kernels/test_dsv4_streamindex_topk.py` -- that file parametrises
    `bkv_p` only where `page_size * bkv_p == 128`, so running it unmodified
    does not exercise production `bkv_p = 3` at page size 1024.

Nothing here is an end-to-end model benchmark.  Every timing is a
microbenchmark, run as a **single unpinned process** on the 4-chip pod rather
than fanned out, because these absolutes have to be comparable with
`bench_prod.json` and a fanned-out wall clock is contended.

Run:  python probe_baseline_device_time.py [--quick] [--skip a,b]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys
import time
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
AP.add_argument("--quick", action="store_true",
                help="fewer trials / shorter chains; a smoke test, not a result")
AP.add_argument("--skip", default="", help="comma-separated sections to skip")
AP.add_argument("--name", default="baseline-device-time")
ARGS = AP.parse_args()

TRIALS = 3 if ARGS.quick else 9
WARMUP = 2 if ARGS.quick else 3
TARGET_S = 0.02 if ARGS.quick else 0.05
CHAIN_R = (1, 2) if ARGS.quick else (1, 2, 3, 4)
CHAIN_R_GRID = (1, 2) if ARGS.quick else (1, 2, 3)
R2_REJECT = 0.999

SKIP = {s for s in ARGS.skip.split(",") if s}
CTX_SWEEP = (8192, 8704, 9216)
BKVP_SWEEP = (2, 3, 4)

RESULTS: dict = {
    "methodology": {
        "trials": TRIALS, "warmup": WARMUP, "target_s": TARGET_S,
        "chain_r": list(CHAIN_R), "chain_r_grid": list(CHAIN_R_GRID),
        "quick": ARGS.quick, "r2_reject_below": R2_REJECT,
        "wall_statistic": "best-of-N batched async (bench_prod's own method)",
        "device_statistic": "chain slope: d t / d R, intercept discarded",
        "note": ("batched per-call wall = max(device, host enqueue); the "
                 "dispatch floor is a clamp, not an addend, and is never "
                 "subtracted anywhere in this file."),
    },
    "baseline_of_record_wall_us": {
        "source": "bench_prod.json, job j-8b1e7acc-ebdc-45d2-b46c",
        "decode_call": 221.4, "prefill_call": 857.3,
        "decode_stage_b": 67.4, "prefill_stage_b": 481.0,
        "decode_stage_a_derived": 109.2, "prefill_stage_a_derived": 335.1,
    },
}


def flush(tag: str) -> None:
    emit_json(ARGS.name, RESULTS)
    p("emitted", tag)


# ---------------------------------------------------------------------------
# chain machinery
# ---------------------------------------------------------------------------

def zero_dep(idx_scalar):
    """An int32 zero XLA will not fold away.

    `min(x, 0)` where `x` is a top-k index.  Every row this probe chains
    through has an effective length far above `k = 2048`, so `out[0, 0]` is a
    real index >= 0 and the minimum is identically 0 -- but XLA cannot prove
    the sign, so it can neither constant-fold the link nor CSE the R copies
    of the call.  `x * 0` *is* folded by the algebraic simplifier and is
    deliberately not used.  §2 verifies numerically that a 2-link chain
    returns exactly what a 1-link chain returns, which is what makes
    "identically 0" a measurement rather than an argument.
    """
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


def chain_slope(build, args, label, rs=None, marker=None):
    """Fit t(R) = intercept + slope * R, with a chain-integrity guard.

    The guard costs a second compile of the arm (`lower().compile()` does not
    hit the tracing cache `time_arm` warmed), so it is taken only at the
    **endpoints** of `rs` -- enough to show the custom-call count scales --
    and it is switched off entirely for the Task 2 grid, whose chains are
    built by the very same `full_chain` / `stage_a_chain` closures that Task 1
    verifies.  Where the guard is off, a collapse would still show as a flat
    `points_us`, which is always reported.
    """
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
            except Exception as e:  # a guard must never kill the probe
                counts[r] = f"ERR {e!r}"
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
             f"markers={counts} intact={ok}")
    return out


def time_serial(fn, args):
    jax.block_until_ready(fn(*args))
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    per = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        per.append(time.perf_counter() - t0)
    return min(per) * 1e6


def wall_statistics(fn, args, label):
    """`bench_prod`'s own statistic, plus the serial one, for comparison."""
    t = time_arm(fn, args, WARMUP, TRIALS, TARGET_S)
    row = {
        "batched_best_us": t["best_s"] * 1e6,
        "batched_median_us": t["median_s"] * 1e6,
        "batched_iters": t["iters"],
        "trial_spread_frac": (max(t["trials_s"]) - min(t["trials_s"])) /
                             t["best_s"],
        "serial_best_us": time_serial(fn, args),
    }
    p(label, f"batched={row['batched_best_us']:.2f}us "
             f"(iters={row['batched_iters']}, "
             f"spread={row['trial_spread_frac'] * 100:.1f}%) "
             f"serial={row['serial_best_us']:.2f}us")
    return row


def cdiv(a, b):
    return -(-a // b)


# ---------------------------------------------------------------------------

def main():  # noqa: C901  -- a probe is a script, not an abstraction
    tpu_banner()
    import bench_streamidx_topk as B
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk as sxmod
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import \
        sparsecore_topk
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        MlaCase, prepare_index_weights, prepare_q_inputs, streamindex_topk)
    import _stage_a as SA

    # md5 read-back. A --map-dir that did not take is otherwise silent, and an
    # earlier job in this exercise measured the image's older kernel behind a
    # green JobStatus.
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
    RESULTS["static_common"] = {k: (list(v) if isinstance(v, tuple) else v)
                                for k, v in static_common.items()}

    dec = shapes["prod_decode_b64_ctx8704"]
    pre = shapes["prod_prefill_q1024_ctx9216"]
    gdec, gpre = B.geometry(dec), B.geometry(pre)
    idec = B.build_inputs(dec, gdec, seed=0)
    ipre = B.build_inputs(pre, gpre, seed=0)
    RESULTS["geometry"] = {
        "decode": dataclasses.asdict(gdec),
        "prefill": dataclasses.asdict(gpre),
        "page_size": B.PAGE_SIZE,
        "num_kv_pages_per_block": list(B.NUM_KV_PAGES_PER_BLOCK),
        "num_queries_per_block": list(B.NUM_QUERIES_PER_BLOCK),
    }
    p("geometry decode", RESULTS["geometry"]["decode"])
    p("geometry prefill", RESULTS["geometry"]["prefill"])
    flush("geometry")

    ARGNAMES = ("q", "indexer_weights", "cache_kv", "seq_lens",
                "page_indices", "cu_q_lens", "distribution")

    def op_args(inputs, ctx=None):
        """Positional args for the full op, optionally at another ctx.

        Overriding `seq_lens` is exactly equivalent to rebuilding at the other
        `prod_decode_b64_ctx*` shape: the block table is sized by
        `max_model_len`, so `pages_per_seq`, `num_sublanes_total` and every
        array are bit-identical at the same seed and only `seq_lens` moves.
        Sharing them removes the PRNG as a between-arm variable.
        """
        args = [inputs[n] for n in ARGNAMES]
        if ctx is not None:
            args[3] = jnp.full_like(inputs["seq_lens"], ctx)
        return tuple(args)

    def full_chain(bkv_tuple):
        def build(r):
            def f(q, w, ckv, sl, pi, cql, dist):
                cur, out = sl, None
                for _ in range(r):
                    out = streamindex_topk(
                        q, w, ckv, cur, pi, cql, dist,
                        num_kv_pages_per_block=bkv_tuple, **static_common)
                    cur = sl + zero_dep(out[0, 0])
                return out
            return jax.jit(f)
        return build

    def stage_a_chain(phase, inputs, geo, bkv_p, nst, ctx=None):
        """Isolated stage A, chained through its own aliased score buffer.

        The real op chains its three `run_scores_kernel` calls exactly this
        way -- each takes the previous one's buffer as `scores_init` and
        aliases it -- so this is the incumbent's own data dependency, not an
        artificial one, and it inserts no copy into the slope.
        """
        if phase == "decode":
            fn = SA.make_stage_a_fn(
                num_q_heads=B.H_IDX, head_dim=B.D_IDX,
                page_size_per_kv_packing=B.PAGE_SIZE // B.KV_PACKING,
                kv_packing=B.KV_PACKING, lkv_dim=B.LKV_DIM,
                num_sublanes_total=nst, num_kv_pages_per_block=bkv_p,
                num_queries_per_block=B.NUM_QUERIES_PER_BLOCK[0],
                static_q_len=1, seq_batch_size=4, case=MlaCase.DECODE,
                compression_ratio=B.COMPRESSION_RATIO)
            start, end = 0, geo.num_tokens        # grid = 64 // 4 = 16
        else:
            fn = SA.make_stage_a_fn(
                num_q_heads=B.H_IDX, head_dim=B.D_IDX,
                page_size_per_kv_packing=B.PAGE_SIZE // B.KV_PACKING,
                kv_packing=B.KV_PACKING, lkv_dim=B.LKV_DIM,
                num_sublanes_total=nst, num_kv_pages_per_block=bkv_p,
                num_queries_per_block=B.NUM_QUERIES_PER_BLOCK[2],
                static_q_len=None, seq_batch_size=1, case=MlaCase.MIXED,
                compression_ratio=B.COMPRESSION_RATIO)
            start, end = 0, 1                      # grid = 1
        ntok = geo.num_tokens

        def build(r):
            def f(q, w, ckv, sl, pi, cql):
                cur = jnp.full((ntok, nst, 128), -jnp.inf, jnp.float32)
                for _ in range(r):
                    cur = fn(q, w, ckv, cur, sl, pi, cql,
                             jnp.int32(start), jnp.int32(end))
                return cur
            return jax.jit(f)

        args = (prepare_q_inputs(inputs["q"]),
                prepare_index_weights(inputs["indexer_weights"],
                                      inputs["q"].dtype),
                inputs["cache_kv"],
                (jnp.full_like(inputs["seq_lens"], ctx) if ctx is not None
                 else inputs["seq_lens"]),
                inputs["page_indices"], inputs["cu_q_lens"])
        return build, args

    def stage_b_chain(shape, geo, inputs):
        sc, rl = B.build_stage_b_scores(shape, geo, inputs, seed=1)
        sc = jax.block_until_ready(sc)

        def one(s, r_):
            return sparsecore_topk(s, B.K_IDX, row_lengths=r_,
                                   write_empty_rows=True)

        def build(r):
            def f(s, r_):
                cur, out = r_, None
                for _ in range(r):
                    out = one(s, cur)
                    cur = r_ + zero_dep(out[0, 0])
                return out
            return jax.jit(f)
        return build, (sc, rl), one

    # =====================================================================
    # 0. shape preflight.  `jax.eval_shape` traces every arm without
    #    compiling or running it, so a wrong argument order or a mis-sized
    #    buffer surfaces in seconds rather than after an hour of compiles.
    #    Systematic breakage is the failure mode that would waste the whole
    #    job; the per-arm try/except below only contains local breakage.
    # =====================================================================
    pf = {}
    try:
        pf["full_decode"] = str(jax.eval_shape(
            full_chain(B.NUM_KV_PAGES_PER_BLOCK)(1), *op_args(idec)))
        pf["full_prefill"] = str(jax.eval_shape(
            full_chain(B.NUM_KV_PAGES_PER_BLOCK)(1), *op_args(ipre)))
        for ph, inp, g in (("decode", idec, gdec), ("prefill", ipre, gpre)):
            bp = (B.NUM_KV_PAGES_PER_BLOCK[0] if ph == "decode"
                  else B.NUM_KV_PAGES_PER_BLOCK[2])
            b_, a_ = stage_a_chain(ph, inp, g, bp, g.num_sublanes_total)
            pf[f"stage_a_{ph}"] = str(jax.eval_shape(b_(1), *a_))
        for ph, sh, g, inp in (("decode", dec, gdec, idec),
                               ("prefill", pre, gpre, ipre)):
            b_, a_, _ = stage_b_chain(sh, g, inp)
            pf[f"stage_b_{ph}"] = str(jax.eval_shape(b_(1), *a_))
        b_, a_ = stage_a_chain("decode", idec, gdec, 4, 96, ctx=9216)
        pf["stage_a_decode_bkvp4_nst96"] = str(jax.eval_shape(b_(1), *a_))
        pf["ok"] = True
    except Exception as e:
        pf["ok"] = False
        pf["error"] = repr(e)
        traceback.print_exc()
    RESULTS["shape_preflight"] = pf
    p("shape_preflight", pf)
    flush("preflight")

    # =====================================================================
    # 1. host dispatch floors, in-process.  CONTEXT ONLY -- never subtracted.
    # =====================================================================
    if "floors" not in SKIP:
        try:
            f1 = jax.jit(lambda x: x + 1.0)
            t1 = time_arm(f1, (jnp.asarray(1.0, jnp.float32), ),
                          WARMUP, TRIALS, TARGET_S)
            sc, rl = B.build_stage_b_scores(dec, gdec, idec, seed=1)
            f2 = jax.jit(lambda s, r_: r_[:16] + s[0, 0].astype(jnp.int32))
            t2 = time_arm(f2, (sc, rl), WARMUP, TRIALS, TARGET_S)
            RESULTS["dispatch_floors"] = {
                "one_arg_scalar_us": t1["best_s"] * 1e6,
                "one_arg_iters": t1["iters"],
                "two_arg_bigbuf_us": t2["best_s"] * 1e6,
                "two_arg_iters": t2["iters"],
                "bigbuf_bytes": int(np.prod(sc.shape)) * 4,
                "bench_prod_recorded_us": 42.005,
                "used_for": "NOTHING. Context only. A clamp, not an addend.",
            }
            p("dispatch_floors", RESULTS["dispatch_floors"])
            del sc, rl
        except Exception as e:
            RESULTS["dispatch_floors"] = {"error": repr(e)}
            traceback.print_exc()
        flush("floors")

    # =====================================================================
    # 2. correctness: the three bkv_p must agree on the top-k index SET,
    #    and the chain perturbation must be provably zero.
    # =====================================================================
    if "correctness" not in SKIP:
        corr = []
        for ctx in CTX_SWEEP:
            args = op_args(idec, ctx=ctx)
            ref = None
            for bp in BKVP_SWEEP:
                row = {"ctx": ctx, "bkv_p": bp}
                try:
                    tup = (bp, 2, 2)

                    def call(*a, _t=tup):
                        return streamindex_topk(
                            *a, num_kv_pages_per_block=_t, **static_common)

                    out = np.asarray(jax.block_until_ready(
                        jax.jit(call)(*args)))
                    sets = [frozenset(int(v) for v in r_ if v >= 0)
                            for r_ in out]
                    row.update(shape=list(out.shape), out_0_0=int(out[0, 0]),
                               n_pad=int((out < 0).sum()),
                               min_idx=int(out.min()), max_idx=int(out.max()))
                    if ref is None:
                        ref = sets
                        row.update(rows_matching_bkvp2=len(sets),
                                   index_sets_match_bkvp2=True)
                    else:
                        n_ok = sum(1 for a, b in zip(ref, sets) if a == b)
                        row.update(rows_matching_bkvp2=n_ok,
                                   index_sets_match_bkvp2=(n_ok == len(ref)))
                    if bp == 2:
                        bld = full_chain(tup)
                        o1 = np.asarray(jax.block_until_ready(bld(1)(*args)))
                        o2 = np.asarray(jax.block_until_ready(bld(2)(*args)))
                        row["chain_perturbation_is_zero"] = bool(
                            np.array_equal(o1, o2))
                except Exception as e:
                    row["error"] = repr(e)
                    traceback.print_exc()
                corr.append(row)
                p("correctness", row)
        RESULTS["bkvp_index_set_equality"] = corr
        flush("correctness")

    # =====================================================================
    # 3. Task 1 -- the baseline of record, restated in device time
    # =====================================================================
    for phase, shape, geo, inputs in (("decode", dec, gdec, idec),
                                      ("prefill", pre, gpre, ipre)):
        if f"task1_{phase}" in SKIP:
            continue
        sect = RESULTS.setdefault("task1", {}).setdefault(phase, {})
        sect["shape"] = shape.name

        try:
            args = op_args(inputs)
            bld = full_chain(B.NUM_KV_PAGES_PER_BLOCK)
            sect["full_op"] = chain_slope(bld, args, f"{phase}/full/chain",
                                          marker="StreamIdxTC")
            sect["full_op"].update(wall_statistics(bld(1), args,
                                                   f"{phase}/full/wall"))
        except Exception as e:
            sect["full_op"] = {"error": repr(e)}
            traceback.print_exc()
        flush(f"task1/{phase}/full")

        try:
            nst = geo.num_sublanes_total
            bkv_p = (B.NUM_KV_PAGES_PER_BLOCK[0] if phase == "decode"
                     else B.NUM_KV_PAGES_PER_BLOCK[2])
            bld, args = stage_a_chain(phase, inputs, geo, bkv_p, nst)
            sect["stage_a_isolated"] = chain_slope(
                bld, args, f"{phase}/stageA/chain", marker="StreamIdxTC")
            sect["stage_a_isolated"].update(
                wall_statistics(bld(1), args, f"{phase}/stageA/wall"))
            sect["stage_a_isolated"].update(
                num_sublanes_total=nst, bkv_p=bkv_p,
                note=("ONE pallas_call: the grouped DECODE call, or the "
                      "MIXED prefill call. The two companion calls the real "
                      "op also issues are empty at these shapes and are NOT "
                      "included here; hlo_stats does include them."))
        except Exception as e:
            sect["stage_a_isolated"] = {"error": repr(e)}
            traceback.print_exc()
        flush(f"task1/{phase}/stageA")

        try:
            bld, args, one = stage_b_chain(shape, geo, inputs)
            sect["stage_b_isolated"] = chain_slope(
                bld, args, f"{phase}/stageB/chain", marker="sc_topk")
            sect["stage_b_isolated"].update(
                wall_statistics(jax.jit(one), args, f"{phase}/stageB/wall"))
            sect["stage_b_isolated"].update(
                p_partition=int(scmod._pick_partition(
                    geo.num_tokens, geo.num_positions, B.K_IDX)),
                eff_len_mean=float(np.asarray(args[1]).mean()),
                num_positions=geo.num_positions)
            del bld, args, one
        except Exception as e:
            sect["stage_b_isolated"] = {"error": repr(e)}
            traceback.print_exc()
        flush(f"task1/{phase}/stageB")

    # =====================================================================
    # 4. Candidate 0 -- bkv_p x ctx, device time, same process
    # =====================================================================
    if "task2" not in SKIP:
        grid = []
        for ctx in CTX_SWEEP:
            args = op_args(idec, ctx=ctx)
            for bp in BKVP_SWEEP:
                tup = (bp, 2, 2)
                nst = max(B.align_to(gdec.pages_per_seq, x) * B.PAGE_SIZE //
                          128 for x in tup)
                nbkv = max(1, cdiv(ctx, bp * B.PAGE_SIZE))
                row = {
                    "ctx": ctx, "bkv_p": bp, "tuple": list(tup),
                    "num_sublanes_total": nst, "num_positions": nst * 128,
                    "like_for_like_with_bkvp2":
                        nst == gdec.num_sublanes_total,
                    "num_bkv": nbkv, "pages_fetched_per_seq": nbkv * bp,
                }
                try:
                    bld = full_chain(tup)
                    row["full_op"] = chain_slope(
                        bld, args, f"task2/full/ctx{ctx}/bkvp{bp}",
                        rs=CHAIN_R_GRID, marker=None)
                    row["full_op"].update(wall_statistics(
                        bld(1), args, f"task2/full/ctx{ctx}/bkvp{bp}/wall"))
                except Exception as e:
                    row["full_op"] = {"error": repr(e)}
                    traceback.print_exc()
                grid.append(row)
                RESULTS["task2_full_op_grid"] = grid
                flush(f"task2/full/ctx{ctx}/bkvp{bp}")

        # Stage A alone at a COMMON num_sublanes_total, so bkv_p = 4 is
        # like-for-like too -- the comparison P7a could not make. Width 96
        # holds every arm; width 80 is the incumbent's own and is run for the
        # arms that fit, so the two widths can be differenced.
        agrid = []
        for nst in (96, 80):
            for ctx in CTX_SWEEP:
                for bp in BKVP_SWEEP:
                    need = max(1, cdiv(ctx, bp * B.PAGE_SIZE)) * (
                        bp * B.PAGE_SIZE // 128)
                    if need > nst:
                        continue  # would write past the buffer: no bounds checks
                    row = {"num_sublanes_total": nst, "ctx": ctx,
                           "bkv_p": bp, "sublanes_written": need}
                    try:
                        bld, args = stage_a_chain("decode", idec, gdec, bp,
                                                  nst, ctx=ctx)
                        row["chain"] = chain_slope(
                            bld, args,
                            f"task2/stageA/nst{nst}/ctx{ctx}/bkvp{bp}",
                            rs=CHAIN_R_GRID, marker=None)
                    except Exception as e:
                        row["chain"] = {"error": repr(e)}
                        traceback.print_exc()
                    agrid.append(row)
                    RESULTS["task2_stage_a_grid"] = agrid
                    flush(f"task2/stageA/nst{nst}/ctx{ctx}/bkvp{bp}")

    flush("done")
    p("done", "all sections complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
