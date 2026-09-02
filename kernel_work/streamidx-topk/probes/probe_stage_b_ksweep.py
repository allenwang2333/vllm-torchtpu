#!/usr/bin/env python3
"""Stage B (`sparsecore_topk`) -- the `k` sweep, the `b` sweep, and a device
time that is not a wall clock.

Three things this exists to settle.

**1. Is prefill stage B vector-issue-bound or output-DMA-bound?**
`briefs/area5-compute-units.md` names the discriminator: hold `b`, `n` and
`row_lengths` fixed and sweep `k` alone.  Reading
`sparsecore_topk.py:154-291`, every `plsc.parallel_loop` -- `cvt`, `hist0_body`,
`ref_body`, `emit_body` -- is bounded by `walk_chunks = cdiv(min(slice_len,
eff_len), 16)`, which does **not** depend on `k`.  The only `k`-dependent costs
in the kernel are

  * `fill_body`, gated on `total < k` -- dead at these shapes, since
    `eff_len >= 8193 > k` for every `k` swept;
  * the final `dma(rowbuf[0:k] -> out_hbm[row*k : +k])`, i.e. `4*k` bytes per
    row;
  * `rowbuf_vmem`'s size, `k + 16` words -- an allocation, not an issue count;
  * how early the three refinement levels terminate, which is `k`- and
    data-dependent and is the one term that is neither flat nor linear.

So: time roughly flat in `k` => issue-bound.  Time roughly linear in `k` =>
output-DMA-bound.

**2. Why does b=8 cost more than b=64?**  `bench_prod.json` has
`prod_decode_b8_ctx8704` at 90.30 us against `prod_decode_b64_ctx8704` at
67.40 us -- eight times less work costing 34% more time.  Reading
`_pick_partition(b, n, k)` (`sparsecore_topk.py:357-372`) predicts the answer
before anything runs: at `n=10240, k=2048` it returns **p=4 for b=8**, **p=2
for b=16**, and **p=1 for every b >= 32**.  p>1 is the two-stage path: two
`_sc_topk_direct` launches plus two `jnp.take_along_axis` gathers on the
critical path.  The `direct_p1` arms below force the single-stage kernel at the
same `b` so the tax can be priced rather than inferred.

**3. What is stage B's device time, separated from host dispatch?**
`instruments/bench_streamidx_topk.py:814` measures a dispatch floor with a
trivial jitted op and `bench_prod.json` records 42.0 us.  §C.7 of the area-5
brief then *subtracts* ~35 us from a 67.4 us measurement to get "30 us device".
**Subtraction is the wrong operation.**  `time_arm` enqueues `iters` calls
asynchronously and blocks once at the end, so the per-call figure it reports is
`max(device_time, host_enqueue_cost)`, not their sum.  A floor under a batched
pipelined measurement is a *clamp*, not an addend.

This probe therefore measures device time a third way that needs no floor at
all: **chain R calls inside one `jax.jit`**, so R kernel executions cost one
dispatch, and take the slope of `t_total(R)` over `R in {1,2,3,4}`.  The slope
is marginal device time per call with the host term differenced out; the
intercept is what the host term actually was.  The chain's data dependency is
`rl + min(out[0,0], 0)` -- identically zero, but XLA's algebraic simplifier
cannot fold `min(x, 0)` the way it folds `x * 0`, so the R calls cannot be CSEd
or reordered.

Every timing here is a **best-of-N host wall clock** on a pod that other agents
may be using.  Best-of-N is optimistic by construction and is a lower bound.
Run unpinned (whole pod, one process) rather than fanned out, because the
absolute microseconds have to be comparable with `bench_prod.json`.

Run:  python probe_stage_b_ksweep.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _probe_util import emit_json, p, time_arm, tpu_banner  # noqa: E402

N = 10240  # num_positions = align_to(9, 2) * 1024, pinned by max_model_len
CTX = 8704  # the decode row report.md models

# k must be a positive multiple of 16 and <= 4096 (sparsecore_topk.py:400).
K_SWEEP = (4096, 2048, 1024, 512, 256)
B_SWEEP = (8, 16, 32, 64, 128, 256, 512, 1024)
CHAIN_R = (1, 2, 3, 4)

TRIALS = 9  # >= 7, per the caller's spec; pod spread is +-8.5%
WARMUP = 3
TARGET_S = 0.05


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
def prefill_row_lengths(q_len: int, ctx: int, num_positions: int):
    """`_effective_row_lengths` for one prefill sequence, compression_ratio=1.

    Reproduces `instruments/bench_streamidx_topk.py:_row_lengths_ref` for the
    `prod_prefill_q1024_ctx9216` row: num_seqs=1, q_len=1024, seq_len=9216, so
    token_pos = 9216-1024+i and visible = min(9216, token_pos+1), i.e.
    8193..9216 with mean 8704.5.  That mean is the `eff_len ~ 8704` the brief's
    1.7 ns/elem constant was fitted at, which is why the number is worth
    reproducing exactly rather than approximating with a constant.
    """
    i = jnp.arange(q_len, dtype=jnp.int32)
    token_pos = ctx - q_len + i
    return jnp.minimum(jnp.minimum(ctx, token_pos + 1), num_positions)


def build_scores(b: int, n: int, row_lengths, seed: int = 1):
    """f32[b, n] N(0,1) with -inf at and past each row's visible length.

    Same construction as `build_stage_b_scores`: the *distribution* is
    synthetic and the radix select is data dependent, so it is held fixed
    across every arm rather than claimed to be what the indexer emits.
    """
    s = jax.random.normal(jax.random.key(seed), (b, n), jnp.float32)
    col = jnp.arange(n, dtype=jnp.int32)
    s = jnp.where(col[None, :] < row_lengths[:, None], s, -jnp.inf)
    return jax.block_until_ready(s)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_serial(fn, args, warmup=WARMUP, trials=TRIALS):
    """Best-of-N with a block after *every* call: device + full round trip."""
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    per = []
    for _ in range(trials):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        per.append(time.perf_counter() - t0)
    return min(per)


def dispatch_floors(scores, row_lengths):
    """Two enqueue floors, measured in this process with `time_arm`.

    `one_arg` reproduces `bench_streamidx_topk.py:measure_dispatch_floor`
    exactly (`jit(x + 1.0)` on a scalar), so it can be compared with
    `bench_prod.json`'s `dispatch_floor_ms = 0.042005`.  `two_arg_bigbuf` is
    the arity and the operand buffers stage B actually gets, with a tiny
    output, so it is the closer floor for stage B's own call.
    """
    f1 = jax.jit(lambda x: x + 1.0)
    x = jnp.asarray(1.0, jnp.float32)
    t1 = time_arm(f1, (x, ), WARMUP, TRIALS, TARGET_S)

    f2 = jax.jit(lambda s, rl: rl[:16] + s[0, 0].astype(jnp.int32))
    t2 = time_arm(f2, (scores, row_lengths), WARMUP, TRIALS, TARGET_S)
    return {
        "one_arg_scalar_us": t1["best_s"] * 1e6,
        "two_arg_bigbuf_us": t2["best_s"] * 1e6,
        "one_arg_iters": t1["iters"],
        "two_arg_iters": t2["iters"],
    }


def make_chain(topk_call, k: int, r: int):
    """R back-to-back stage-B calls in one jit, dependency-chained.

    `min(out[0,0], 0)` is 0 for every row whose eff_len >= k (all of them
    here, so the chain is numerically a no-op), but XLA cannot prove it and
    therefore cannot CSE or reorder the R launches.  `x * 0` would be folded
    away by the algebraic simplifier and is deliberately not used.
    """

    def f(scores, rl):
        cur = rl
        out = None
        for _ in range(r):
            out = topk_call(scores, k, cur)
            cur = rl + jnp.minimum(out[0, 0], jnp.int32(0))
        return out

    return jax.jit(f)


def chain_slope(topk_call, k, scores, rl):
    """Fit t(R) = intercept + slope*R over R in CHAIN_R.

    `slope` is marginal device time per stage-B call with the host dispatch
    differenced out; `intercept` is what that host term measured.  `r2` is
    reported so a non-linear chain (compiler fusing the links, or the pod
    interfering) is visible rather than silently averaged in.
    """
    pts = {}
    for r in CHAIN_R:
        fn = make_chain(topk_call, k, r)
        t = time_arm(fn, (scores, rl), WARMUP, TRIALS, TARGET_S)
        pts[r] = {"best_us": t["best_s"] * 1e6, "iters": t["iters"]}
    xs = np.array(CHAIN_R, float)
    ys = np.array([pts[r]["best_us"] for r in CHAIN_R], float)
    a = np.vstack([np.ones_like(xs), xs]).T
    (intercept, slope), *_ = np.linalg.lstsq(a, ys, rcond=None)
    pred = intercept + slope * xs
    ss_res = float(((ys - pred)**2).sum())
    ss_tot = float(((ys - ys.mean())**2).sum())
    return {
        "points": pts,
        "intercept_us": float(intercept),
        "slope_us_per_call": float(slope),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def measure_point(topk_call, label, b, n, k, scores, rl, do_chain=True):
    row = {"label": label, "b": b, "n": n, "k": k}
    try:
        fn = jax.jit(lambda s, r: topk_call(s, k, r))
        t = time_arm(fn, (scores, rl), WARMUP, TRIALS, TARGET_S)
        row["batched_best_us"] = t["best_s"] * 1e6
        row["batched_median_us"] = t["median_s"] * 1e6
        row["batched_iters"] = t["iters"]
        row["batched_trials_us"] = [x * 1e6 for x in t["trials_s"]]
        row["compile_s"] = t["compile_s"]
        spread = (max(t["trials_s"]) - min(t["trials_s"])) / t["best_s"]
        row["trial_spread_frac"] = spread
        row["serial_best_us"] = time_serial(fn, (scores, rl)) * 1e6
        if do_chain:
            row["chain"] = chain_slope(topk_call, k, scores, rl)
        p(label,
          f"batched={row['batched_best_us']:.2f}us "
          f"serial={row['serial_best_us']:.2f}us "
          + (f"chain_slope={row['chain']['slope_us_per_call']:.2f}us "
             f"(r2={row['chain']['r2']:.4f}, "
             f"intercept={row['chain']['intercept_us']:.2f}us) "
             if do_chain else "")
          + f"iters={row['batched_iters']} spread={spread*100:.1f}%")
    except Exception as e:
        row["error"] = repr(e)
        p(label, "FAILED", repr(e))
        traceback.print_exc()
    return row


# ---------------------------------------------------------------------------
def main():
    tpu_banner()
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    p("sparsecore_topk_module", scmod.__file__)

    def public(s, k, rl):
        return scmod.sparsecore_topk(s, k, row_lengths=rl,
                                     write_empty_rows=True)

    def direct(s, k, rl):
        return scmod._sc_topk_direct(s, k, rl, write_empty=True)

    results = {
        "n": N,
        "methodology": {
            "trials": TRIALS,
            "warmup": WARMUP,
            "target_s": TARGET_S,
            "chain_r": list(CHAIN_R),
            "statistic": "best-of-N host wall clock, batched (async enqueue)",
            "note": ("batched per-call = max(device, host enqueue); the "
                     "dispatch floor is a clamp, not an addend. chain slope "
                     "is the dispatch-free device time."),
        },
        "arms": {},
    }

    # ---- prefill k sweep -------------------------------------------------
    q_len, ctx_pf = 1024, 9216
    rl_pf = prefill_row_lengths(q_len, ctx_pf, N)
    rl_pf = jax.block_until_ready(rl_pf)
    scores_pf = build_scores(q_len, N, rl_pf)
    p("prefill row_lengths",
      f"b={q_len} min={int(rl_pf.min())} max={int(rl_pf.max())} "
      f"mean={float(rl_pf.mean()):.1f}")

    results["dispatch_floors"] = dispatch_floors(scores_pf, rl_pf)
    p("dispatch_floors", results["dispatch_floors"])

    pf = []
    for k in K_SWEEP:
        p("pick_partition",
          f"prefill b={q_len} n={N} k={k} -> p="
          f"{scmod._pick_partition(q_len, N, k)}")
        pf.append(
            measure_point(public, f"prefill_b{q_len}_k{k}", q_len, N, k,
                          scores_pf, rl_pf))
        pf[-1]["p"] = scmod._pick_partition(q_len, N, k)
        pf[-1]["num_waves"] = -(-q_len // 32)
        pf[-1]["eff_len_mean"] = float(rl_pf.mean())
    results["arms"]["prefill_k_sweep"] = pf

    # ---- decode k sweep --------------------------------------------------
    b_dec = 64
    rl_dec = jnp.full((b_dec, ), CTX, jnp.int32)
    scores_dec = build_scores(b_dec, N, rl_dec)
    dk = []
    for k in K_SWEEP:
        dk.append(
            measure_point(public, f"decode_b{b_dec}_k{k}", b_dec, N, k,
                          scores_dec, rl_dec))
        dk[-1]["p"] = scmod._pick_partition(b_dec, N, k)
        dk[-1]["num_waves"] = -(-b_dec // 32)
        dk[-1]["eff_len_mean"] = float(CTX)
    results["arms"]["decode_k_sweep"] = dk

    # ---- b sweep at fixed k ----------------------------------------------
    k_fix = 2048
    bs = []
    for b in B_SWEEP:
        rl = jnp.full((b, ), CTX, jnp.int32)
        sc = build_scores(b, N, rl)
        pp = scmod._pick_partition(b, N, k_fix)
        p("pick_partition", f"b={b} n={N} k={k_fix} -> p={pp}")
        row = measure_point(public, f"bsweep_b{b}_k{k_fix}_public", b, N,
                            k_fix, sc, rl)
        row["p"] = pp
        row["path"] = "single-stage" if pp == 1 else f"two-stage(p={pp})"
        row["num_waves_stage1"] = -(-(b * pp) // 32)
        bs.append(row)
        # The p=1 control: same b, same data, single-stage kernel forced.
        # Only informative where the public path chose p>1.
        if pp > 1:
            row2 = measure_point(direct, f"bsweep_b{b}_k{k_fix}_direct_p1", b,
                                 N, k_fix, sc, rl)
            row2["p"] = 1
            row2["path"] = "single-stage (forced, _sc_topk_direct)"
            row2["num_waves_stage1"] = -(-b // 32)
            bs.append(row2)
        del sc
    results["arms"]["b_sweep"] = bs

    emit_json("stage-b-ksweep", results)


if __name__ == "__main__":
    main()
