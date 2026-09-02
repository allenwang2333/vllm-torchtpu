#!/usr/bin/env python3
"""Stage B, part 2 -- separate the un-bounded DMA from the length-bounded scan.

`probe_stage_b_ksweep.py` established that stage B's device time is flat in `k`
(issue-bound, not output-DMA-bound) and that it is linear in the wave count:
`t ~ 8.6 us + 10.2 us x waves` at `eff_len = 8704, n = 10240, k = 2048`, over a
32x lever arm in waves with r^2 ~ 1.0000.

That is a wave model, not a cost model: **nothing in it varied `eff_len` or
`n`**, so the 10.2 us/wave cannot yet be split into its parts.  Splitting it is
the point of this probe, and the reason it can be split at all is the asymmetry
at `sparsecore_topk.py:154-161`:

  * every scan (`cvt`, `hist0_body`, `ref_body`, `emit_body`) is bounded by
    `walk_chunks = cdiv(min(slice_len, eff_len), 16)` -- it tracks **eff_len**;
  * the resident-slice DMA is `dma(scores_hbm.at[pl.ds(s, slice_len)], ...)`
    with `slice_len = align_to(n, 16)`, a **trace-time constant** -- it tracks
    **n** and is blind to `row_lengths`.

So the two terms are independently addressable:

  **Arm A -- eff_len sweep, n fixed.**  n = 10240 throughout, so the DMA is
  byte-for-byte identical at every point; only the scans move.  The slope in
  eff_len is the scan cost per element; the intercept is everything that does
  not track eff_len, which includes the full-width DMA.

  **Arm B -- n sweep, eff_len fixed.**  eff_len = 2048 throughout, so
  `walk_chunks` is 128 at every point and the scan work is identical; only the
  DMA grows.  The slope in n is the DMA cost per element, and it is the price
  of the un-bounded read stated as a number rather than as a code reading.

Together they give `t_wave = fixed + alpha*n + beta*eff_len`, fitted over two
orthogonal lever arms rather than to a single operating point.

Methodology is `probe_stage_b_ksweep.py`'s and the caveats carry over verbatim:
best-of-N host wall clock is a lower bound; the batched figure is
`max(device, host enqueue)` so a dispatch floor is a clamp and not an addend;
the dispatch-free device time is the **slope of R chained calls in one jit**,
not any wall clock.  Run unpinned, one process.

Run:  python probe_stage_b_dma_scan.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from _probe_util import emit_json, p, tpu_banner  # noqa: E402
from probe_stage_b_ksweep import (build_scores, dispatch_floors,  # noqa: E402
                                  measure_point)

K = 2048

# Arm A: n fixed at the production buffer width, eff_len swept.
A_N = 10240
A_EFF = (1024, 2048, 4096, 6144, 8192, 8704, 10240)
A_B = (64, 1024)  # 2 waves and 32 waves

# Arm B: eff_len fixed, n swept. n must be a multiple of 16 and >= eff_len;
# `_pick_partition` stays at p=1 for b >= 32 and n <= MAX_SLICE_WORDS (65536).
B_EFF = 2048
B_N_SMALL = (2048, 4096, 8192, 10240, 16384, 32768, 65536)  # b = 64
B_N_LARGE = (2048, 4096, 8192, 10240, 16384)  # b = 1024, HBM-bounded


def fit(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    a = np.vstack([np.ones_like(xs), xs]).T
    (icpt, slope), *_ = np.linalg.lstsq(a, ys, rcond=None)
    pred = icpt + slope * xs
    ss_res = float(((ys - pred)**2).sum())
    ss_tot = float(((ys - ys.mean())**2).sum())
    return {
        "intercept_us": float(icpt),
        "slope": float(slope),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def main():
    tpu_banner()
    from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as scmod
    p("sparsecore_topk_module", scmod.__file__)

    def public(s, k, rl):
        return scmod.sparsecore_topk(s, k, row_lengths=rl,
                                     write_empty_rows=True)

    results = {"k": K, "arms": {}}
    floors_done = False

    # ---- Arm A: eff_len sweep at fixed n ---------------------------------
    arm_a = []
    for b in A_B:
        for eff in A_EFF:
            rl = jnp.full((b, ), eff, jnp.int32)
            sc = build_scores(b, A_N, rl)
            if not floors_done:
                results["dispatch_floors"] = dispatch_floors(sc, rl)
                p("dispatch_floors", results["dispatch_floors"])
                floors_done = True
            row = measure_point(public, f"A_b{b}_n{A_N}_eff{eff}", b, A_N, K,
                                sc, rl)
            row["eff_len"] = eff
            row["waves"] = -(-b // 32)
            row["p"] = scmod._pick_partition(b, A_N, K)
            arm_a.append(row)
            del sc
    results["arms"]["A_eff_len_sweep"] = arm_a

    # ---- Arm B: n sweep at fixed eff_len ---------------------------------
    arm_b = []
    for b, ns in ((64, B_N_SMALL), (1024, B_N_LARGE)):
        for n in ns:
            rl = jnp.full((b, ), B_EFF, jnp.int32)
            sc = build_scores(b, n, rl)
            pp = scmod._pick_partition(b, n, K)
            if pp != 1:
                p("skip", f"b={b} n={n} would take p={pp}, not comparable")
                del sc
                continue
            row = measure_point(public, f"B_b{b}_n{n}_eff{B_EFF}", b, n, K, sc,
                                rl)
            row["eff_len"] = B_EFF
            row["waves"] = -(-b // 32)
            row["p"] = pp
            arm_b.append(row)
            del sc
    results["arms"]["B_n_sweep"] = arm_b

    # ---- fits, per (arm, b), on the dispatch-free chain slope ------------
    fits = {}
    for name, rows, xkey in (("A_eff_len", arm_a, "eff_len"),
                             ("B_n", arm_b, "n")):
        for b in sorted({r["b"] for r in rows}):
            sel = [r for r in rows if r["b"] == b and "chain" in r]
            if len(sel) < 3:
                continue
            waves = sel[0]["waves"]
            xs = [r[xkey] for r in sel]
            # per-wave device time: total slope divided by the wave count
            ys = [r["chain"]["slope_us_per_call"] / waves for r in sel]
            f = fit(xs, ys)
            f["waves"] = waves
            f["ns_per_element_per_wave"] = f["slope"] * 1e3
            fits[f"{name}_b{b}"] = f
            p(f"fit {name} b={b}",
              f"per-wave = {f['intercept_us']:.2f}us + "
              f"{f['ns_per_element_per_wave']:.4f}ns x {xkey}  "
              f"(r2={f['r2']:.4f}, waves={waves})")
    results["fits"] = fits

    emit_json("stage-b-dma-scan", results)


if __name__ == "__main__":
    main()
