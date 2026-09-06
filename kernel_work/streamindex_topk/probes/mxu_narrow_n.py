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
"""P4 -- what a narrow-N matmul costs per unit of useful work on the MXU.

The question, in one line: at fixed M and K, how much more does the MXU cost
PER USEFUL FLOP when the output has N=32 lanes instead of N=128?

Why the primitive and not the variant
-------------------------------------
`step2/combined.md` raises this for C6, but the answer is a property of the MXU,
not of C6. The 128x128 systolic array is fed 128 lanes wide; an N=32 output
plausibly leaves three quarters of it idle, in which case a design that splits
work into narrow-N pieces pays 4x for the same FLOPs. combined.md already
carries a MEASURED 3.57x derate on the *M* axis at M=32. Nobody has measured the
*N* axis. Probing the primitive means the number is reusable by any later
narrow-N design, and it cannot be contaminated by whatever else C6 does.

The ladder
----------
M = 128 throughout. N walks 32 -> 64 -> 128 -> 256. R = 512 / N repetitions, so
across the whole ladder:

    useful FLOPs    = 2 * M * N * K * R = 2 * 128 * 512 * K   -- CONSTANT
    output elements = M * N * R         = 128 * 512           -- CONSTANT

Holding both constant is the point. If narrow N were free, every rung would take
the same time. Anything above flat at N=32 is the derate, and because the output
element count is pinned, it cannot be an artifact of storing more or fewer
results.

Separating the MXU from the store
---------------------------------
At K=128 the vector store of the result is a large fraction of the arm, so
`time / FLOP` at a single K measures the MXU and the store together and would
understate the derate. So each rung is run at TWO values of K (128 and 512) and
the MXU cost is recovered as the SLOPE:

    mxu_time(per K=128 unit) = (t_K512 - t_K128) / 3

The store term, the loop overhead and the fixed per-grid-step cost are identical
at both K values and cancel out of the difference. `t_K128` alone is reported
too, and the intercept `t_K128 - mxu_time` is the store-plus-overhead residue --
useful, but it is a by-product, not the answer.

Confounds this is built to dodge
--------------------------------
  * Slicing an operand on a non-128 lane boundary would inject a relayout, which
    is precisely what P3 measures separately and must not leak in here. `b` is
    therefore indexed on a LEADING axis, `b_ref[r]`, which `hardware-facts.md`
    calls free bookkeeping.
  * A constant `index_map` on every operand can let the compiler hoist the whole
    body out of the grid. `b` is cycled through two distinct blocks with
    `index_map=lambda i: (i % 2, 0, 0)` so the body cannot be loop-invariant.
  * `a` is reused across all R matmuls on purpose: that is the shape of the
    contraction in question, and it keeps HBM traffic far below the roofline so
    nothing here is bandwidth bound.

Two dtypes are run. bf16 is the native pair `combined.md` already has a Pallas
rate for (569.9 Tops/s), so it anchors the ladder against a known number. fp8
e4m3 is run because combined.md records NO Pallas measurement for fp8 at all,
and the whole 48.4 us vs 120.6 us spread in the mixed floor turns on whether fp8
reaches the MXU. That second rung is a by-product of this probe, not its
question -- P1 answers the fp8 question properly, off the Mosaic MLIR.

Sign convention: every percentage is a change in TIME, (new - old)/old.
Negative is faster.
"""

import argparse
import functools
import json
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M = 128
NK_TOTAL = 512  # N * R, held constant across the ladder
GRID = 64
WARMUP = 3
REPS = 20
BATCH = 8
VMEM_LIMIT = 128 * 2**20

DTYPES = {
    "bf16": jnp.bfloat16,
    "fp8_e4m3": jnp.float8_e4m3fn,
    "f32": jnp.float32,
}


def _median_spread(samples):
    med = statistics.median(samples)
    spread = (max(samples) - min(samples)) / med * 100.0 if med else float("nan")
    return med, spread


def _time(fn, *args):
    jax.block_until_ready(fn(*args))
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(BATCH):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) / BATCH)
    return _median_spread(samples)


def _kernel(a_ref, b_ref, o_ref, *, reps):
    """`b_ref` carries a leading block axis of 1; `b_ref[0, r]` indexes only
    leading axes, which hardware-facts.md calls free bookkeeping. Slicing the
    lane axis here would inject the relayout P3 measures and must not happen."""
    a = a_ref[...]
    for r in range(reps):
        o_ref[r] = jnp.dot(a, b_ref[0, r], preferred_element_type=jnp.float32)


def build(n, k, dtype_name):
    dt = DTYPES[dtype_name]
    reps = NK_TOTAL // n
    rng = np.random.default_rng(7)
    a = jnp.asarray(rng.standard_normal((M, k)).astype(np.float32)).astype(dt)
    # Two distinct b blocks so the grid body is not loop-invariant.
    b = jnp.asarray(
        rng.standard_normal((2, reps, k, n)).astype(np.float32)).astype(dt)

    f2 = pl.pallas_call(
        functools.partial(_kernel, reps=reps),
        grid=(GRID, ),
        in_specs=[
            pl.BlockSpec((M, k), lambda i: (0, 0)),
            pl.BlockSpec((1, reps, k, n), lambda i: (i % 2, 0, 0, 0)),
        ],
        out_specs=pl.BlockSpec((reps, M, n), lambda i: (0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((reps, M, n), jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
    )
    return jax.jit(f2), (a, b)


def run(dtype_names, ns, ks):
    rows = []
    for dtype_name in dtype_names:
        for n in ns:
            for k in ks:
                reps = NK_TOTAL // n
                flops = 2 * M * n * k * reps
                outel = M * n * reps
                rec = dict(dtype=dtype_name, n=n, k=k, reps=reps,
                           flops_per_step=flops, out_elems_per_step=outel)
                try:
                    fn, args = build(n, k, dtype_name)
                    med, spread = _time(fn, *args)
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = repr(exc)[:400]
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
                    continue
                rec.update(total_s=med, ns_per_step=med / GRID * 1e9,
                           spread_pct=spread,
                           tops_per_s=flops / (med / GRID) / 1e12)
                rows.append(rec)
                print(json.dumps(rec), flush=True)
    return rows


def summarise(rows, ns, ks):
    ok = [r for r in rows if "error" not in r]
    print("\n=== P4 matmul ladder: M=%d, N*R=%d fixed, grid=%d ==="
          % (M, NK_TOTAL, GRID))
    print("FLOPs/step and output elements/step are constant down each dtype "
          "block at fixed K. Every row is one arm; no row is a control.")
    print(f"\n{'dtype':<10}{'N':>5}{'R':>4}{'K':>6}{'ns/step':>11}"
          f"{'spread%':>9}{'GFLOP/step':>12}{'Tops/s':>10}")
    print("-" * 67)
    for r in rows:
        if "error" in r:
            print(f"{r['dtype']:<10}{r['n']:>5}{r['reps']:>4}{r['k']:>6}"
                  f"   ERROR {r['error'][:50]}")
            continue
        print(f"{r['dtype']:<10}{r['n']:>5}{r['reps']:>4}{r['k']:>6}"
              f"{r['ns_per_step']:>11.1f}{r['spread_pct']:>9.2f}"
              f"{r['flops_per_step']/1e9:>12.3f}{r['tops_per_s']:>10.1f}")

    if len(ks) < 2:
        return
    k_lo, k_hi = min(ks), max(ks)
    ratio = k_hi / k_lo
    print("\n=== MXU cost isolated by the K slope ===")
    print(f"mxu_ns = (t(K={k_hi}) - t(K={k_lo})) / {ratio - 1:.0f}  -- the "
          f"store, the loop and the per-grid-step fixed cost are identical at "
          f"both K and cancel.")
    print("`d time %` is (N arm - N=128 arm)/N=128 arm on the SAME dtype. "
          "Negative is faster. `derate x` >1 means the narrow arm costs more "
          "per useful FLOP.")
    print(f"\n{'dtype':<10}{'N':>5}{'t(Klo) ns':>11}{'t(Khi) ns':>11}"
          f"{'mxu ns':>10}{'resid ns':>10}{'mxu Tops/s':>12}"
          f"{'derate x':>10}{'d time %':>10}")
    print("-" * 89)
    for dtype_name in sorted({r["dtype"] for r in ok}):
        base = {}
        block = []
        for n in ns:
            lo = next((r for r in ok
                       if r["dtype"] == dtype_name and r["n"] == n
                       and r["k"] == k_lo), None)
            hi = next((r for r in ok
                       if r["dtype"] == dtype_name and r["n"] == n
                       and r["k"] == k_hi), None)
            if not lo or not hi:
                continue
            mxu = (hi["ns_per_step"] - lo["ns_per_step"]) / (ratio - 1)
            resid = lo["ns_per_step"] - mxu
            tops = lo["flops_per_step"] / (mxu * 1e-9) / 1e12
            block.append((n, lo, hi, mxu, resid, tops))
            if n == 128:
                base = dict(mxu=mxu, total=lo["ns_per_step"])
        for n, lo, hi, mxu, resid, tops in block:
            derate = mxu / base["mxu"] if base else float("nan")
            pct = ((lo["ns_per_step"] - base["total"]) / base["total"] * 100.0
                   if base else float("nan"))
            print(f"{dtype_name:<10}{n:>5}{lo['ns_per_step']:>11.1f}"
                  f"{hi['ns_per_step']:>11.1f}{mxu:>10.1f}{resid:>10.1f}"
                  f"{tops:>12.1f}{derate:>10.2f}{pct:>10.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="bf16,fp8_e4m3")
    ap.add_argument("--ns", default="32,64,128,256")
    ap.add_argument("--ks", default="128,512")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    dtypes = [d for d in a.dtypes.split(",") if d]
    ns = [int(x) for x in a.ns.split(",") if x]
    ks = [int(x) for x in a.ks.split(",") if x]
    rows = run(dtypes, ns, ks)
    summarise(rows, ns, ks)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
