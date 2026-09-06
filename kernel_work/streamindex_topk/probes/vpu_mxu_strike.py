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
"""§7.4 of `measurement_and_design.md` -- is the MIXED band VPU-saturated?

The claim under test
--------------------
`probes/llo_capture_RESULT.md`'s STATIC LLO analysis says VALU 94.66% slot
utilization against MXU 35.26%, and area 5's op-slot model predicts 624 us
against a MEASURED 269.5 us of TensorCore time at `mixed_prod`
[measured: j-5f8f4a08-20b4-4fa0-bfc3]. Inverting 624/269.5 = 2.32 gives "a VPU
running 2.32x the reference rate, saturated, accounts for 100% of mixed's
scoring time".

Both of those are static / modelled. The static spill signal ships
self-labelled [TRACK RECORD 0/5 -- directly refuted twice]. So this probe is a
DYNAMIC measurement or it is nothing. No spill-reduction candidate is opened on
the result whichever way it lands -- that is a standing instruction, not this
probe's call.

The target number: 269.51 us / 32 inner iterations = **8.42 us per inner step**
at `mixed_prod` [measured: j-5f8f4a08-20b4-4fa0-bfc3]. CONFIRMED if the
non-MXU part of the step lands near 8.4 us. REFUTED if it is <~2 us, in which
case ~6.2 us/step is somewhere nobody has looked and THAT is the finding.

Why this is not literally §7.4's two arms
-----------------------------------------
§7.4 says: arm 1 replaces the einsum with "a compile-time constant of the same
shape"; arm 2 replaces the VPU chain with "a slice". Both spellings have a
defect that would manufacture the answer:

  * A compile-time constant `s` lets the compiler hoist relu(s), the `w`
    multiply and even the head sum straight out of the grid loop -- arm 1 then
    reads ~0 and the hypothesis is "refuted" by dead-code elimination.
  * A slice epilogue lets DCE delete 31 of the 32 head-rows of the matmul --
    arm 2 then reads 1/32 of the MXU cost and the residual is "confirmed" by
    the same mechanism in the other direction.

So the decomposition is done by a ladder instead, which cannot be folded:

    K = head_dim walks 128 -> 256 -> 512 with everything else pinned.
    MXU FLOPs are proportional to K.  The VPU chain, the output store and the
    per-grid-step fixed cost do not depend on K at all.

    t(K) = mxu_per_K * K + rest      ->  slope = MXU, intercept = everything else

Two epilogues are run over that same ladder:

    `full` -- today's body: relu, the `w` broadcast-multiply, the 32-head sum,
              the scale multiply, the two masks and the `where`.
              (streamindex_topk.py:304-354, seq_batch_size = 1)
    `sum`  -- the head sum ALONE. This is the minimal epilogue that still
              consumes every row of the matmul, so the MXU is not DCE'd. It is
              the honest stand-in for §7.4's "replaced by a slice".

    intercept(full) - intercept(sum) = relu + w-mul + scale + mask, isolated.
    intercept(sum) itself = head sum + out-store + fixed per-step cost.

§7.4's literal arm 1 is ALSO run, as `gen=bcast`: `s` built by broadcasting a
runtime (1, bkv_sz) vector instead of by the einsum. It is reported next to the
K=0 intercept as a cross-check. If the two disagree the broadcast arm is the
one to distrust, for the hoisting reason above.

What is a control here and must not be lifted out
-------------------------------------------------
Every `sum`-epilogue row is a CONTROL ARM. It is engineered to be subtracted,
it is not a realistic kernel, and its absolute time means nothing on its own.
The results are the DIFFERENCES and the slope. The per-arm absolute times are
printed so the arithmetic can be re-derived, prefixed `_control_` where they
are subtrahends.

Sign convention: percentages are changes in TIME, (new - old)/old. Negative is
faster.
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

# The MIXED band's inner step at `mixed_prod` and at `mixed_64k` alike:
# seq_batch_size = 1 (streamindex_topk.py:801 passes seq_batch_size=1 for
# MlaCase.MIXED), bq_sz = 128 (num_queries_per_block[2]), num_q_heads = 32
# (config.json index_n_heads), bkv_sz = page_size * bkv_p = 1024 * 2 = 2048.
BQ_SZ = 128
NUM_HEADS = 32
BKV_SZ = 2048
HEAD_DIM = 128  # the real K; the ladder's other rungs are synthetic

GRID = 32  # = mixed_prod's inner iteration count (num_bq 8 * num_bkv 4)
WARMUP = 3
REPS = 10
BATCH = 4
VMEM_LIMIT = 128 * 2**20

# PROBING.md's dry-run path: `--tiny --interpret` shrinks every extent and runs
# through `pltpu.InterpretParams()`, so the whole arm matrix can be traced,
# lowered and executed under `JAX_PLATFORMS=cpu` before a TPU slot is spent.
# A dry-run number is NOT a measurement and is never quoted.
INTERPRET = False


def _tiny():
    global BQ_SZ, NUM_HEADS, BKV_SZ, HEAD_DIM, GRID, WARMUP, REPS, BATCH
    BQ_SZ, NUM_HEADS, BKV_SZ, HEAD_DIM = 8, 4, 128, 8
    GRID, WARMUP, REPS, BATCH = 2, 0, 1, 1


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


def _body(q_ref, kv_ref, w_ref, sc_ref, pos_ref, klen_ref, o_ref, *,
          gen, epilogue, k):
    """Mirror of `compute_scores` (streamindex_topk.py:304-354) at
    seq_batch_size = 1, with two ablation switches."""
    if gen == "mxu":
        q = q_ref[...]                  # (BQ_SZ * NUM_HEADS, k)  fp8
        kv = kv_ref[0]                  # (BKV_SZ, k)             fp8
        s = jnp.einsum("nd,md->nm", q, kv,
                       preferred_element_type=jnp.float32)
    else:
        # §7.4's literal arm 1: `s` with no MXU behind it. Built from a runtime
        # ref rather than a literal so it is not a foldable constant -- but see
        # the module docstring on why this arm is a cross-check, not the answer.
        s = jnp.broadcast_to(sc_ref[...], (BQ_SZ * NUM_HEADS, BKV_SZ))

    s = s.reshape(BQ_SZ, NUM_HEADS, BKV_SZ)

    if epilogue == "full":
        s = jnp.maximum(s, 0.0)
        s = s * w_ref[...].astype(jnp.float32)[:, :, None]
        ss = s.sum(axis=1)
        ss = ss * sc_ref[...]
        k_span = jnp.arange(BKV_SZ, dtype=jnp.int32)
        valid = k_span[None, :] < klen_ref[...]
        causal = k_span[None, :] <= pos_ref[...][:, None]
        ss = jnp.where(jnp.logical_and(valid, causal), ss, -jnp.inf)
    else:
        # CONTROL epilogue: the head sum alone. Minimal work that still reads
        # every row of `s`, so the matmul survives DCE.
        ss = s.sum(axis=1)

    o_ref[0] = ss


def build(gen, epilogue, k):
    rng = np.random.default_rng(11)
    q = jnp.asarray(
        rng.standard_normal((BQ_SZ * NUM_HEADS, k)).astype(np.float32)
    ).astype(jnp.float8_e4m3fn)
    # Two kv blocks, cycled, so the grid body is not loop-invariant and the
    # compiler cannot hoist the whole step out of the loop.
    kv = jnp.asarray(
        rng.standard_normal((2, BKV_SZ, k)).astype(np.float32)
    ).astype(jnp.float8_e4m3fn)
    w = jnp.asarray(rng.standard_normal((BQ_SZ, NUM_HEADS)).astype(np.float32))
    sc = jnp.asarray(rng.random((1, BKV_SZ)).astype(np.float32))
    pos = jnp.asarray(rng.integers(1024, 2048, size=(BQ_SZ, )).astype(np.int32))
    klen = jnp.asarray(np.full((1, 1), 2048, dtype=np.int32))

    fn = pl.pallas_call(
        functools.partial(_body, gen=gen, epilogue=epilogue, k=k),
        grid=(GRID, ),
        in_specs=[
            pl.BlockSpec((BQ_SZ * NUM_HEADS, k), lambda i: (0, 0)),
            pl.BlockSpec((1, BKV_SZ, k), lambda i: (i % 2, 0, 0)),
            pl.BlockSpec((BQ_SZ, NUM_HEADS), lambda i: (0, 0)),
            pl.BlockSpec((1, BKV_SZ), lambda i: (0, 0)),
            pl.BlockSpec((BQ_SZ, ), lambda i: (0, )),
            pl.BlockSpec((1, 1), lambda i: (0, 0)),
        ],
        # The store moves to a new offset every step, exactly as the real
        # kernel's does (1 MiB per inner step at mixed). Identical across every
        # arm, so it cancels out of every difference below.
        out_specs=pl.BlockSpec((1, BQ_SZ, BKV_SZ), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((GRID, BQ_SZ, BKV_SZ), jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
        **({"interpret": pltpu.InterpretParams()} if INTERPRET else {}),
    )
    return jax.jit(fn), (q, kv, w, sc, pos, klen)


def run(ks):
    arms = [("mxu", "full", k) for k in ks]
    arms += [("mxu", "sum", k) for k in ks]
    arms += [("bcast", "full", HEAD_DIM), ("bcast", "sum", HEAD_DIM)]

    rows = []
    for gen, epi, k in arms:
        rec = dict(gen=gen, epilogue=epi, k=k, grid=GRID,
                   is_control=(epi == "sum"))
        if gen == "mxu":
            rec["mxu_flops_per_step"] = 2 * BQ_SZ * NUM_HEADS * BKV_SZ * k
        try:
            fn, args = build(gen, epi, k)
            med, spread = _time(fn, *args)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)[:600]
            rows.append(rec)
            print(json.dumps(rec), flush=True)
            continue
        rec.update(total_s=med, us_per_step=med / GRID * 1e6,
                   spread_pct=spread)
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    return rows


def summarise(rows, ks):
    def get(gen, epi, k):
        return next((r for r in rows
                     if r["gen"] == gen and r["epilogue"] == epi
                     and r["k"] == k and "error" not in r), None)

    print("\n=== raw arms: us per inner step, grid=%d, "
          "shape (bq_sz=%d, heads=%d, bkv_sz=%d) ==="
          % (GRID, BQ_SZ, NUM_HEADS, BKV_SZ))
    print("Rows with epilogue=sum are CONTROL ARMS. They exist to be "
          "subtracted; their absolute time is not a finding.")
    hdr = (f"{'gen':<8}{'epilogue':<10}{'K':>6}{'us/step':>10}{'spread%':>9}"
           f"{'GFLOP/step':>12}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['gen']:<8}{r['epilogue']:<10}{r['k']:>6}   ERROR "
                  f"{r['error'][:70]}")
            continue
        fl = r.get("mxu_flops_per_step", 0) / 1e9
        print(f"{r['gen']:<8}{r['epilogue']:<10}{r['k']:>6}"
              f"{r['us_per_step']:>10.3f}{r['spread_pct']:>9.2f}"
              f"{fl:>12.3f}")

    if len(ks) < 2:
        return
    k_lo, k_hi = min(ks), max(ks)

    print("\n=== the decomposition ===")
    print(f"slope   = (t(K={k_hi}) - t(K={k_lo})) / ({k_hi} - {k_lo})  "
          "us per unit of K")
    print(f"mxu(K={HEAD_DIM}) = slope * {HEAD_DIM}")
    print(f"rest    = t(K={k_lo}) - slope * {k_lo}   "
          "-- the K=0 intercept: everything that is not the matmul")
    hdr2 = (f"{'epilogue':<10}{'t(Klo) us':>12}{'t(Khi) us':>12}"
            f"{'mxu us':>11}{'rest us':>11}{'mxu Tops/s':>13}")
    print(hdr2)
    print("-" * len(hdr2))
    inter = {}
    for epi in ("full", "sum"):
        lo, hi = get("mxu", epi, k_lo), get("mxu", epi, k_hi)
        if not lo or not hi:
            continue
        slope = (hi["us_per_step"] - lo["us_per_step"]) / (k_hi - k_lo)
        mxu = slope * HEAD_DIM
        rest = lo["us_per_step"] - slope * k_lo
        inter[epi] = dict(mxu=mxu, rest=rest, t_lo=lo["us_per_step"])
        tops = (2 * BQ_SZ * NUM_HEADS * BKV_SZ * HEAD_DIM) / (mxu * 1e-6) / 1e12
        print(f"{epi:<10}{lo['us_per_step']:>12.3f}{hi['us_per_step']:>12.3f}"
              f"{mxu:>11.3f}{rest:>11.3f}{tops:>13.1f}")

    if "full" in inter and "sum" in inter:
        vpu_extra = inter["full"]["rest"] - inter["sum"]["rest"]
        print(f"\nrelu + w-multiply + scale + mask, isolated "
              f"= rest(full) - rest(sum) = "
              f"{inter['full']['rest']:.3f} - {inter['sum']['rest']:.3f} = "
              f"{vpu_extra:.3f} us/step")
        print(f"head sum + out-store + fixed per-step  = rest(sum) = "
              f"{inter['sum']['rest']:.3f} us/step")
        print(f"non-MXU total at K={HEAD_DIM}             = rest(full) = "
              f"{inter['full']['rest']:.3f} us/step")
        print(f"MXU at K={HEAD_DIM}                       = "
              f"{inter['full']['mxu']:.3f} us/step")
        print(f"control (mxu, full, K={HEAD_DIM})         = "
              f"{inter['full']['t_lo']:.3f} us/step  "
              f"-- target 8.42 us/step [measured j-5f8f4a08]")

    bc_full, bc_sum = get("bcast", "full", HEAD_DIM), get("bcast", "sum",
                                                          HEAD_DIM)
    if bc_full and bc_sum:
        print(f"\n§7.4 literal arm 1 (no MXU, broadcast s): full "
              f"{bc_full['us_per_step']:.3f} us/step, sum-only control "
              f"{bc_sum['us_per_step']:.3f} us/step, difference "
              f"{bc_full['us_per_step'] - bc_sum['us_per_step']:.3f} us/step.")
        print("Cross-check against `rest(full)` and the isolated VPU line "
              "above. Disagreement means the broadcast was hoisted; believe "
              "the ladder, not this arm.")

    print("\nVERDICT RULE (§7.4, stated before the numbers were seen): "
          "CONFIRMED if the non-MXU part of the step is near 8.4 us; REFUTED "
          "if it is below ~2 us. Between the two, the residual is split and "
          "neither reading is supported.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="128,256,512")
    ap.add_argument("--json", default="")
    ap.add_argument("--interpret", action="store_true",
                    help="dry run through pltpu.InterpretParams(); not a "
                         "measurement")
    ap.add_argument("--tiny", action="store_true",
                    help="shrink every extent for the CPU dry run")
    a = ap.parse_args(argv)
    global INTERPRET
    INTERPRET = a.interpret
    if a.tiny:
        _tiny()
    ks = [int(x) for x in a.ks.split(",") if x]
    rows = run(ks)
    summarise(rows, ks)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
