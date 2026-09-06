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
"""Option F -- move the 32-head reduction off the VPU onto the idle MXU.

Step-level probe, at the MIXED band's exact inner-step shape. This is the
instrument that prices F against the number that opened it:

    32-head-sum bundle          3.858 us/step   52.5% of a 7.343 us step
    relu + w + scale + masks    2.401 us/step   32.7%
    MXU at K=128                1.084 us/step   14.8%
        [measured (j-b54e9716-3db5-4f1f-b8d4), shard 3]

It is a direct descendant of `probes/vpu_mxu_strike.py` -- same body, same grid,
same store, same K-ladder decomposition -- with the head reduction swapped for a
matmul in the F arms. Keeping the harness identical is deliberate: the
incumbent arm here must reproduce that 7.343 us, and if it does not, nothing
below is comparable to the decomposition it is arguing with.

The formulations, and why these ones
------------------------------------
`step2/sheets/area5-compute-units.md` alternative H, and `measurement_and_design.md`
Sec 2.5, spell F as "the 32-head sum as a block-diagonal ones-matmul" and reject it
on MXU FILL (3.1%). Fill is the wrong criterion when the MXU is idle, so the
rejection is re-opened -- but only the rejection, not the arithmetic. The
arithmetic is:

    incumbent   s_summed[q, m] = sum_h relu(s)[q,h,m] * w[q,h]
                                 -- 32-deep reduction over the SUBLANE axis

    F1 (ones)   S2[n, m] = relu(s)[n, m] * w_flat[n],   n = 32q + h
                s_summed = ONES_BD @ S2,  ONES_BD[q, n] = 1 iff n // 32 == q
                shape (128, 4096) @ (4096, 2048) -> (128, 2048)

    F2 (w-fold) s_summed = W_BD @ relu(s),  W_BD[q, n] = w[q, n % 32] iff n//32 == q
                -- the same matmul, with the VPU w-multiply folded into the
                   operand. Needs NO sign guarantee on w, which is what killed
                   candidate C8's E1 arm (`weights_proj.weight` is ~50% negative
                   on the real GLM-5.2-FP8 checkpoint; 0 of 32 rows all-non-neg).

Three things about the shape are worth stating before the numbers. The
reduction matmul is (128, 4096) x (4096, 2048): M/128 = 1, K/128 = 32,
N/128 = 16, so **512 MXU tile-ops -- exactly the same count as the scoring
einsum's** (M/128 = 32, K/128 = 1, N/128 = 16). F does not add a small matmul
beside a big one; it adds a second matmul of the same size. Its FLOPs are also
identical: 2*128*4096*2048 = 2*4096*2048*128 = 2.147 GFLOP. And its arithmetic
intensity is dire -- 3.1% of the multiplies are against a non-zero -- which is
exactly the objection F is testing, restated.

A formulation that is NOT here, and why
---------------------------------------
"Reshape so the head axis is a contraction dimension of the existing einsum."
The ReLU sits between the two contractions (`streamindex_topk.py:333`), so the
head axis has to be materialised before it can be reduced. That is precisely
what makes F exact and what makes A1 (drop the ReLU) an approximation. It is
not an option; it is a different candidate, owned elsewhere.

The bf16x2 split (s = bf16(s) + bf16(s - bf16(s)), two matmuls, ~2^-17
relative) is also not here, and it is worth recording WHY rather than leaving
it as an unexplored branch: it needs two matmuls of the shape above, and the
split itself costs ~4 VPU ops per vreg of `s` against the 2.25 ops/vreg that
the sublane reduction it replaces costs (area 5's op-slot model). It pays more
VPU than it saves and doubles the MXU. It is self-defeating on its own
arithmetic and needs no arm.

The exactness question this probe has to answer
-----------------------------------------------
F's selling point over A1 is that it is exact. Area 6 records that pure
reassociation of the 32-term head sum is bounded by 2.46e-06 relative
[verified j-ed517191], four decades below the 61-rank churn already present -- so
a reduction that only REASSOCIATES needs no gate.

But a matmul only reaches the MXU through its operand dtypes, and the scoring
einsum reaches it as native fp8 (`f8E4M3FN x f8E4M3FN -> f32`, zero bf16
matmuls) [measured (j-689b87cd-7460-4b90-bd87)]. If F's operands land as bf16
then F rounds `relu(s)*w` to 8 mantissa bits before summing and it is NOT a
reassociation -- it is a different approximation from A1's, with no gate. The
arms below therefore include:

    F4  f32 operands, precision=DEFAULT   -- what a naive `jnp.einsum` on the
                                             f32 array emits
    F5  f32 operands, precision=HIGHEST   -- the only spelling that could be
                                             exact; priced so the trade is on
                                             the record rather than assumed

and every F arm is checked for max relative error against the incumbent's own
output on identical inputs, in a separate untimed pass. `llo_f.py` confirms the
`vmatmul` operand types at the opcode level; the dtype printed by this file is
the JAX-level dtype and is NOT evidence of what the MXU received.

Controls
--------
`sum_vpu` and `sum_ones_bf16` are CONTROL ARMS. They are the head reduction
alone -- the minimal epilogue that still consumes every row of the matmul, so
the scoring einsum is not DCE'd. They exist to be subtracted; their absolute
times are not findings. `incumbent_repeat` is the DRIFT CONTROL: an arm
identical to the first, run last. Variant E's job showed within-arm spread is
not a bound on arm-to-arm drift, and a "free at mixed" reading there turned into
a real ~2% cost once a repeat was added.

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

# The MIXED band's inner step, identical to `probes/vpu_mxu_strike.py`:
# seq_batch_size = 1 (streamindex_topk.py:801), bq_sz = 128
# (num_queries_per_block[2]), num_q_heads = 32 (config.json index_n_heads),
# bkv_sz = page_size * bkv_p = 1024 * 2 = 2048.
BQ_SZ = 128
NUM_HEADS = 32
BKV_SZ = 2048
HEAD_DIM = 128  # the real K; the ladder's other rung is synthetic

GRID = 32  # = mixed_prod's inner iteration count (num_bq 8 * num_bkv 4)
WARMUP = 3
REPS = 10
BATCH = 4
VMEM_LIMIT = 128 * 2**20

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


def _ones_bd_iota(dtype):
    """(BQ_SZ, BQ_SZ*NUM_HEADS) block-diagonal ones, built from iota.

    Loop-invariant, so Mosaic may hoist it out of the grid. Whether it does is
    the difference between the `_iota` and `_ref` arms.
    """
    shape = (BQ_SZ, BQ_SZ * NUM_HEADS)
    r = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    c = jax.lax.broadcasted_iota(jnp.int32, shape, 1) // NUM_HEADS
    return (r == c).astype(dtype)


def _body(q_ref, kv_ref, w_ref, sc_ref, pos_ref, klen_ref, ones_ref, wbd_ref,
          o_ref, *, arm, k):
    """Mirror of `compute_scores` (streamindex_topk.py:304-354) at
    seq_batch_size = 1, with the head reduction swappable."""
    q = q_ref[...]                  # (BQ_SZ * NUM_HEADS, k)  fp8
    kv = kv_ref[0]                  # (BKV_SZ, k)             fp8
    s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)

    tail = True  # scale + the two masks + the where

    if arm in ("incumbent", "incumbent_repeat", "sum_vpu"):
        # ---- incumbent: relu, w-multiply, 32-deep VPU sublane reduction ----
        s = s.reshape(BQ_SZ, NUM_HEADS, BKV_SZ)
        if arm == "sum_vpu":
            ss = s.sum(axis=1)      # CONTROL: the reduction alone
            tail = False
        else:
            s = jnp.maximum(s, 0.0)
            s = s * w_ref[...].astype(jnp.float32)[:, :, None]
            ss = s.sum(axis=1)
    elif arm in ("F1_iota", "F1_ref", "F4_ones_f32", "F5_ones_f32_highest",
                 "sum_ones_bf16"):
        # ---- F1: relu and w on the VPU, head sum as ONES_BD @ S2 ----
        if arm == "sum_ones_bf16":
            tail = False            # CONTROL: the matmul reduction alone
            s2 = s
        else:
            s = s.reshape(BQ_SZ, NUM_HEADS, BKV_SZ)
            s = jnp.maximum(s, 0.0)
            s = s * w_ref[...].astype(jnp.float32)[:, :, None]
            # (BQ_SZ, NUM_HEADS, BKV_SZ) -> (BQ_SZ*NUM_HEADS, BKV_SZ). BKV_SZ
            # stays in lanes and two sublane axes merge, so no data moves --
            # the C5 result [measured (j-efb89184-020f-495f-9d4b)] is that this
            # class of reshape is below the noise floor.
            s2 = s.reshape(BQ_SZ * NUM_HEADS, BKV_SZ)
        if arm in ("F4_ones_f32", "F5_ones_f32_highest"):
            ones = _ones_bd_iota(jnp.float32)
            prec = (jax.lax.Precision.HIGHEST
                    if arm == "F5_ones_f32_highest" else
                    jax.lax.Precision.DEFAULT)
            ss = jnp.einsum("qn,nm->qm", ones, s2,
                            preferred_element_type=jnp.float32,
                            precision=prec)
        else:
            ones = (ones_ref[...] if arm == "F1_ref"
                    else _ones_bd_iota(jnp.bfloat16))
            ss = jnp.einsum("qn,nm->qm", ones, s2.astype(jnp.bfloat16),
                            preferred_element_type=jnp.float32)
    elif arm in ("F2_wfold_bf16", "F3_wfold_fp8"):
        # ---- F2/F3: relu only on the VPU; w folded into the matmul operand ----
        # W_BD is handed in already built. Building it in-kernel from the
        # (BQ_SZ, NUM_HEADS) `w` needs a lane<->sublane relayout of a 2 MiB
        # tensor, which is a producer-side question (candidate C13's route),
        # not a question about whether the MXU reduction pays. Pricing the
        # operand's construction is OUT OF SCOPE for this arm and is recorded
        # as `unmodelled`.
        s = jnp.maximum(s, 0.0)
        if arm == "F3_wfold_fp8":
            wbd = wbd_ref[...].astype(jnp.float8_e4m3fn)
            s2 = s.astype(jnp.float8_e4m3fn)
        else:
            wbd = wbd_ref[...].astype(jnp.bfloat16)
            s2 = s.astype(jnp.bfloat16)
        ss = jnp.einsum("qn,nm->qm", wbd, s2,
                        preferred_element_type=jnp.float32)
    else:
        raise ValueError(f"unknown arm {arm}")

    if tail:
        ss = ss * sc_ref[...]
        k_span = jnp.arange(BKV_SZ, dtype=jnp.int32)
        valid = k_span[None, :] < klen_ref[...]
        causal = k_span[None, :] <= pos_ref[...][:, None]
        ss = jnp.where(jnp.logical_and(valid, causal), ss, -jnp.inf)

    o_ref[0] = ss


def build(arm, k):
    rng = np.random.default_rng(11)
    q = jnp.asarray(
        rng.standard_normal((BQ_SZ * NUM_HEADS, k)).astype(np.float32)
    ).astype(jnp.float8_e4m3fn)
    # Two kv blocks, cycled, so the grid body is not loop-invariant.
    kv = jnp.asarray(
        rng.standard_normal((2, BKV_SZ, k)).astype(np.float32)
    ).astype(jnp.float8_e4m3fn)
    w_np = rng.standard_normal((BQ_SZ, NUM_HEADS)).astype(np.float32)
    w = jnp.asarray(w_np)
    sc = jnp.asarray(rng.random((1, BKV_SZ)).astype(np.float32))
    pos = jnp.asarray(rng.integers(1024, 2048, size=(BQ_SZ, )).astype(np.int32))
    klen = jnp.asarray(np.full((1, 1), 2048, dtype=np.int32))

    # ONES_BD and W_BD, host-built, in f32; the kernel casts.
    n = BQ_SZ * NUM_HEADS
    blk = (np.arange(BQ_SZ)[:, None] == (np.arange(n)[None, :] // NUM_HEADS))
    ones_bd = jnp.asarray(blk.astype(np.float32))
    wbd = jnp.asarray((blk * np.tile(w_np.reshape(1, n), (BQ_SZ, 1))
                       ).astype(np.float32))

    fn = pl.pallas_call(
        functools.partial(_body, arm=arm, k=k),
        grid=(GRID, ),
        in_specs=[
            pl.BlockSpec((BQ_SZ * NUM_HEADS, k), lambda i: (0, 0)),
            pl.BlockSpec((1, BKV_SZ, k), lambda i: (i % 2, 0, 0)),
            pl.BlockSpec((BQ_SZ, NUM_HEADS), lambda i: (0, 0)),
            pl.BlockSpec((1, BKV_SZ), lambda i: (0, 0)),
            pl.BlockSpec((BQ_SZ, ), lambda i: (0, )),
            pl.BlockSpec((1, 1), lambda i: (0, 0)),
            pl.BlockSpec((BQ_SZ, n), lambda i: (0, 0)),
            pl.BlockSpec((BQ_SZ, n), lambda i: (0, 0)),
        ],
        # The store moves to a new offset every step, exactly as the real
        # kernel's does (1 MiB per inner step at mixed). Identical across every
        # arm, so it cancels out of every difference.
        out_specs=pl.BlockSpec((1, BQ_SZ, BKV_SZ), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((GRID, BQ_SZ, BKV_SZ), jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
        **({"interpret": pltpu.InterpretParams()} if INTERPRET else {}),
    )
    return jax.jit(fn), (q, kv, w, sc, pos, klen, ones_bd, wbd)


# label, arm, K. Order matters: `incumbent` at K=128 is the denominator and
# `incumbent_repeat` is the drift control, run LAST.
ARMS = [
    ("incumbent",            "incumbent",           HEAD_DIM),
    ("incumbent_K256",       "incumbent",           2 * HEAD_DIM),
    ("F1_iota",              "F1_iota",             HEAD_DIM),
    ("F1_iota_K256",         "F1_iota",             2 * HEAD_DIM),
    ("F1_ref",               "F1_ref",              HEAD_DIM),
    ("F2_wfold_bf16",        "F2_wfold_bf16",       HEAD_DIM),
    ("F3_wfold_fp8",         "F3_wfold_fp8",        HEAD_DIM),
    ("F4_ones_f32",          "F4_ones_f32",         HEAD_DIM),
    ("F5_ones_f32_highest",  "F5_ones_f32_highest", HEAD_DIM),
    ("sum_vpu_CONTROL",      "sum_vpu",             HEAD_DIM),
    ("sum_ones_bf16_CONTROL", "sum_ones_bf16",      HEAD_DIM),
    ("incumbent_repeat",     "incumbent",           HEAD_DIM),
]

CONTROLS = {"sum_vpu_CONTROL", "sum_ones_bf16_CONTROL"}


def numerics():
    """Untimed: max relative error of every F arm against the incumbent.

    Same inputs, same kernel, one difference. The incumbent is the reference
    because the question F asks is whether it changed the arithmetic, not
    whether the arithmetic is right -- `correctness.py` asks that, against a
    float64 oracle at the real shapes.
    """
    ref = None
    out = []
    for label, arm, k in ([("incumbent", "incumbent", HEAD_DIM)]
                          + [a for a in ARMS
                             if a[0].startswith("F") and a[2] == HEAD_DIM]):
        try:
            fn, args = build(arm, k)
            v = np.asarray(jax.block_until_ready(fn(*args)))
        except Exception as exc:  # noqa: BLE001
            out.append(dict(label=label, error=repr(exc)[:300]))
            continue
        if ref is None:
            ref = v
            out.append(dict(label=label, norm_err=0.0, big_rel=0.0, exact=True,
                            note="reference"))
            continue
        fin = np.isfinite(ref) & np.isfinite(v)
        a, b = ref[fin], v[fin]
        scale = float(np.max(np.abs(a))) if a.size else 1.0
        # Normalised error -- max absolute deviation over the tensor's own
        # scale. This is the quantity a top-k cares about: an error small
        # against the largest score cannot move a rank far. A plain elementwise
        # relative error is not usable here, because `s_summed` passes through
        # zero (w is ~50% negative) and near a cancellation any absolute error
        # divides to O(1) while moving no rank at all.
        norm = float(np.max(np.abs(b - a))) / max(scale, 1e-30) if a.size else float("nan")
        # ...and the elementwise relative error restricted to the elements that
        # can actually be in a top-k: those within 100x of the row maximum.
        big = np.abs(a) > 0.01 * scale
        big_rel = (float(np.max(np.abs(b[big] - a[big]) / np.abs(a[big])))
                   if big.any() else float("nan"))
        out.append(dict(label=label, norm_err=norm, big_rel=big_rel,
                        exact=bool(np.array_equal(a, b)),
                        inf_match=bool(np.array_equal(np.isfinite(ref),
                                                      np.isfinite(v)))))
    return out


def run():
    rows = []
    for label, arm, k in ARMS:
        rec = dict(label=label, arm=arm, k=k, grid=GRID,
                   is_control=label in CONTROLS,
                   scoring_gflop_per_step=2 * BQ_SZ * NUM_HEADS * BKV_SZ * k / 1e9)
        if arm.startswith("F") or arm == "sum_ones_bf16":
            rec["reduce_gflop_per_step"] = (
                2 * BQ_SZ * (BQ_SZ * NUM_HEADS) * BKV_SZ / 1e9)
        try:
            fn, args = build(arm, k)
            med, spread = _time(fn, *args)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)[:900]
            rows.append(rec)
            print(json.dumps(rec), flush=True)
            continue
        rec.update(total_s=med, us_per_step=med / GRID * 1e6, spread_pct=spread)
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    return rows


def summarise(rows, nrows):
    by = {r["label"]: r for r in rows}
    base = by.get("incumbent")

    print("\n=== raw arms: us per inner step, grid=%d, "
          "shape (bq_sz=%d, heads=%d, bkv_sz=%d) ===" %
          (GRID, BQ_SZ, NUM_HEADS, BKV_SZ))
    print("Rows marked CONTROL are engineered to be subtracted. Their absolute "
          "time is not a finding.")
    print("`Delta time %` = (new - old)/old against the `incumbent` arm at "
          "K=128 measured in this same process. NEGATIVE IS FASTER.")
    hdr = (f"{'label':<24}{'K':>5}{'us/step':>10}{'spread%':>9}"
           f"{'Delta time %':>14}{'speedup x':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['label']:<24}{r['k']:>5}   ERROR {r['error'][:120]}")
            continue
        if base and "us_per_step" in base:
            d = ((r["us_per_step"] - base["us_per_step"])
                 / base["us_per_step"] * 100.0)
            sp = base["us_per_step"] / r["us_per_step"]
            ds, ss = f"{d:+.2f}", f"{sp:.3f}"
        else:
            ds, ss = "-", "-"
        print(f"{r['label']:<24}{r['k']:>5}{r['us_per_step']:>10.3f}"
              f"{r['spread_pct']:>9.2f}{ds:>14}{ss:>11}")

    # ---- drift control -------------------------------------------------
    rep = by.get("incumbent_repeat")
    if base and rep and "us_per_step" in base and "us_per_step" in rep:
        drift = ((rep["us_per_step"] - base["us_per_step"])
                 / base["us_per_step"] * 100.0)
        print(f"\nDRIFT CONTROL: incumbent {base['us_per_step']:.3f} us/step "
              f"(first) vs {rep['us_per_step']:.3f} us/step (last), "
              f"Delta time {drift:+.2f}%. Any arm-to-arm delta smaller than "
              f"|{drift:.2f}%| is not readable as an effect.")

    # ---- the attribution against the 3.858 us target -------------------
    sv = by.get("sum_vpu_CONTROL")
    so = by.get("sum_ones_bf16_CONTROL")
    print("\n=== step attribution ===")
    print("shard-3 reference decomposition of the 7.343 us step "
          "[measured (j-b54e9716-3db5-4f1f-b8d4)]:")
    print("   32-head-sum bundle 3.858 us | relu+w+scale+masks 2.401 us | "
          "MXU 1.084 us")
    if base and sv and "us_per_step" in sv:
        print(f"   this job, incumbent step         = "
              f"{base['us_per_step']:.3f} us/step")
        print(f"   this job, sum_vpu CONTROL        = "
              f"{sv['us_per_step']:.3f} us/step   "
              f"(scoring MXU + VPU head sum + store + fixed)")
    if so and "us_per_step" in so:
        print(f"   this job, sum_ones_bf16 CONTROL  = "
              f"{so['us_per_step']:.3f} us/step   "
              f"(scoring MXU + MXU head sum + store + fixed)")
    if sv and so and "us_per_step" in sv and "us_per_step" in so:
        d = so["us_per_step"] - sv["us_per_step"]
        pct = d / sv["us_per_step"] * 100.0
        print(f"   reduction swapped, everything else held: "
              f"{sv['us_per_step']:.3f} -> {so['us_per_step']:.3f} us/step, "
              f"Delta time {pct:+.2f}%")
        print("   ^ this pair is the cleanest read of 'what does moving the "
              "reduction cost or buy', because the two controls differ in "
              "exactly one thing. Both are CONTROLS; only the difference is a "
              "result.")

    # ---- did the MXU become the critical path? -------------------------
    print("\n=== is the MXU still off the critical path? (K ladder) ===")
    print("Doubling K doubles the SCORING matmul's FLOPs and changes nothing "
          "else. Shard 3 measured +2.18% on the control arm at K=128->256, "
          "i.e. hidden [measured (j-b54e9716-3db5-4f1f-b8d4)]. If F puts the "
          "MXU on the critical path, the same ladder gets steep.")
    for lo_l, hi_l in (("incumbent", "incumbent_K256"),
                       ("F1_iota", "F1_iota_K256")):
        lo, hi = by.get(lo_l), by.get(hi_l)
        if not lo or not hi or "us_per_step" not in lo or "us_per_step" not in hi:
            continue
        d = ((hi["us_per_step"] - lo["us_per_step"]) / lo["us_per_step"]
             * 100.0)
        print(f"   {lo_l:<12} K=128 {lo['us_per_step']:7.3f} us -> K=256 "
              f"{hi['us_per_step']:7.3f} us   Delta time {d:+.2f}%")

    print("\n=== numerics: max relative error vs the incumbent's own output ===")
    print("Same inputs, same kernel, one difference. Exactness is F's whole "
          "selling point; `bit-identical` is the only clean pass.")
    for r in nrows:
        if "error" in r:
            print(f"   {r['label']:<24} ERROR {r['error'][:110]}")
            continue
        print(f"   {r['label']:<24} norm_err={r['norm_err']:.3e}  "
              f"rel_err_on_top_1pct={r['big_rel']:.3e}  "
              f"bit_identical={r['exact']}  "
              f"inf_pattern_match={r.get('inf_match')}")
    print("\nReference for judging those: pure reassociation of the 32-term "
          "head sum is bounded by 2.46e-06 relative [verified j-ed517191], "
          "four decades below the 61-rank churn already present. bf16 rounding "
          "of the summands is ~4e-03 and is NOT a reassociation.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    global INTERPRET
    INTERPRET = a.interpret
    if a.tiny:
        _tiny()
        for i, (lab, arm, k) in enumerate(ARMS):
            ARMS[i] = (lab, arm, HEAD_DIM if k <= 128 else 2 * HEAD_DIM)
    print(f"jax {jax.__version__} devices {jax.devices()}", flush=True)
    rows = run()
    nrows = numerics()
    summarise(rows, nrows)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(timing=rows, numerics=nrows), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
