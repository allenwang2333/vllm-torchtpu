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
"""P3 -- what the two relayouts C5 and C8 exist to delete actually cost.

`step2/combined.md` applies the seams file's strike test to C5 and C8 and gets
the same answer for both: strike the relayout term and the two options are
IDENTICAL in FLOPs and in HBM bytes. The relayout is the entire difference, and
`hardware-facts.md` says plainly that no floor in the suite has a term for it.
So it cannot be modelled; it has to be measured, at this kernel's shapes.

The two moves, read off `streamindex_topk.py` at the `mixed_prod` shape
(seq_batch_size=1, bq_sz=128, num_q_heads=32, bkv_sz=2048):

  C5 / S1-c  `s_summed` (128, 2048) f32 -> (128, 16, 128)
             `streamindex_topk.py:416-418`, the reshape into
             `scores_block_x2_ref`. 262,144 elements per inner iteration.
             Area 4's largest layout row.

  C8 / S1-b  `bq_weights_stacked[:, :, :, None]`, `streamindex_topk.py:331`.
             At mixed the value is (128, 32) f32 and it is broadcast against
             an s of (128, 32, 2048). A lane->sublane move on 4,096 elements
             followed by a lane broadcast.

Why the arms are shaped the way they are
----------------------------------------
`hardware-facts.md`'s relayout table measured a move by putting it between two
arithmetic ops so nothing could fold it into a contraction, and reported each
arm against a control with the same output element count that performs no move.
This file follows that, with one addition: the C5 pair is measured in BOTH
directions, so the answer has an internal cross-check rather than resting on a
single subtraction.

  GROUP 1 -- C5, four arms. Every arm loads 262,144 f32, does the same
  `v*2+1` then `*3` arithmetic on it, and stores 262,144 f32.

    flat_flat    in (128,2048)   -> arith in (128,2048)   -> store (128,2048)
    flat_tiled   in (128,2048)   -> arith in (128,2048)   -> RESHAPE -> store (128,16,128)
    tiled_tiled  in (128,16,128) -> arith in (128,16,128) -> store (128,16,128)
    tiled_flat   in (128,16,128) -> arith in (128,16,128) -> RESHAPE -> store (128,2048)

  `flat_tiled - tiled_tiled` is C5's prize: today's kernel computes in the flat
  layout and reshapes; C5 computes in the tiled layout and does not.
  `tiled_flat - flat_flat` is the same move measured the other way round. The
  two must agree or the number is not a relayout cost.

  GROUP 2 -- C8, three arms at two lane widths. Every arm loads the same
  (128, 32, W) f32 and stores the same (128, 32, W) f32.

    w_bcast   o = s * w[:, :, None]     w_ref is (128, 32)      <- today
    w_pre     o = s * w3                w3_ref is (128, 32, 1)  <- C8's L13 arm:
                                        the weight arrives already on sublanes,
                                        so the move is deleted and the multiply
                                        is kept
    w_none    o = s                     <- C8's E1 arm upper bound: BOTH the
                                        move and the multiply are gone, which is
                                        what folding w into q buys

  `w_bcast - w_pre` prices L13. `w_bcast - w_none` prices E1 (minus the small
  multiply E1 adds on q, which is 4,096 elements and not measured here).

  NEGATIVE CONTROL -- `leading_axis`, (128,2048) -> (1,128,2048). Adding a
  leading axis is free per `hardware-facts.md`. If this arm reads as a cost, the
  harness is measuring something other than relayouts and no row in the file may
  be quoted.

What is a control and must not be lifted out
--------------------------------------------
`flat_flat`, `tiled_tiled` and `w_none` are engineered to be CANCELLABLE, not
realistic. `w_none` in particular is not "the kernel without the weight" -- it
is an arm that exists to be subtracted. Only the differences are results, and
the printout marks the arms accordingly.

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

# Grid length. Constant index_maps mean the blocks are fetched once and the
# loop is DMA-free, so this multiplies the body's cost without adding traffic
# -- the same construction `hardware-facts.md`'s relayout table used.
GRID = 64

WARMUP = 3
REPS = 20
BATCH = 8

VMEM_LIMIT = 128 * 2**20


def _median_spread(samples):
    med = statistics.median(samples)
    spread = (max(samples) - min(samples)) / med * 100.0 if med else float("nan")
    return med, spread


def _time(fn, *args, batch=BATCH, reps=REPS, warmup=WARMUP):
    """Per-call wall time. `batch` dispatches then one block, so the ~157 us
    host round trip is paid once per region rather than once per call."""
    jax.block_until_ready(fn(*args))
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(batch):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) / batch)
    return _median_spread(samples)


def _const(*shape):
    return lambda i: tuple(0 for _ in shape)


# --------------------------------------------------------------------------
# Group 1 -- the s_summed reshape
# --------------------------------------------------------------------------

def _g1_kernel(x_ref, o_ref, *, out_shape):
    v = x_ref[...]
    v = v * 2.0 + 1.0
    v = v.reshape(out_shape)
    o_ref[...] = v * 3.0


def build_g1(in_shape, out_shape):
    x = jnp.asarray(
        np.random.default_rng(0).standard_normal(in_shape).astype(np.float32))
    f = pl.pallas_call(
        functools.partial(_g1_kernel, out_shape=out_shape),
        grid=(GRID, ),
        in_specs=[pl.BlockSpec(in_shape, _const(*in_shape))],
        out_specs=pl.BlockSpec(out_shape, _const(*out_shape)),
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
    )
    return jax.jit(f), (x, )


# --------------------------------------------------------------------------
# Group 2 -- the w broadcast
# --------------------------------------------------------------------------

def _g2_bcast(s_ref, w_ref, o_ref):
    s = s_ref[...] * 2.0 + 1.0
    w = w_ref[...]
    o_ref[...] = s * w[:, :, None]


def _g2_pre(s_ref, w3_ref, o_ref):
    s = s_ref[...] * 2.0 + 1.0
    w3 = w3_ref[...]
    o_ref[...] = s * w3


def _g2_none(s_ref, o_ref):
    s = s_ref[...] * 2.0 + 1.0
    o_ref[...] = s


def build_g2(arm, rows, heads, width):
    rng = np.random.default_rng(1)
    s_shape = (rows, heads, width)
    s = jnp.asarray(rng.standard_normal(s_shape).astype(np.float32))
    specs = [pl.BlockSpec(s_shape, _const(*s_shape))]
    args = [s]
    if arm == "w_bcast":
        body = _g2_bcast
        w_shape = (rows, heads)
        args.append(jnp.asarray(rng.standard_normal(w_shape).astype(np.float32)))
        specs.append(pl.BlockSpec(w_shape, _const(*w_shape)))
    elif arm == "w_pre":
        body = _g2_pre
        w_shape = (rows, heads, 1)
        args.append(jnp.asarray(rng.standard_normal(w_shape).astype(np.float32)))
        specs.append(pl.BlockSpec(w_shape, _const(*w_shape)))
    elif arm == "w_none":
        body = _g2_none
    else:
        raise ValueError(arm)

    f = pl.pallas_call(
        body,
        grid=(GRID, ),
        in_specs=specs,
        out_specs=pl.BlockSpec(s_shape, _const(*s_shape)),
        out_shape=jax.ShapeDtypeStruct(s_shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
    )
    return jax.jit(f), tuple(args)


# --------------------------------------------------------------------------

def run(interpret_check=False, widths=(128, 512)):
    rows = []

    def add(name, group, elems, builder, note=""):
        try:
            fn, args = builder()
            med, spread = _time(fn, *args)
        except Exception as exc:  # noqa: BLE001
            r = dict(name=name, group=group, error=repr(exc)[:400])
            rows.append(r)
            print(json.dumps(r), flush=True)
            return
        r = dict(name=name, group=group, elements=elems,
                 total_s=med, ns_per_step=med / GRID * 1e9,
                 spread_pct=spread, note=note)
        rows.append(r)
        print(json.dumps(r), flush=True)

    # --- group 1: the (128, 2048) <-> (128, 16, 128) move ---
    FLAT = (128, 2048)
    TILED = (128, 16, 128)
    N1 = 128 * 2048
    add("g1_flat_flat", "c5", N1, lambda: build_g1(FLAT, FLAT),
        "CONTROL ARM -- no move, flat layout. Not a result on its own.")
    add("g1_flat_tiled", "c5", N1, lambda: build_g1(FLAT, TILED),
        "today: arithmetic flat, then reshape to (128,16,128)")
    add("g1_tiled_tiled", "c5", N1, lambda: build_g1(TILED, TILED),
        "CONTROL ARM -- no move, tiled layout. Not a result on its own.")
    add("g1_tiled_flat", "c5", N1, lambda: build_g1(TILED, FLAT),
        "the same move in reverse; cross-check on g1_flat_tiled")
    add("g1_leading_axis", "control", N1,
        lambda: build_g1(FLAT, (1, 128, 2048)),
        "NEGATIVE CONTROL -- adding a leading axis is free per "
        "hardware-facts.md. A cost here invalidates the whole file.")

    # --- group 2: the w broadcast ---
    for width in widths:
        n2 = 128 * 32 * width
        for arm, note in (
            ("w_bcast", "today: s * w[:, :, None]"),
            ("w_pre", "L13 arm: w arrives as (128,32,1), move deleted"),
            ("w_none", "CONTROL ARM -- no move and no multiply. E1 upper "
                       "bound. Not a result on its own."),
        ):
            add(f"g2_{arm}_w{width}", f"c8_w{width}", n2,
                functools.partial(build_g2, arm, 128, 32, width), note)

    return rows


def summarise(rows):
    by = {r["name"]: r for r in rows if "error" not in r}

    print("\n=== P3 arms (grid=%d, DMA-free, one TensorCore) ===" % GRID)
    hdr = (f"{'arm':<22}{'elements':>10}{'ns/step':>12}{'spread%':>9}  note")
    print(hdr)
    print("-" * 100)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<22}  ERROR {r['error'][:70]}")
            continue
        print(f"{r['name']:<22}{r['elements']:>10}{r['ns_per_step']:>12.1f}"
              f"{r['spread_pct']:>9.2f}  {r['note']}")

    print("\n=== the differences, which are the results ===")
    print("`d time %` is (arm - control)/control. Negative is faster. Every "
          "delta sits beside the two absolute ns/step it came from.")
    print(f"\n{'pair':<34}{'arm ns':>10}{'ctrl ns':>10}{'delta ns':>10}"
          f"{'ps/elem':>10}{'d time %':>10}")
    print("-" * 84)

    def pair(label, arm, ctrl):
        a, c = by.get(arm), by.get(ctrl)
        if not a or not c:
            print(f"{label:<34}  (missing arm)")
            return None
        d = a["ns_per_step"] - c["ns_per_step"]
        ps = d * 1e3 / a["elements"]
        pct = d / c["ns_per_step"] * 100.0
        print(f"{label:<34}{a['ns_per_step']:>10.1f}{c['ns_per_step']:>10.1f}"
              f"{d:>10.1f}{ps:>10.3f}{pct:>10.2f}")
        return d

    pair("C5 reshape (flat->tiled)", "g1_flat_tiled", "g1_tiled_tiled")
    pair("C5 reshape, reverse", "g1_tiled_flat", "g1_flat_flat")
    pair("layout-only (tiled vs flat arith)", "g1_tiled_tiled", "g1_flat_flat")
    pair("NEG CTRL leading axis (want ~0)", "g1_leading_axis", "g1_flat_flat")
    for width in (128, 512, 1024, 2048):
        if f"g2_w_bcast_w{width}" in by:
            pair(f"C8 L13: broadcast only, W={width}",
                 f"g2_w_bcast_w{width}", f"g2_w_pre_w{width}")
            pair(f"C8 E1: broadcast+multiply, W={width}",
                 f"g2_w_bcast_w{width}", f"g2_w_none_w{width}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--widths", default="128,512")
    a = ap.parse_args(argv)
    widths = tuple(int(x) for x in a.widths.split(",") if x)
    rows = run(widths=widths)
    summarise(rows)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
