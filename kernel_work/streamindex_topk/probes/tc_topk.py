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
"""§9.4 item 1 of `measurement_and_design.md` -- what does a TensorCore-side
top-2048 cost?

The question, in one line: Option D moves top-`k` selection off the SparseCore
onto the TensorCore, and the cost of doing so is `unmodelled`. Struck from the
ledger, D goes to rank 1 with -58.82%. Priced above ~394 us at `mixed_prod`,
D loses. One number spans the whole ranking, which is the definition of a
defect at step 2.

Probe the PRIMITIVE, not the variant
------------------------------------
The answer wanted is "what does top-2048-of-N cost on the TensorCore at 1024
rows", not "how fast is Option D". Written as the primitive it is reusable by
any later design that wants selection on the TC, and it cannot be contaminated
by whatever else D does (the second einsum pass, the deleted round trip).

The arms
--------
Reference, measured in the SAME job so the comparison is a ratio and not a
cross-job quote:

  `sc_topk`     `sparsecore_topk` on the same synthetic scores tensor. This is
                exactly `instruments/microbench.py`'s `topk` arm and it is the
                incumbent. It is a RESULT here, not a control -- it is the
                thing D proposes to delete.

TensorCore formulations:

  `xla_topk`    `jax.lax.top_k(x, 2048)` in plain XLA. The simplest TC-side
                formulation that exists, and exact.
  `xla_approx`  `jax.lax.approx_max_k(x, 2048, recall_target=0.99)`. An
                approximation -- it would need C's rank gate before it could
                ship -- but D's area 6 already contemplates an approximate
                mechanism, so it belongs on the ladder.
  `pallas_topk` `jax.lax.top_k` inside a `pallas_call`. FEASIBILITY only: D
                needs the selection fused into the scoring kernel, and if the
                op has no Mosaic lowering that is a hard structural answer, not
                a slow one. UNSUPPORTED and FAIL are different findings.

The reusable per-pass primitive, for any counting / bisection scheme (D's C11
threshold-first mechanism is one):

  `pass_count`  one full compare-and-count pass over the array in Pallas:
                compare against a per-row threshold, reduce along the row.
  `pass_copy`   CONTROL for the above. Identical BlockSpec, so the identical
                bytes are DMA'd into VMEM, with the compare and the reduction
                removed. Its absolute time is NOT a finding -- it exists so
                that `pass_count - pass_copy` isolates the selection work from
                the memory traffic. A bisection over the 32 bits of a monotone
                int32 key costs 32 x `pass_count`, and that product is the
                number to compare against ~394 us.

Shapes
------
Rows are 1024 throughout: `max_num_batched_tokens` at the A/B shape, which is
also `mixed_prod`'s and `mixed_64k`'s `num_tokens`.

  W = 10240   `mixed_prod`'s scores width (num_sublanes_total 80 x 128). This
              is the width behind the 399.60 us SparseCore figure D is priced
              against [measured: j-5f8f4a08-20b4-4fa0-bfc3].
  W = 67584   the A/B shape's width (num_sublanes_total 528 x 128, from
              max_model_len 66560). This is the one the end-to-end benchmark
              would actually run.

Sign convention: percentages are changes in TIME, (new - old)/old. Negative is
faster.
"""

import argparse
import json
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROWS = 1024
K = 2048
WIDTHS = [10240, 67584]

WARMUP = 2
REPS = 5
BATCH = 2
RTILE = 8
VMEM_LIMIT = 128 * 2**20

# PROBING.md's dry-run path -- see `vpu_mxu_strike.py`. A dry-run number is not
# a measurement and is never quoted.
INTERPRET = False


def _tiny():
    global ROWS, K, RTILE, WARMUP, REPS, BATCH
    ROWS, K, RTILE = 16, 8, 8
    WARMUP, REPS, BATCH = 0, 1, 1

# `report.md`'s whole-kernel achieved HBM ceiling, measured, not the spec rate.
HBM_BYTES_PER_S = 2.97e12


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


def _scores(width, live):
    """The distribution the scoring kernel actually emits: random normals in
    the live columns, -inf in the padding. `microbench.py` records why an
    all-zeros buffer is not usable -- every element ties and `sparsecore_topk`
    goes pathological."""
    rng = np.random.default_rng(1234)
    arr = np.full((ROWS, width), -np.inf, dtype=np.float32)
    n = min(live, width)
    arr[:, :n] = (rng.standard_normal((ROWS, n)) * 10.0).astype(np.float32)
    return jnp.asarray(arr), n


def _pass_body(x_ref, thr_ref, o_ref, *, count):
    if count:
        x = x_ref[...]
        m = (x > thr_ref[:, :1]).astype(jnp.int32)
        c = jnp.sum(m, axis=1)
        o_ref[...] = jnp.broadcast_to(c[:, None], o_ref.shape)
    else:
        # CONTROL: the BlockSpec still DMAs the whole (RTILE, W) block into
        # VMEM; only the compare and the reduction are gone.
        o_ref[...] = x_ref[:, :o_ref.shape[1]].astype(jnp.int32)


def _build_pass(width, count):
    import functools
    grid = ROWS // RTILE
    fn = pl.pallas_call(
        functools.partial(_pass_body, count=count),
        grid=(grid, ),
        in_specs=[
            pl.BlockSpec((RTILE, width), lambda i: (i, 0)),
            pl.BlockSpec((RTILE, 128), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((RTILE, 128), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((ROWS, 128), jnp.int32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
        **({"interpret": pltpu.InterpretParams()} if INTERPRET else {}),
    )
    return jax.jit(fn)


def _build_pallas_topk(width):
    def body(x_ref, o_ref):
        o_ref[...] = jax.lax.top_k(x_ref[...], K)[1]

    fn = pl.pallas_call(
        body,
        grid=(ROWS, ),
        in_specs=[pl.BlockSpec((1, width), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, K), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((ROWS, K), jnp.int32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
        **({"interpret": pltpu.InterpretParams()} if INTERPRET else {}),
    )
    return jax.jit(fn)


def run(widths, arms):
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
        sparsecore_topk)

    rows = []
    for width in widths:
        scores, live = _scores(width, width)
        row_lengths = jnp.full((ROWS, ), live, jnp.int32)
        thr = jnp.zeros((ROWS, 128), jnp.float32)
        read_bytes = ROWS * width * 4

        builders = {
            "sc_topk": lambda: (
                jax.jit(lambda s, r: sparsecore_topk(
                    s, K, row_lengths=r, write_empty_rows=True)),
                (scores, row_lengths)),
            "xla_topk": lambda: (
                jax.jit(lambda s: jax.lax.top_k(s, K)[1]), (scores, )),
            "xla_approx": lambda: (
                jax.jit(lambda s: jax.lax.approx_max_k(
                    s, K, recall_target=0.99)[1]), (scores, )),
            "pallas_topk": lambda: (_build_pallas_topk(width), (scores, )),
            "pass_count": lambda: (_build_pass(width, True), (scores, thr)),
            "pass_copy": lambda: (_build_pass(width, False), (scores, thr)),
        }

        for arm in arms:
            rec = dict(arm=arm, rows=ROWS, width=width, k=K,
                       read_MB=read_bytes / 1e6,
                       hbm_read_floor_us=read_bytes / HBM_BYTES_PER_S * 1e6,
                       is_control=(arm == "pass_copy"))
            try:
                fn, args = builders[arm]()
                med, spread = _time(fn, *args)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = repr(exc)[:900]
                rows.append(rec)
                print(json.dumps(rec), flush=True)
                continue
            rec.update(us=med * 1e6, spread_pct=spread)
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    return rows


def summarise(rows, widths):
    print(f"\n=== TensorCore-side top-{K} of W, at {ROWS} rows ===")
    print("`pass_copy` is a CONTROL ARM: same DMA, no compare, no reduction. "
          "Its absolute time is not a finding.")
    hdr = (f"{'width':>8}  {'arm':<14}{'us':>13}{'spread%':>10}"
           f"{'readMB':>10}{'hbm floor us':>15}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['width']:>8}  {r['arm']:<14}   ERROR (full string in "
                  f"the json line above)")
            continue
        print(f"{r['width']:>8}  {r['arm']:<14}{r['us']:>13.2f}"
              f"{r['spread_pct']:>10.2f}{r['read_MB']:>10.1f}"
              f"{r['hbm_read_floor_us']:>15.1f}")

    print("\n=== ratios against the incumbent, same job, same shape ===")
    print("`D time %` is (TC arm - sc_topk) / sc_topk. Negative is faster, "
          "i.e. the TensorCore would win.")
    for width in widths:
        sc = next((r for r in rows if r["width"] == width
                   and r["arm"] == "sc_topk" and "error" not in r), None)
        if not sc:
            print(f"  width {width}: sc_topk did not produce a number; no "
                  f"ratio is quotable here.")
            continue
        for r in rows:
            if (r["width"] != width or r["arm"] == "sc_topk"
                    or "error" in r or r.get("is_control")):
                continue
            d = (r["us"] - sc["us"]) / sc["us"] * 100.0
            print(f"  width {width:>6}  {r['arm']:<14} {r['us']:>10.2f} us "
                  f"vs sc_topk {sc['us']:>10.2f} us   D time % {d:>+9.2f}")

    print("\n=== the reusable per-pass primitive ===")
    for width in widths:
        cnt = next((r for r in rows if r["width"] == width
                    and r["arm"] == "pass_count" and "error" not in r), None)
        cpy = next((r for r in rows if r["width"] == width
                    and r["arm"] == "pass_copy" and "error" not in r), None)
        if not cnt or not cpy:
            continue
        diff = cnt["us"] - cpy["us"]
        print(f"  width {width:>6}: pass_count {cnt['us']:.2f} us - "
              f"pass_copy(control) {cpy['us']:.2f} us = {diff:.2f} us of "
              f"compare+reduce per pass.")
        print(f"                 whole pass (traffic included) "
              f"{cnt['us']:.2f} us; a 32-step bisection over a monotone "
              f"int32 key = {cnt['us'] * 32:.1f} us [modelled from this "
              f"measured pass].")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    ap.add_argument("--arms",
                    default="sc_topk,xla_topk,xla_approx,pallas_topk,"
                            "pass_count,pass_copy")
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
    widths = [int(x) for x in a.widths.split(",") if x]
    arms = [x for x in a.arms.split(",") if x]
    rows = run(widths, arms)
    summarise(rows, widths)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
