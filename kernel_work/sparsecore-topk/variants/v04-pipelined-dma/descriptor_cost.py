"""What does one extra SparseCore DMA descriptor cost, at constant bytes?

This is the one number v4 turns on and the only one nothing in the tree
measures. `sctopk_2stage_bench.py --bw` sweeps the *wave* count at a fixed
slice, so bytes and descriptors move together and its 0.613 us/wave slope is
`transfer + descriptor` inseparably. v4 holds the bytes roughly fixed and
multiplies the descriptor count by `n / chunk`, so the two have to be split
before the plan's "-7.1%" means anything: at `g = 0` v4 wins, at `g = 100 ns`
it is roughly a wash at production occupancy and a loss at rho = 1.

Design: **measure a slope, do not divide a total.** Total bytes per wave are
held at `slice_len` words per subcore for every arm; only the number of copies
that move them changes. `time(C) = a + g_eff * C` and `g_eff` is the answer.

  raw/c{C}    C blocking copies of slice_len/C words per subcore per wave.
              Slope = the raw descriptor cost.
  pipe/c{C}   the same C copies through `pltpu.emit_pipeline`, trivial body.
              Slope - raw slope = the pipeline's own per-step bookkeeping.
              With a trivial body there is nothing to overlap, so this is
              overhead only, which is the point: it is v4's floor cost.

`C` runs over divisors of `slice_len / LANES * LANES` so no arm pays a ragged
tail the others do not. Three drift nulls at the same `C` register the noise
floor -- one null is a number, not a floor.

Harness rules are inherited by IMPORT from the frozen instruments
(`glm_dsa_block_bench.measure`/`dispatch_floor`): block once per rep batch, two
rep counts with a paired slope, interleaved rounds, every variant warmed before
round 0. Read `us_median`. Nothing in `instruments/` is modified.

  uv_venv/bin/python descriptor_cost.py [--ctx 9216] [--waves 16] [--rounds 9]

Needs a TPU. Not run by the agent that wrote it.
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import sys

import vllm_torchtpu  # noqa: F401  (env_override must precede jax)
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

INSTRUMENTS = pathlib.Path(__file__).parents[2] / "instruments"
sys.path.insert(0, str(INSTRUMENTS))
from glm_dsa_block_bench import dispatch_floor, measure  # noqa: E402
from sctopk_2stage_bench import describe, topk_n  # noqa: E402

LANES = 16
SEED = 20260910
PREFILL_CHUNK = 1024


def _kernel(scores_hbm, out_hbm, resident_vmem, acc_vmem, sems, *, b,
            slice_len, chunk, num_waves, pipelined):
    """Move `slice_len` words per subcore per wave in `slice_len/chunk` copies.

    The accumulate keeps the copies alive: one lane vector added per chunk,
    against `chunk` words of transfer, is not a perturbation of the thing being
    timed but it is enough that nothing can be dead-code eliminated.
    """
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * 16 + sub
    nc = slice_len // chunk

    def wave_body(w, _):
        row = jnp.minimum(w * 32 + flat, b - 1)

        if pipelined:

            def body(chunk_vmem):
                acc_vmem[pl.ds(0, LANES)] = (acc_vmem[pl.ds(0, LANES)] +
                                             chunk_vmem[pl.ds(0, LANES)])

            pltpu.emit_pipeline(
                body,
                grid=(nc, ),
                in_specs=(pl.BlockSpec((chunk, ), lambda c: (row * nc + c, )), ),
            )(scores_hbm)
        else:

            def one(c):
                cp = pltpu.make_async_copy(
                    scores_hbm.at[pl.ds(row * slice_len + c * chunk, chunk)],
                    resident_vmem.at[pl.ds(c * chunk, chunk)], sems.at[0])
                cp.start()
                cp.wait()
                acc_vmem[pl.ds(0, LANES)] = (
                    acc_vmem[pl.ds(0, LANES)] +
                    resident_vmem[pl.ds(c * chunk, LANES)])

            for c in range(nc):
                one(c)
        return 0

    lax.fori_loop(0, num_waves, wave_body, 0)

    # One writer: 32 concurrent DMAs into the same 16 words is a race whose
    # cost would land inside the number being measured.
    @pl.when(flat == 0)
    def _():
        cp = pltpu.make_async_copy(acc_vmem.at[pl.ds(0, LANES)],
                                   out_hbm.at[pl.ds(0, LANES)], sems.at[1])
        cp.start()
        cp.wait()


def make_thunk(src, slice_len, chunk, num_waves, pipelined):
    """Build the kernel ONCE and return a 0-arg jitted thunk.

    Constructing `pl.kernel` inside the thunk re-compiles on every timed call
    and `measure`'s paired slope then reads as minus the compile time -- the
    arm comes out fastest on the board.
    """
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    kern = pl.kernel(
        functools.partial(_kernel,
                          b=src.shape[0],
                          slice_len=slice_len,
                          chunk=chunk,
                          num_waves=num_waves,
                          pipelined=pipelined),
        out_type=jax.ShapeDtypeStruct((LANES, ), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        scratch_types=(
            pltpu.VMEM((slice_len, ), jnp.int32),
            pltpu.VMEM((LANES, ), jnp.int32),
            pltpu.SemaphoreType.DMA((2, )),
        ),
        mesh=plsc.VectorSubcoreMesh(num_cores=sc.num_cores,
                                    num_subcores=sc.num_subcores,
                                    core_axis_name="core",
                                    subcore_axis_name="subcore"),
        # The arm is in the name, so a profile scope proves which one ran.
        name=f"sc_desc_{'pipe' if pipelined else 'raw'}_c{chunk}_w{num_waves}",
    )
    f = jax.jit(kern)
    flat = src.reshape(-1)
    return lambda: f(flat)


def fit(xs, ys):
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    slope, icpt = np.polyfit(xs, ys, 1)
    resid = float(np.abs(ys - (slope * xs + icpt)).max())
    return float(slope), float(icpt), resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=9216,
                    help="max_model_len; 9216 is the production P=1 shape and "
                         "is NOT the bench default")
    ap.add_argument("--waves", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    n = topk_n(args.ctx)
    slice_len = describe(PREFILL_CHUNK, n)["slice_len"]
    counts = [c for c in (1, 2, 4, 5, 8, 10, 20) if slice_len % c == 0
              and (slice_len // c) % LANES == 0]
    print(f"=== DESCRIPTOR COST  slice_len={slice_len} words "
          f"({slice_len * 4 / 1024:.0f} KiB/subcore/wave), waves={args.waves}")
    print(f"  chunk counts: {counts}  "
          f"(chunk words: {[slice_len // c for c in counts]})")
    print(f"  bytes per wave is CONSTANT at "
          f"{32 * slice_len * 4 / 2**20:.2f} MiB across every arm")

    rng = np.random.default_rng(SEED)
    rows = args.waves * 32
    keep, variants = [], {}
    for pipelined in (False, True):
        tag = "pipe" if pipelined else "raw"
        for c in counts:
            src = jnp.asarray(
                rng.standard_normal((rows, slice_len),
                                    dtype=np.float32).view(np.int32))
            keep.append(src)
            variants[f"{tag}/c{c}"] = make_thunk(src, slice_len, slice_len // c,
                                                 args.waves, pipelined)
    # Three nulls at the largest chunk count, where the slope is read.
    for j in range(3):
        src = jnp.asarray(
            rng.standard_normal((rows, slice_len),
                                dtype=np.float32).view(np.int32))
        keep.append(src)
        variants[f"raw/c{counts[-1]}#null{j}"] = make_thunk(
            src, slice_len, slice_len // counts[-1], args.waves, False)

    floor = dispatch_floor()
    print(f"  dispatch floor: {floor:.2f} us/call", flush=True)
    res = measure(variants, rounds=args.rounds)

    nulls = [res[k]["us_median"] for k in res if "#null" in k]
    spread = (max(nulls) - min(nulls)) / np.mean(nulls) * 100
    print(f"  three-null spread: {spread:.2f}% "
          f"({min(nulls):.3f} .. {max(nulls):.3f} us)")

    out = {"slice_len": slice_len, "waves": args.waves, "counts": counts,
           "null_spread_pct": float(spread), "dispatch_floor_us": floor,
           "rows": {}, "fits": {}}
    for tag in ("raw", "pipe"):
        xs, ys = [], []
        for c in counts:
            r = res[f"{tag}/c{c}"]
            xs.append(c)
            ys.append(r["us_median"])
            out["rows"][f"{tag}/c{c}"] = r
            print(f"  {tag}/c{c:<3} chunk={slice_len // c:<6} "
                  f"us_median={r['us_median']:9.3f}  us={r['us']:9.3f}  "
                  f"hostbound={r['hostbound_pct']:5.1f}%  "
                  f"suspect={r['suspect']}")
        slope, icpt, resid = fit(xs, ys)
        per_desc_ns = slope / args.waves * 1000
        out["fits"][tag] = {"slope_us_per_chunkcount": slope,
                            "intercept_us": icpt, "max_resid_us": resid,
                            "per_descriptor_ns": per_desc_ns}
        print(f"  -> {tag}: {slope:+.4f} us per extra chunk-per-wave "
              f"(intercept {icpt:.3f} us, max resid {resid:.3f} us) "
              f"= {per_desc_ns:+.1f} ns per descriptor per subcore")

    g_raw = out["fits"]["raw"]["per_descriptor_ns"]
    g_pipe = out["fits"]["pipe"]["per_descriptor_ns"]
    print(f"\n  g (raw descriptor)      = {g_raw:+.1f} ns")
    print(f"  g (through emit_pipeline)= {g_pipe:+.1f} ns   "
          f"[bookkeeping = {g_pipe - g_raw:+.1f} ns]")
    print("  v4 at production occupancy adds ~3.5 descriptors per row per "
          f"subcore at chunk=1024, i.e. ~{3.5 * 32 * g_pipe / 1000:+.2f} us "
          "over 32 waves, against a 13.469 us score-DMA prize.")
    print("  READ THIS AS: if that figure exceeds ~11 us, v4 cannot win at "
          "chunk=1024 and the sweep must move to a larger chunk.")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
