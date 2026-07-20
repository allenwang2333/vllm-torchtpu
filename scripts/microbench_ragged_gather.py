"""Micro-benchmark of vllm_torchtpu.kernels.sparse_core.ragged_gather at
production shapes with vs without needs_layout_passes=False.

Hypothesis: JAX 0.10.2 defaulted CompilerParams.needs_layout_passes to True
("The layout passes are still a work in progress" — pallas changelog).
ragged_gather.py did not add the explicit override, so on this branch the
kernel now runs through the WIP layout passes. If this is the regression's
source, forcing False here should yield materially faster per-call latency
and/or a cleaner emitted MLIR/HLO.

Shapes: Qwen3.5-397B-A17B FP8, hidden=4096, topk=10.
- Prefill worst-case (per DP rank): (num_tokens=1024, hidden=4096) →
  indices size = num_tokens * topk = 10240.
- Decode (per DP rank, MAX_NUM_SEQS=64): num_tokens=64 → indices size=640.

Usage:
  python scripts/microbench_ragged_gather.py [--dump-hlo]
"""

import argparse
import functools
import os
import time

# Route HLO dumps to a per-invocation subdir when requested.
_DUMP_BASE = os.environ.get("HLO_DUMP_BASE")


import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper
from vllm_torchtpu.kernels.sparse_core import ragged_gather as rg


def _build_kernel(needs_layout_passes: bool | None):
    """Return a jitted function that runs the ragged_gather kernel with the
    given needs_layout_passes setting. None = leave the default (True on 0.10.2)."""

    @jax.jit
    def _run(x, indices, start, end):
        assert x.ndim == 2
        assert indices.ndim == 1

        dtype = x.dtype
        sc_info = pltpu.get_tpu_info().sparse_core
        assert sc_info is not None, "SparseCore required"

        hidden_size = x.shape[-1]
        out_size = indices.size
        num_simd_lanes = sc_info.num_lanes
        num_cores = sc_info.num_cores * sc_info.num_subcores
        block_size = num_simd_lanes * num_cores
        col_size = rg.calculate_col_size(hidden_size)

        out_pad_size = pl.cdiv(out_size, block_size) * block_size - out_size
        indices = jnp.pad(indices, ((0, out_pad_size)))
        aligned_hidden_size = pl.cdiv(hidden_size, col_size) * col_size

        vector_mesh = plsc.VectorSubcoreMesh(
            num_cores=sc_info.num_cores,
            num_subcores=sc_info.num_subcores,
            core_axis_name="core",
            subcore_axis_name="subcore",
        )

        cp_kwargs = dict(use_tc_tiling_on_sc=True, disable_bounds_checks=True)
        if needs_layout_passes is not None:
            cp_kwargs["needs_layout_passes"] = needs_layout_passes
        compiler_params = pltpu.CompilerParams(**cp_kwargs)

        return core_map_helper.kernel(
            functools.partial(
                rg.main_kernel,
                core_axis_name=vector_mesh.core_axis_name,
                subcore_axis_name=vector_mesh.subcore_axis_name,
            ),
            out_type=jax.ShapeDtypeStruct(
                (out_size + out_pad_size, aligned_hidden_size), dtype),
            compiler_params=compiler_params,
            scratch_types=dict(
                start_vmem_ref=pltpu.VMEM((num_simd_lanes, ), jnp.int32),
                end_vmem_ref=pltpu.VMEM((num_simd_lanes, ), jnp.int32),
                out_vmem_ref=pltpu.VMEM((num_simd_lanes, col_size), jnp.uint32),
                indices_vmem_ref=pltpu.VMEM((num_simd_lanes, ), jnp.int32),
                sem_ref=pltpu.SemaphoreType.DMA((2, )),
            ),
            mesh=vector_mesh,
            name="sc_ragged_gather",
        )(start, end, x, indices)[:out_size, :hidden_size]

    return _run


def _make_inputs(num_tokens, hidden, topk, seed):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (num_tokens, hidden), dtype=jnp.bfloat16)
    # Random gather indices in [0, num_tokens)
    indices = jax.random.randint(
        k2, (num_tokens * topk, ), minval=0, maxval=num_tokens, dtype=jnp.int32)
    start = jnp.array([0], dtype=jnp.int32)
    end = jnp.array([indices.size], dtype=jnp.int32)
    return x, indices, start, end


def _time_run(run, args, iters=200, warmup=20):
    # Warmup — includes compile on the first call.
    for _ in range(warmup):
        out = run(*args)
    out.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = run(*args)
    out.block_until_ready()
    dt = time.perf_counter() - t0
    return dt / iters * 1e6  # µs per call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-hlo", action="store_true",
                    help="Also dump HLO to $HLO_DUMP_BASE/{default,forced_false}")
    args = ap.parse_args()

    shapes = [
        ("prefill_1024", 1024, 4096, 10),
        ("decode_64", 64, 4096, 10),
    ]

    only_false = os.environ.get("BENCH_ONLY_FALSE") == "1"
    if only_false:
        print(f"{'shape':<16} {'needs_layout=False':>22}")
        print("-" * 44)
    else:
        print(f"{'shape':<16} {'default(0.10.2)':>18} {'needs_layout=False':>22} {'delta':>10}")
        print("-" * 70)
    for name, num_tokens, hidden, topk in shapes:
        # Different keys per shape so gather is nontrivial.
        inputs = _make_inputs(num_tokens, hidden, topk, seed=hash(name) & 0xFFFF)

        run_forced = _build_kernel(needs_layout_passes=False)
        us_forced = _time_run(run_forced, inputs)

        if only_false:
            print(f"{name:<16} {us_forced:>18.2f} µs")
            continue

        # Default: needs_layout_passes=None means "use the JAX default", which
        # is True on 0.10.2 for pallas.tpu.CompilerParams.
        run_default = _build_kernel(needs_layout_passes=None)
        us_default = _time_run(run_default, inputs)

        pct = (us_default - us_forced) / us_default * 100.0
        print(f"{name:<16} {us_default:>15.2f} µs {us_forced:>18.2f} µs {pct:>8.1f}%")


if __name__ == "__main__":
    main()
