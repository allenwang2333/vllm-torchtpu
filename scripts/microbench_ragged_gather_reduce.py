"""Micro-benchmark of ragged_gather_reduce with needs_layout_passes True vs False.

The branch's ragged_gather_reduce.py explicitly sets needs_layout_passes=False
because vector.bitcast broke under 0.10.2's default. This bench checks whether
the override *also* has a perf side effect (for or against) at production shape.
"""

import argparse
import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper
from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce as rgr


def _build_kernel(needs_layout_passes: bool | None):

    @jax.jit
    def _run(x, indices, topk_weights, valid_rows_mask, num_src_rows_per_partition):
        sc_info = pltpu.get_tpu_info().sparse_core
        assert sc_info is not None
        hidden_size = x.shape[-1]
        num_simd_lanes = sc_info.num_lanes
        num_cores = sc_info.num_cores * sc_info.num_subcores
        # Same partitioning helpers as production.
        num_column_partitions = rgr._calculate_num_col_column_partitions(
            hidden_size, num_cores, pltpu.get_tpu_info().num_lanes)
        num_row_partitions = num_cores // num_column_partitions
        col_size = x.shape[-1] // num_column_partitions
        vector_mesh = plsc.VectorSubcoreMesh(
            num_cores=sc_info.num_cores,
            num_subcores=sc_info.num_subcores,
            core_axis_name="core",
            subcore_axis_name="subcore",
        )
        cp_kwargs = dict(use_tc_tiling_on_sc=True, disable_bounds_checks=True)
        if needs_layout_passes is not None:
            cp_kwargs["needs_layout_passes"] = needs_layout_passes
        return core_map_helper.kernel(
            functools.partial(
                rgr.main_kernel,
                core_axis_name=vector_mesh.core_axis_name,
                subcore_axis_name=vector_mesh.subcore_axis_name,
                num_row_partitions=num_row_partitions,
                num_column_partitions=num_column_partitions,
            ),
            out_type=jax.ShapeDtypeStruct(
                (x.shape[0] // 10, x.shape[1]), jnp.float32),
            compiler_params=pltpu.CompilerParams(**cp_kwargs),
            scratch_types=dict(
                num_rows_per_row_partition_vmem_ref=pltpu.VMEM(
                    (num_simd_lanes,), jnp.int32),
                out_vmem_ref=pltpu.VMEM(
                    (num_simd_lanes, col_size), jnp.uint32),
                prev_iter_last_row_vmem_ref=pltpu.VMEM(
                    (1, col_size), jnp.uint32),
                src_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
                dst_indices_vmem_ref=pltpu.VMEM((num_simd_lanes,), jnp.int32),
                topk_weights_vmem_ref=pltpu.VMEM(
                    (num_simd_lanes,), jnp.float32),
                sorted_by_validity_vmem_ref=pltpu.VMEM(
                    (num_simd_lanes,), jnp.int32),
                sem_ref=pltpu.SemaphoreType.DMA((2,)),
            ),
            mesh=vector_mesh,
            name="sc_ragged_gather_reduce",
        )(num_src_rows_per_partition, x, indices, topk_weights, valid_rows_mask)

    return _run


def _make_inputs(num_tokens, hidden, topk, seed):
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
    # After the MoE experts, we have (num_tokens*topk, hidden).
    x = jax.random.normal(k1, (num_tokens * topk, hidden), dtype=jnp.bfloat16)
    # indices into `num_tokens` (which slot each pre-expert row belongs to).
    indices = jax.random.randint(
        k2, (num_tokens * topk,), 0, num_tokens, dtype=jnp.int32)
    topk_weights = jax.random.uniform(
        k3, (num_tokens * topk,), dtype=jnp.float32)
    valid_mask = jax.random.bernoulli(k4, 0.5, (num_tokens * topk,))
    # Faked-up row-partition counts; production's _preprocess computes real
    # values but for latency bench random plausible values are fine.
    num_row = jax.numpy.array([1] * 8, dtype=jnp.int32)
    return x, indices, topk_weights, valid_mask, num_row


def _time_run(run, args, iters=100, warmup=10):
    for _ in range(warmup):
        out = run(*args)
    out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = run(*args)
    out.block_until_ready()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    print(f"{'shape':<16} {'default(=False)':>18} {'True':>16} {'delta':>10}")
    print("-" * 66)
    # Only prefill is meaningful — decode goes to XLA fallback per the size guard.
    for name, num_tokens, hidden, topk in [
        ("prefill_1024", 1024, 4096, 10),
    ]:
        inputs = _make_inputs(num_tokens, hidden, topk, seed=hash(name) & 0xFFFF)
        # Default: needs_layout_passes=False (as branch currently sets)
        run_false = _build_kernel(needs_layout_passes=False)
        us_false = _time_run(run_false, inputs)
        # True (default): note this may fail to compile due to vector.bitcast
        try:
            run_true = _build_kernel(needs_layout_passes=True)
            us_true = _time_run(run_true, inputs)
            pct = (us_true - us_false) / us_true * 100.0
            print(f"{name:<16} {us_false:>15.2f} µs {us_true:>13.2f} µs {pct:>8.1f}%")
        except Exception as exc:
            print(f"{name:<16} {us_false:>15.2f} µs   True=<compile FAIL>: {type(exc).__name__}: {str(exc)[:80]}")


if __name__ == "__main__":
    main()
