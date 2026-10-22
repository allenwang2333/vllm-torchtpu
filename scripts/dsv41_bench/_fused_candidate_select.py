# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental SC score-compaction/selection fusion for the 96K/16K fixture.

Two VMEM scopes reuse the full-row staging allocation after gathering. This
prototype is benchmark-only until its full-path correctness and timing pass.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    RADIX_BITS,
    RADIX_BUCKETS,
    _compact,
    _sort_selected,
)


def fused_select(scores, candidates, *, k=512, block_size=8, unroll_selection=False):
    rows, width = scores.shape
    count = candidates.shape[1]
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    alignment = pltpu.get_tpu_info().num_lanes
    capacity = count * block_size + lanes
    unroll = alignment // lanes if unroll_selection else 1
    resident = 4 * (capacity + count)
    scratch_peak = 4 * max(width, 4 * capacity + 2 * k + RADIX_BUCKETS)
    assert resident + scratch_peak <= info.vmem_capacity_bytes
    assert k % alignment == count % alignment == width % alignment == 0
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        mesh=mesh,
        out_type=jax.ShapeDtypeStruct((rows, k), jnp.int32),
        scratch_types=(
            pltpu.VMEM((capacity,), jnp.float32),
            pltpu.VMEM((count,), jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="candidate_select_fused",
    )
    def kernel(src, ids_src, out, compact, metadata, sem):
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        first = jax.lax.axis_index(("core", "subcore"))

        @pl.loop(first, rows, step=mesh.num_cores * mesh.num_subcores)
        def row(r):
            def gather_row(raw):
                copy = pltpu.async_copy(src.at[r], raw, sem.at[0])
                pltpu.async_copy(ids_src.at[r], metadata, sem.at[1]).wait()
                copy.wait()

                @plsc.parallel_loop(
                    0, count * block_size, lanes, unroll=alignment // lanes
                )
                def gather(i):
                    slots = (i + lane_ids).astype(jnp.uint32)
                    blocks = plsc.load_gather(
                        metadata, [(slots // block_size).astype(jnp.int32)]
                    )
                    logical = blocks * block_size + (slots % block_size).astype(
                        jnp.int32
                    )
                    valid = (blocks >= 0) & (logical < width)
                    values = plsc.load_gather(raw, [logical], mask=valid)
                    compact[pl.ds(i, lanes)] = jnp.where(valid, values, -jnp.inf)

            pl.run_scoped(gather_row, pltpu.VMEM((width,), jnp.float32))

            def select_row(keys, ids, selected, selected_ids, hist):
                @plsc.parallel_loop(0, count * block_size, lanes, unroll=unroll)
                def populate(i):
                    keys[pl.ds(i, lanes)] = compact[pl.ds(i, lanes)]
                    ids[pl.ds(i, lanes)] = i + lane_ids

                digits = max(
                    1, pl.cdiv((count * block_size - 1).bit_length(), RADIX_BITS)
                )
                _compact(
                    keys,
                    ids,
                    selected,
                    selected_ids,
                    hist,
                    jnp.int32(count * block_size),
                    candidate_buffers=2,
                    id_digits=digits,
                    unroll=unroll,
                )

                @plsc.parallel_loop(0, k, lanes, unroll=unroll)
                def remove_invalid(i):
                    ids[pl.ds(i, lanes)] = jnp.where(
                        keys[pl.ds(i, lanes)] > -jnp.inf,
                        ids[pl.ds(i, lanes)],
                        -1,
                    )

                _sort_selected(ids, selected_ids, hist, id_digits=digits, unroll=unroll)

                @plsc.parallel_loop(0, k, lanes, unroll=unroll)
                def remap(i):
                    slots = ids[pl.ds(i, lanes)]
                    safe = jnp.maximum(slots, 0).astype(jnp.uint32)
                    blocks = plsc.load_gather(
                        metadata, [(safe // block_size).astype(jnp.int32)]
                    )
                    ids[pl.ds(i, lanes)] = jnp.where(
                        slots >= 0,
                        blocks * block_size + (safe % block_size).astype(jnp.int32),
                        -1,
                    )

                pltpu.async_copy(ids.at[:k], out.at[r], sem.at[0]).wait()

            pl.run_scoped(
                select_row,
                pltpu.VMEM((2 * capacity,), jnp.float32),
                pltpu.VMEM((2 * capacity,), jnp.int32),
                pltpu.VMEM((k,), jnp.float32),
                pltpu.VMEM((k,), jnp.int32),
                pltpu.VMEM((RADIX_BUCKETS,), jnp.int32),
            )

    return kernel(scores, candidates)


def verify_fused_selection(*, unroll_selection):
    """Exercise empty rows, ties, short rows and noncontiguous block IDs."""
    from vllm_torchtpu.kernels.deepseek_v41.index_selection._packed_fp8_compact import (
        candidate_topk,
        compact_candidate_scores,
    )

    positions = jnp.arange(98304, dtype=jnp.int32)
    scores = jnp.stack(
        (
            jnp.full((98304,), -jnp.inf),
            jnp.zeros((98304,)),
            jnp.where(positions < 17, 0.0, -jnp.inf),
            (positions % 17).astype(jnp.float32),
        )
    )
    candidates = jnp.broadcast_to(jnp.arange(2048), (4, 2048))
    candidates = candidates.at[0].set(-1).at[3].multiply(3)

    def reference(values, ids):
        compact = compact_candidate_scores(values, ids, block_size=8)
        return candidate_topk(compact, ids, k=512, block_size=8)

    expected = jax.jit(reference)(scores, candidates)
    actual = jax.jit(
        functools.partial(fused_select, unroll_selection=unroll_selection)
    )(scores, candidates)
    assert bool(jnp.array_equal(actual, expected)), "Fused selection edge case changed"
    print("Fused selection edge cases passed", flush=True)
