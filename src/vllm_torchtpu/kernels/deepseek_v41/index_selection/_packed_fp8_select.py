# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather candidate scores and select exact TopK in one SparseCore call.

Separate VMEM scopes reuse full-score staging memory for radix selection.
"""

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

from ._packed_fp8_compact import candidate_topk, compact_candidate_scores


def select_candidate_scores(scores, candidates, *, k: int, block_size: int):
    """Select logical IDs from unique ascending candidates, padded with -1."""
    rows, width = scores.shape
    count = candidates.shape[1]

    def separate():
        return candidate_topk(
            compact_candidate_scores(scores, candidates, block_size=block_size),
            candidates,
            k=k,
            block_size=block_size,
        )

    if jax.default_backend() != "tpu":
        return separate()
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    alignment = pltpu.get_tpu_info().num_lanes
    capacity = count * block_size + lanes
    unroll = alignment // lanes
    resident = 4 * (capacity + count)
    scratch_peak = 4 * max(width, 4 * capacity + 2 * k + RADIX_BUCKETS)
    if resident + scratch_peak > info.vmem_capacity_bytes or any(
        n % alignment for n in (k, count, width)
    ):
        return separate()
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

                # Selecting padding is harmless when fewer than k values exist;
                # remove those IDs before sorting so -1 remains a suffix.
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
