# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SparseCore candidate membership and compact FP32 score output."""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from .streaming_topk import TopKEpilogue, streaming_topk


def candidate_topk(
    scores: jax.Array, candidates: jax.Array, *, k: int, block_size: int
) -> jax.Array:
    """Select compact scores and return ascending logical token IDs."""
    if jax.default_backend() != "tpu" or k % plsc.get_sparse_core_info().num_lanes:
        values, indices = jax.lax.top_k(scores, k)
        blocks = jnp.take_along_axis(candidates, indices // block_size, axis=1)
        logical = blocks * block_size + indices % block_size
        limit = jnp.iinfo(jnp.int32).max
        logical = jnp.sort(jnp.where(jnp.isfinite(values), logical, limit), axis=1)
        return jnp.where(logical < limit, logical, -1)

    lanes = plsc.get_sparse_core_info().num_lanes

    def remap(inputs, indices, metadata, sem, sort_selected):
        del inputs, sem
        sort_selected()

        @plsc.parallel_loop(0, k, lanes)
        def block(i):
            slots = indices[pl.ds(i, lanes)]
            safe = jnp.maximum(slots, 0).astype(jnp.uint32)
            block_slots = (safe // block_size).astype(jnp.int32)
            offsets = (safe % block_size).astype(jnp.int32)
            selected = plsc.load_gather(metadata, [lanes + block_slots])
            indices[pl.ds(i, lanes)] = jnp.where(
                slots >= 0, selected * block_size + offsets, -1
            )

    return streaming_topk(
        scores,
        k=k,
        exclude_neg_inf=True,
        epilogue=TopKEpilogue((), candidates, 0, remap),
    )


def candidate_membership(
    candidates: jax.Array, blocks: int
) -> tuple[jax.Array, jax.Array]:
    """Return block presence and ascending unique candidates, padded with -1."""
    rows, capacity = candidates.shape
    if jax.default_backend() != "tpu" or capacity == 0:
        valid = (candidates >= 0) & (candidates < blocks)
        mask = jnp.zeros((rows, blocks + 1), jnp.int32)
        mask = mask.at[
            jnp.arange(rows)[:, None], jnp.where(valid, candidates, blocks)
        ].set(1)[:, :blocks]
        canonical = jax.vmap(
            lambda row: jnp.nonzero(row, size=capacity, fill_value=-1)[0]
        )(mask)
        return mask, canonical.astype(jnp.int32)

    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    alignment = pltpu.get_tpu_info().num_lanes
    histogram_width = pl.cdiv(blocks, alignment) * alignment
    candidate_width = pl.cdiv(capacity, alignment) * alignment
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        mesh=mesh,
        out_type=(
            jax.ShapeDtypeStruct((rows, histogram_width), jnp.int32),
            jax.ShapeDtypeStruct((rows, candidate_width), jnp.int32),
        ),
        scratch_types=(
            pltpu.VMEM((histogram_width,), jnp.int32),
            pltpu.VMEM((candidate_width,), jnp.int32),
            pltpu.VMEM((candidate_width + lanes,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="packed_fp8_candidate_membership_sc",
    )
    def kernel(src, mask_out, canonical_out, hist, ids, canonical, sem):
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        first = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(first, rows, step=mesh.num_cores * mesh.num_subcores)
        def row(row_id):
            @plsc.parallel_loop(0, histogram_width, lanes, unroll=alignment // lanes)
            def clear(i):
                hist[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            pltpu.async_copy(src.at[row_id], ids, sem).wait()

            @plsc.parallel_loop(0, candidate_width, lanes, unroll=alignment // lanes)
            def scatter(i):
                keys = ids[pl.ds(i, lanes)]
                valid = (keys >= 0) & (keys < blocks)
                inc, mask = plsc.scan_count(keys, mask=valid)
                plsc.addupdate_scatter(
                    hist, [jnp.clip(keys, 0, histogram_width - 1)], inc, mask=mask
                )

            @plsc.parallel_loop(0, candidate_width, lanes, unroll=alignment // lanes)
            def clear_ids(i):
                canonical[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            def compact(i, count):
                valid = (hist[pl.ds(i, lanes)] != 0) & (i + lane_ids < blocks)
                hist[pl.ds(i, lanes)] = valid.astype(jnp.int32)
                plsc.store_compressed(
                    canonical.at[pl.ds(count, lanes)], i + lane_ids, mask=valid
                )
                return count + plsc.all_reduce_population_count(valid)[0]

            compact_loop = cast(
                Callable[[Callable[[jax.Array, jax.Array], jax.Array]], jax.Array],
                plsc.parallel_loop(
                    0,
                    histogram_width,
                    lanes,
                    unroll=alignment // lanes,
                    carry=jnp.int32(0),
                ),
            )
            compact_loop(compact)
            pltpu.async_copy(hist, mask_out.at[row_id], sem).wait()
            pltpu.async_copy(
                canonical.at[:candidate_width], canonical_out.at[row_id], sem
            ).wait()

    mask, canonical = kernel(
        jnp.pad(
            candidates, ((0, 0), (0, candidate_width - capacity)), constant_values=-1
        )
    )
    return mask[:, :blocks], canonical[:, :capacity]


def compact_candidate_scores(
    scores: jax.Array, candidates: jax.Array, *, block_size: int
) -> jax.Array:
    """Gather canonical candidate blocks into [T, candidates * block_size].

    The TensorCore scorer's full-width intermediate is read into SparseCore
    VMEM; only compact scores leave this stage. No full-width masked copy is made.
    """
    rows, width = scores.shape
    capacity = candidates.shape[1]
    output_width = capacity * block_size

    def gather_scores():
        indices = candidates[:, :, None] * block_size + jnp.arange(block_size)
        valid = (candidates[:, :, None] >= 0) & (indices < width)
        indices = indices.reshape(rows, output_width)
        result = jnp.take_along_axis(scores, jnp.clip(indices, 0, width - 1), axis=1)
        return jnp.where(valid.reshape(rows, output_width), result, -jnp.inf)

    if jax.default_backend() != "tpu" or output_width == 0:
        return gather_scores()

    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    alignment = pltpu.get_tpu_info().num_lanes
    id_width = pl.cdiv(capacity, alignment) * alignment
    out_width = pl.cdiv(output_width, alignment) * alignment
    if 4 * (width + id_width + out_width) > info.vmem_capacity_bytes:
        return gather_scores()
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

    @pl.kernel(
        mesh=mesh,
        out_type=jax.ShapeDtypeStruct((rows, out_width), jnp.float32),
        scratch_types=(
            pltpu.VMEM((width,), jnp.float32),
            pltpu.VMEM((id_width,), jnp.int32),
            pltpu.VMEM((out_width,), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="packed_fp8_candidate_scores_sc",
    )
    def kernel(src, ids_src, out, values, ids, compact, sem):
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        first = jax.lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name))

        @pl.loop(first, rows, step=mesh.num_cores * mesh.num_subcores)
        def row(row_id):
            value_copy = pltpu.async_copy(src.at[row_id], values, sem.at[0])
            pltpu.async_copy(ids_src.at[row_id], ids, sem.at[1]).wait()
            value_copy.wait()

            @plsc.parallel_loop(0, out_width, lanes, unroll=alignment // lanes)
            def gather(i):
                slots = i + lane_ids
                unsigned_slots = slots.astype(jnp.uint32)
                block_slots = (unsigned_slots // block_size).astype(jnp.int32)
                offsets = (unsigned_slots % block_size).astype(jnp.int32)
                blocks = plsc.load_gather(ids, [block_slots])
                logical = blocks * block_size + offsets
                valid = (slots < output_width) & (blocks >= 0) & (logical < width)
                selected = plsc.load_gather(values, [logical], mask=valid)
                compact[pl.ds(i, lanes)] = jnp.where(valid, selected, -jnp.inf)

            pltpu.async_copy(compact, out.at[row_id], sem.at[0]).wait()

    return kernel(
        scores,
        jnp.pad(candidates, ((0, 0), (0, id_width - capacity)), constant_values=-1),
    )[:, :output_width]
