# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-query merge-path metadata in SparseCore VMEM.

Adapted from DeepGEMM scheduler/sm100_sparse_mqa_logits_metadata.cuh.
Merge diagonals and duplicate detection use vector lanes; binary-search
rounds are static. Prefix counts compact directly into the output metadata.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


class PairedMQAMetadata(NamedTuple):
    # [T,4,2C]: physical page, logical page, Q0 output slot, Q1 output slot.
    pages: jax.Array
    counts: jax.Array  # [T] union count; only pair-start rows are populated.


def _merge_partition(
    read_a, read_b, diagonal: jax.Array, capacity: int, count_a, count_b
):
    lo, hi = jnp.maximum(diagonal - count_b, 0), jnp.minimum(diagonal, count_a)
    for _ in range(capacity.bit_length()):
        i = (lo + hi) // 2
        j = diagonal - i
        advance = (lo < hi) & (i < count_a) & (j > 0) & (read_b(j - 1) >= read_a(i))
        lo, hi = jnp.where(advance, i + 1, lo), jnp.where(advance, hi, i)
    return lo, diagonal - lo


@jax.jit
def paired_mqa_metadata(
    candidates: jax.Array, block_table: jax.Array, query_start_loc: jax.Array
) -> PairedMQAMetadata:
    """Merge adjacent queries within each offset interval, preserving slots.

    INT32 candidates [T,C] are strictly increasing logical page IDs with -1
    tail padding. C and the page-table width must be TC-lane aligned. Empty
    and odd-sized intervals are supported; no query permutation is needed.
    """
    tokens, capacity = candidates.shape
    groups, page_count = block_table.shape
    alignment = pltpu.get_tpu_info().num_lanes
    if (
        min(tokens, capacity, groups, page_count) < 1
        or capacity % alignment
        or page_count % alignment
        or candidates.dtype != jnp.int32
        or block_table.dtype != jnp.int32
        or query_start_loc.dtype != jnp.int32
        or query_start_loc.shape != (groups + 1,)
    ):
        raise ValueError(
            "Expected lane-aligned INT32 candidates/page tables and group offsets"
        )
    info = plsc.get_sparse_core_info()
    lanes, width = info.num_lanes, 2 * capacity
    rows = jnp.arange(tokens, dtype=jnp.int32)
    owners = jnp.searchsorted(
        query_start_loc[1:], rows, side="right", method="scan_unrolled"
    )
    sizes = jnp.where(
        (rows - query_start_loc[owners]) % 2 == 0,
        jnp.minimum(2, query_start_loc[owners + 1] - rows),
        0,
    )
    header_width = pl.cdiv(tokens, alignment) * alignment
    header = jnp.pad(
        jnp.stack(
            (
                owners,
                sizes,
                ((candidates >= 0) & (candidates < page_count))
                .sum(1)
                .astype(jnp.int32),
            )
        ),
        ((0, 0), (0, header_width - tokens)),
    )
    if (
        4 * (3 * header_width + 2 * capacity + page_count + 7 * width + alignment)
        >= info.vmem_capacity_bytes
    ):
        raise ValueError("Paired candidate metadata exceeds SparseCore VMEM")
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, pl.cdiv(tokens, info.num_subcores)),
    )

    @pl.kernel(
        out_type=(
            jax.ShapeDtypeStruct((tokens, 4, width), jnp.int32),
            jax.ShapeDtypeStruct((tokens, alignment), jnp.int32),
        ),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((header_width,), jnp.int32),
            pltpu.VMEM((header_width,), jnp.int32),
            pltpu.VMEM((header_width,), jnp.int32),
            pltpu.VMEM((capacity,), jnp.int32),
            pltpu.VMEM((capacity,), jnp.int32),
            pltpu.VMEM((page_count,), jnp.int32),
            pltpu.VMEM((3 * width,), jnp.int32),
            pltpu.VMEM((4 * width,), jnp.int32),
            pltpu.VMEM((alignment,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="paired_mqa_metadata_sc",
    )
    def kernel(
        src,
        table,
        headers,
        out,
        count_out,
        owner_cache,
        size_cache,
        candidate_counts,
        a,
        b,
        pages,
        temporary,
        merged,
        counts,
        sem,
    ):
        copies = (
            pltpu.make_async_copy(headers.at[0, 0], owner_cache, sem),
            pltpu.make_async_copy(headers.at[1, 0], size_cache, sem),
            pltpu.make_async_copy(headers.at[2, 0], candidate_counts, sem),
        )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()
        lane = jnp.arange(lanes, dtype=jnp.int32)
        first = jax.lax.axis_index("core") * mesh.num_subcores + jax.lax.axis_index(
            "subcore"
        )

        @pl.loop(first, tokens, step=mesh.num_cores * mesh.num_subcores)
        def row(query):
            size = plsc.load_gather(size_cache, [jnp.full(lanes, query, jnp.int32)])[0]

            @plsc.parallel_loop(0, 4 * width, lanes)
            def clear(i):
                merged[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            @plsc.parallel_loop(0, alignment, lanes)
            def clear_count(i):
                counts[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            @pl.when(size > 0)
            def merge():
                owner = plsc.load_gather(
                    owner_cache, [jnp.full(lanes, query, jnp.int32)]
                )[0]
                transfers = (
                    pltpu.make_async_copy(src.at[query], a, sem),
                    pltpu.make_async_copy(
                        src.at[jnp.minimum(query + 1, tokens - 1)], b, sem
                    ),
                    pltpu.make_async_copy(table.at[owner], pages, sem),
                )
                for transfer in transfers:
                    transfer.start()
                for transfer in transfers:
                    transfer.wait()

                def read(ref, indices):
                    value = plsc.load_gather(ref, [jnp.clip(indices, 0, capacity - 1)])
                    valid = (
                        (indices >= 0)
                        & (indices < capacity)
                        & (value >= 0)
                        & (value < page_count)
                    )
                    return jnp.where(valid, value, jnp.iinfo(jnp.int32).max)

                def read_b(index):
                    return jnp.where(
                        size == 2, read(b, index), jnp.iinfo(jnp.int32).max
                    )

                steps = width // lanes
                count_a = plsc.load_gather(
                    candidate_counts, [jnp.full(lanes, query, jnp.int32)]
                )
                count_b = jnp.where(
                    size == 2,
                    plsc.load_gather(
                        candidate_counts,
                        [
                            jnp.full(
                                lanes, jnp.minimum(query + 1, tokens - 1), jnp.int32
                            )
                        ],
                    ),
                    0,
                )
                diagonal = lane * (count_a + count_b) // lanes
                i, j = _merge_partition(
                    lambda index: read(a, index),
                    read_b,
                    diagonal,
                    capacity,
                    count_a,
                    count_b,
                )
                remaining = (lane + 1) * (count_a + count_b) // lanes - diagonal
                duplicate = (
                    (remaining > 0)
                    & (i > 0)
                    & (j < count_b)
                    & (read(a, i - 1) == read_b(j))
                )
                j = j + duplicate.astype(jnp.int32)
                remaining = remaining - duplicate.astype(jnp.int32)
                local_count = jnp.zeros(lanes, jnp.int32)

                def merge_step(_, state):
                    i, j, remaining, local_count = state
                    av, bv = read(a, i), read_b(j)
                    in_a, in_b = av <= bv, bv <= av
                    value = jnp.minimum(av, bv)
                    keep = (remaining > 0) & (value != jnp.iinfo(jnp.int32).max)
                    destination = lane * steps + local_count
                    for field, values in enumerate(
                        (value, jnp.where(in_a, i, -1), jnp.where(in_b, j, -1))
                    ):
                        plsc.store_scatter(
                            temporary, [field * width + destination], values, mask=keep
                        )
                    consume_a = (remaining > 0) & in_a
                    consume_b = (remaining > consume_a.astype(jnp.int32)) & in_b
                    i, j = (
                        i + consume_a.astype(jnp.int32),
                        j + consume_b.astype(jnp.int32),
                    )
                    remaining = (
                        remaining
                        - consume_a.astype(jnp.int32)
                        - consume_b.astype(jnp.int32)
                    )
                    local_count = local_count + keep.astype(jnp.int32)
                    return i, j, remaining, local_count

                _, _, _, local_count = jax.lax.fori_loop(
                    0, steps, merge_step, (i, j, remaining, local_count)
                )
                prefix = plsc.cumsum(local_count)
                count = prefix[-1]
                prefix = prefix - local_count

                def compact(item, _):
                    keep = item < local_count
                    index = lane * steps + item
                    logical, slot0, slot1 = (
                        plsc.load_gather(temporary, [field * width + index])
                        for field in range(3)
                    )
                    physical = plsc.load_gather(
                        pages, [jnp.clip(logical, 0, page_count - 1)]
                    )
                    for field, value in enumerate((physical, logical, slot0, slot1)):
                        plsc.store_scatter(
                            merged, [field * width + prefix + item], value, mask=keep
                        )

                jax.lax.fori_loop(0, steps, compact, None)

                @plsc.parallel_loop(0, alignment, lanes)
                def save_count(i):
                    counts[pl.ds(i, lanes)] = jnp.full(lanes, count, jnp.int32)

            transfers = tuple(
                pltpu.make_async_copy(
                    merged.at[pl.ds(field * width, width)], out.at[query, field], sem
                )
                for field in range(4)
            )
            transfers += (pltpu.make_async_copy(counts, count_out.at[query], sem),)
            for transfer in transfers:
                transfer.start()
            for transfer in transfers:
                transfer.wait()

    pages, counts = kernel(candidates, block_table, header[:, None])
    return PairedMQAMetadata(pages, counts[:, 0])
