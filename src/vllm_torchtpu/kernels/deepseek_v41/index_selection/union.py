# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build candidate memberships and ascending request unions in SparseCore VMEM."""

import functools
from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


class MetadataLayout(NamedTuple):
    histogram_width: int
    output_width: int
    count_width: int
    scratch_bytes: int


def metadata_layout(
    candidate_shape: tuple[int, ...], logical_blocks: int, capacity: int
) -> MetadataLayout:
    """Shared dispatch/allocation geometry for TC-tiled HBM metadata."""
    _, queries, candidates = candidate_shape
    alignment = pltpu.get_tpu_info().num_lanes
    lanes = plsc.get_sparse_core_info().num_lanes
    histogram = pl.cdiv(logical_blocks, alignment) * alignment
    output = pl.cdiv(capacity, alignment) * alignment
    scratch_bytes = jnp.dtype(jnp.int32).itemsize * (
        queries * histogram + candidates + 2 * (output + lanes) + alignment
    )
    return MetadataLayout(histogram, output, alignment, scratch_bytes)


@functools.partial(jax.jit, static_argnames=("logical_blocks", "capacity"))
def sparse_union_metadata(
    candidate_block_ids: jax.Array, logical_blocks: int, capacity: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build sorted IDs and query bits; retain the full count on overflow.

    Each subcore accumulates query-local histograms in VMEM, then streams
    their membership masks in ID order without an HBM scatter or sort.
    """
    requests, queries, candidates_per_query = candidate_block_ids.shape
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    layout = metadata_layout(candidate_block_ids.shape, logical_blocks, capacity)
    histogram_width, output_width = layout.histogram_width, layout.output_width
    if (
        candidate_block_ids.dtype != jnp.int32
        or candidates_per_query % layout.count_width
        or not 1 <= queries <= jnp.iinfo(jnp.int32).bits
        or min(requests, candidates_per_query, logical_blocks, capacity) < 1
    ):
        raise ValueError(
            "Expected aligned int32 candidates with query memberships fitting int32"
        )
    if layout.scratch_bytes > info.vmem_capacity_bytes:
        raise ValueError(
            f"Candidate union needs {layout.scratch_bytes} bytes per subcore, "
            f"available {info.vmem_capacity_bytes}"
        )
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, pl.cdiv(requests, info.num_subcores)),
    )
    shape = jax.ShapeDtypeStruct((requests, output_width), jnp.int32)

    @pl.kernel(
        out_type=(
            shape,
            shape,
            jax.ShapeDtypeStruct((requests, layout.count_width), jnp.int32),
        ),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((queries * histogram_width,), jnp.int32),
            pltpu.VMEM((candidates_per_query,), jnp.int32),
            pltpu.VMEM((output_width + lanes,), jnp.int32),
            pltpu.VMEM((output_width + lanes,), jnp.int32),
            pltpu.VMEM((layout.count_width,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="candidate_union_metadata_sc",
    )
    def kernel(src, ids_out, bits_out, count_out, hist, block, ids, bits, counts, sem):
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        first = jax.lax.axis_index("core") * mesh.num_subcores + jax.lax.axis_index(
            "subcore"
        )

        @pl.loop(first, requests, step=mesh.num_cores * mesh.num_subcores)
        def request(row):
            @plsc.parallel_loop(0, queries * histogram_width, lanes)
            def clear(i):
                hist[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            @pl.loop(0, queries)
            def query(query_index):
                pltpu.async_copy(src.at[row, query_index], block, sem).wait()

                @plsc.parallel_loop(0, candidates_per_query, lanes)
                def scatter(i):
                    keys = block[pl.ds(i, lanes)]
                    valid = (keys >= 0) & (keys < logical_blocks)
                    inc, mask = plsc.scan_count(keys, mask=valid)
                    addresses = query_index * histogram_width + jnp.clip(
                        keys, 0, histogram_width - 1
                    )
                    plsc.addupdate_scatter(hist, [addresses], inc, mask=mask)

            @plsc.parallel_loop(0, output_width, lanes)
            def clear_out(i):
                ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)
                bits[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

            def compact(i: jax.Array, n: jax.Array) -> jax.Array:
                membership = jnp.zeros(lanes, jnp.int32)
                for query_index in range(queries):
                    membership = membership | (
                        (
                            hist[pl.ds(query_index * histogram_width + i, lanes)] > 0
                        ).astype(jnp.int32)
                        << query_index
                    )
                valid = (membership != 0) & (i + lane_ids < logical_blocks)
                keep = valid & (n + jnp.cumsum(valid.astype(jnp.int32)) <= capacity)
                off = jnp.minimum(n, output_width)
                plsc.store_compressed(
                    ids.at[pl.ds(off, lanes)], i + lane_ids, mask=keep
                )
                plsc.store_compressed(bits.at[pl.ds(off, lanes)], membership, mask=keep)
                return n + plsc.all_reduce_population_count(valid)[0]

            # The public annotation omits parallel_loop's carry overload.
            compact_loop = cast(
                Callable[[Callable[[jax.Array, jax.Array], jax.Array]], jax.Array],
                plsc.parallel_loop(0, histogram_width, lanes, carry=jnp.int32(0)),
            )
            count = compact_loop(compact)

            @plsc.parallel_loop(0, layout.count_width, lanes)
            def save_count(i):
                counts[pl.ds(i, lanes)] = jnp.full(lanes, count, jnp.int32)

            pltpu.async_copy(
                ids.at[pl.ds(0, output_width)], ids_out.at[row], sem
            ).wait()
            pltpu.async_copy(
                bits.at[pl.ds(0, output_width)], bits_out.at[row], sem
            ).wait()
            pltpu.async_copy(counts, count_out.at[row], sem).wait()

    ids, bits, count = kernel(candidate_block_ids)
    return ids[:, :capacity], bits[:, :capacity], count[:, 0], count[:, 0] > capacity


def _ragged_scratch_sizes(tokens, requests, candidates, logical_blocks, capacity):
    return (
        pl.cdiv(logical_blocks, 128) * 128,
        candidates,
        pl.cdiv(capacity, 128) * 128 + 128,
        pl.cdiv(capacity, 128) * 128,
        pl.cdiv(tokens, 128) * 128,
        2 * pl.cdiv(requests + 1, 128) * 128,
        128,
    )


def is_supported_by_sparse_ragged_union(
    candidates: jax.Array, offsets: jax.Array, logical_blocks: int, capacity: int
) -> bool:
    info = pltpu.get_tpu_info()
    if info.chip_version != pltpu.ChipVersion.TPU_7X or info.sparse_core is None:
        return False
    tokens, width = candidates.shape
    sizes = _ragged_scratch_sizes(
        tokens, offsets.size - 1, width, logical_blocks, capacity
    )
    return (
        tokens > 0
        and width > 0
        and width % 128 == 0
        and candidates.dtype == jnp.int32
        and 4 * sum(sizes) <= info.sparse_core.vmem_capacity_bytes
    )


@functools.partial(jax.jit, static_argnames=("logical_blocks", "capacity"))
def sparse_ragged_union(
    candidates: jax.Array,
    lengths: jax.Array,
    query_offsets: jax.Array,
    word_offsets: jax.Array,
    logical_blocks: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build request unions and query words in the scorer's native membership layout."""
    tokens, candidate_width = candidates.shape
    requests = query_offsets.size - 1
    hist_width, _, id_scratch, bits_width, length_width, offset_scratch, _ = (
        _ragged_scratch_sizes(
            tokens, requests, candidate_width, logical_blocks, capacity
        )
    )
    id_width = id_scratch - 128
    offset_width = offset_scratch // 2
    word_rows = (tokens + requests * 31) // 32 + 1
    info = plsc.get_sparse_core_info()
    lanes = info.num_lanes
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=min(info.num_cores, pl.cdiv(requests, info.num_subcores)),
    )

    @pl.kernel(
        out_type=(
            jax.ShapeDtypeStruct((requests, id_width), jnp.int32),
            jax.ShapeDtypeStruct((word_rows, bits_width), jnp.int32),
            jax.ShapeDtypeStruct((requests, 128), jnp.int32),
        ),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((hist_width,), jnp.int32),
            pltpu.VMEM((candidate_width,), jnp.int32),
            pltpu.VMEM((id_scratch,), jnp.int32),
            pltpu.VMEM((bits_width,), jnp.int32),
            pltpu.VMEM((length_width,), jnp.int32),
            pltpu.VMEM((offset_scratch,), jnp.int32),
            pltpu.VMEM((128,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=False),
        name="candidate_ragged_union_metadata_sc",
    )
    def kernel(
        src,
        length_src,
        offset_src,
        ids_out,
        bits_out,
        count_out,
        hist,
        block,
        ids,
        bits,
        query_lengths,
        offsets,
        counts,
        sem,
    ):
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        first = jax.lax.axis_index("core") * mesh.num_subcores + jax.lax.axis_index(
            "subcore"
        )
        stride = mesh.num_cores * mesh.num_subcores
        pltpu.async_copy(length_src, query_lengths, sem).wait()
        pltpu.async_copy(offset_src, offsets, sem).wait()

        def clear_hist():
            @plsc.parallel_loop(0, hist_width, lanes)
            def clear(i):
                hist[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

        def accumulate(position, *, membership=False):
            active = (
                plsc.load_gather(query_lengths, [jnp.full(lanes, position, jnp.int32)])[
                    0
                ]
                > 0
            )

            pltpu.async_copy(src.at[position], block, sem).wait()

            @plsc.parallel_loop(0, candidate_width, lanes)
            def scatter(i):
                keys = block[pl.ds(i, lanes)]
                valid = active & (keys >= 0) & (keys < logical_blocks)
                if membership:
                    keys = plsc.load_gather(hist, [jnp.clip(keys, 0, hist_width - 1)])
                    valid &= keys >= 0
                inc, mask = plsc.scan_count(keys, mask=valid)
                target = ids if membership else hist
                plsc.addupdate_scatter(
                    target, [jnp.clip(keys, 0, target.shape[0] - 1)], inc, mask=mask
                )

        @pl.loop(first, requests, step=stride)
        def request(row):
            clear_hist()
            begin = plsc.load_gather(offsets, [jnp.full(lanes, row, jnp.int32)])[0]
            end = plsc.load_gather(offsets, [jnp.full(lanes, row + 1, jnp.int32)])[0]

            @pl.loop(begin, end)
            def collect(position):
                accumulate(position)

            @plsc.parallel_loop(0, id_width, lanes)
            def clear_ids(i):
                ids[pl.ds(i, lanes)] = jnp.full(lanes, -1, jnp.int32)

            def compact(i, n):
                valid = (hist[pl.ds(i, lanes)] > 0) & (i + lane_ids < logical_blocks)
                ranks = n + jnp.cumsum(valid.astype(jnp.int32))
                keep = valid & (ranks <= capacity)
                # Reuse the histogram as logical-block -> compact-slot lookup.
                hist[pl.ds(i, lanes)] = jnp.where(keep, ranks - 1, -1)
                plsc.store_compressed(
                    ids.at[pl.ds(jnp.minimum(n, id_width), lanes)],
                    i + lane_ids,
                    mask=keep,
                )
                return n + plsc.all_reduce_population_count(valid)[0]

            compact_loop = cast(
                Callable[[Callable[[jax.Array, jax.Array], jax.Array]], jax.Array],
                plsc.parallel_loop(0, hist_width, lanes, carry=jnp.int32(0)),
            )
            count = compact_loop(compact)

            @plsc.parallel_loop(0, 128, lanes)
            def save_count(i):
                counts[pl.ds(i, lanes)] = jnp.full(lanes, count, jnp.int32)

            pltpu.async_copy(ids.at[:id_width], ids_out.at[row], sem).wait()
            pltpu.async_copy(counts, count_out.at[row], sem).wait()

            first_word = plsc.load_gather(
                offsets, [jnp.full(lanes, offset_width + row, jnp.int32)]
            )[0]

            @pl.loop(0, pl.cdiv(end - begin, 32))
            def query_word(word):
                @plsc.parallel_loop(0, bits_width, lanes)
                def clear_word(i):
                    bits[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

                @pl.loop(begin + word * 32, jnp.minimum(begin + (word + 1) * 32, end))
                def membership(position):
                    # The union IDs are already in HBM; reuse their scratch buffer.
                    @plsc.parallel_loop(0, id_width, lanes)
                    def clear_presence(i):
                        ids[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

                    accumulate(position, membership=True)

                    @plsc.parallel_loop(0, bits_width, lanes)
                    def include(i):
                        present = ids[pl.ds(i, lanes)] > 0
                        bit = jnp.int32(1) << ((position - begin) % 32)
                        bits[pl.ds(i, lanes)] |= jnp.where(present, bit, 0)

                pltpu.async_copy(bits, bits_out.at[first_word + word], sem).wait()

        # Unused capacity includes a sentinel word for scorer boundary reads.
        @plsc.parallel_loop(0, bits_width, lanes)
        def clear_bits(i):
            bits[pl.ds(i, lanes)] = jnp.zeros(lanes, jnp.int32)

        @pl.loop(
            plsc.load_gather(
                offsets, [jnp.full(lanes, offset_width + requests, jnp.int32)]
            )[0]
            + first,
            word_rows,
            step=stride,
        )
        def padding(position):
            pltpu.async_copy(bits, bits_out.at[position], sem).wait()

    ids, bits, counts = kernel(
        candidates,
        jnp.pad(lengths, (0, length_width - tokens)),
        jnp.concatenate(
            (
                jnp.pad(query_offsets, (0, offset_width - requests - 1)),
                jnp.pad(word_offsets, (0, offset_width - requests - 1)),
            )
        ),
    )
    return (
        ids[:, :capacity],
        bits[:, :capacity],
        counts[:, 0],
        counts[:, 0] > capacity,
    )
