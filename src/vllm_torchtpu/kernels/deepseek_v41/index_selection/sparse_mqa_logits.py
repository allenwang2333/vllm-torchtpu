# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MQA scoring and Top-K over flat queries and request-local keys."""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS, SLOTS
from vllm_torchtpu.kernels.deepseek_v41.index_selection import tuned_block_sizes
from vllm_torchtpu.kernels.deepseek_v41.index_selection.gather import (
    gather_packed_candidate_keys,
    is_supported_by_gather_packed_candidate_keys,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.sparse_mqa_metadata import (
    membership_offsets,
    query_metadata,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    TopKEpilogue,
    streaming_topk,
)


def _expand_metadata(values, repeats, group_width):
    if repeats == 1:
        return values
    # Lane-local gather avoids repeat's unsupported narrow-minor reshape.
    width = math.gcd(values.shape[0], group_width)
    source = values.reshape(-1, width)
    indices = jnp.broadcast_to(
        (jnp.arange(width * repeats) // repeats)[None, :, None],
        (source.shape[0], width * repeats, 1),
    )
    return jax.lax.gather(
        source,
        indices,
        jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(1,),
            start_index_map=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
        ),
        slice_sizes=(1, 1),
        mode="promise_in_bounds",
    ).reshape(-1)


def _sparse_mqa_logits_kernel(
    rows,
    q,
    weights,
    keys,
    ids,
    membership,
    offsets,
    out,
    key_buffers,
    input_sems,
    *,
    compute_tile,
):
    group = q.shape[0]
    first = pl.program_id(1) * group
    tile_start = pl.program_id(0) * out.shape[1]
    repeat = out.shape[1] // ids.shape[-1]
    dma_tile = key_buffers.shape[1] * SLOTS
    used = jnp.int32(0)
    for local in range(group):
        row = jnp.minimum(first + local, rows.shape[1] - 1)
        used = jnp.maximum(used, jnp.where(rows[3, row] > 0, rows[2, row], 0))
    fragments = jnp.minimum(
        out.shape[1] // dma_tile,
        pl.cdiv(jnp.maximum(used - tile_start, 0), dma_tile),
    )
    tasks = jnp.minimum(group, rows.shape[1] - first) * fragments
    out[...] = jnp.full(out.shape, -jnp.inf, jnp.float32)

    def copy(task):
        row = first + task // fragments
        key_start = tile_start + task % fragments * dma_tile
        return pltpu.make_async_copy(
            keys.at[rows[0, row], pl.ds(key_start // SLOTS, dma_tile // SLOTS), :],
            key_buffers.at[task % 2],
            input_sems.at[task % 2],
        )

    @pl.when(tasks > 0)
    def prefetch_first():
        copy(0).start()

    def body(local, task, unused):
        copy(task).wait()

        @pl.when(task + 1 < tasks)
        def prefetch():
            copy(task + 1).start()

        row = first + local
        owner, bit_position = rows[0, row], rows[1, row]
        fragment_start = task % fragments * dma_tile
        for offset in range(0, dma_tile, compute_tile):
            section = pl.ds(fragment_start + offset, compute_tile)
            kk = pltpu.bitcast(
                key_buffers[task % 2, pl.ds(offset // SLOTS, compute_tile // SLOTS), :],
                jnp.bfloat16,
            )
            logits = jax.lax.dot_general(
                q[local],
                kk,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            values = (jnp.maximum(logits, 0) * weights[local][:, None]).sum(0)
            meta_start = task % fragments * (dma_tile // repeat) + offset // repeat
            meta_section = pl.ds(meta_start, compute_tile // repeat)
            logical = _expand_metadata(ids[owner, meta_section], repeat, DIM)
            logical += offsets[section]
            members = membership[bit_position // 32, meta_section]
            allow = ((members >> (bit_position % 32)) & 1) != 0
            allow = _expand_metadata(allow, repeat, DIM)
            allow &= (logical >= 0) & (logical < rows[3, row])
            out[local, section] = jnp.where(allow, values, -jnp.inf)
        return unused

    for local in range(group):
        jax.lax.fori_loop(
            local * fragments,
            jnp.minimum((local + 1) * fragments, tasks),
            functools.partial(body, local),
            None,
        )


@functools.partial(
    jax.jit,
    static_argnames=("block_size", "key_tile", "compute_tile", "dma_tile", "interpret"),
)
def sparse_mqa_logits(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    rows: jax.Array,
    *,
    block_size: int,
    key_tile: int = 4096,
    compute_tile: int | None = None,
    dma_tile: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Return FP32 [T, capacity] scores; rows holds owner, bit, union length, length."""
    tokens, heads, dim = q.shape
    width = ids.shape[1] * block_size
    compute_tile = key_tile if compute_tile is None else compute_tile
    dma_tile = math.lcm(compute_tile, SLOTS) if dma_tile is None else dma_tile
    if (
        dim != DIM
        or q.dtype != jnp.bfloat16
        or weights.shape != (tokens, heads)
        or weights.dtype != jnp.float32
        or keys.dtype != jnp.int32
        or keys.shape != (ids.shape[0], width // SLOTS, ROW_WORDS)
        or rows.shape != (4, tokens)
        or rows.dtype != jnp.int32
        or membership.ndim != 2
        or membership.shape[1] != ids.shape[1]
        or membership.dtype != jnp.int32
        or min(block_size, key_tile, compute_tile, dma_tile) < 1
        or width % key_tile
        or key_tile % dma_tile
        or dma_tile % compute_tile
        or dma_tile % SLOTS
    ):
        raise ValueError("Expected flat BF16 Q, packed int32 keys and per-row metadata")
    if tokens == 0:
        return jnp.empty((0, width), jnp.float32)
    row_group = 1 if interpret else min(tokens, pltpu.get_tpu_info().num_sublanes)
    metadata_stride = math.gcd(block_size, key_tile // min(key_tile, dim))
    metadata_tile = key_tile // metadata_stride
    if compute_tile % metadata_stride:
        raise ValueError("compute_tile must align with compact metadata")
    bases = jnp.where(ids >= 0, ids * block_size, -block_size)
    bases = jnp.repeat(bases, block_size // metadata_stride, axis=1)
    members = jnp.repeat(membership, block_size // metadata_stride, axis=1)
    return pl.pallas_call(
        functools.partial(_sparse_mqa_logits_kernel, compute_tile=compute_tile),
        out_shape=jax.ShapeDtypeStruct((tokens, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(width // key_tile, pl.cdiv(tokens, row_group)),
            in_specs=[
                pl.BlockSpec((row_group, heads, dim), lambda t, g, rows: (g, 0, 0)),
                pl.BlockSpec((row_group, heads), lambda t, g, rows: (g, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((ids.shape[0], metadata_tile), lambda t, g, rows: (0, t)),
                pl.BlockSpec(
                    (membership.shape[0], metadata_tile), lambda t, g, rows: (0, t)
                ),
                pl.BlockSpec((key_tile,), lambda t, g, rows: (t,)),
            ],
            out_specs=pl.BlockSpec((row_group, key_tile), lambda t, g, rows: (g, t)),
            scratch_shapes=(
                pltpu.VMEM((2, dma_tile // SLOTS, ROW_WORDS), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",) * 2),
        interpret=interpret,
        name="sparse_mqa_logits",
    )(
        rows,
        q,
        weights,
        keys,
        bases,
        members,
        jnp.arange(width, dtype=jnp.int32) % block_size,
    )


@functools.partial(
    jax.jit,
    static_argnames=("k", "block_size", "reconstruct_threshold"),
)
def sparse_topk_remap(
    scores: jax.Array,
    *,
    valid_lengths: jax.Array,
    logical_blocks: jax.Array,
    row_requests: jax.Array,
    k: int,
    block_size: int,
    reconstruct_threshold: int,
) -> jax.Array:
    """Return ascending logical token IDs, with -1 padding, excluding -inf.

    logical_blocks [requests, blocks] maps compact score columns to token IDs;
    the block count must divide the score width. Valid blocks must increase,
    padded blocks must have ineligible scores, and mapped IDs must fit int32.
    row_requests assigns each score row to its request.
    """
    rows, width = scores.shape
    if row_requests.shape != (rows,) or row_requests.dtype != jnp.int32:
        raise ValueError("row_requests must contain one int32 owner per score row")
    if (
        logical_blocks.ndim != 2
        or logical_blocks.dtype != jnp.int32
        or not logical_blocks.shape[1]
        or width % logical_blocks.shape[1]
    ):
        raise ValueError(
            "logical_blocks must be int32 [requests, blocks], with blocks dividing N"
        )
    records_per_block = width // logical_blocks.shape[1]
    alignment = pltpu.get_tpu_info().num_lanes
    lanes = plsc.get_sparse_core_info().num_lanes
    output_width = pl.cdiv(k, alignment) * alignment
    mapping_width = pl.cdiv(logical_blocks.shape[1], alignment) * alignment
    logical_blocks = jnp.pad(
        logical_blocks,
        ((0, 0), (0, mapping_width - logical_blocks.shape[1])),
        constant_values=-1,
    )
    row_metadata = jnp.broadcast_to(row_requests[:, None], (rows, lanes))

    def finish_row(refs, candidate_ids, row_metadata_ref, sem, sort_selected):
        # Selection is finished: reuse the ID tail for mapping metadata.
        # Sorting touches only the first k entries and overlaps this DMA.
        mapping_copy = pltpu.make_async_copy(
            refs[0].at[row_metadata_ref[pl.ds(lanes, lanes)][0]],
            candidate_ids.at[pl.ds(output_width, mapping_width)],
            sem,
        )
        mapping_copy.start()
        sort_selected()
        mapping_copy.wait()

        @plsc.parallel_loop(0, k, lanes)
        def map_selected(i):
            slots = candidate_ids[pl.ds(i, lanes)]
            blocks = plsc.load_gather(
                candidate_ids,
                [output_width + jnp.maximum(slots, 0) // records_per_block],
            )
            logical = blocks * records_per_block + slots % records_per_block
            candidate_ids[pl.ds(i, lanes)] = jnp.where(slots >= 0, logical, -1)

    return streaming_topk(
        scores,
        valid_lengths=valid_lengths,
        k=k,
        block_size=block_size,
        reconstruct_threshold=reconstruct_threshold,
        exclude_neg_inf=True,
        scheduling_group_id=None,
        epilogue=TopKEpilogue(
            (logical_blocks,), row_metadata, mapping_width, finish_row
        ),
    )


def _gather_query_keys(keys, starts, block_size):
    bank = keys.reshape(-1, ROW_WORDS)
    if is_supported_by_gather_packed_candidate_keys(bank, starts, block_size):
        gathered = gather_packed_candidate_keys(bank, starts, block_size)
    else:
        indices = jnp.maximum(starts, 0)[..., None] // SLOTS
        indices += jnp.arange(block_size // SLOTS)
        gathered = bank[indices]
    return gathered.reshape(starts.shape[0], -1, ROW_WORDS)


@functools.partial(jax.jit, static_argnames=("block_size", "key_tile", "row_tile", "k"))
def sparse_mqa_indexer(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    lengths: jax.Array,
    *,
    query_start_loc: jax.Array,
    union_counts: jax.Array,
    block_size: int,
    key_tile: int | None = None,
    row_tile: int | None = None,
    k: int = 512,
    candidate_block_ids: jax.Array | None = None,
) -> jax.Array:
    """Select [T,k] logical IDs in input query order.

    candidate_block_ids selects per-query score columns [T,C*block_size].
    None retains request-union columns. Both modes reuse the shared key cache.
    """
    if q.ndim != 3 or query_start_loc.shape != (keys.shape[0] + 1,):
        raise ValueError("Expected flat Q[T,H,D] and request boundaries [R+1]")
    tokens = q.shape[0]
    if candidate_block_ids is not None and (
        candidate_block_ids.ndim != 2
        or candidate_block_ids.shape[0] != tokens
        or candidate_block_ids.shape[1] < 1
        or candidate_block_ids.dtype != jnp.int32
    ):
        raise ValueError("candidate_block_ids must be int32 [T,C] with C > 0")
    if tokens == 0:
        return jnp.empty((0, k), jnp.int32)
    positions = jnp.arange(tokens, dtype=jnp.int32)
    owners = jnp.minimum(
        jnp.searchsorted(query_start_loc[1:], positions, side="right"),
        keys.shape[0] - 1,
    )
    lengths = jnp.where(positions < query_start_loc[-1], lengths, 0)
    starts = None
    if candidate_block_ids is None:
        bits = membership_offsets(query_start_loc)[owners] * 32
        bits += positions - query_start_loc[owners]
        counts = jnp.minimum(union_counts[owners], ids.shape[1])
    else:
        ids, starts, counts = query_metadata(
            candidate_block_ids, ids, owners, lengths, block_size=block_size
        )
        owners = positions
        bits = jnp.zeros(tokens, jnp.int32)
        membership = jnp.ones((1, ids.shape[1]), jnp.int32)
    rows = jnp.stack((owners, bits, counts * block_size, lengths))
    config = tuned_block_sizes.get_sparse_mqa_block_sizes(
        tuned_block_sizes.SparseMQATuningKey(
            jax.devices()[0].device_kind,
            q.shape[1],
            ids.shape[1] * block_size,
            key_tile,
            k,
        ),
        tokens,
    )
    if row_tile is None:
        row_tile = config.row_tile
    if type(row_tile) is not int or row_tile < 1:
        raise ValueError("row_tile must be a positive integer")
    row_tile = min(row_tile, tokens)

    def prepare(start):
        end = min(start + row_tile, tokens)
        metadata = rows[:, start:end]
        if starts is None:
            return q[start:end], weights[start:end], metadata
        batch_keys = _gather_query_keys(keys, starts[start:end], block_size)
        metadata = metadata.at[0].set(jnp.arange(end - start, dtype=jnp.int32))
        return (
            q[start:end],
            weights[start:end],
            batch_keys,
            ids[start:end],
            membership,
            metadata,
        )

    def score(inputs):
        if starts is None:
            queries, row_weights, metadata = inputs
            batch_keys, batch_ids, members = keys, ids, membership
        else:
            queries, row_weights, batch_keys, batch_ids, members, metadata = inputs
        return sparse_mqa_logits(
            queries,
            row_weights,
            batch_keys,
            batch_ids,
            members,
            metadata,
            block_size=block_size,
            key_tile=config.key_tile,
            compute_tile=config.compute_tile,
            dma_tile=config.dma_tile,
        )

    def select(values, start):
        end = start + values.shape[0]
        return sparse_topk_remap(
            values,
            valid_lengths=jnp.where(rows[3, start:end] > 0, rows[2, start:end], 0),
            logical_blocks=ids,
            row_requests=owners[start:end],
            k=k,
            block_size=config.topk_block_size,
            reconstruct_threshold=config.reconstruct_threshold,
        )

    selected = []
    previous = None
    current = score(prepare(0))
    for start in range(row_tile, tokens, row_tile):
        inputs = prepare(start)
        if previous is None:
            inputs, current = jax.lax.optimization_barrier((inputs, current))
        else:
            inputs, current, previous = jax.lax.optimization_barrier(
                (inputs, current, previous)
            )
            selected.append(previous)
        following = score(inputs)
        previous = select(current, start - row_tile)
        current = following
    if previous is not None:
        selected.append(previous)
    selected.append(select(current, (tokens - 1) // row_tile * row_tile))
    return selected[0] if len(selected) == 1 else jnp.concatenate(selected)


def cache_capacity(
    query_shape: tuple[int, ...],
    record_shape: tuple[int, ...],
    candidate_count: int,
    block_size: int,
    page_records: int,
) -> int | None:
    """Static key-cache envelope; query ownership comes from runtime boundaries."""
    _, width = record_shape
    _, heads, dim = query_shape
    if block_size < SLOTS or block_size % SLOTS:
        return None
    capacity = tuned_block_sizes.SPARSE_MQA_KEY_CAPACITY // block_size * block_size
    if (
        dim != DIM
        or heads < 1
        or candidate_count < 1
        or candidate_count * block_size > capacity
        or not 0 < capacity < width
        or page_records < block_size
        or page_records % block_size
        or width % page_records
    ):
        return None
    return capacity
