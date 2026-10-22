# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MQA metadata for contiguous, variable-length request query intervals."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import SLOTS

from . import tuned_block_sizes


class UnionMetadata(NamedTuple):
    union_block_ids: jax.Array
    membership: jax.Array
    union_block_count: jax.Array
    overflow: jax.Array


def _sorted_union_metadata(
    candidate_block_ids: jax.Array, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    requests, queries, _ = candidate_block_ids.shape
    valid = (candidate_block_ids >= 0) & (candidate_block_ids < logical_blocks)
    # Include query ownership in the sort key so duplicate IDs within one
    # query contribute its membership bit only once.
    encoded = jnp.where(
        valid,
        candidate_block_ids * queries + jnp.arange(queries)[None, :, None],
        logical_blocks * queries,
    )
    encoded = jnp.sort(encoded.reshape(requests, -1), axis=1)
    blocks = encoded // queries
    valid = blocks < logical_blocks
    previous = jnp.pad(encoded[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    first = valid & (blocks != previous // queries)
    count = first.sum(1).astype(jnp.int32)
    slots = jnp.where(valid, jnp.cumsum(first, axis=1) - 1, capacity_blocks)
    rows = jnp.arange(requests)[:, None]
    ids = jnp.full((requests, capacity_blocks), -1, jnp.int32)
    ids = ids.at[rows, slots].max(jnp.where(valid, blocks, -1), mode="drop")
    bits = jnp.where(valid & (encoded != previous), 1 << (encoded % queries), 0)
    membership = jnp.zeros_like(ids).at[rows, slots].add(bits, mode="drop")
    return UnionMetadata(ids, membership, count, count > capacity_blocks)


def grouped_union_metadata(
    candidate_block_ids: jax.Array, *, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    """Compact request-local logical block IDs and retain full union counts."""
    requests, queries, _ = candidate_block_ids.shape
    if (
        not 1 <= queries <= jnp.iinfo(jnp.int32).bits
        or min(logical_blocks, capacity_blocks) < 1
    ):
        raise ValueError("Query memberships must fit int32 and capacities be positive")
    if logical_blocks > jnp.iinfo(jnp.int32).max // queries:
        raise ValueError("Encoded block/query IDs must fit int32")
    # For longer contexts, bound metadata work by candidate count, not ID space.
    if logical_blocks > tuned_block_sizes.CANDIDATE_DENSE_UNION_MAX_BLOCKS:
        return _sorted_union_metadata(
            candidate_block_ids, logical_blocks, capacity_blocks
        )
    if (
        jax.default_backend() == "tpu"
        and candidate_block_ids.dtype == jnp.int32
        and requests > 0
        and candidate_block_ids.shape[-1] > 0
    ):
        from .union import metadata_layout, sparse_union_metadata

        tpu_info = pltpu.get_tpu_info()
        info = tpu_info.sparse_core
        if (
            info is None
            or tpu_info.chip_version != pltpu.ChipVersion.TPU_7X
            or candidate_block_ids.shape[-1] % tpu_info.num_lanes
        ):
            return _dense_union_metadata(
                candidate_block_ids, logical_blocks, capacity_blocks
            )
        layout = metadata_layout(
            candidate_block_ids.shape, logical_blocks, capacity_blocks
        )
        if layout.scratch_bytes <= info.vmem_capacity_bytes:
            return UnionMetadata(
                *sparse_union_metadata(
                    candidate_block_ids, logical_blocks, capacity_blocks
                )
            )
    return _dense_union_metadata(candidate_block_ids, logical_blocks, capacity_blocks)


def _dense_union_metadata(
    candidate_block_ids: jax.Array, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    requests, queries, _ = candidate_block_ids.shape
    valid = (candidate_block_ids >= 0) & (candidate_block_ids < logical_blocks)
    safe = jnp.where(valid, candidate_block_ids, logical_blocks)
    presence = jnp.zeros((requests, queries, logical_blocks + 1), jnp.int32)
    presence = presence.at[
        jnp.arange(requests)[:, None, None], jnp.arange(queries)[None, :, None], safe
    ].max(valid.astype(jnp.int32))
    bits = (presence[:, :, :-1] << jnp.arange(queries)[None, :, None]).sum(1)
    count = (bits != 0).sum(1).astype(jnp.int32)
    ids = jax.vmap(
        lambda row: jnp.nonzero(row, size=capacity_blocks, fill_value=-1)[0]
    )(bits)
    membership = jnp.take_along_axis(bits, jnp.maximum(ids, 0), axis=1)
    return UnionMetadata(
        ids, jnp.where(ids >= 0, membership, 0), count, count > capacity_blocks
    )


def union_record_starts(
    metadata: UnionMetadata,
    records: jax.Array,
    *,
    block_size: int,
    pool_records: int,
) -> tuple[jax.Array, jax.Array]:
    """Resolve logical blocks to aligned physical runs; invalidate bad pages."""
    logical = jnp.maximum(metadata.union_block_ids, 0) * block_size
    starts = jnp.take_along_axis(records, jnp.minimum(logical, records.shape[1] - 1), 1)
    valid = (metadata.union_block_ids >= 0) & (logical < records.shape[1])
    valid &= (starts >= 0) & (starts + block_size <= pool_records)
    # D128 BF16 packs two records into one pool word row. Logical blocks
    # need not start at a multiple of block_size in the physical allocation.
    valid &= starts % SLOTS == 0
    return jnp.where(valid, starts, -1), jnp.where(valid, metadata.membership, 0)


def membership_rows(tokens: int, requests: int) -> int:
    """Bound request-local query words, plus a zero word for boundary reads."""
    return (tokens + requests * 31) // 32 + 1


def membership_offsets(query_offsets: jax.Array) -> jax.Array:
    """Locate each request's query words; each bit names one local query."""
    return jnp.concatenate(
        (jnp.zeros(1, jnp.int32), jnp.cumsum((jnp.diff(query_offsets) + 31) // 32))
    )


def query_metadata(
    candidates: jax.Array,
    union_ids: jax.Array,
    row_requests: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return sorted unique query block IDs, shared-cache key starts and counts."""
    rows, capacity = candidates.shape
    sentinel = jnp.iinfo(jnp.int32).max
    valid = (candidates >= 0) & (
        candidates < (lengths[:, None] + block_size - 1) // block_size
    )
    ordered = jnp.sort(jnp.where(valid, candidates, sentinel), axis=1)
    previous = jnp.pad(ordered[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    valid = (ordered != sentinel) & (ordered != previous)

    # Invalid physical pages can leave holes in the otherwise sorted union.
    lookup, cache_slots = jax.lax.sort_key_val(
        jnp.where(union_ids >= 0, union_ids, sentinel),
        jnp.broadcast_to(
            jnp.arange(union_ids.shape[1], dtype=jnp.int32), union_ids.shape
        ),
        dimension=1,
    )
    slots = jax.vmap(jnp.searchsorted)(lookup[row_requests], ordered)
    slots = jnp.minimum(slots, union_ids.shape[1] - 1)
    valid &= jnp.take_along_axis(lookup[row_requests], slots, axis=1) == ordered
    cache_slots = jnp.take_along_axis(cache_slots[row_requests], slots, axis=1)
    starts = (row_requests[:, None] * union_ids.shape[1] + cache_slots) * block_size
    targets = jnp.where(valid, jnp.cumsum(valid, axis=1) - 1, capacity)
    row_ids = jnp.arange(rows)[:, None]
    ids = (
        jnp.full(candidates.shape, -1, jnp.int32)
        .at[row_ids, targets]
        .set(ordered, mode="drop")
    )
    starts = (
        jnp.full(candidates.shape, -1, jnp.int32)
        .at[row_ids, targets]
        .set(starts, mode="drop")
    )
    return ids, starts, valid.sum(1).astype(jnp.int32)


def union_metadata(
    candidates: jax.Array,
    lengths: jax.Array,
    query_offsets: jax.Array,
    *,
    logical_blocks: int,
    capacity_blocks: int,
) -> UnionMetadata:
    """Union each request interval, ignoring queries whose length is zero."""
    if min(logical_blocks, capacity_blocks) < 1:
        raise ValueError("Union requires positive block capacities")
    if jax.default_backend() == "tpu":
        from .union import is_supported_by_sparse_ragged_union, sparse_ragged_union

        if is_supported_by_sparse_ragged_union(
            candidates, query_offsets, logical_blocks, capacity_blocks
        ):
            return UnionMetadata(
                *sparse_ragged_union(
                    candidates,
                    lengths,
                    query_offsets,
                    membership_offsets(query_offsets),
                    logical_blocks,
                    capacity_blocks,
                )
            )
    # Portable implementation; the TPU kernel keeps these histograms in SC VMEM.
    tokens = candidates.shape[0]
    requests = query_offsets.size - 1
    positions = jnp.arange(tokens)
    owners = jnp.searchsorted(query_offsets[1:], positions, side="right")
    valid = (
        (candidates >= 0)
        & (candidates < logical_blocks)
        & (owners[:, None] < requests)
        & (lengths[:, None] > 0)
    )
    slots = jnp.where(valid, candidates, logical_blocks)
    presence = jnp.zeros((tokens, logical_blocks + 1), jnp.int32)
    presence = presence.at[positions[:, None], slots].set(valid.astype(jnp.int32))
    union = jnp.zeros((requests, logical_blocks), jnp.int32)
    union = union.at[owners].max(presence[:, :-1], mode="drop")
    counts = union.sum(1).astype(jnp.int32)
    ids = jax.vmap(
        lambda row: jnp.nonzero(row, size=capacity_blocks, fill_value=-1)[0]
    )(union)
    row_ids = ids[jnp.minimum(owners, requests - 1)]
    members = jnp.take_along_axis(presence, jnp.maximum(row_ids, 0), axis=1)
    members = (members != 0) & (row_ids >= 0) & (owners[:, None] < requests)
    local_queries = positions - query_offsets[jnp.minimum(owners, requests)]
    word_offsets = membership_offsets(query_offsets)
    words = word_offsets[jnp.minimum(owners, requests)] + local_queries // 32
    word_count = membership_rows(tokens, requests)
    words = jnp.where(owners < requests, words, word_count)
    values = (
        members.astype(jnp.uint32) << (local_queries % 32).astype(jnp.uint32)[:, None]
    )
    bits = jnp.zeros((word_count, capacity_blocks), jnp.uint32)
    bits = bits.at[words].add(values, mode="drop").astype(jnp.int32)
    return UnionMetadata(ids, bits, counts, counts > capacity_blocks)
