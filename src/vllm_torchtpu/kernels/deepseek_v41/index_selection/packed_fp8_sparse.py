# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-driven page tasks without materializing another K cache."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .packed_fp8_cache import PAGE_RECORDS
from .sparse_mqa_metadata import union_metadata
from .tuned_block_sizes import PACKED_FP8_MQA_CONFIG


class CandidatePages(NamedTuple):
    logical: jax.Array
    physical: jax.Array
    query_offsets: jax.Array
    query_groups: jax.Array
    lengths: jax.Array


def candidate_pages(
    candidates: jax.Array,
    lengths: jax.Array,
    requests: jax.Array,
    page_table: jax.Array,
    *,
    block_size: int,
    cache_pages: int,
    topk: int,
) -> CandidatePages:
    """Union pages within at most six adjacent queries of the same request.

    Queries must be sorted by request. Candidate IDs are logical blocks;
    physical page holes and blocks beyond each query's length are excluded.
    Capacity covers the complete union, so no overflow fallback is necessary.
    """
    rows, candidate_count = candidates.shape
    query_tile = PACKED_FP8_MQA_CONFIG.query_tile
    logical_pages = page_table.shape[1]
    if min(rows, block_size, logical_pages, page_table.shape[0], topk) < 1:
        raise ValueError("Candidate page tasks require nonempty queries and pages")
    positions = jnp.arange(rows, dtype=jnp.int32)
    first = jnp.concatenate((jnp.ones((1,), jnp.bool_), requests[1:] != requests[:-1]))
    request_start = jax.lax.associative_scan(
        jnp.maximum, jnp.where(first, positions, 0)
    )
    first |= (positions - request_start) % query_tile == 0
    groups = jnp.cumsum(first, dtype=jnp.int32) - 1
    max_groups = min(
        rows, (rows + query_tile - 1) // query_tile + page_table.shape[0] - 1
    )
    offsets = jnp.full((max_groups + 1,), rows, jnp.int32).at[groups].min(positions)
    owners = requests[jnp.minimum(offsets[:-1], rows - 1)]

    if PAGE_RECORDS % block_size == 0:
        pages_per_block = 1
    elif block_size % PAGE_RECORDS == 0:
        pages_per_block = block_size // PAGE_RECORDS
    else:
        pages_per_block = (block_size + 2 * PAGE_RECORDS - 2) // PAGE_RECORDS
    # Clip invalid IDs before multiplication, including INT_MAX sentinels.
    max_blocks = (logical_pages * PAGE_RECORDS + block_size - 1) // block_size
    valid = (candidates >= 0) & (candidates < max_blocks)
    starts = jnp.clip(candidates, 0, max_blocks - 1) * block_size
    ids = starts[:, :, None] // PAGE_RECORDS + jnp.arange(pages_per_block)
    end = jnp.minimum(starts + block_size, lengths[:, None])
    valid = valid[:, :, None] & (starts[:, :, None] < end[:, :, None])
    valid &= (ids < logical_pages) & (ids * PAGE_RECORDS < end[:, :, None])
    ids = ids.reshape(rows, -1)
    physical = jnp.take_along_axis(
        page_table[requests], jnp.clip(ids, 0, logical_pages - 1), axis=1
    )
    valid = valid.reshape(ids.shape) & (physical >= 0) & (physical < cache_pages)
    ids = jnp.where(valid, ids, -1).astype(jnp.int32)
    # The existing SparseCore union consumes 128-aligned candidate metadata.
    aligned = max(128, (ids.shape[1] + 127) // 128 * 128)
    ids = jnp.pad(ids, ((0, 0), (0, aligned - ids.shape[1])), constant_values=-1)
    capacity = max(
        1, min(logical_pages, query_tile * candidate_count * pages_per_block)
    )
    union = union_metadata(
        ids, lengths, offsets, logical_blocks=logical_pages, capacity_blocks=capacity
    )
    logical = union.union_block_ids
    physical = jnp.take_along_axis(page_table[owners], jnp.maximum(logical, 0), axis=1)
    physical = jnp.where(logical >= 0, physical, -1)
    page_tile = PACKED_FP8_MQA_CONFIG.key_tile // PAGE_RECORDS
    padded = (
        (max(capacity, (topk + PAGE_RECORDS - 1) // PAGE_RECORDS) + page_tile - 1)
        // page_tile
        * page_tile
    )
    logical, physical = (
        jnp.pad(x, ((0, 0), (0, padded - capacity)), constant_values=-1)
        for x in (logical, physical)
    )
    compact_lengths = jnp.where(
        lengths > 0, union.union_block_count[groups] * PAGE_RECORDS, 0
    )
    return CandidatePages(logical, physical, offsets, groups, compact_lengths)


def mask_candidate_scores(
    scores: jax.Array,
    pages: CandidatePages,
    candidates: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int,
    width: int,
) -> jax.Array:
    """Keep query-local candidates and causal tails within the loaded page union."""
    logical = (
        pages.logical[pages.query_groups, :, None] * PAGE_RECORDS
        + jnp.arange(PAGE_RECORDS)
    ).reshape(scores.shape)
    blocks = (width + block_size - 1) // block_size
    valid = (candidates >= 0) & (candidates < blocks)
    membership = jnp.zeros((scores.shape[0], blocks + 1), jnp.bool_)
    membership = membership.at[
        jnp.arange(scores.shape[0])[:, None], jnp.where(valid, candidates, blocks)
    ].set(True)
    allowed = jnp.take_along_axis(
        membership, jnp.clip(logical // block_size, 0, blocks), axis=1
    )
    allowed &= (logical >= 0) & (logical < jnp.minimum(lengths, width)[:, None])
    return jnp.where(allowed, scores, -jnp.inf)


def restore_candidate_indices(indices: jax.Array, pages: CandidatePages) -> jax.Array:
    """Map compact top-k positions to original logical token IDs; preserve -1."""
    logical = jnp.take_along_axis(
        pages.logical[pages.query_groups],
        jnp.maximum(indices, 0) // PAGE_RECORDS,
        axis=1,
    )
    return jnp.where(
        (indices >= 0) & (logical >= 0),
        logical * PAGE_RECORDS + indices % PAGE_RECORDS,
        -1,
    ).astype(jnp.int32)
