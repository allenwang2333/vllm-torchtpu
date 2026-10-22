# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate page masks that preserve logical token positions."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .packed_fp8_cache import PAGE_RECORDS
from .tuned_block_sizes import PACKED_FP8_MQA_CONFIG


class CandidatePages(NamedTuple):
    physical: jax.Array
    query_offsets: jax.Array
    lengths: jax.Array
    block_mask: jax.Array


def candidate_pages(
    candidates: jax.Array,
    lengths: jax.Array,
    requests: jax.Array,
    page_table: jax.Array,
    *,
    block_size: int,
    cache_pages: int,
) -> CandidatePages:
    """Mark pages needed by at most six adjacent queries of the same request.

    Queries must be sorted by request. Page slots retain their logical positions;
    the scorer skips unused pages, so neither compaction nor ID remapping is needed.
    """
    rows = candidates.shape[0]
    query_tile = PACKED_FP8_MQA_CONFIG.query_tile
    logical_pages = page_table.shape[1]
    if min(rows, block_size, logical_pages, page_table.shape[0]) < 1:
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

    width = logical_pages * PAGE_RECORDS
    blocks = (width + block_size - 1) // block_size
    valid = (candidates >= 0) & (candidates < blocks)
    block_mask = (
        jnp.zeros((rows, blocks + 1), jnp.int32)
        .at[positions[:, None], jnp.where(valid, candidates, blocks)]
        .set(1)[:, :blocks]
    )
    # Only expand to the common block/page boundary, not individual tokens.
    stride = math.gcd(PAGE_RECORDS, block_size)
    covered = jnp.repeat(block_mask, block_size // stride, axis=1)[:, : width // stride]
    covered *= jnp.arange(width // stride)[None, :] * stride < lengths[:, None]
    query_pages = covered.reshape(rows, logical_pages, PAGE_RECORDS // stride).max(-1)
    needed = (
        jnp.zeros((max_groups, logical_pages), jnp.int32).at[groups].max(query_pages)
    )
    physical = page_table[owners]
    physical = jnp.where(
        (needed != 0) & (physical >= 0) & (physical < cache_pages), physical, -1
    )
    page_tile = PACKED_FP8_MQA_CONFIG.key_tile // PAGE_RECORDS
    padded = (logical_pages + page_tile - 1) // page_tile * page_tile
    physical = jnp.pad(
        physical, ((0, 0), (0, padded - logical_pages)), constant_values=-1
    )
    lengths = jnp.where(query_pages.any(axis=1), lengths, 0)
    return CandidatePages(physical, offsets, lengths, block_mask)


def mask_candidate_scores(
    scores: jax.Array,
    block_mask: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int,
    width: int,
) -> jax.Array:
    """Broadcast query-local block membership over logically ordered scores."""
    blocks = (scores.shape[1] + block_size - 1) // block_size
    block_mask = jnp.pad(block_mask, ((0, 0), (0, blocks - block_mask.shape[1])))
    allowed = jnp.repeat(block_mask != 0, block_size, axis=1)[:, : scores.shape[1]]
    allowed &= (
        jnp.arange(scores.shape[1])[None, :] < jnp.minimum(lengths, width)[:, None]
    )
    return jnp.where(allowed, scores, -jnp.inf)
