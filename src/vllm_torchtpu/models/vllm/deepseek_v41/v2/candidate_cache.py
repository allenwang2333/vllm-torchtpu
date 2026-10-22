# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare request-local candidate keys and metadata for reuse across a stage."""

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS, SLOTS
from vllm_torchtpu.kernels.deepseek_v41.index_selection import sparse_mqa_metadata
from vllm_torchtpu.kernels.deepseek_v41.index_selection.gather import (
    gather_packed_candidate_keys,
    is_supported_by_gather_packed_candidate_keys,
)

from .cache import IndexCacheRows, _cache_num_rows, gather_cache_page_words


class SparseMQACache(NamedTuple):
    """Request-local keys/unions and contiguous query intervals.

    Each membership word names 32 request-local queries for one union block.
    """

    keys: torch.Tensor
    union_block_ids: torch.Tensor
    membership: torch.Tensor
    union_block_count: torch.Tensor
    usable: torch.Tensor
    query_offsets: torch.Tensor


_prepare_op = None
_prepare_2bank_op = None


def prepare_sparse_mqa_cache_jax(
    pool: jax.Array,
    records: jax.Array,
    candidate_block_ids: jax.Array,
    query_offsets: jax.Array,
    lengths: jax.Array,
    block_size: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Gather request-local candidate blocks; capacity counts Index-K records."""
    if block_size <= 0 or block_size % SLOTS or capacity <= 0 or capacity % block_size:
        raise ValueError("Candidate capacity must contain whole even-sized blocks")
    groups = records.shape[0]
    key_shape = (groups, capacity // SLOTS, ROW_WORDS)
    assert query_offsets.shape == (groups + 1,)
    meta = sparse_mqa_metadata.union_metadata(
        candidate_block_ids,
        lengths,
        query_offsets,
        logical_blocks=(records.shape[1] + block_size - 1) // block_size,
        capacity_blocks=capacity // block_size,
    )
    address_meta = meta._replace(membership=jnp.ones_like(meta.union_block_ids))
    starts, valid = sparse_mqa_metadata.union_record_starts(
        address_meta,
        records,
        block_size=block_size,
        pool_records=_cache_num_rows(pool) * SLOTS,
    )
    usable = ~meta.overflow
    starts = jnp.where(usable[:, None], starts, -1)
    ids = jnp.where(valid != 0, meta.union_block_ids, -1)
    use_sparsecore_kernel = is_supported_by_gather_packed_candidate_keys(
        pool, starts, block_size
    )

    keys = (
        gather_packed_candidate_keys(pool, starts, block_size)
        if use_sparsecore_kernel
        else gather_cache_page_words(pool, starts, block_size, DIM)
    ).reshape(key_shape)
    return keys, ids, meta.membership, meta.union_block_count, usable


def prepare_sparse_mqa_cache_2bank_jax(
    pool_lo: jax.Array,
    pool_hi: jax.Array,
    records: jax.Array,
    candidate_block_ids: jax.Array,
    query_offsets: jax.Array,
    lengths: jax.Array,
    block_size: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return prepare_sparse_mqa_cache_jax(
        cast(Any, (pool_lo, pool_hi)),
        records,
        candidate_block_ids,
        query_offsets,
        lengths,
        block_size,
        capacity,
    )


def initialize_sparse_mqa_cache() -> None:
    global _prepare_op, _prepare_2bank_op
    if _prepare_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
    )

    # The pinned bridge annotation omits valid PartitionSpec and static None entries.
    _prepare_op = pallas.jax_op(
        "pallas::dsv41_sparse_mqa_prepare",
        prepare_sparse_mqa_cache_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 5 + (None,) * 2)
        ),
    )
    _prepare_2bank_op = pallas.jax_op(
        "pallas::dsv41_sparse_mqa_prepare_2bank",
        prepare_sparse_mqa_cache_2bank_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 6 + (None,) * 2)
        ),
    )

    def fake_prepare(
        pool: torch.Tensor,
        records: torch.Tensor,
        candidate_block_ids: torch.Tensor,
        query_offsets: torch.Tensor,
        lengths: torch.Tensor,
        block_size: int,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        groups = records.shape[0]
        return (
            pool.new_empty((groups, capacity // SLOTS, ROW_WORDS), dtype=torch.int32),
            records.new_empty((groups, capacity // block_size)),
            records.new_empty(
                (
                    sparse_mqa_metadata.membership_rows(
                        candidate_block_ids.shape[0], groups
                    ),
                    capacity // block_size,
                ),
                dtype=torch.int32,
            ),
            records.new_empty((groups,)),
            records.new_empty((groups,), dtype=torch.bool),
        )

    def fake_prepare_2bank(
        pool_lo: torch.Tensor,
        pool_hi: torch.Tensor,
        records: torch.Tensor,
        candidate_block_ids: torch.Tensor,
        query_offsets: torch.Tensor,
        lengths: torch.Tensor,
        block_size: int,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return fake_prepare(
            pool_lo,
            records,
            candidate_block_ids,
            query_offsets,
            lengths,
            block_size,
            capacity,
        )

    _prepare_op.register_fake(fake_prepare)
    _prepare_2bank_op.register_fake(fake_prepare_2bank)


def prepare_sparse_mqa_cache(
    cache: IndexCacheRows,
    records: torch.Tensor,
    candidate_block_ids: torch.Tensor,
    query_offsets: torch.Tensor,
    lengths: torch.Tensor,
    *,
    block_size: int,
    capacity: int,
) -> SparseMQACache:
    """Gather Index-K for contiguous request intervals from attention metadata."""
    assert _prepare_op is not None and _prepare_2bank_op is not None
    query_offsets = query_offsets.int()
    args = (
        records.int(),
        candidate_block_ids.int(),
        query_offsets,
        lengths.flatten().int(),
        block_size,
        capacity,
    )
    result = (
        _prepare_2bank_op(cache.raw, cache.raw_hi, *args)
        if cache.raw_hi is not None
        else _prepare_op(cache.raw, *args)
    )
    return SparseMQACache(*result, query_offsets=query_offsets)
