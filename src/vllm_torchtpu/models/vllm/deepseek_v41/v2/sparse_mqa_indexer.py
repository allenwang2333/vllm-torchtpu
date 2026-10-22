# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-consuming indexer with stage-local key reuse."""

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    sparse_mqa_logits,
    tuned_block_sizes,
)

from . import attention_kernels
from .cache import IndexCacheRows, split_halves
from .candidate_cache import initialize_sparse_mqa_cache, prepare_sparse_mqa_cache
from .packed_fp8_indexer import initialize_packed_fp8_indexer

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache

_select_op = None
_select_2bank_op = None


def _select_indices_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidate_block_ids: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    keys: jax.Array,
    union_block_ids: jax.Array,
    membership: jax.Array,
    union_block_count: jax.Array,
    usable: jax.Array,
    query_offsets: jax.Array,
    block_size: int,
    page_records: int,
    row_tile: int | None,
    per_query_candidates: bool = False,
) -> jax.Array:
    if q.shape[0] == 0:
        return jnp.empty((0, 512), jnp.int32)

    def cached():
        result = sparse_mqa_logits.sparse_mqa_indexer(
            split_halves(q),
            weights.astype(jnp.float32),
            keys,
            union_block_ids,
            membership,
            lengths,
            query_start_loc=query_offsets,
            union_counts=union_block_count,
            block_size=block_size,
            row_tile=row_tile,
            candidate_block_ids=candidate_block_ids if per_query_candidates else None,
        )

        # Repair only overflowing requests, in bounded batches of flat queries.
        bad = ~usable & (jnp.diff(query_offsets) > 0)
        bad_requests = jnp.nonzero(bad, size=bad.size, fill_value=0)[0]
        tile = min(q.shape[0], tuned_block_sizes.DEFAULT_SPARSE_MQA_ROW_TILE)

        def repair(index, output):
            owner = bad_requests[index]
            begin, end = query_offsets[owner], query_offsets[owner + 1]

            def repair_tile(index, output):
                rows = begin + index * tile + jnp.arange(tile)
                valid = rows < end
                rows = jnp.minimum(rows, q.shape[0] - 1)
                selected = attention_kernels.paged_indexer_jax(
                    q[rows],
                    pool,
                    jax.lax.dynamic_slice_in_dim(records, owner, 1),
                    weights[rows],
                    jnp.where(valid, lengths[rows], 0),
                    candidate_block_ids[rows],
                    jnp.zeros(rows.shape, jnp.int32),
                    512,
                    block_size,
                    False,
                    True,
                    True,
                    batch_max,
                    page_records,
                )[0]
                return output.at[jnp.where(valid, rows, q.shape[0])].set(
                    selected, mode="drop"
                )

            return jax.lax.fori_loop(
                0, (end - begin + tile - 1) // tile, repair_tile, output
            )

        return jax.lax.fori_loop(0, jnp.sum(bad), repair, result)

    def fallback():
        return attention_kernels.paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidate_block_ids,
            requests,
            512,
            block_size,
            False,
            True,
            True,
            batch_max,
            page_records,
        )[0]

    return jax.lax.cond(batch_max.max() > 512, cached, fallback)


def _select_indices_2bank_jax(
    q: jax.Array,
    pool_lo: jax.Array,
    pool_hi: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidate_block_ids: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    keys: jax.Array,
    union_block_ids: jax.Array,
    membership: jax.Array,
    union_block_count: jax.Array,
    usable: jax.Array,
    query_offsets: jax.Array,
    block_size: int,
    page_records: int,
    row_tile: int | None,
    per_query_candidates: bool = False,
) -> jax.Array:
    return _select_indices_jax(
        q,
        cast(Any, (pool_lo, pool_hi)),
        records,
        weights,
        lengths,
        candidate_block_ids,
        requests,
        batch_max,
        keys,
        union_block_ids,
        membership,
        union_block_count,
        usable,
        query_offsets,
        block_size,
        page_records,
        row_tile,
        per_query_candidates,
    )


def initialize_indexer() -> None:
    """Register plain and cached indexer paths at model startup."""
    global _select_op, _select_2bank_op
    attention_kernels.initialize_indexer()
    initialize_packed_fp8_indexer()
    initialize_sparse_mqa_cache()
    if _select_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
    )

    # The pinned bridge annotation omits valid PartitionSpec and static None entries.
    _select_op = pallas.jax_op(
        "pallas::dsv41_sparse_mqa_select",
        _select_indices_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 14 + (None,) * 4)
        ),
    )
    _select_2bank_op = pallas.jax_op(
        "pallas::dsv41_sparse_mqa_select_2bank",
        _select_indices_2bank_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 15 + (None,) * 4)
        ),
    )

    def fake_select(
        q: torch.Tensor,
        pool: torch.Tensor,
        records: torch.Tensor,
        weights: torch.Tensor,
        lengths: torch.Tensor,
        candidate_block_ids: torch.Tensor,
        requests: torch.Tensor,
        batch_max: torch.Tensor,
        keys: torch.Tensor,
        union_block_ids: torch.Tensor,
        membership: torch.Tensor,
        union_block_count: torch.Tensor,
        usable: torch.Tensor,
        query_offsets: torch.Tensor,
        block_size: int,
        page_records: int,
        row_tile: int | None,
        per_query_candidates: bool = False,
    ) -> torch.Tensor:
        return records.new_empty((q.shape[0], 512))

    def fake_select_2bank(
        q: torch.Tensor,
        pool_lo: torch.Tensor,
        pool_hi: torch.Tensor,
        records: torch.Tensor,
        weights: torch.Tensor,
        lengths: torch.Tensor,
        candidate_block_ids: torch.Tensor,
        requests: torch.Tensor,
        batch_max: torch.Tensor,
        keys: torch.Tensor,
        union_block_ids: torch.Tensor,
        membership: torch.Tensor,
        union_block_count: torch.Tensor,
        usable: torch.Tensor,
        query_offsets: torch.Tensor,
        block_size: int,
        page_records: int,
        row_tile: int | None,
        per_query_candidates: bool = False,
    ) -> torch.Tensor:
        return fake_select(
            q,
            pool_lo,
            records,
            weights,
            lengths,
            candidate_block_ids,
            requests,
            batch_max,
            keys,
            union_block_ids,
            membership,
            union_block_count,
            usable,
            query_offsets,
            block_size,
            page_records,
            row_tile,
            per_query_candidates,
        )

    _select_op.register_fake(fake_select)
    _select_2bank_op.register_fake(fake_select_2bank)


class SparseMQAIndexer(torch.nn.Module):
    """Consume candidates after Q projection and K-cache writes."""

    def __init__(
        self,
        k_cache: "DeepseekV4IndexerCache",
        topk_tokens: int,
        candidate_blocks: torch.Tensor,
        candidate_block_size: int,
        *,
        candidate_source_layer: int,
        per_query_candidates: bool = False,
    ) -> None:
        super().__init__()
        self.k_cache = k_cache
        self.topk_tokens = topk_tokens
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_source_layer = candidate_source_layer
        self.per_query_candidates = per_query_candidates

    def forward(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        cache: IndexCacheRows,
        records: torch.Tensor,
        lengths: torch.Tensor,
        requests: torch.Tensor,
        query_start_loc: torch.Tensor,
        *,
        batch_max: torch.Tensor,
        page_records: int,
        capacity: int,
        stage_cache: dict[Any, Any],
    ) -> torch.Tensor:
        assert self.topk_tokens == 512
        candidates = self.candidate_blocks[: q.shape[0]]
        cache_key = (
            "candidate_index_keys",
            self.k_cache.prefix,
            stage_cache.get(("candidate_index_version", self.k_cache.prefix), 0),
            self.candidate_source_layer,
            stage_cache.get(("candidate_publication", self.candidate_source_layer), 0),
            capacity,
        )
        prepared = stage_cache.get(cache_key)
        if prepared is None:
            prepared = prepare_sparse_mqa_cache(
                cache,
                records,
                candidates,
                query_start_loc[: records.shape[0] + 1],
                lengths,
                block_size=self.candidate_block_size,
                capacity=capacity,
            )
            stage_cache[cache_key] = prepared

        assert _select_op is not None and _select_2bank_op is not None
        args = (
            records.int(),
            weights,
            lengths.flatten().int(),
            candidates.int(),
            requests.int(),
            batch_max.reshape(1).int(),
            prepared.keys,
            prepared.union_block_ids,
            prepared.membership,
            prepared.union_block_count,
            prepared.usable,
            prepared.query_offsets,
            self.candidate_block_size,
            page_records,
            None,
            self.per_query_candidates,
        )
        return (
            _select_2bank_op(q, cache.raw, cache.raw_hi, *args)
            if cache.raw_hi is not None
            else _select_op(q, cache.raw, *args)
        )
