# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kernel-owned routing for GLM-5.2 sparse and masked-dense MLA.

The layer-facing sparse MLA interface stays unchanged. Eligibility, profile
validation, cost selection, and device-side dispatch live here so the shared
runner, attention metadata, and attention implementation do not acquire
model-specific routing state.
"""

import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.mla.masked_dense.cost_model import (
    GLM52_TPU7X_PREFILL_COST_MODEL, KV_BLOCK_SIZE, QUERY_BLOCK_SIZE,
    MaskedDensePrefillCostModel)
from vllm_torchtpu.kernels.mla.masked_dense.kernel import (
    _masked_dense_suffix_ragged_paged_attention,
    masked_dense_ragged_paged_attention)
from vllm_torchtpu.kernels.mla.sparse.kernel import (
    _prepare_sparse_first_chunk, _sparse_prefix_ragged_paged_attention)
from vllm_torchtpu.kernels.mla.sparse.kernel import \
    prepare_q_inputs as _prepare_query
from vllm_torchtpu.kernels.mla.sparse.kernel import \
    sparse_ragged_paged_attention

# This profile is deliberately exact. The opt-in is process-wide, while the
# sparse MLA wrapper is generic to vLLM's DeepSeekV2-style model family. An
# unrecognized sparse model must therefore retain the original gather kernel.
GLM52_NUM_HEADS = 64
GLM52_NOPE_DIM = 512
GLM52_ROPE_DIM = 64
GLM52_INDEX_TOPK = 2048
GLM52_PAGE_SIZE = 1024

MASKED_DENSE_ANALYTIC_MAX_KV_LEN = 2048
MASKED_DENSE_MAX_KV_LEN = 6144
MASKED_DENSE_KV_BLOCK = 1024
MASKED_DENSE_KV_PAGES_PER_BLOCK = (1, 1, 1)
MASKED_DENSE_QUERIES_PER_BLOCK = (1, 32, 32)
MASKED_DENSE_PREFILL_COST_MODEL = GLM52_TPU7X_PREFILL_COST_MODEL
# Keep decode-sized compiled buckets on the original sparse graph. The
# masked-dense serving target uses at least 512 query rows; below that point a
# dynamic dispatcher would add branch and compilation cost to the common
# decode path while serving no targeted prefill workload.
MASKED_DENSE_MIN_TOKEN_BUCKET = 512
# A 1,024-token TPU7x sweep found a split with 256 masked-dense suffix rows
# slower than sparse-only, while 512 suffix rows was faster.
MASKED_DENSE_GROUPED_MIN_SUFFIX_TOKENS = 512


def _resolve_masked_dense_limit(name: str, default: int) -> int:
    value = getattr(envs, name)
    if value is None:
        return default
    if value < 0 or value % MASKED_DENSE_KV_BLOCK != 0:
        raise ValueError(
            f"{name}={value} must be 0 (disabled) or a positive multiple of "
            f"{MASKED_DENSE_KV_BLOCK}")
    return value


# Process-static configuration: servers set the environment before importing
# model code. Tests may override these module attributes directly.
MASKED_DENSE_ANALYTIC_LIMIT = _resolve_masked_dense_limit(
    "TPU_MLA_MASKED_DENSE_ANALYTIC_MAX_KV_LEN",
    MASKED_DENSE_ANALYTIC_MAX_KV_LEN)
MASKED_DENSE_LIMIT = _resolve_masked_dense_limit(
    "TPU_MLA_MASKED_DENSE_MAX_KV_LEN", MASKED_DENSE_MAX_KV_LEN)


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def matches_glm52_tpu7x_profile(
    q: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    topk_indices: jax.Array,
) -> bool:
    """Whether static operands match the calibrated GLM-5.2 kernel profile."""
    return bool(
        q.ndim == 3
        and q.shape[1:] == (GLM52_NUM_HEADS, GLM52_NOPE_DIM + GLM52_ROPE_DIM)
        and topk_indices.ndim == 2
        and topk_indices.shape[-1] == GLM52_INDEX_TOPK
        and kv_cache_nope.ndim == 4
        and kv_cache_nope.shape[1] == GLM52_PAGE_SIZE
        and kv_cache_rope.ndim == 4
        and kv_cache_rope.shape[1] * kv_cache_rope.shape[2] == GLM52_PAGE_SIZE)


def matches_glm52_tpu7x_dimensions(
    *,
    num_heads: int,
    nope_dim: int,
    rope_dim: int,
    kv_dim: int,
    k_pe_dim: int,
    topk: int,
    page_size: int,
    rope_page_size: int,
) -> bool:
    """Static counterpart used to scope the runner's neutral reordering."""
    return (num_heads == GLM52_NUM_HEADS and nope_dim == GLM52_NOPE_DIM
            and rope_dim == GLM52_ROPE_DIM and kv_dim == GLM52_NOPE_DIM
            and k_pe_dim == GLM52_ROPE_DIM and topk == GLM52_INDEX_TOPK
            and page_size == GLM52_PAGE_SIZE
            and rope_page_size == GLM52_PAGE_SIZE)


def masked_dense_prefill_mla_tier(
    seq_lens: jax.Array,
    request_distribution: jax.Array,
    query_start_loc: jax.Array,
    limits: tuple[int, int],
    cost_model: MaskedDensePrefillCostModel | None = None,
) -> jax.Array:
    """Select analytic suffix, bitmap suffix, or sparse for a reordered step.

    The runner establishes ``[decode][long][medium][short]`` order only. This
    function independently derives and validates each tier's boundary from
    existing device tensors; no split point or selected tier crosses the
    attention interface.
    """
    if cost_model is None:
        cost_model = MASKED_DENSE_PREFILL_COST_MODEL
    analytic_limit, bitmap_limit = limits
    num_seqs = request_distribution[2]
    seq_ids = jnp.arange(seq_lens.shape[0])
    active = seq_ids < num_seqs
    seq_ok = jnp.where(active, seq_lens, 0)
    q_lens = (query_start_loc[1:seq_lens.shape[0] + 1] -
              query_start_loc[:seq_lens.shape[0]])
    kv_blocks = (seq_ok + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    q_blocks = (q_lens + QUERY_BLOCK_SIZE - 1) // QUERY_BLOCK_SIZE
    num_tokens = query_start_loc[num_seqs]
    sparse_cost = (cost_model.sparse_fixed_ns +
                   cost_model.sparse_token_ns * num_tokens)

    def candidate(bound, fixed_ns, token_ns, *, bitmap=False):
        prefix = active & ((seq_ids < request_distribution[0]) |
                           (seq_ok > bound))
        split_seq = jnp.sum(prefix).astype(jnp.int32)
        expected_prefix = active & (seq_ids < split_seq)
        partitioned = jnp.all((~active) | (prefix == expected_prefix))
        suffix = active & (seq_ids >= split_seq)
        suffix_tokens = num_tokens - query_start_loc[split_seq]
        prefix_tokens = num_tokens - suffix_tokens
        suffix_grid = jnp.sum(jnp.where(suffix, q_blocks * kv_blocks, 0))
        dense_tokens = jnp.where(bitmap & (split_seq > 0), num_tokens,
                                 suffix_tokens)
        dense_cost = (fixed_ns + token_ns * dense_tokens +
                      cost_model.q_kv_block_ns * suffix_grid)
        prefix_cost = (cost_model.sparse_fixed_ns +
                       cost_model.sparse_token_ns * prefix_tokens)
        total_cost = dense_cost + jnp.where(split_seq > 0, prefix_cost, 0)
        split_worthwhile = ((split_seq == 0) |
                            (suffix_tokens
                             >= MASKED_DENSE_GROUPED_MIN_SUFFIX_TOKENS))
        valid = ((bound > 0) & partitioned & (split_seq < num_seqs)
                 & split_worthwhile)
        return jnp.where(valid, total_cost, jnp.iinfo(jnp.int32).max)

    analytic_cost = candidate(analytic_limit, cost_model.analytic_fixed_ns,
                              cost_model.analytic_token_ns)
    bitmap_cost = candidate(bitmap_limit,
                            cost_model.bitmap_fixed_ns,
                            cost_model.bitmap_token_ns,
                            bitmap=True)
    unavailable = jnp.iinfo(jnp.int32).max
    costs = jnp.stack(
        (jnp.where(analytic_limit > 0, analytic_cost, unavailable),
         jnp.where(bitmap_limit > 0, bitmap_cost, unavailable), sparse_cost))
    return jnp.argmin(costs).astype(jnp.int32)


def _ragged_paged_attention(
    q: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    topk_indices: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    sm_scale: float | None = None,
    k_scale: float | None = None,
    dense_first: bool = False,
    shared_query_padding: bool = False,
) -> jax.Array:
    """Dispatch one traced DSA step without changing the layer interface."""
    limits = (MASKED_DENSE_ANALYTIC_LIMIT, MASKED_DENSE_LIMIT)
    enable_masked_dense = bool(
        envs.TPU_MLA_MASKED_DENSE_ENABLED
        and q.shape[0] >= MASKED_DENSE_MIN_TOKEN_BUCKET
        and matches_glm52_tpu7x_profile(q, kv_cache_nope, kv_cache_rope,
                                        topk_indices))

    def _sparse():
        return sparse_ragged_paged_attention(
            q,
            kv_cache_nope,
            kv_cache_rope,
            topk_indices,
            block_tables,
            query_start_loc,
            request_distribution,
            sm_scale=sm_scale or 1.0,
            k_scale=k_scale or 1.0,
        )

    if not enable_masked_dense:
        return _sparse()

    def _masked_dense(max_kv_len):
        seq_ids = jnp.arange(seq_lens.shape[0])
        active = seq_ids < request_distribution[2]
        prefix = active & ((seq_ids < request_distribution[0]) |
                           (seq_lens > max_kv_len))
        split_seq = jnp.sum(prefix).astype(jnp.int32)

        def _whole(_):
            return masked_dense_ragged_paged_attention(
                q,
                kv_cache_nope,
                kv_cache_rope,
                seq_lens,
                topk_indices,
                block_tables,
                query_start_loc,
                request_distribution,
                sm_scale=sm_scale or 1.0,
                k_scale=k_scale or 1.0,
                max_kv_len=max_kv_len,
                num_kv_pages_per_block=MASKED_DENSE_KV_PAGES_PER_BLOCK,
                num_queries_per_block=MASKED_DENSE_QUERIES_PER_BLOCK,
            )

        def _split(_):
            query = _prepare_query(q) if shared_query_padding else q

            def finish(output):
                return output[:, :q.shape[1], :q.shape[2]]

            split_token = query_start_loc[split_seq]
            prepared_inputs = first_gather = None
            if dense_first:
                prepared_inputs, first_gather = _prepare_sparse_first_chunk(
                    kv_cache_nope, kv_cache_rope, topk_indices, block_tables,
                    query_start_loc)

            def sparse_prefix(query):
                return _sparse_prefix_ragged_paged_attention(
                    query,
                    kv_cache_nope,
                    kv_cache_rope,
                    topk_indices,
                    block_tables,
                    query_start_loc,
                    request_distribution,
                    split_token,
                    prepared_inputs=prepared_inputs,
                    first_gather=first_gather,
                    sm_scale=sm_scale or 1.0,
                    k_scale=k_scale or 1.0,
                )

            def dense_suffix(query):
                return _masked_dense_suffix_ragged_paged_attention(
                    query,
                    kv_cache_nope,
                    kv_cache_rope,
                    seq_lens,
                    topk_indices,
                    block_tables,
                    query_start_loc,
                    request_distribution,
                    split_seq,
                    sm_scale=sm_scale or 1.0,
                    k_scale=k_scale or 1.0,
                    max_kv_len=max_kv_len,
                    num_kv_pages_per_block=MASKED_DENSE_KV_PAGES_PER_BLOCK,
                    num_queries_per_block=MASKED_DENSE_QUERIES_PER_BLOCK,
                )

            if dense_first:
                return finish(sparse_prefix(dense_suffix(query)))
            return finish(dense_suffix(sparse_prefix(query)))

        return jax.lax.cond(split_seq > 0, _split, _whole, None)

    page_span = (block_tables.shape[0] // seq_lens.shape[0] *
                 kv_cache_nope.shape[1])
    branches, tier_limits = [], list(limits)
    for slot, bound in enumerate(limits):
        bound = min(_align_to(bound, MASKED_DENSE_KV_BLOCK), page_span)
        if bound == 0 or (slot == 0 and bound > topk_indices.shape[-1]):
            tier_limits[slot] = 0
        else:
            branches.append(functools.partial(_masked_dense, bound))
    branches.append(_sparse)

    if len(branches) == 1:
        return _sparse()

    tier = masked_dense_prefill_mla_tier(seq_lens, request_distribution,
                                         query_start_loc, tuple(tier_limits))
    if len(branches) == 2:
        kept = 0 if tier_limits[0] else 1
        return jax.lax.cond(tier == kept, branches[0], branches[1])
    return jax.lax.switch(tier, branches)


def ragged_paged_attention(
    q: jax.Array,
    kv_cache_nope: jax.Array,
    kv_cache_rope: jax.Array,
    topk_indices: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    sm_scale: float | None = None,
    k_scale: float | None = None,
) -> jax.Array:
    """Dispatch one traced DSA step with the established layer interface."""
    return _ragged_paged_attention(q,
                                   kv_cache_nope,
                                   kv_cache_rope,
                                   topk_indices,
                                   seq_lens,
                                   block_tables,
                                   query_start_loc,
                                   request_distribution,
                                   sm_scale=sm_scale,
                                   k_scale=k_scale)
