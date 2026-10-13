# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SparseCore Pallas scatter of MLA latents into the paged KV cache."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.mla import kv_cache_utils, scatter_utils


@functools.cache
def _spec_meta(
    spec: kv_cache_utils.SparseMLAKVCacheSpec,
) -> tuple[tuple[int, ...], jnp.dtype, int, int, bool, str]:
    """Caches static layout metadata derived from an immutable KV cache spec."""
    return (
        spec.shape,
        spec.jax_dtype,
        spec.token_bytes,
        spec.page_size,
        spec.layout is kv_cache_utils.KVCacheLayout.TENSORCORE,
        f"scatter_{spec.cache_type.value}_{spec.layout.value}_cache",
    )


def _scatter_impl(
    cache: jax.Array,
    u8: jax.Array,
    dst_rows: jax.Array,
    token_bytes: int,
    is_tc: bool,
    kernel_name: str,
    scatter_rows_fn: Callable[..., jax.Array],
) -> jax.Array:
    u8 = jax.lax.optimization_barrier(u8)
    src = (
        kv_cache_utils.as_token_bytes(u8, token_bytes)
        if is_tc
        else kv_cache_utils.pack_tokens(u8, token_bytes)
    )
    return scatter_rows_fn(
        cache,
        src,
        dst_rows,
        name=kernel_name,
    )


@functools.cache
def _jit_scatter_dst(
    token_bytes: int,
    is_tc: bool,
    kernel_name: str,
    scatter_rows_fn: Callable[..., jax.Array],
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def _fn(
        cache: jax.Array,
        u8: jax.Array,
        dst_rows: jax.Array,
    ) -> jax.Array:
        return _scatter_impl(
            cache,
            u8,
            dst_rows,
            token_bytes,
            is_tc,
            kernel_name,
            scatter_rows_fn,
        )

    return _fn


@functools.cache
def _jit_scatter_addr(
    page_size: int,
    token_bytes: int,
    is_tc: bool,
    kernel_name: str,
    scatter_rows_fn: Callable[..., jax.Array],
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def _fn(
        cache: jax.Array,
        u8: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
    ) -> jax.Array:
        dst_rows = kv_cache_utils.get_dst_rows(
            num_tokens=u8.shape[0],
            seq_lens=seq_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            page_size=page_size,
        )
        return _scatter_impl(
            cache,
            u8,
            dst_rows,
            token_bytes,
            is_tc,
            kernel_name,
            scatter_rows_fn,
        )

    return _fn


def scatter(
    cache: jax.Array,
    values: jax.Array,
    spec: kv_cache_utils.SparseMLAKVCacheSpec,
    seq_lens: jax.Array | None = None,
    block_tables: jax.Array | None = None,
    query_start_loc: jax.Array | None = None,
    dst_rows: jax.Array | None = None,
) -> jax.Array:
    """Packs `values` for `spec`'s layout and scatters one row per token.

    Args:
      cache: paged KV cache in HBM, allocated as `spec.shape` / `spec.jax_dtype`.
      values: [num_tokens, unpadded head_dim] fp8 latents -- `kv_c_normed` for a
        NOPE cache, `k_pe` for a ROPE one.
      spec: layout descriptor the cache was allocated from.
      seq_lens: per-sequence KV length including this step's tokens.
      block_tables: flattened per-sequence-padded page table.
      query_start_loc: cumulative new-token counts, [num_seqs + 1].
      dst_rows: precomputed destination row per token; derived from the three
        arrays above when None.

    Returns:
      The updated cache, same shape and dtype as `cache`.
    """
    expected_shape, expected_dtype, token_bytes, page_size, is_tc, kernel_name = (
        _spec_meta(spec)
    )
    assert cache.shape == expected_shape and cache.dtype == expected_dtype, (
        f"cache {cache.shape} {cache.dtype} does not match spec"
        f" {expected_shape} {expected_dtype}"
    )
    u8 = (
        values
        if values.dtype == jnp.uint8
        else jax.lax.bitcast_convert_type(values, jnp.uint8)
    )
    scatter_rows_fn = scatter_utils.scatter_rows
    if dst_rows is None:
        assert (
            seq_lens is not None
            and block_tables is not None
            and query_start_loc is not None
        ), "pass dst_rows or the addressing arrays"
        if not isinstance(cache, jax.core.Tracer):
            return _jit_scatter_addr(
                page_size, token_bytes, is_tc, kernel_name, scatter_rows_fn
            )(cache, u8, seq_lens, block_tables, query_start_loc)
        dst_rows = kv_cache_utils.get_dst_rows(
            num_tokens=u8.shape[0],
            seq_lens=seq_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            page_size=page_size,
        )
    elif not isinstance(cache, jax.core.Tracer):
        return _jit_scatter_dst(token_bytes, is_tc, kernel_name, scatter_rows_fn)(
            cache, u8, dst_rows
        )
    return _scatter_impl(
        cache, u8, dst_rows, token_bytes, is_tc, kernel_name, scatter_rows_fn
    )
