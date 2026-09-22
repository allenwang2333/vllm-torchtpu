# SPDX-License-Identifier: Apache-2.0
"""Decode over a page-striped dense latent cache.

This first implementation uses bounded JAX tiles: it never materializes the
whole context or a dense top-k list. Communication stays in the torch adapter.
The caller supplies replicated queries and one token per active request.
"""

import jax
import jax.numpy as jnp
from jax import lax


def local_context_lengths(lengths, rank: int, size: int, page_size: int):
    cycle = size * page_size
    return (lengths // cycle * page_size +
            jnp.clip(lengths % cycle - rank * page_size, 0, page_size))


def decode_attention(
    cache: jax.Array,
    q_nope: jax.Array,
    q_pe: jax.Array,
    new_kv: jax.Array,
    new_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    rank: int,
    size: int,
    scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    tile_size: int = 2048,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Insert owned decode tokens; return cache, local output, natural-log LSE.

    Cache is the existing dense MLA [pages, page/packing, packing, width]
    layout, with latent and RoPE components concatenated before row padding.
    seq_lens includes the new token. Empty/padded requests contribute no KV.
    """
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"Invalid decode CP rank/size: {rank}/{size}")
    if cache.ndim != 4 or tile_size <= 0:
        raise ValueError("Expected packed four-dimensional MLA cache")
    page_size = cache.shape[1] * cache.shape[2]
    num_tokens, num_heads, latent_dim = q_nope.shape
    rope_dim = q_pe.shape[-1]
    num_reqs = seq_lens.shape[0]
    tables = block_tables.reshape(num_reqs, -1)
    # Token-major mappings also cover padded buckets and zero-token requests.
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    req_ids = jnp.searchsorted(query_start_loc[1:], token_ids, side="right")
    safe_reqs = jnp.minimum(req_ids, num_reqs - 1)
    active = (req_ids < num_reqs) & (token_ids < query_start_loc[-1])
    active &= seq_lens[safe_reqs] > 0
    positions = jnp.maximum(seq_lens[safe_reqs] - 1, 0)
    owner = positions // page_size % size
    local_pos = positions // (page_size *
                              size) * page_size + positions % page_size
    table_col = local_pos // page_size
    page_ids = tables[safe_reqs, jnp.minimum(table_col, tables.shape[1] - 1)]
    write = active & (owner == rank) & (table_col < tables.shape[1])
    # Out-of-bounds scatter indices are dropped, including padded token rows.
    page_ids = jnp.where(write, page_ids, cache.shape[0])
    values = jnp.concatenate((new_kv, new_pe), axis=-1)
    values = jnp.pad(values, ((0, 0), (0, cache.shape[-1] - values.shape[-1])))
    cache = cache.at[page_ids, local_pos % page_size // cache.shape[2],
                     local_pos % cache.shape[2], :].set(values, mode="drop")
    lengths = local_context_lengths(seq_lens, rank, size, page_size)

    def attend(t):
        req = safe_reqs[t]
        length = jnp.where(active[t], lengths[req], 0)
        q = jnp.concatenate((q_nope[t], q_pe[t]), axis=-1)
        initial = (jnp.full((num_heads, ), -jnp.inf,
                            jnp.float32), jnp.zeros((num_heads, ),
                                                    jnp.float32),
                   jnp.zeros((num_heads, latent_dim), jnp.float32))

        def step(i, state):
            m, denominator, acc = state
            pos = i * tile_size + jnp.arange(tile_size)
            if tile_size % page_size == 0:
                # Read each physical page once instead of repeating its index
                # for every token. This preserves contiguous packed KV reads.
                cols = i * (tile_size // page_size) + jnp.arange(
                    tile_size // page_size)
                pages = tables[req, jnp.minimum(cols, tables.shape[1] - 1)]
                rows = cache[pages].reshape(tile_size, cache.shape[-1])
            else:
                col = jnp.minimum(pos // page_size, tables.shape[1] - 1)
                pages = tables[req, col]
                rows = cache[pages, pos % page_size // cache.shape[2],
                             pos % cache.shape[2], :]
            valid = pos < length
            rows = jnp.where(valid[:, None], rows, 0)
            keys = rows[:, :latent_dim + rope_dim].astype(q.dtype)
            scores = jnp.matmul(q, keys.T, preferred_element_type=jnp.float32)
            scores *= scale * k_scale
            scores = jnp.where(valid[None, :], scores, -jnp.inf)
            next_m = jnp.maximum(m, scores.max(axis=-1))
            alpha = jnp.exp(m - next_m)
            p = jnp.exp(scores - next_m[:, None])
            values = rows[:, :latent_dim].astype(q.dtype)
            acc = acc * alpha[:, None] + jnp.matmul(
                p.astype(q.dtype), values, preferred_element_type=jnp.float32)
            return next_m, denominator * alpha + p.sum(axis=-1), acc

        m, denominator, acc = lax.fori_loop(0, (length + tile_size - 1) //
                                            tile_size, step, initial)
        out = acc / jnp.maximum(denominator[:, None], 1e-30) * v_scale
        lse = jnp.where(denominator > 0, m + jnp.log(denominator), -jnp.inf)
        return out.astype(q_nope.dtype), lse

    output, lse = lax.map(attend, token_ids)
    return cache, output, lse
