"""Backend-agnostic half of streamindex_topk: input prep and the XLA reference.

Nothing here knows about Pallas, Mosaic or the SparseCore.

# Dimensions.
#   T   padded token count      H   index_n_heads, 32     D   index_head_dim, 128
#   PSZ page_size               PPS pages_per_seq          k   index_topk, 2048
"""
import jax
import jax.numpy as jnp

import tiling


def get_dtype_packing(dtype):
    return 32 // jax.dtypes.itemsize_bits(dtype)


def prepare_q_inputs(q):  # [T, H_actual, D_actual] fp8 -> [T, H, D] fp8
    _, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = tiling.align_to(actual_num_q_heads, q_packing)
    head_dim = tiling.align_to(actual_head_dim, tiling.LANES)
    return jnp.pad(q, ((0, 0), (0, num_q_heads - actual_num_q_heads),
                       (0, head_dim - actual_head_dim)), constant_values=0)


def prepare_index_weights(index_weights, q_dtype):  # [T, H_actual] -> [T, H]
    _, actual_num_q_heads = index_weights.shape
    index_weights = index_weights.astype(jnp.float32)
    num_q_heads = tiling.align_to(actual_num_q_heads, get_dtype_packing(q_dtype))
    return jnp.pad(index_weights,
                   ((0, 0), (0, num_q_heads - actual_num_q_heads)),
                   constant_values=0)


def effective_row_lengths(seq_lens, cu_q_lens, distribution, num_tokens,
                          num_positions, compression_ratio):
    """Visible compressed KV positions per token row of the score matrix."""
    max_num_seqs = seq_lens.shape[0]
    num_seqs = distribution[2]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_mask = token_ids[:, None] >= cu_q_lens[None, 1:max_num_seqs + 1]
    seq_mask = jnp.where(
        jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False)
    seq_ids = jnp.sum(seq_mask, axis=1)

    q_start = cu_q_lens[seq_ids]
    q_len = cu_q_lens[seq_ids + 1] - q_start
    seq_len = seq_lens[seq_ids]
    token_pos = seq_len - q_len + (token_ids - q_start)
    kv_len = seq_len // compression_ratio
    visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
    valid = token_ids < cu_q_lens[num_seqs]
    return jnp.where(valid, jnp.minimum(visible, num_positions), 0)


def fast_path(q, seq_lens, cu_q_lens, compression_ratio, k):
    """Every visible position fits in k, so the answer is [0, 1, 2, ...]."""
    token_idx = jnp.arange(q.shape[0])
    seq_idx = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], token_idx, side="right"),
        seq_lens.shape[0] - 1)
    seq_len = seq_lens[seq_idx]
    q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
    q_abs_pos = (seq_len - q_len) + (token_idx - cu_q_lens[seq_idx])
    max_valid_idx = jnp.minimum(seq_len // compression_ratio - 1,
                                q_abs_pos // compression_ratio)
    max_valid_idx = jnp.where(token_idx < cu_q_lens[-1], max_valid_idx, -1)
    s_idx = jnp.arange(k, dtype=jnp.int32)[None, :]
    return jnp.where(s_idx <= max_valid_idx[:, None], s_idx, -1)


def _fwd(q, indexer_weights, cache_kv, seq_lens, page_indices, cu_q_lens,
         distribution, *, k, compression_ratio):
    """Plain-XLA reference: score every visible position, then `lax.top_k`.

    O(T * N_pad) in f32 and materialises the whole score matrix, so it is a
    small-shape reference only -- it is here because code-style.md 12 says the
    reference lives in `base.py`, and it is what a unit test at a toy shape
    compares against.

    NOT the gate of record. That is `instruments/correctness.py`, whose numpy
    reference was written independently and whose two tolerance-carrying
    metrics were calibrated on hardware (`j-ae538ec1-d606-48f4-8c11`). Two
    references disagreeing is informative; this one silently agreeing with a
    kernel it was written beside is not.
    """
    max_num_seqs = seq_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    _, page_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    n_pad = pages_per_seq * page_size
    T, H, D = q.shape

    # Gather the [PPS*PSZ, 256] cache rows each sequence sees, then split the
    # 256-byte row into 128 fp8 values and the one e8m0 scale byte.
    pi = page_indices.reshape(max_num_seqs, pages_per_seq)
    rows = cache_kv.reshape(-1, page_size_per_kv_packing, kv_packing, lkv_dim)
    j = jnp.arange(n_pad, dtype=jnp.int32)
    slot = j % page_size

    def one_row(tok):
        seq_id = jnp.sum(tok >= cu_q_lens[1:max_num_seqs + 1])
        page = pi[seq_id, j // page_size]
        row = rows[page, slot // kv_packing, slot % kv_packing]   # [N_pad,256]
        k8 = jax.lax.bitcast_convert_type(
            row[:, :D].astype(jnp.uint8), jnp.float8_e4m3fn).astype(jnp.float32)
        scale = jax.lax.bitcast_convert_type(
            row[:, D:D + 1].astype(jnp.uint8),
            jnp.float8_e8m0fnu)[:, 0].astype(jnp.float32)
        dots = jnp.maximum(q[tok].astype(jnp.float32) @ k8.T, 0.0)   # [H,N_pad]
        s = (indexer_weights[tok][:, None] * dots).sum(0) * scale
        # Visibility: the kernel's causal + length mask, in compressed space.
        q_start, q_end = cu_q_lens[seq_id], cu_q_lens[seq_id + 1]
        pos = seq_lens[seq_id] - (q_end - q_start) + (tok - q_start)
        live = jnp.minimum(seq_lens[seq_id] // compression_ratio,
                           pos // compression_ratio + 1)
        live = jnp.where(tok < cu_q_lens[distribution[2]], live, 0)
        return jnp.where(j < live, s, -jnp.inf)

    scores = jax.vmap(one_row)(jnp.arange(T))
    vals, idx = jax.lax.top_k(scores, k)
    return jnp.where(jnp.isfinite(vals), idx.astype(jnp.int32), -1)
