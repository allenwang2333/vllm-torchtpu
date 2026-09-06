"""The plain-XLA reference for streamindex top-k: no Pallas, no SparseCore.
"""

# Dimensions.
#   T          padded token count (rows of q, and of the output)
#   R          sequences in the batch; the first D of them are decoding
#   H, DH      index heads (32) and head dim (128)
#   L          one sequence's live KV length
#   N          padded score extent per row; N = ceil(pps/bkv_p)*bkv_p*psz
#   K          selected count per row
#   psz        page size, in KV slots
#
# What this file is for. code-style.md §12 says the reference a kernel is tested
# against is the plain-XLA `_fwd` in `base.py` -- the same operation in ordinary
# JAX, in the same repo, rather than a golden file that records yesterday's bug
# as today's expectation. This is that function.
#
# It is NOT the gate. The gate is the shared instrument at
# `../instruments/correctness.py`, which is read-only, runs against a NumPy
# reference of its own, and is calibrated by a negative control
# (j-ae538ec1-d606-48f4-8c11). Two references rather than one is deliberate:
# the instrument's is per-row NumPy and exists to be obviously right; this one
# is batched XLA and exists to be runnable at the same call shape as the
# candidate, so `_fwd` and the kernel can be handed identical arguments.
#
# Cost. This materialises a dense f32 [T, N] score matrix and calls
# `lax.top_k` on it. At (B) geometry that is a 4 GiB intermediate and a 1M-wide
# sort per row, so `_fwd` is for the correctness shapes only. Nothing here is
# on a benchmarked path.

import jax
import jax.numpy as jnp

from tiling import cdiv


def _dequantize_page_slots(cache_kv, page_ids, slot_ids, head_dim):
    """Gather [., DH] f32 keys out of the paged uint8 cache.

    Layout, and the reason this is not a reshape: slot `s` of a page lives at
    `cache_kv[page, s // 4, s % 4]`, and the trailing 256 bytes hold `DH` fp8
    mantissa bytes, then one UE8M0 exponent byte, then padding. The scale is
    applied *after* the dot product and the ReLU, not folded into the key --
    see `score_rows`.
    """
    raw = cache_kv[page_ids, slot_ids // 4, slot_ids % 4]  # [., 256] u8
    keys = jax.lax.bitcast_convert_type(
        raw[..., :head_dim], jnp.float8_e4m3fn).astype(jnp.float32)
    scale = jax.lax.bitcast_convert_type(
        raw[..., head_dim:head_dim + 1], jnp.float8_e8m0fnu).astype(jnp.float32)
    return keys, scale[..., 0]


def score_rows(q_tokens, weights_tokens, keys, scale):
    """One sequence's scores for a set of query rows.

    q_tokens      # [t, H, DH] f32   dequantized queries, t rows of this seq
    weights_tokens# [t, H] f32
    keys          # [L, DH] f32      UNSCALED fp8 key values
    scale         # [L] f32          the per-key UE8M0 exponent
    -> [t, L] f32

    The operation order is the kernel's, and the order is the whole point of
    the comment: dot on the unscaled fp8 values, ReLU, weight, sum over heads,
    and only then multiply by the key's scale. Folding the scale into the key
    first is mathematically identical -- it is a positive power of two -- but
    leaves room to wonder whether a mismatch is the kernel's or the reference's,
    and a reference exists to remove that room.
    """
    dots = jax.nn.relu(jnp.einsum("thd,ld->thl", q_tokens, keys))
    return jnp.einsum("th,thl->tl", weights_tokens, dots) * scale[None, :]


def _fwd(
    q,              # [T, H, DH] f8_e4m3   queries, one row per token
    weights,        # [T, H] f32           softmax and head scales pre-folded
    cache_kv,       # [nb, psz//4, 4, 256] u8   paged fp8 KV with UE8M0 scales
    seq_lens,       # [S] i32              live KV length per sequence
    page_indices,   # [S * pages_per_seq] i32
    cu_q_lens,      # [S + 1] i32          exclusive prefix sum of query lengths
    distribution,   # [3] i32              (D, D, R)
    *,
    k: int,
    compression_ratio: int = 1,
    page_size: int,
    pages_per_seq: int,
    padded_n: int,
):
    """Exact top-k KV positions per token row, in plain XLA. [T, k] i32.

    Rows past the batch's live token count, and rows of a sequence with fewer
    than `k` visible positions, are -1-suffixed -- the same contract the kernel
    holds, because a caller downstream masks on the -1.

    `padded_n` is passed rather than derived. It is the extent the kernel's
    score buffer actually has, which is `align_to(pages_per_seq, bkv_p)*psz` and
    at (A) geometry is 10240 against a 9216 `max_model_len`. Deriving it here
    from `pages_per_seq * page_size` would make the reference score a narrower
    row than the kernel does, and the two would disagree only in the padding --
    which is exactly the disagreement nobody would think to look for.
    """
    # Not jitted, and it loops over sequences in Python. Both are deliberate:
    # a reference is allowed to be slow and is not allowed to be clever, and
    # the ragged per-sequence gather is the part of this operation a reader
    # most needs to see written out. Calling it under `jax.jit` fails here,
    # loudly, rather than silently tracing the wrong number of sequences.
    T = q.shape[0]
    num_seqs = int(distribution[2])

    q_f32 = q.astype(jnp.float32)
    scores = jnp.full((T, padded_n), -jnp.inf, jnp.float32)

    for seq_id in range(num_seqs):
        L = int(seq_lens[seq_id])
        if L == 0:
            continue
        lo, hi = int(cu_q_lens[seq_id]), int(cu_q_lens[seq_id + 1])
        if hi <= lo:
            continue
        j = jnp.arange(L, dtype=jnp.int32)
        pages = page_indices[seq_id * pages_per_seq + j // page_size]
        keys, scale = _dequantize_page_slots(cache_kv, pages, j % page_size,
                                             q.shape[-1])
        row = score_rows(q_f32[lo:hi], weights[lo:hi], keys, scale)

        # Causal visibility, in the incumbent's own form
        # (`_effective_row_lengths`): a token at position `pos` sees
        # `min(L // cr, pos // cr + 1)` compressed positions. The last query of
        # a sequence sits at L-1, the one before it at L-2, so
        # `pos = L - qlen + (tok - lo)`. Written as a COUNT and compared with
        # `<`, not as an index compared with `<=`, because that is what the
        # kernel computes and an off-by-one here is invisible in the total.
        qlen = hi - lo
        pos = (L - qlen + jnp.arange(qlen, dtype=jnp.int32))[:, None]
        n_visible = jnp.minimum(L // compression_ratio,
                                pos // compression_ratio + 1)
        row = jnp.where(j[None, :] < n_visible, row, -jnp.inf)
        scores = scores.at[lo:hi, :L].set(row)

    # `lax.top_k` is a full sort of a `padded_n`-wide row. That is the reference
    # being deliberately dumb: it is the definition of the answer, and it is why
    # this function is not allowed near a benchmark.
    _, idx = jax.lax.top_k(scores, k)
    live = jnp.isfinite(jnp.take_along_axis(scores, idx, axis=1))
    return jnp.where(live, idx.astype(jnp.int32), jnp.int32(-1))


def n_pages(seq_len: int, page_size: int) -> int:
    """Pages one sequence occupies. Here so callers do not re-derive it."""
    return cdiv(seq_len, page_size)
