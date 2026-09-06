"""Argument validation and the plain-XLA reference. No Pallas below this line.

`_fwd` is the operation written in ordinary JAX, which `code-style.md` S12
makes the reference the kernel is judged against. `bind` is the boundary check
`code-style.md` S4 asks jaxtyping for; jaxtyping is not installed in the
benchmark image, so it is spelled out (see `DEVIATIONS.md`).
"""

import jax
import jax.numpy as jnp


def bind(scores: jax.Array, k: int, row_lengths: jax.Array | None):
    """Normalise `row_lengths` and re-assert the selector's argument contract.

    `ValueError` for an argument that is wrong; `NotImplementedError` is
    reserved for a shape this backend merely cannot serve, and is raised by
    `pallas_mosaic_tpu.select_supported_on`, which knows the chip.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if scores.dtype != jnp.float32:
        raise ValueError(f"scores must be f32, got {scores.dtype}")
    b, _ = scores.shape
    if k <= 0:
        raise ValueError(f"{k=} must be positive")
    if row_lengths is None:
        return None
    if row_lengths.shape != (b, ):
        raise ValueError(
            f"row_lengths must be [B] = {(b,)}, got {row_lengths.shape}")
    return row_lengths.astype(jnp.int32)


def check_shapes(q, indexer_weights, cache_kv, seq_lens, page_indices,
                 cu_q_lens, distribution, k):
    """The boundary check jaxtyping would have done for the scoring op."""
    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, DH], got {q.shape}")
    T, H, _ = q.shape
    if indexer_weights.shape != (T, H):
        raise ValueError(
            f"indexer_weights must be [T, H] = {(T, H)}, got "
            f"{indexer_weights.shape}; it is indexed by the same token rows "
            f"as q")
    if cache_kv.ndim != 4:
        raise ValueError(
            f"cache_kv must be [nb, psz // kv_packing, kv_packing, lkv], got "
            f"{cache_kv.shape}")
    S = seq_lens.shape[0]
    if cu_q_lens.shape != (S + 1, ):
        raise ValueError(
            f"cu_q_lens must be [S + 1] = {(S + 1,)} to match seq_lens' "
            f"S = {S}, got {cu_q_lens.shape}")
    if page_indices.ndim != 1 or page_indices.shape[0] % S != 0:
        raise ValueError(
            f"page_indices must be a flat [S * pages_per_seq]; "
            f"{page_indices.shape} is not a multiple of S = {S}")
    if distribution.shape != (3, ):
        raise ValueError(
            f"distribution must be i32[3] = (num_decode, "
            f"num_decode_and_prefill, num_seqs), got {distribution.shape}")
    if k <= 0:
        raise ValueError(f"{k=} must be positive")


def fwd(scores: jax.Array, k: int, row_lengths: jax.Array,
        *, write_empty_rows: bool = True) -> jax.Array:
    """Exact top-k indices per row in ordinary JAX. `-1` suffix-padded.

    `jax.lax.top_k`, not `jnp.top_k`: the latter is gone from this image's JAX.
    Ties break to the lowest index here, which the SparseCore path does not
    promise -- so this is a SET reference, not an ordering one, exactly as the
    shared `instruments/correctness.py` treats the kernel.
    """
    _, n = scores.shape
    col = jnp.arange(n, dtype=jnp.int32)[None, :]
    live = col < row_lengths[:, None]
    masked = jnp.where(live & jnp.isfinite(scores), scores, -jnp.inf)
    vals, idx = jax.lax.top_k(masked, k)
    out = jnp.where(jnp.isfinite(vals), idx.astype(jnp.int32), -1)
    if not write_empty_rows:
        out = jnp.where((row_lengths > 0)[:, None], out, -1)
    return out
