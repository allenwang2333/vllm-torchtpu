"""Validation and the plain-XLA reference for the O8 selector.

No Pallas, no Mosaic, no SparseCore below this line. `_fwd` is the operation
written in ordinary JAX, which `code-style.md` §12 makes the test reference.
"""

import jax
import jax.numpy as jnp

LANES = 16


def bind(scores: jax.Array, k: int, row_lengths: jax.Array | None):
    """Re-assert the legality ladder and normalise `row_lengths`.

    Raises `NotImplementedError` (not `ValueError`) for shapes the backend does
    not support, per the checklist; `ValueError` is reserved for arguments that
    are wrong rather than merely unimplemented.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.shape}")
    if scores.dtype != jnp.float32:
        raise ValueError(f"scores must be f32, got {scores.dtype}")
    b, n = scores.shape
    if n % LANES != 0:
        raise NotImplementedError(f"{n=} must be a multiple of {LANES}")
    if k % LANES != 0 or not 0 < k <= 4096:
        raise NotImplementedError(
            f"{k=} must be a positive multiple of {LANES}, at most 4096")
    if row_lengths is None:
        row_lengths = jnp.full((b, ), n, jnp.int32)
    return row_lengths.astype(jnp.int32)


def _fwd(scores: jax.Array, k: int, row_lengths: jax.Array,
         *, write_empty_rows: bool = True) -> jax.Array:
    """Exact top-k indices per row in ordinary JAX. `-1` suffix-padded.

    `jax.lax.top_k`, not `jnp.top_k`: the latter is gone from this image's JAX.
    Ties are broken by the lowest index, which the SparseCore path does not
    promise -- so this is a set reference, not an ordering one, exactly as
    `instruments/correctness.py` treats the kernel.
    """
    b, n = scores.shape
    col = jnp.arange(n, dtype=jnp.int32)[None, :]
    live = col < row_lengths[:, None]
    masked = jnp.where(live & jnp.isfinite(scores), scores, -jnp.inf)
    vals, idx = jax.lax.top_k(masked, k)
    out = jnp.where(jnp.isfinite(vals), idx.astype(jnp.int32), -1)
    if not write_empty_rows:
        out = jnp.where((row_lengths > 0)[:, None], out, -1)
    return out
