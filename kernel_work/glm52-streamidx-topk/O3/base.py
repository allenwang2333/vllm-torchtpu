"""Plain-XLA reference for the selector O3 replaces: exact top-k per masked row."""

# Dimensions.
#   B   rows of the score matrix
#   N   padded score extent per row
#   K   top-k
#
# code-style.md 4 asks for jaxtyping on the public wrapper. jaxtyping is NOT
# installed in the target image (checked: site-packages has pydantic, not
# jaxtyping), so shapes are carried in trailing comments and asserted at the
# boundary instead. Declared here rather than silently dropped.

import jax
import jax.numpy as jnp


def _fwd(
    scores: jax.Array,  # [B, N] f32
    k: int,
    row_lengths: jax.Array,  # [B] i32
) -> jax.Array:  # [B, K] i32
    """Exact top-k column indices per row, -1 suffix-padded, unsorted.

    The contract the SparseCore kernel implements, written in ordinary JAX so a
    test can hold the kernel to something in this repo rather than to a golden
    file (code-style.md 12). Positions at or beyond `row_lengths`, and -inf
    scores, are never selected; a row with fewer than k live positions gets a
    -1 suffix.
    """
    B, N = scores.shape
    col = jnp.arange(N, dtype=jnp.int32)[None, :]
    live = col < row_lengths.astype(jnp.int32)[:, None]
    masked = jnp.where(live & jnp.isfinite(scores), scores, -jnp.inf)
    vals, idx = jax.lax.top_k(masked, k)
    return jnp.where(jnp.isfinite(vals), idx.astype(jnp.int32), -1)
