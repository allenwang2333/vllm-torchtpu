"""Plain-XLA reference top-k for O9. No Pallas, no SparseCore, no config.

`_fwd` is the shape/padding reference a CPU test can check against in a second.
It is NOT the gate: the gate is `../instruments/correctness.py`, which scores
the whole streamindex op against an f32 recomputation and owns the calibrated
deficit and churn tolerances. Nothing here carries a tolerance.

Declared deviation from `code-style.md` (checklist item "jaxtyping on the public
wrapper"): `jaxtyping` is not installed in the measurement image, so shapes are
carried as trailing comments in the glossary's letters instead. Same deviation
as `api.py`.

Dimension glossary
  B  score rows (token rows, or token rows x partition slices)
  N  score columns, PADDED extent -- the width the kernel allocates
  K  selected indices per row
"""
import jax
import jax.numpy as jnp


def _fwd(
    scores,  # [B, N] f32
    k,
    row_lengths=None,  # [B] i32
    *,
    write_empty_rows=True,
):  # -> [B, K] i32
    """Exact top-k indices per row, -1-padded past the live length."""
    b, n = scores.shape
    if row_lengths is None:
        row_lengths = jnp.full((b, ), n, jnp.int32)
    col = jnp.arange(n, dtype=jnp.int32)[None, :]
    masked = jnp.where(col < row_lengths[:, None], scores, -jnp.inf)
    # `jax.lax.top_k`, not `jnp.top_k`: the latter is removed in this image's
    # JAX and resolves through the deprecation shim to an AttributeError.
    _, idx = jax.lax.top_k(masked, k)
    taken = jnp.take_along_axis(masked, idx, axis=1)
    out = jnp.where(jnp.isfinite(taken), idx, -1)
    if not write_empty_rows:
        return out
    return jnp.where(row_lengths[:, None] > 0, out, -1)
