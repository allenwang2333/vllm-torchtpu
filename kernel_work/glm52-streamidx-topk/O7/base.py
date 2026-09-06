"""Reference top-k for O7. Plain XLA, no Pallas, no config.

`_fwd` is what the shape/algebra tests compare against. It is NOT the gate --
the gate is `../instruments/correctness.py`, which scores against an f32
recomputation of the whole streamindex op and carries the calibrated deficit and
churn tolerances. This file exists so a shape or a padding bug can be found on
CPU in a second rather than on a TPU pod in twenty minutes.
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
    b, n = scores.shape
    if row_lengths is None:
        row_lengths = jnp.full((b, ), n, jnp.int32)
    col = jnp.arange(n, dtype=jnp.int32)[None, :]
    masked = jnp.where(col < row_lengths[:, None], scores, -jnp.inf)
    # jax.lax.top_k, not jnp.top_k: the latter is removed in this image's JAX
    # (it resolves through jax/_src/deprecations.py and raises AttributeError).
    _, idx = jax.lax.top_k(masked, k)
    taken = jnp.take_along_axis(masked, idx, axis=1)
    out = jnp.where(jnp.isfinite(taken), idx, -1)
    if not write_empty_rows:
        return out
    return jnp.where(row_lengths[:, None] > 0, out, -1)
