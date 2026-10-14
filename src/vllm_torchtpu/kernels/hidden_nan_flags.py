# SPDX-License-Identifier: Apache-2.0
"""Accumulate one NaN flag per decoder layer for a scheduled round."""

import jax
import jax.numpy as jnp


def record_hidden_nan(
    flags: jax.Array,
    armed: jax.Array,
    hidden: jax.Array,
    layer_slot: jax.Array,
) -> jax.Array:
    """OR all chunks and repeated draft forwards into the layer's dense slot."""
    assert flags.ndim == 1 and flags.size > 0, (
        f"Layer NaN flags must be a nonempty vector: {flags.shape}"
    )
    assert armed.shape == (1,), (
        f"Layer NaN arm flag must have shape (1,): {armed.shape}"
    )
    assert hidden.ndim == 2 and hidden.size > 0, (
        f"Layer NaN detection requires nonempty 2D hidden: {hidden.shape}"
    )

    def inspect(current):
        return current.at[layer_slot].set(current[layer_slot] | jnp.isnan(hidden).any())

    return jax.lax.cond(armed[0], inspect, lambda current: current, flags)
