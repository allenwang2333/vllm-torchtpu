# SPDX-License-Identifier: Apache-2.0
"""Capture the first NaN layer, sampling three of every hundred bad rounds."""

import jax
import jax.numpy as jnp


def record_hidden_nan(
    state: jax.Array,
    hidden_snapshot: jax.Array,
    token_snapshot: jax.Array,
    armed: jax.Array,
    hidden: jax.Array,
    input_ids: jax.Array,
    input_rows: jax.Array,
    layer_slot: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """State is [first slot, event phase modulo 100, hidden rows, input rows]."""
    assert state.shape == (4,) and armed.shape == (1,), (state.shape, armed.shape)
    assert hidden.ndim == 2 and hidden.size > 0, (
        f"Layer NaN detection requires nonempty 2D hidden: {hidden.shape}"
    )
    assert (
        hidden.shape[0] <= hidden_snapshot.shape[0]
        and hidden.shape[1] == hidden_snapshot.shape[1]
        and hidden.dtype == hidden_snapshot.dtype
    ), (
        "Hidden snapshot capacity/dtype mismatch: "
        f"source={hidden.shape}/{hidden.dtype}, "
        f"snapshot={hidden_snapshot.shape}/{hidden_snapshot.dtype}"
    )
    assert input_ids.ndim == 1 and input_ids.size <= token_snapshot.size, (
        f"Input snapshot capacity mismatch: {input_ids.shape}, {token_snapshot.shape}"
    )

    def inspect(buffers):
        def capture(current):
            metadata, saved_hidden, saved_tokens = current

            def copy_snapshot(snapshots):
                return (
                    jax.lax.dynamic_update_slice(snapshots[0], hidden, (0, 0)),
                    jax.lax.dynamic_update_slice(
                        snapshots[1], input_ids.astype(snapshots[1].dtype), (0,)
                    ),
                )

            saved_hidden, saved_tokens = jax.lax.cond(
                metadata[1] < 3,
                copy_snapshot,
                lambda snapshots: snapshots,
                (saved_hidden, saved_tokens),
            )
            metadata = jnp.stack(
                (layer_slot, (metadata[1] + 1) % 100, hidden.shape[0], input_rows[0])
            ).astype(metadata.dtype)
            return metadata, saved_hidden, saved_tokens

        return jax.lax.cond(jnp.isnan(hidden).any(), capture, lambda x: x, buffers)

    return jax.lax.cond(
        armed[0] & (state[0] < 0),
        inspect,
        lambda buffers: buffers,
        (state, hidden_snapshot, token_snapshot),
    )
