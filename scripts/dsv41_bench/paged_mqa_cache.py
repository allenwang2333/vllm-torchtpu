# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental page format: 128 FP8 keys followed by their FP32 scales."""

import jax
import jax.numpy as jnp


@jax.jit
def pack_pages(keys, scales):
    values = jax.lax.bitcast_convert_type(keys, jnp.uint8).astype(jnp.uint32)
    values = values.reshape(-1, 32, 4, 128)
    words = values[:, :, 0] | (values[:, :, 1] << 8)
    words |= (values[:, :, 2] << 16) | (values[:, :, 3] << 24)
    scale_words = jax.lax.bitcast_convert_type(scales, jnp.int32).reshape(-1, 1, 128)
    return jnp.concatenate((words.astype(jnp.int32), scale_words), axis=1).reshape(
        -1, 128
    )
