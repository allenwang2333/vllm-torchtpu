# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-vector FP8 quantization shared by index cache writers and queries."""

import jax
import jax.numpy as jnp


def quantize_index_rows(values: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return E4M3 values, FP32 scales and E8M0 scale bytes for D128 vectors."""
    # Same power-of-two quantizer as the existing V4.1 adapter; scales belong
    # to the entire post-RoPE vector, independently of its physical layout.
    if values.shape[-1] != 128:
        raise ValueError("V4.1 FP8 index vectors must have 128 channels")
    values = values.astype(jnp.float32)
    amax = jnp.maximum(jnp.abs(values).max(axis=-1), 1e-4)
    exponent = jnp.ceil(jnp.log2(amax / 448.0)).astype(jnp.int32)
    scale_bytes = (exponent + 127).astype(jnp.uint8)
    scales = jax.lax.bitcast_convert_type(
        scale_bytes.astype(jnp.uint32) << 23, jnp.float32
    )
    quantized = (values / scales[..., None]).astype(jnp.float8_e4m3fn)
    return quantized, scales, scale_bytes
