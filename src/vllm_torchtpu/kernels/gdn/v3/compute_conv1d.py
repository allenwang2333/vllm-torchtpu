# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3 import config


def extract_final_conv_state(
    real_sizes: jax.Array,  # [seq]
    lhs: jax.Array,  # [seq, prev_kernel_size + chunk, q, dim_size]
    cfg: config.GDNConfig,
) -> jax.Array:
    """Extract masked conv state for the final tile of a sequence."""
    real_sizes = real_sizes.reshape(-1, 1, 1, 1)
    state_list = []
    for w_idx in range(cfg.window_size):
        last_row = 1 + cfg.chunk_size - cfg.window_size + w_idx
        new_conv_state = lhs[:, last_row:last_row + cfg.prev_kernel_size]
        for c_idx in range(1, last_row):
            new_conv_state = jnp.where(
                c_idx == real_sizes,
                lhs[:, c_idx:c_idx + cfg.prev_kernel_size],
                new_conv_state,
            )
        state_list.append(new_conv_state)
    return jnp.stack(state_list, axis=1)


def causal_conv1d(
    real_sizes: jax.Array,  # [seq]
    lhs: jax.Array,  # [seq, chunk, q, dim_size]
    conv_weight: jax.Array,  # [prev_kernel_size, 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    cfg: config.GDNConfig,
    skip_partial_mask: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Perform causal Conv1D. Returns Conv1D output and convolution states.

    States are [seq, window_size, prev_kernel_size, q, dim_size]: checkpoint
    w holds the last prev_kernel_size inputs ending at token
    `chunk_size - window_size + w`, clamped to the last real token, i.e. the
    state a sequence resuming right after that token must start from.
    """

    assert lhs.ndim == 4

    out = jnp.zeros((cfg.seq_tile_size, cfg.chunk_size, 1, cfg.dim_size), jnp.float32)
    for k in range(cfg.kernel_size):
        w_k = conv_weight[k:k + 1].reshape(1, 1, 1, cfg.dim_size)
        out = out + lhs[:, k:k + cfg.chunk_size] * w_k

    if conv_bias is not None:
        out = out + conv_bias.reshape(1, 1, 1, cfg.dim_size)

    if skip_partial_mask:
        state_list = [
            lhs[:, (1 + cfg.chunk_size - cfg.window_size + w_idx):(1 + cfg.chunk_size - cfg.window_size + w_idx + cfg.prev_kernel_size)]
            for w_idx in range(cfg.window_size)
        ]
        return out, jnp.stack(state_list, axis=1)

    return out, extract_final_conv_state(real_sizes, lhs, cfg)

