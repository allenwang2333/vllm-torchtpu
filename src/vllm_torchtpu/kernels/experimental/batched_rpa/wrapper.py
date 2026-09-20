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
"""Input packing shared with the PCP streaming RPA kernel.

All that is left of this wrapper is `prepare_inputs`: the in-tree batched RPA
kernel it used to drive is gone, and attention now runs on tokamax's
`PallasMosaicTpuBatchedRpa` (see `layers/adapter/attention.py`). The packing
below survives because `pcp_streaming_rpa.wrapper` builds its ring buffers with
it, and `configs.KVLayout` because several modules still name a KV layout.
"""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, utils


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
) -> tuple[jax.Array, jax.Array]:

    total_q_tokens, actual_num_q_heads, actual_head_dim = q.shape
    _, actual_num_kv_heads, _ = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = utils.get_dtype_packing(q_dtype)
    kv_packing = utils.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head,
                                                     q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_q_head_dim = utils.align_to(actual_head_dim, num_lanes)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        aligned_kv_head_dim = utils.align_to(actual_head_dim,
                                             num_sublanes * kv_packing)
    else:
        aligned_kv_head_dim = utils.align_to(actual_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    o_hbm_alias_q_hbm = (jnp.pad(
        q.reshape(
            total_q_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head,
            actual_head_dim,
        ),
        (
            (0, 0),
            (0, 0),
            (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
            (0, aligned_q_head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        total_q_tokens,
        actual_num_kv_heads,
        aligned_num_q_heads_per_kv_head // q_packing,
        q_packing,
        aligned_q_head_dim,
    ).swapaxes(0, 1))

    # Pad keys and values head_dim
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2_aligned = utils.align_to(actual_num_kv_heads_x2,
                                             kv_packing)

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_lanes = pltpu.get_tpu_info().num_lanes
        padded_total_tokens = utils.align_to(total_q_tokens, num_lanes)
        new_kv_hbm = (jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(total_q_tokens,
                                                     actual_num_kv_heads_x2,
                                                     actual_head_dim),
            (
                (0, padded_total_tokens - total_q_tokens),
                (0, 0),
                (0, aligned_kv_head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            padded_total_tokens,
            actual_num_kv_heads_x2,
            aligned_kv_head_dim // kv_packing,
            kv_packing,
        ).transpose(1, 2, 3, 0))
    else:
        new_kv_hbm = jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(total_q_tokens,
                                                     actual_num_kv_heads_x2,
                                                     actual_head_dim),
            (
                (0, 0),
                (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
                (0, aligned_kv_head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            total_q_tokens,
            num_kv_heads_x2_aligned // kv_packing,
            kv_packing,
            aligned_kv_head_dim,
        )
    return o_hbm_alias_q_hbm, new_kv_hbm
