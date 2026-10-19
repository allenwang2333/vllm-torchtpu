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
"""The serving pool's GDN state plan, built exactly as layers/core/gdn_attention.py's
_build_v3_pool_state_plan does (same helpers), without importing the torch-side layers."""

import math

import jax
import jax.numpy as jnp

from vllm_torchtpu import gdn_pool_layout
from vllm_torchtpu.kernels import pool_adapters


def build_gdn_plan(pool_shape, pool_dtype, *, conv_dim, n_v, d_k, d_v, kernel_size,
                   pool_block_tokens, qk_pair_layout=True, recurrent_state_dtype=jnp.bfloat16):
    """Returns (plan, stride): the v3 StateSourcePlan of one GDN layer's state in a unified pool
    of `pool_shape` (SEQ_ALONG_LANE: (pages, 2 kvh, 64, 4, 128)), and the kernel pages per manager
    block (state of block b starts at page b * stride)."""
    pool = jax.ShapeDtypeStruct(pool_shape, pool_dtype)
    block_size, payload, lanes = pool_adapters._pool_geometry(pool)
    assert pool_block_tokens % block_size == 0, (pool_block_tokens, block_size)
    split = pool_block_tokens // block_size
    if pool_adapters.is_seq_along_lane_pool():
        tok_bytes = math.prod(payload) * jnp.dtype(pool_dtype).itemsize
    else:
        tok_bytes = math.prod(payload) * lanes * jnp.dtype(pool_dtype).itemsize
    layout = gdn_pool_layout.derive_pooled_gdn_state_layout(
        ssm_bytes=gdn_pool_layout.pooled_gdn_ssm_state_bytes(
            num_v_heads=n_v, head_k_dim=d_k, head_v_dim=d_v,
            dtype=jnp.dtype(recurrent_state_dtype)),
        conv_bytes=gdn_pool_layout.pooled_gdn_conv_and_scale_state_bytes(
            kernel_size=kernel_size, conv_dim=conv_dim, ssm_dtype=jnp.dtype(recurrent_state_dtype)),
        token_bytes=tok_bytes)
    assert layout.required_tokens <= pool_block_tokens, (layout.required_tokens, pool_block_tokens)
    plan = pool_adapters.v3_state_source(
        pool, split=split, ssm_ntok=layout.ssm_tokens, conv_tok0=layout.ssm_tokens,
        conv_ntok=layout.conv_tokens, conv_dim=conv_dim, n_v=n_v, d_k=d_k, d_v=d_v,
        kernel_size=kernel_size, qk_pair_layout=qk_pair_layout,
        recurrent_state_dtype=recurrent_state_dtype)
    return plan, plan.stride
