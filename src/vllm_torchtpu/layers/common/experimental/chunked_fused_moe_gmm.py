# Copyright 2026 The vLLM-TorchTPU Authors.
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
"""Experimental chunked MoE execution for sequence token chunking."""

import functools

import jax
from jax import numpy as jnp

from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "activation",
        "use_ep",
        "use_sparse_core",
        "onehot_moe_permute_threshold",
        "rhs_quant_dtype",
        "skip_padded_tokens",
        "chunk_size",
    ),
)
def chunked_fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    experts_start: jax.Array | None = None,
    topk: int = 1,
    activation: str = "silu",
    use_ep: bool = False,
    use_sparse_core: bool = True,
    onehot_moe_permute_threshold: int = 0,
    rhs_quant_dtype: jnp.dtype | None = None,
    skip_padded_tokens: bool = False,
    chunk_size: int = 0,
) -> jax.Array:
    """Run MoE with dynamic balanced sequence chunking across tokens.

    When sequence token count N exceeds the configured chunk size C,
    split tokens into k = ceil(N / C) balanced slices, process each chunk
    using fused_moe_func, and concatenate the resulting output tensors
    along axis 0.

    Args:
        chunk_size: Sequence token chunk size for dynamic balanced chunking
            (0 disables chunking).
        See `fused_moe_func` for the description of all other arguments.

    Returns:
        Concatenated MoE output tensor of shape [num_tokens, hidden_size].
    """
    num_tokens = hidden_states.shape[0]

    if chunk_size <= 0 or num_tokens <= chunk_size:
        return fused_moe_func(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            topk_weights,
            topk_ids,
            experts_start=experts_start,
            topk=topk,
            activation=activation,
            use_ep=use_ep,
            use_sparse_core=use_sparse_core,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
        )

    num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    base_chunk_size = num_tokens // num_chunks
    remainder = num_tokens % num_chunks

    chunk_outputs = []
    start_idx = 0
    for i in range(num_chunks):
        s_i = base_chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + s_i

        chunk_hidden_states = hidden_states[start_idx:end_idx]
        chunk_topk_weights = topk_weights[start_idx:end_idx]
        chunk_topk_ids = topk_ids[start_idx:end_idx]

        out_chunk = fused_moe_func(
            chunk_hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            chunk_topk_weights,
            chunk_topk_ids,
            experts_start=experts_start,
            topk=topk,
            activation=activation,
            use_ep=use_ep,
            use_sparse_core=use_sparse_core,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
        )
        chunk_outputs.append(out_chunk)
        start_idx = end_idx

    return jnp.concatenate(chunk_outputs, axis=0)
