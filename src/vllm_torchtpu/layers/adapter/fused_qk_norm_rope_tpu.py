# Copyright 2025 Google LLC
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
"""vLLM's fused split + QK-RMSNorm + RoPE + gate for the Qwen3-Next family,
on TPU: one Pallas call per attention layer (kernels/qk_norm_rope_gate) in
place of the eager path's forty small operations, behind
TPU_FUSED_QK_NORM_ROPE=1. Same signature and contract as
vllm.model_executor.layers.fused_qk_norm_rope.fused_qk_rmsnorm_rope_gate.
"""

import functools

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.qk_norm_rope_gate import qk_norm_rope_gate_rows

_ops: dict = {}


def _jax_fn(
    q_gate: jax.Array,
    k: jax.Array,
    q_w: jax.Array,
    k_w: jax.Array,
    cos_sin: jax.Array,
    *,
    heads: int,
    kv_heads: int,
    dim: int,
    rot: int,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # cos_sin: the rope rows for these tokens, gathered by position at the torch level
    return qk_norm_rope_gate_rows(
        q_gate,
        k,
        q_w,
        k_w,
        cos_sin,
        heads=heads,
        kv_heads=kv_heads,
        dim=dim,
        rot=rot,
        eps=eps,
    )


def _get_op(heads, kv_heads, dim, rot, eps):
    """The op for one geometry. Built eagerly by prebuild() at model
    construction; the traced forward only looks it up (Dynamo cannot enter a
    lock or register an op while tracing)."""
    key = (heads, kv_heads, dim, rot, float(eps))
    op = _ops.get(key)
    if op is not None:
        return op
    fn = functools.partial(
        _jax_fn, heads=heads, kv_heads=kv_heads, dim=dim, rot=rot, eps=float(eps)
    )
    op = pallas.jax_op(
        f"pallas::qk_norm_rope_gate_h{heads}_kv{kv_heads}_d{dim}_r{rot}", fn
    )

    def _fake(q_gate, k, q_w, k_w, cos_sin):
        n = q_gate.shape[0]
        return (
            torch.empty(n, heads * dim, dtype=q_gate.dtype, device=q_gate.device),
            torch.empty(n, kv_heads * dim, dtype=k.dtype, device=k.device),
            torch.empty(n, heads * dim, dtype=q_gate.dtype, device=q_gate.device),
        )

    op.register_fake(_fake)
    _ops[key] = op
    return op


def prebuild(heads, kv_heads, dim, rot, eps):
    """Register the op for this geometry outside any traced region."""
    _get_op(heads, kv_heads, dim, rot, eps)


def fused_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
):
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be a positive even integer <= head_dim, got {rotary_dim} / {head_dim}"
        )
    op = _get_op(num_q_heads, num_kv_heads, head_dim, rotary_dim, eps)
    pos = positions[0] if positions.dim() == 2 else positions
    cos_sin = cos_sin_cache.index_select(0, pos.to(torch.int64))[:, :rotary_dim]
    q, kk, gate = op(q_gate, k, q_weight, k_weight, cos_sin)
    return q, kk, gate
