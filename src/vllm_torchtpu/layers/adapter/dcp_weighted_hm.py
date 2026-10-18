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
"""An ordering edge for the chunked DCP merge (TPU_DCP_MERGE_RS_LAG): `tie`
hands back its first operand once the second is available."""

import jax
import torch
from jax import lax
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.mla import dcp_weighted_hm as _k


def use_kernel(chunk_rows: int) -> bool:
    """TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS on the bucket: the op sees one
    merge chunk (static rows); the chunked merge splits every bucket it takes
    into TPU_DCP_MERGE_CHUNKED equal chunks."""
    bucket = chunk_rows * max(int(envs.TPU_DCP_MERGE_CHUNKED), 1)
    return bucket > envs.bucket_gate_edge(envs.TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS)


def _build(o: jax.Array, w: jax.Array, out_dtype) -> jax.Array:
    if use_kernel(o.shape[0]):
        return _k.weighted_heads_major(o, w, out_dtype=out_dtype)
    return _k.weighted_heads_major_xla(o, w, out_dtype=out_dtype)


def _fn(o: jax.Array, w: jax.Array) -> jax.Array:
    return _build(o, w, jnp.float32)


def _fake(o: torch.Tensor, w: torch.Tensor):
    t, h, d = o.shape
    return torch.empty((h, t, d), dtype=torch.float32, device=o.device)


_op = pallas.jax_op("pallas::dcp_weighted_heads_major", _fn)
_op.register_fake(_fake)


def _fn_bf16(o: jax.Array, w: jax.Array) -> jax.Array:
    return _build(o, w, jnp.bfloat16)


def _fake_bf16(o: torch.Tensor, w: torch.Tensor):
    t, h, d = o.shape
    return torch.empty((h, t, d), dtype=torch.bfloat16, device=o.device)


_op_bf16 = pallas.jax_op("pallas::dcp_weighted_heads_major_bf16", _fn_bf16)
_op_bf16.register_fake(_fake_bf16)


def weighted_heads_major(
    output: torch.Tensor, weight: torch.Tensor, bf16: bool = False
) -> torch.Tensor:
    """[T, H, D], f32[T, H] -> f32[H, T, D] = where(w > 0, o * w, 0), heads-major.
    `bf16` (TPU_DCP_MERGE_RS_BF16): the same values rounded once to bf16."""
    op = _op_bf16 if bf16 else _op
    return op(output.contiguous(), weight.to(torch.float32).contiguous())


def _tie_fn(a: jax.Array, b: jax.Array) -> jax.Array:
    out, _ = lax.optimization_barrier((a, b))
    return out


def _tie_fake(a: torch.Tensor, b: torch.Tensor):
    return torch.empty_like(a)


_tie_op = pallas.jax_op("pallas::dcp_tie", _tie_fn)
_tie_op.register_fake(_tie_fake)


def tie(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a`, unchanged, available only once `b` is: an optimization barrier."""
    return _tie_op(a.contiguous(), b.contiguous())
