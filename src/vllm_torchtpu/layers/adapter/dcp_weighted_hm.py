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
"""Torch bridge for the DCP merge's heads-major weighted operand
(`kernels/mla/dcp_weighted_hm.py`, TPU_DCP_MERGE_WEIGHTED_HM)."""

import jax
import torch
from jax import lax
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.mla import dcp_weighted_hm as _k


def _fn(o: jax.Array, w: jax.Array) -> jax.Array:
    return _k.weighted_heads_major(o, w)


def _fake(o: torch.Tensor, w: torch.Tensor):
    t, h, d = o.shape
    return torch.empty((h, t, d), dtype=torch.float32, device=o.device)


_op = pallas.jax_op("pallas::dcp_weighted_heads_major", _fn)
_op.register_fake(_fake)


def weighted_heads_major(output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[T, H, D], f32[T, H] -> f32[H, T, D] = where(w > 0, o * w, 0), heads-major."""
    return _op(output.contiguous(), weight.to(torch.float32).contiguous())


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
