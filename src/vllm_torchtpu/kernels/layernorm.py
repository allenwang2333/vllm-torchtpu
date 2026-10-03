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
"""TPU-native implementations of vLLM IR layernorm operations."""

import torch
from torch import Tensor
from vllm.ir import ops


def _tpu_rms_ok(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    return x.dtype == torch.bfloat16 and (weight is None or weight.dtype == x.dtype)


def _tpu_add_rms_ok(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    return (
        x.dtype == torch.bfloat16
        and x_residual.dtype == x.dtype
        and (weight is None or weight.dtype == x.dtype)
    )


@ops.rms_norm.register_impl("tpu", supports_args=_tpu_rms_ok)
def rms_norm_tpu(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor:
    """TPU-optimized RMSNorm casting the 1D rsqrt scale to orig_dtype before scaling."""
    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    x_var = x_f32 if variance_size is None else x_f32[..., :variance_size]
    variance = (x_var * x_var).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + epsilon).to(orig_dtype)
    out = x * inv_rms
    if weight is not None:
        out = out * weight.to(orig_dtype)
    return out


@ops.fused_add_rms_norm.register_impl("tpu", supports_args=_tpu_add_rms_ok)
def fused_add_rms_norm_tpu(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """TPU-optimized fused residual add + RMSNorm."""
    orig_dtype = x.dtype
    x_residual = x + x_residual
    x_f32 = x_residual.to(torch.float32)
    x_var = x_f32 if variance_size is None else x_f32[..., :variance_size]
    variance = (x_var * x_var).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + epsilon).to(orig_dtype)
    out = x_residual * inv_rms
    if weight is not None:
        out = out * weight.to(orig_dtype)
    return out, x_residual
