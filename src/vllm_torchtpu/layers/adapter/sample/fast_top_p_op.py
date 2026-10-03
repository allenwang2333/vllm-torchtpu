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
"""Torch bridge for the proven-prefilter top-p threshold (kernels/top_p.py)."""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.top_p import top_p_threshold as _top_p_threshold_jax


def _fake(probs: torch.Tensor, p: torch.Tensor, k: int, method: str, accum: str):
    return torch.empty((probs.shape[0], 1), dtype=torch.float32, device=probs.device)


# Registered at import: called from inside compiled sampler graphs, where
# Dynamo cannot trace a lock-guarded registration.
_op = pallas.jax_op("vllm_tpu::top_p_threshold", _top_p_threshold_jax)
_op.register_fake(_fake)


def top_p_threshold(
    probs: torch.Tensor, p: torch.Tensor, k: int, method: str, accum: str
) -> torch.Tensor:
    """probs [B, V] (bf16|f32), p [B, 1] f32 -> threshold [B, 1] f32."""
    if probs.dim() != 2:
        raise ValueError(f"fast top-p expects [rows, vocab], got {probs.shape}")
    return _op(probs, p.reshape(-1, 1), k, method, accum)
