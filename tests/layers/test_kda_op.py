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

"""Unit tests for TPU KDA Attention layer (VllmKimiK3DeltaAttention)."""

import torch
import torch.nn as nn

from vllm.v1.kv_cache_interface import MambaSpec
from vllm_torchtpu.layers.vllm.custom_ops.kda_attention_op import (
    VllmKimiK3DeltaAttention,
)


def test_vllm_kda_spec_shapes():
    """Verify VllmKimiK3DeltaAttention MambaSpec shape calculation."""
    local_projection_size = 128
    local_num_heads = 8
    head_dim = 16
    conv_width = 4

    layer = object.__new__(VllmKimiK3DeltaAttention)
    nn.Module.__init__(layer)
    layer.local_projection_size = local_projection_size
    layer.local_num_heads = local_num_heads
    layer.head_dim = head_dim
    layer.conv1d = nn.Conv1d(
        3 * local_projection_size,
        3 * local_projection_size,
        kernel_size=conv_width,
        groups=3 * local_projection_size,
    )

    spec = layer.get_kv_cache_spec(None)
    assert isinstance(spec, MambaSpec)

    conv_shape, recurrent_shape = spec.shapes
    assert conv_shape == (3 * local_projection_size, conv_width - 1)
    assert recurrent_shape == (local_num_heads, head_dim, head_dim)


def test_vllm_kda_forward_execution():
    """Verify forward recurrence execution and KV cache state mutation."""
    num_tokens = 4
    local_proj_size = 64
    local_num_heads = 4
    head_dim = 16
    conv_width = 4
    hidden_dim = local_num_heads * head_dim

    layer = object.__new__(VllmKimiK3DeltaAttention)
    nn.Module.__init__(layer)

    layer.local_projection_size = local_proj_size
    layer.local_num_heads = local_num_heads
    layer.head_dim = head_dim
    layer.in_proj_padding = 0

    proj_dim = (
        3 * local_proj_size + local_proj_size + head_dim + local_num_heads
    )
    layer.in_proj_qkvgfab = lambda x: (
        torch.randn((num_tokens, proj_dim)),
        None,
    )
    layer.f_b_proj = lambda x: (
        torch.randn((num_tokens, local_num_heads * head_dim)),
        None,
    )
    layer.o_proj = lambda x: (torch.randn((num_tokens, hidden_dim)), None)

    layer.conv1d = nn.Conv1d(
        3 * local_proj_size,
        3 * local_proj_size,
        kernel_size=conv_width,
        groups=3 * local_proj_size,
    )
    layer.dt_bias = nn.Parameter(torch.randn(local_num_heads))
    layer.A_log = nn.Parameter(torch.randn(local_num_heads))

    conv_state = torch.zeros((3 * local_proj_size, conv_width - 1))
    recurrent_state = torch.zeros((local_num_heads, head_dim, head_dim))
    layer.kv_cache = (conv_state, recurrent_state)

    hidden_states = torch.randn((num_tokens, hidden_dim))
    positions = torch.arange(num_tokens)

    out = layer.forward(hidden_states, positions)

    assert out.shape == (num_tokens, hidden_dim)
    assert not torch.allclose(conv_state, torch.zeros_like(conv_state))
    assert not torch.allclose(
        recurrent_state, torch.zeros_like(recurrent_state)
    )
