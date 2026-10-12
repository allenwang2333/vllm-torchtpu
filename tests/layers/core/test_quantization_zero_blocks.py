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
"""Regression tests for zero-valued quantization blocks."""

import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu.layers.core.quantization import (
    dequantize_tensor,
    dtype_max,
    is_floating_dtype,
    pack_fp4_indices,
    quantize_kv,
    quantize_tensor,
    quantize_tensor_to_fp4,
    static_per_tensor_quantize_tensor,
)


@pytest.fixture
def device():
    return torch.device("cpu")


def test_quantize_tensor_keeps_zero_rows_finite(device):
    tensor = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 3.0, -4.0]],
        device=device,
        dtype=torch.float32,
    )

    quantized, scale = quantize_tensor(
        tensor, quant_dtype=torch.float8_e4m3fn, axis=-1, block_size=None
    )

    assert not torch.isnan(quantized.float()).any()
    assert not torch.isinf(quantized.float()).any()
    assert torch.equal(quantized[0].float(), torch.zeros_like(quantized[0].float()))
    assert scale[0].item() == 0.0


def test_quantize_tensor_uses_reciprocal_multiply_scale_rounding():
    tensor = torch.tensor([[0.0010000200709328055]], device="cpu", dtype=torch.float32)
    max_val = float(torch.finfo(torch.float8_e4m3fn).max)
    expected_scale = tensor.abs().amax(dim=-1, keepdim=True) * torch.tensor(
        1.0 / max_val, dtype=torch.float32
    )

    _, scale = quantize_tensor(
        tensor, quant_dtype=torch.float8_e4m3fn, axis=-1, block_size=None
    )

    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


def test_quantize_tensor_to_fp4_keeps_zero_blocks_finite(device):
    tensor = torch.tensor(
        [[0.0, 0.0, 1.0, -1.0], [2.0, -2.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )

    indices, scale = quantize_tensor_to_fp4(tensor, axis=-1, block_size=2)
    dequantized = indices.float()

    assert not torch.isnan(dequantized).any()
    assert not torch.isinf(dequantized).any()
    assert torch.equal(indices[0, :2], torch.zeros_like(indices[0, :2]))
    assert torch.equal(indices[1, 2:], torch.zeros_like(indices[1, 2:]))
    assert torch.allclose(
        scale,
        torch.tensor(
            [[0.0, 1.0 / 6.0], [2.0 / 6.0, 0.0]], device=device, dtype=torch.float32
        ),
    )


def test_quantization_helpers_and_validation_branches():
    assert dtype_max(jnp.int8) == 127.0
    assert dtype_max(jnp.float8_e4m3fn) == 448.0
    assert is_floating_dtype(torch.float32) is True
    assert is_floating_dtype(torch.int8) is False

    with pytest.raises(ValueError, match="non-zero"):
        static_per_tensor_quantize_tensor(torch.ones(2), torch.int8, 0.0)

    q_fp8 = static_per_tensor_quantize_tensor(
        torch.tensor([1.0, -2.0]), torch.float8_e4m3fn, 0.5
    )
    assert q_fp8.dtype == torch.float8_e4m3fn
    q_int8 = static_per_tensor_quantize_tensor(
        torch.tensor([1.0, -2.0]), torch.int8, 0.5
    )
    assert q_int8.dtype == torch.int8

    k_only, v_none = quantize_kv(torch.int8, torch.tensor([2.0, -4.0]), None, 0.5)
    assert k_only.dtype == torch.int8 and v_none is None
    k_q, v_q = quantize_kv(
        torch.int8,
        torch.tensor([2.0, -4.0]),
        torch.tensor([1.0, -1.0]),
        k_scale=0.5,
        v_scale=0.25,
    )
    assert k_q.dtype == torch.int8 and v_q is not None and v_q.dtype == torch.int8

    # dequantize_tensor when tensor_q.ndim != scale.ndim (per-tensor scale)
    dq = dequantize_tensor(
        torch.tensor([[2, -4]], dtype=torch.int8),
        torch.tensor(0.5, dtype=torch.float32),
        axis=-1,
    )
    torch.testing.assert_close(dq, torch.tensor([[1.0, -2.0]]))

    with pytest.raises(ValueError, match="divisible by block_size"):
        quantize_tensor(torch.ones(2, 5), torch.int8, axis=-1, block_size=2)

    with pytest.raises(ValueError, match="not divisible by block_size"):
        quantize_tensor_to_fp4(torch.ones(2, 5), axis=[-1], block_size=2)

    with pytest.raises(ValueError, match="Last dimension must be even"):
        pack_fp4_indices(torch.zeros(2, 3, dtype=torch.int8))
