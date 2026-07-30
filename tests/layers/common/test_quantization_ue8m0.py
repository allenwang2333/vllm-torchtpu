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
"""Tests for the UE8M0 (power-of-two) block scales in `quantize_tensor`."""

import pytest
import torch

from vllm_torchtpu.layers.common.quantization import (e8m0_to_fp32,
                                                      quantize_tensor)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def _quantize_ue8m0(tensor: torch.Tensor, block_size: int | None = None):
    return quantize_tensor(tensor,
                           FP8_DTYPE,
                           axis=-1,
                           block_size=block_size,
                           use_ue8m0=True)


def test_scale_is_a_ue8m0_byte(device):
    tensor = torch.randn(4, 128, device=device, dtype=torch.bfloat16)

    _, scale = _quantize_ue8m0(tensor, block_size=128)

    assert scale.dtype == torch.uint8
    assert scale.shape == (4, 1)
    # Decoded scales are exact powers of two, and the byte round-trips through
    # torch's own e8m0 encoder.
    decoded = e8m0_to_fp32(scale)
    assert torch.equal(torch.exp2(torch.log2(decoded).round()), decoded)
    assert torch.equal(
        decoded.to(torch.float8_e8m0fnu).view(torch.uint8), scale)


@pytest.mark.parametrize("dtype",
                         [torch.bfloat16, torch.float16, torch.float32])
def test_no_value_clips_against_dtype_max(device, dtype):
    # Values chosen so `abs_max / dtype_max` lands just above a power of two:
    # rounding the exponent down (or computing log2 in the input dtype) would
    # scale the block's largest magnitude past dtype_max and clip it.
    tensor = torch.tensor(
        [[7.0625, -1.0], [7.03125, 0.5], [4.0, 4.0], [1e-4, -1e-4]],
        device=device,
        dtype=dtype,
    )

    quantized, scale = _quantize_ue8m0(tensor, block_size=2)
    decoded = e8m0_to_fp32(scale)
    abs_max = tensor.float().abs().amax(dim=-1, keepdim=True)

    # Nothing saturated: the largest magnitude in each block stays inside the
    # fp8 range once divided by its scale.
    assert (abs_max / decoded <= FP8_MAX).all(), abs_max / decoded
    assert (quantized.float().abs() <= FP8_MAX).all()

    # And the ceil-rounded scale stays within one octave of the tight one.
    tight = abs_max / FP8_MAX
    assert (decoded >= tight).all()
    assert (decoded < 2 * tight).all()


def test_dequantized_values_are_close(device):
    torch.manual_seed(0)
    tensor = torch.randn(16, 512, device=device, dtype=torch.bfloat16) * 3.0

    quantized, scale = _quantize_ue8m0(tensor, block_size=128)
    dequantized = quantized.float().reshape(16, 4, 128) * e8m0_to_fp32(
        scale).unsqueeze(-1)
    dequantized = dequantized.reshape(16, 512)

    # A ue8m0 scale wastes up to one octave of fp8 range, so the relative
    # error budget is ~2x the 4-bit-mantissa fp8 step.
    reference = tensor.float()
    error = (dequantized - reference).abs()
    tolerance = reference.abs().amax() * 2 / 2**4
    assert (error <= tolerance).all(), error.max()


def test_zero_blocks_stay_exactly_zero(device):
    tensor = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 3.0, -4.0]],
        device=device,
        dtype=torch.float32,
    )

    quantized, scale = _quantize_ue8m0(tensor, block_size=4)

    assert torch.isfinite(quantized.float()).all()
    assert torch.equal(quantized[0].float(),
                       torch.zeros_like(quantized[0], dtype=torch.float32))
    # An all-zero block hits log2(0) = -inf, which clamps to the smallest
    # e8m0 exponent rather than wrapping around to a huge scale.
    assert scale[0].item() == 0
    # The non-zero block uses its full fp8 range, up to the one octave a
    # power-of-two scale can give away.
    peak = quantized[1].float().abs().max().item()
    assert FP8_MAX / 2 < peak <= FP8_MAX


def test_subnormal_and_huge_blocks_stay_representable(device):
    tensor = torch.tensor([[1e-38], [3e38]],
                          device=device,
                          dtype=torch.float32)

    _, scale = _quantize_ue8m0(tensor)

    # Both ends clamp into the encodable byte range instead of wrapping.
    assert scale.min().item() >= 0
    assert scale.max().item() <= 254


def test_fp32_scale_path_is_unchanged(device):
    tensor = torch.randn(4, 128, device=device, dtype=torch.bfloat16)

    _, scale = quantize_tensor(tensor, FP8_DTYPE, axis=-1, block_size=128)

    assert scale.dtype == torch.float32
    expected = (tensor.abs().amax(dim=-1, keepdim=True) *
                torch.tensor(1.0 / FP8_MAX, dtype=torch.float32,
                             device=device)).to(torch.float32)
    torch.testing.assert_close(scale, expected, rtol=0, atol=0)


def test_matches_jax_reference_formula(device):
    """Parity with `quantize_fp8_ue8m0` in kernels/deepseek_v4."""
    torch.manual_seed(0)
    tensor = torch.randn(8, 256, device=device, dtype=torch.float32)

    quantized, scale = _quantize_ue8m0(tensor, block_size=128)

    blocked = tensor.reshape(8, 2, 128)
    abs_max = blocked.abs().amax(dim=-1, keepdim=True)
    expected_scale = torch.exp2(torch.ceil(torch.log2(abs_max / FP8_MAX)))
    expected_q = (blocked / expected_scale).to(FP8_DTYPE)

    torch.testing.assert_close(e8m0_to_fp32(scale).unsqueeze(-1),
                               expected_scale,
                               rtol=0,
                               atol=0)
    assert torch.equal(quantized.float(), expected_q.float().reshape(8, 256))


def test_indexer_packed_row_layout(device):
    """The indexer packs `head_dim` fp8 bytes + one scale byte per token."""
    head_dim = 128
    k = torch.randn(6, head_dim, device=device, dtype=torch.bfloat16)

    k_quant, k_scale = quantize_tensor(k,
                                       FP8_DTYPE,
                                       axis=-1,
                                       block_size=head_dim,
                                       use_ue8m0=True)
    packed = torch.cat([k_quant.view(torch.uint8), k_scale], dim=-1)

    assert packed.shape == (6, head_dim + 1)
    assert packed.dtype == torch.uint8
    assert torch.equal(packed[:, :head_dim], k_quant.view(torch.uint8))
    assert torch.equal(packed[:, head_dim:], k_scale)
