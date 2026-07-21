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
"""Tests for MXFP4 quantization weight processing and packing correctness on CPU."""

import pytest
import torch

from vllm_torchtpu.layers.common.quantization import (
    quantize_tensor_to_fp4,
    pack_fp4_indices,
    unpack_uint8_to_fp4
)

def test_mxfp4_quantize_dequantize_correctness():
    # Shape matching w2 down projection: [num_experts, intermediate_size, hidden_size]
    # intermediate_size = 2048, hidden_size = 4096
    num_experts = 4
    in_features = 1024
    out_features = 2048
    block_size = 512
    
    # Generate random floats
    generator = torch.Generator().manual_seed(42)
    original_weights = torch.randn(num_experts, in_features, out_features, generator=generator)
    
    # 1. Quantize along axis 1 (contracting dimension)
    indices, scale = quantize_tensor_to_fp4(
        original_weights, axis=1, block_size=block_size
    )
    
    # Verify scale dimensions
    num_blocks = in_features // block_size
    assert scale.shape == (num_experts, num_blocks, out_features)
    assert indices.shape == original_weights.shape
    
    # 2. Pack indices into uint8 nibbles
    packed_indices = pack_fp4_indices(indices)
    assert packed_indices.shape == (num_experts, in_features, out_features // 2)
    assert packed_indices.dtype == torch.uint8
    
    # 3. Unpack packed indices back to FP4 floats
    unpacked_weights = unpack_uint8_to_fp4(packed_indices)
    assert unpacked_weights.shape == original_weights.shape
    
    # 4. Dequantize manually by broadcasting scales
    w_blocked = unpacked_weights.reshape(num_experts, num_blocks, block_size, out_features)
    scale_expanded = scale.unsqueeze(2) # [num_experts, num_blocks, 1, out_features]
    dequantized_blocked = w_blocked * scale_expanded
    dequantized_weights = dequantized_blocked.reshape(original_weights.shape)
    
    # 5. Verify mathematical grid correctness
    # All dequantized values mapped divided by their corresponding block scale
    # must lie exactly on the FP4 representable grid: [0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6]
    FP4_GRID = set([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    
    for e in range(num_experts):
        for b in range(num_blocks):
            for k_block in range(block_size):
                k = b * block_size + k_block
                for n in range(out_features):
                    val = dequantized_weights[e, k, n].item()
                    s_val = scale[e, b, n].item()
                    
                    if val == 0.0:
                        assert s_val == 0.0 or val / s_val == 0.0
                        continue
                        
                    assert s_val > 0.0
                    rel_val = abs(val / s_val)
                    
                    # Find closest grid point
                    closest = min(FP4_GRID, key=lambda g: abs(g - rel_val))
                    # Math check: must align precisely onto the closest representable value
                    assert abs(rel_val - closest) < 1e-5, (
                        f"Value {val} with scale {s_val} yields relative value {rel_val}, "
                        f"which does not lie on the FP4 representable grid (closest: {closest})"
                    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
