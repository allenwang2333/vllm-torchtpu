import torch
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import os

import sys

# Ensure src/ is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from vllm_torchtpu.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod
from vllm_torchtpu.layers.common.quantization import dequantize_mxfp4_packed, unpack_uint8_to_fp4
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

def test_moe_weight_loading_and_quantization():
    # Setup dummy vllm config
    from vllm.config.vllm import set_current_vllm_config
    dummy_vllm_config = SimpleNamespace(
        model_config=None,
        kernel_config=SimpleNamespace(moe_backend="auto"),
        lora_config=None,
        device_config=SimpleNamespace(device=torch.device("cpu")),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
            expert_placement_strategy="linear",
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={},
            static_all_moe_layers=[],
            custom_ops=["all"],
            enabled_custom_ops=set(),
            disabled_custom_ops=set(),
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
        ),
    )
    
    # Setup dimensions
    # Hidden size = 2048, Intermediate size = 2048.
    # num_experts = 256, ep_size = 8 -> num_experts per rank = 32
    num_experts_global = 256
    ep_size = 8
    num_experts_local = num_experts_global // ep_size
    hidden_size = 2048
    intermediate_size = 2048
    top_k = 6
    
    with set_current_vllm_config(dummy_vllm_config), \
         patch("vllm_torchtpu.layers.vllm.quantization.mxfp4.get_tensor_model_parallel_world_size", return_value=1):
        # Instantiate real FusedMoE layer

        layer = FusedMoE(
            num_experts=num_experts_global,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=torch.bfloat16,
            renormalize=True,
            use_grouped_topk=True,
            num_expert_group=4,
            topk_group=2,
            tp_size=1,
            ep_size=ep_size,
            dp_size=1,
            pcp_size=1,
            activation="swigluoai",
        )
        
        layer.layer_idx = 10

        
        # 1. Create mock checkpoint inputs:
        # w13_weight_packed has shape [32, 2 * intermediate_size, hidden_size // 2] = [32, 4096, 1024]
        w13_packed = torch.randint(0, 256, (num_experts_local, 2 * intermediate_size, hidden_size // 2), dtype=torch.uint8)
        # w13_weight_scale has shape [32, 2 * intermediate_size, hidden_size // 32] = [32, 4096, 64]
        # e8m0 representation: e.g. exponent = 127 + 2 = 129 -> scale = 2^2 = 4.0
        w13_scale_u8 = torch.ones(num_experts_local, 2 * intermediate_size, hidden_size // 32, dtype=torch.uint8) * 129 # exponent = 129 -> scale = 4.0
        
        # w2_weight_packed has shape [32, hidden_size, intermediate_size // 2] = [32, 2048, 1024]
        w2_packed = torch.randint(0, 256, (num_experts_local, hidden_size, intermediate_size // 2), dtype=torch.uint8)
        # w2_weight_scale has shape [32, hidden_size, intermediate_size // 32] = [32, 2048, 64]
        w2_scale_u8 = torch.ones(num_experts_local, hidden_size, intermediate_size // 32, dtype=torch.uint8) * 130 # exponent = 130 -> scale = 8.0
        
        # Bind parameters via create_weights to allocate final parameters and temp CPU buffers
        moe_method = VllmMxfp4MoEMethod(layer.moe_config)
        moe_method.create_weights(
            layer=layer,
            num_experts=num_experts_global,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.bfloat16,
        )

        # Dequantize packed inputs to get raw floats for temp CPU weight loading
        w13_float = dequantize_mxfp4_packed(w13_packed, w13_scale_u8, axis=2, out_dtype=torch.bfloat16)
        w1_float = w13_float[:, ::2, :]
        w3_float = w13_float[:, 1::2, :]
        w2_float = dequantize_mxfp4_packed(w2_packed, w2_scale_u8, axis=2, out_dtype=torch.bfloat16)

        # Load raw floats into temp CPU buffers (mimicking FusedMoE weight loader)
        layer.w1_temp.copy_(w1_float)
        layer.w3_temp.copy_(w3_float)
        layer.w2_temp.copy_(w2_float)
        
        # Run the weight processing (performing on-the-fly quantization and packing)
        moe_method.process_weights_after_loading(layer)
        
    # --- Verify Outputs ---
    # 1. Output shapes
    # w13_weight should be padded/processed shape: [32, hidden_size // 2, 2 * intermediate_size] = [32, 1024, 4096]
    assert layer.w13_weight.shape == (num_experts_local, hidden_size // 2, 2 * intermediate_size)
    assert layer.w2_weight.shape == (num_experts_local, intermediate_size // 2, hidden_size)
    
    # 2. Output scale shapes
    # w13_weight_scale should be: [32, K // 512, 1, N] = [32, 2048 // 512, 1, 4096] = [32, 4, 1, 4096]
    assert layer.w13_weight_scale.shape == (num_experts_local, 4, 1, 4096)
    assert layer.w2_weight_scale.shape == (num_experts_local, 4, 1, 2048)


    
    # 4. Values checks:
    # Transpose axis 1 to last axis, unpack, and transpose back
    w13_temp = layer.w13_weight.cpu().transpose(1, 2).contiguous() # [32, 4096, 1024]
    w13_output_unpacked = unpack_uint8_to_fp4(w13_temp).transpose(1, 2).contiguous() # [32, 2048, 4096]
    scales_expanded = layer.w13_weight_scale.cpu().squeeze(2).repeat_interleave(512, dim=1) # [32, 2048, 4096]

    print("w13_output_unpacked shape:", w13_output_unpacked.shape)
    print("scales_expanded shape:", scales_expanded.shape)
    w13_output_dequant = (w13_output_unpacked * scales_expanded).to(torch.float32)


    
    # Compute manual reference dequantization on the checkpoint input:
    w13_ref_dequant = dequantize_mxfp4_packed(w13_packed, w13_scale_u8, axis=2, out_dtype=torch.float32) # [32, 4096, 2048]
    w1_ref = w13_ref_dequant[:, ::2, :]
    w3_ref = w13_ref_dequant[:, 1::2, :]
    w13_ref_dequant = torch.cat([w1_ref, w3_ref], dim=1) # [32, 4096, 2048]
    w13_ref_dequant = w13_ref_dequant.transpose(1, 2).contiguous()
    
    # Compare ref and output dequantized weights (MAE should be small)
    mae = (w13_output_dequant - w13_ref_dequant).abs().mean().item()
    print(f"MAE between ref and output: {mae}")
    assert mae < 0.15, f"MAE is too large: {mae}"
    
    print("MoE weight loading test passed successfully!")


if __name__ == "__main__":
    test_moe_weight_loading_and_quantization()
