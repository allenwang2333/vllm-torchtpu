import torch
import os
import sys
from types import SimpleNamespace

# Ensure src/ is in python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))

from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

def test_linear_weight_loading_quant_config_resolving():
    # Setup global dummy config
    dummy_quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[1, 32],
    )
    
    # 1. Test Dense/Non-Expert layer (attention/projections)
    dense_prefix = "model.layers.0.self_attn.fused_wqa_wkv"
    dense_method = VllmFp8LinearMethodTPU(
        quant_config=dummy_quant_config,
        prefix=dense_prefix,
    )
    
    # Non-expert layer MUST have block_quant disabled
    assert dense_method.block_quant is False, f"Expected block_quant to be False for prefix {dense_prefix!r}"
    assert dense_method.weight_block_size is None, f"Expected weight_block_size to be None for prefix {dense_prefix!r}"
    assert dense_method._linear_quant_config[2] is None, f"Expected requant_block_size to be None for prefix {dense_prefix!r}"
    
    # 2. Test Routed Expert layer
    expert_prefix = "model.layers.0.mlp.experts.0.w13"
    expert_method = VllmFp8LinearMethodTPU(
        quant_config=dummy_quant_config,
        prefix=expert_prefix,
    )
    
    # Expert layer MUST have block_quant enabled
    assert expert_method.block_quant is True, f"Expected block_quant to be True for prefix {expert_prefix!r}"
    assert expert_method.weight_block_size == [1, 32], f"Expected weight_block_size to be [1, 32] for prefix {expert_prefix!r}"

    print("VllmFp8LinearMethodTPU weight loading quant config resolving test passed successfully!")

if __name__ == "__main__":
    test_linear_weight_loading_quant_config_resolving()
