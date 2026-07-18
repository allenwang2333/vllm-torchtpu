import os
import sys
import importlib.util

# Dynamic hot-patch to import our local patched pallas module before any vllm imports occur.
try:
    patch_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "vllm_torchtpu", "patches", "pallas_patched.py")
    if os.path.exists(patch_path):
        import torch_tpu._internal.pallas as original_pallas
        import torch_tpu._internal.pallas.pallas as original_pallas_submodule
        
        spec = importlib.util.spec_from_file_location("patched_pallas_temp", patch_path)
        patched_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patched_module)
        
        for target in [original_pallas, original_pallas_submodule]:
            target.jax_op = patched_module.jax_op
            target.custom_jax_kernel = patched_module.custom_jax_kernel
            target.JaxCallable = patched_module.JaxCallable
            target.torch_placeholder = patched_module.torch_placeholder
            target._convert_shape = patched_module._convert_shape
            
        print(f"=== [HOT PATCH] Successfully hot-patched pallas package and submodule from {patch_path} ===")
except Exception as e:
    print(f"=== [HOT PATCH ERROR] Failed to hot-patch pallas.py: {e} ===")

import torch
import torch_tpu
from torch_tpu._internal.compile._backend import TpuBackend
import jax
import numpy as np
from unittest.mock import MagicMock

# Mock vLLM distributed state BEFORE importing layers that use it
import vllm.distributed.parallel_state as ps
mock_tp = MagicMock()
mock_tp.world_size = 1
mock_tp.rank = 0
mock_tp.rank_in_group = 0
# Mock all_gather to be identity for TP=1
mock_tp.all_gather = lambda tensor, dim=0: tensor
ps._TP = mock_tp

# Mock get_world_group if needed
mock_world = MagicMock()
mock_world.world_size = 1
mock_world.rank = 0
ps._WORLD = mock_world

from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4 import VllmDeepseekV4MLAAttention
from vllm.config import VllmConfig, ModelConfig, LoadConfig, ParallelConfig, CacheConfig, SchedulerConfig, LoRAConfig, set_current_vllm_config
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context
from vllm.forward_context import set_forward_context
from vllm_torchtpu.layers.vllm.quantization import get_tpu_quantization_config
from vllm_torchtpu.models.vllm.deepseek_v4_patch import _maybe_patch_for_deepseek_v4

def test_attention_compilation():
    print("Setting up dummy vLLM config...")
    # Minimal config to initialize the layer with TP=1
    model_config = ModelConfig(
        model="/root/cloud-devkit/torchtpu-vllm/examples/dsv4-flash-4layer-fp8", # Use the local toy model path
        tokenizer="/root/cloud-devkit/torchtpu-vllm/examples/dsv4-flash-4layer-fp8",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
    )
    
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    
    # Mock VllmConfig
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.9, cache_dtype="fp8"),
        parallel_config=parallel_config,
        scheduler_config=SchedulerConfig(max_num_seqs=256, max_model_len=4096, is_encoder_decoder=False),
        load_config=LoadConfig(),
        lora_config=None,
    )
    
    # Get TPU specific quantization config and set it
    print("Setting up TPU quantization config...")
    tpu_quant_config = get_tpu_quantization_config(vllm_config)
    vllm_config.quant_config = tpu_quant_config
    
    # Initialize sharding config
    from vllm_torchtpu.layers.common.sharding import ShardingConfigManager, build_mesh
    sharding_manager = ShardingConfigManager.initialize(vllm_config)
    
    # Setup JAX mesh (will be size 1)
    devices = jax.devices()
    print(f"Available JAX devices: {devices}")
    mesh = build_mesh(devices[:1], sharding_manager.sharding_strategy.__dict__)
    print(f"Built JAX mesh: {mesh}")
    
    # Mock the attention metadata (explicitly on TPU)
    from unittest.mock import MagicMock
    mock_metadata = MagicMock()
    mock_metadata.seq_lens = torch.tensor([32], dtype=torch.int32, device="tpu")
    mock_metadata.block_tables = torch.zeros((1, 10), dtype=torch.int32, device="tpu")
    mock_metadata.query_start_loc = torch.tensor([0, 32], dtype=torch.int32, device="tpu")
    mock_metadata.request_distribution = torch.tensor([0, 0, 1], dtype=torch.int32, device="tpu")
    
    metadata_dict = {
        "model.layers.0.attn.swa_cache": mock_metadata,
        "model.layers.0.attn": mock_metadata,
    }
    
    # Use the context managers to set the mesh, config, patches, forward context, and default device
    with set_vllm_model_wrapper_context(mesh=mesh), \
         set_current_vllm_config(vllm_config), \
         _maybe_patch_for_deepseek_v4(vllm_config), \
         set_forward_context(attn_metadata=metadata_dict, vllm_config=vllm_config), \
         torch.device("tpu"):
         
        # Initialize the layer
        print("Initializing VllmDeepseekV4MLAAttention...")
        layer = VllmDeepseekV4MLAAttention(vllm_config, prefix="model.layers.0.attn")
        
        # Initialize parameters and buffers to non-NaN constants to prevent arithmetic overflow/NaNs
        print("Initializing weights and scales to valid constants...")
        for name, tensor in list(layer.named_parameters()) + list(layer.named_buffers()):
            if tensor.numel() > 0:
                if tensor.dtype.is_floating_point:
                    # Fill weight scales with 1.0, other floats with a small non-zero value
                    temp = torch.empty_like(tensor, dtype=torch.bfloat16)
                    if "scale" in name:
                        temp.fill_(1.0)
                    else:
                        temp.uniform_(-0.02, 0.02)
                    tensor.data.copy_(temp)
                elif tensor.dtype == torch.uint8 or tensor.dtype == torch.int8:
                    tensor.data.fill_(1)
                elif tensor.dtype == torch.int32:
                    tensor.data.fill_(0)
        
        
        # Allocate dummy KV caches using the correct PallasAttentionBackend shape logic (explicitly on TPU)
        print("Allocating dummy KV caches...")
        from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
        
        swa_spec = layer.mla_attn.swa_cache_layer.get_kv_cache_spec(vllm_config)
        swa_shape = PallasAttentionBackend.get_kv_cache_shape(
            num_blocks=10,
            block_size=swa_spec.block_size,
            num_kv_heads=swa_spec.num_kv_heads,
            head_size=swa_spec.head_size,
            cache_dtype_str=swa_spec.dtype,
        )
        layer.mla_attn.swa_cache_layer.kv_cache = torch.randint(1, 255, swa_shape, dtype=torch.uint8, device="tpu")
        print(f"Allocated SWA KV cache with shape {swa_shape}")

        main_spec = layer.get_kv_cache_spec(vllm_config)
        if main_spec is not None:
            main_shape = PallasAttentionBackend.get_kv_cache_shape(
                num_blocks=10,
                block_size=main_spec.block_size,
                num_kv_heads=main_spec.num_kv_heads,
                head_size=main_spec.head_size,
                cache_dtype_str=main_spec.dtype,
            )
            layer.kv_cache = torch.randint(1, 255, main_shape, dtype=torch.uint8, device="tpu")
            print(f"Allocated Main KV cache with shape {main_shape}")
        
        # Create dummy inputs (explicitly on TPU)
        print("Creating dummy inputs...")
        t = 16
        hidden_states = torch.randn(t, 4096, dtype=torch.bfloat16, device="tpu")
        # Compile the forward pass
        print("Compiling layer.forward...")
        compiled_forward = torch.compile(layer.forward, backend=TpuBackend(), dynamic=False)

        # Run compiled forward (this will trigger XLA compilation)
        print("Running compiled forward for shape 32 (this will trigger XLA compilation)...")
        # Allocate dummy inputs for shape 32
        hidden_states32 = torch.randn(32, 4096, dtype=torch.bfloat16, device="tpu")
        positions32 = torch.randint(0, 100, (32,), dtype=torch.int64, device="tpu")
        out32 = compiled_forward(positions32, hidden_states32)
        print(f"Compilation 32 successful! Output shape: {out32.shape}")

        # Now try to compile and run for a smaller shape (16) to test caching
        print("Creating dummy inputs for shape 16...")
        hidden_states16 = torch.randn(16, 4096, dtype=torch.bfloat16, device="tpu")
        positions16 = torch.zeros(16, dtype=torch.int32, device="tpu")
        
        # Update metadata to match shape 16
        mock_metadata.seq_lens.copy_(torch.tensor([16], dtype=torch.int32, device="tpu"))
        mock_metadata.query_start_loc.copy_(torch.tensor([0, 16], dtype=torch.int32, device="tpu"))
        
        print("Running compiled forward for shape 16 (this should trigger new XLA compilation)...")
        out16 = compiled_forward(positions16, hidden_states16)
        print(f"Compilation 16 successful! Output shape: {out16.shape}")


if __name__ == "__main__":
    test_attention_compilation()
