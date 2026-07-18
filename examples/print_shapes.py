import vllm_torchtpu # Import our plugin first to register the tpu platform override!
import torch
import numpy as np
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, DeviceConfig, LoadConfig, LoRAConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs

# Mock parallel state groups since distributed system is not initialized in standalone script
class MockParallelGroup:
    @property
    def is_first_rank(self):
        return True
    @property
    def is_last_rank(self):
        return True
    @property
    def rank_in_group(self):
        return 0
    @property
    def rank(self):
        return 0
    @property
    def world_size(self):
        return 1

import vllm.distributed as dist
dist.get_pp_group = lambda: MockParallelGroup()
dist.get_tp_group = lambda: MockParallelGroup()
dist.get_dp_group = lambda: MockParallelGroup()
dist.get_pcp_group = lambda: MockParallelGroup()
dist.get_sp_group = lambda: MockParallelGroup()
dist.get_ep_group = lambda: MockParallelGroup()
dist.get_tensor_model_parallel_rank = lambda: 0
dist.get_tensor_model_parallel_world_size = lambda: 1

import vllm.distributed.parallel_state as ps
ps.get_pp_group = lambda: MockParallelGroup()
ps.get_tp_group = lambda: MockParallelGroup()
ps.get_dp_group = lambda: MockParallelGroup()
ps.get_pcp_group = lambda: MockParallelGroup()
ps.get_sp_group = lambda: MockParallelGroup()
ps.get_ep_group = lambda: MockParallelGroup()
ps.get_tensor_model_parallel_group = lambda: MockParallelGroup()
ps.get_tensor_model_parallel_rank = lambda: 0
ps.get_tensor_model_parallel_world_size = lambda: 1

def main():
    engine_args = EngineArgs(
        model="/root/cloud-devkit/torchtpu-vllm/examples/dsv4-flash-4layer-fp8",
        tokenizer="/root/cloud-devkit/torchtpu-vllm/examples/dsv4-flash-4layer-fp8",
        trust_remote_code=True,
        load_format="dummy",
        kv_cache_dtype="fp8",
        quantization="deepseek_v4_fp8",
        tensor_parallel_size=1,
    )
    vllm_config = engine_args.create_engine_config()
    from vllm_torchtpu.layers.vllm.quantization import get_tpu_quantization_config
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config)
    
    # Mock hf config
    from transformers import AutoConfig
    # We can create a dummy config with DeepSeek-V2/V3/V4 properties
    hf_config = AutoConfig.for_model("llama") # fallback
    hf_config.architectures = ["DeepseekV4ForCausalLM"]
    hf_config.hidden_size = 4096
    hf_config.num_attention_heads = 64
    hf_config.num_key_value_heads = 64
    hf_config.head_dim = 512
    hf_config.qk_rope_head_dim = 64
    hf_config.o_groups = 8
    hf_config.o_lora_rank = 1024
    hf_config.q_lora_rank = 1024
    hf_config.sliding_window = 1024
    hf_config.rms_norm_eps = 1e-6
    hf_config.num_hidden_layers = 1
    hf_config.max_position_embeddings = 4096
    hf_config.quantization_config = {
        "quant_method": "deepseek_v4_fp8",
        "scale_fmt": "weight_scale",
    }
    hf_config.compress_ratios = [0, 0, 4, 128, 0]
    hf_config.compress_rope_theta = 160000
    hf_config.rope_theta = 10000
    hf_config.rope_scaling = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    }
    hf_config.rope_parameters = {
        "rope_type": "default",
    }
    
    vllm_config.model_config.hf_config = hf_config
    vllm_config.cache_config.block_size = 2048
    
    from vllm_torchtpu.models.vllm.deepseek_v4_patch import _maybe_patch_for_deepseek_v4
    from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4 import VllmDeepseekV4MLAAttention
    
    metadata_dict = {
        "model.layers.0.attn": None,
    }
    
    import jax
    from jax.sharding import Mesh
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context
    
    devices = jax.devices()[:1]
    mesh = Mesh(np.array(devices), axis_names=("model",))
    
    with _maybe_patch_for_deepseek_v4(vllm_config), \
         set_vllm_model_wrapper_context(mesh=mesh), \
         set_current_vllm_config(vllm_config), \
         torch.device("cpu"):
         
        layer = VllmDeepseekV4MLAAttention(vllm_config, prefix="model.layers.0.attn")
        
        # Initialize weights and scales to dummy values so dequantize/requantize works
        for param in list(layer.parameters()) + list(layer.buffers()):
            if param.numel() > 0:
                if param.dtype == torch.float32 or param.dtype == torch.bfloat16:
                    param.data.fill_(1.0)
                else:
                    param.data.fill_(1)
                    
        # Process weights
        layer.wo_a.quant_method.process_weights_after_loading(layer.wo_a)
        layer.wo_b.quant_method.process_weights_after_loading(layer.wo_b)
        
        print("wo_a.weight shape:", list(layer.wo_a.weight.shape))
        print("wo_a.weight_scale shape:", list(layer.wo_a.weight_scale.shape) if hasattr(layer.wo_a, "weight_scale") else "None")
        print("wo_b.weight shape:", list(layer.wo_b.weight.shape))
        print("wo_b.weight_scale shape:", list(layer.wo_b.weight_scale.shape) if hasattr(layer.wo_b, "weight_scale") else "None")

if __name__ == "__main__":
    main()
