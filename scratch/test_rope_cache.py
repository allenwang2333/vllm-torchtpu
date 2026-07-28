import torch
import torch_tpu._loader as loader
try:
    loader._init_device('tpu')
except Exception as e:
    print('Failed to init device:', e)

import unittest.mock as mock
from vllm.platforms import current_platform

# Mock get_cached_compilation_config to return a mock config
mock_config = mock.MagicMock()
mock_config.enabled_custom_ops = set()
mock_config.disabled_custom_ops = set()
mock_config.mode.name = 'NONE'
mock_config.backend = 'eager'
mock_config.custom_ops = ['all']

import vllm.model_executor.custom_op as custom_op
custom_op.get_cached_compilation_config = lambda: mock_config

import vllm.config.vllm as vllm_cfg
vllm_cfg.get_cached_compilation_config = lambda: mock_config

from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import DeepseekV4ScalingRotaryEmbedding

def test_rope():
    # Instantiate CPU instance
    orig_dev = current_platform.device_type
    current_platform.device_type = 'cpu'
    
    emb_cpu = DeepseekV4ScalingRotaryEmbedding(
        head_size=128,
        rotary_dim=64,
        max_position_embeddings=4096,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        scaling_factor=1,
    )
    
    # Instantiate another one for TPU
    emb_tpu_cpu = DeepseekV4ScalingRotaryEmbedding(
        head_size=128,
        rotary_dim=64,
        max_position_embeddings=4096,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        scaling_factor=1,
    )
    
    # Restore platform device type to tpu
    current_platform.device_type = orig_dev
    
    # Move CPU embedding to TPU
    emb_tpu = emb_tpu_cpu.to('tpu')
    
    # Check cos_sin_cache
    cpu_cache = emb_cpu.cos_sin_cache
    tpu_cache = emb_tpu.cos_sin_cache
    
    print('CPU Cache shape:', cpu_cache.shape, 'mean:', cpu_cache.mean().item(), 'std:', cpu_cache.std().item())
    print('TPU Cache shape:', tpu_cache.shape, 'mean:', tpu_cache.cpu().mean().item(), 'std:', tpu_cache.cpu().std().item())
    
    diff = torch.abs(cpu_cache - tpu_cache.cpu()).max().item()
    print('Cache Max Diff:', diff)
    
    # Run forward pass comparison
    positions = torch.tensor([0, 1, 2, 3], device='cpu')
    query = torch.randn(4, 1, 128, device='cpu')
    
    # Run CPU forward
    q_cpu, _ = emb_cpu(positions, query)
    
    # Run TPU forward
    q_tpu, _ = emb_tpu(positions.to('tpu'), query.to('tpu'))
    
    forward_diff = torch.abs(q_cpu - q_tpu.cpu()).max().item()
    print('Forward Output Max Diff:', forward_diff)
    print('q_cpu snippet:', q_cpu[0, 0, -5:])
    print('q_tpu snippet:', q_tpu.cpu()[0, 0, -5:])

if __name__ == '__main__':
    test_rope()
