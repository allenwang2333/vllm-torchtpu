# Copyright 2026 Google LLC
import torch
import sys
import inspect
from unittest.mock import MagicMock
from vllm.config.vllm import VllmConfig, CompilationConfig, ModelConfig, ParallelConfig, CacheConfig
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4 import VllmDeepseekV4MLAAttention
from vllm.platforms import current_platform

current_platform.device_type = "cpu"
mock_cap = MagicMock()
mock_cap.major = 9
mock_cap.minor = 0
current_platform.get_device_capability = MagicMock(return_value=mock_cap)
import torch.cuda
torch.cuda.Event = MagicMock()

comp_config = CompilationConfig()
comp_config.custom_ops = ["all"]

mock_vllm_config = MagicMock()
mock_vllm_config.compilation_config = comp_config

model_config = MagicMock()
model_config.hf_config.num_hidden_layers = 4
model_config.hf_config.hidden_size = 4096
model_config.hf_config.num_attention_heads = 64
model_config.hf_config.q_lora_rank = 1024
model_config.hf_config.o_lora_rank = 1024
model_config.hf_config.index_head_dim = 128
model_config.hf_config.index_topk = 8
model_config.hf_config.head_dim = 512
model_config.hf_config.qk_rope_head_dim = 64
model_config.hf_config.o_groups = 8
model_config.hf_config.sliding_window = 128
model_config.hf_config.compress_ratios = [4, 4, 4, 4]
model_config.hf_config.rms_norm_eps = 1e-6
model_config.hf_config.max_position_embeddings = 65536
model_config.hf_config.rope_theta = 10000.0
model_config.hf_config.compress_rope_theta = 10000.0
model_config.hf_config.rope_scaling = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "type": "yarn"
}
model_config.hf_config.quantization_config = {
    "scale_fmt": "ue8m0"
}
model_config.hf_config.rope_parameters = {
    "rope_type": "default"
}

parallel_config = MagicMock()
parallel_config.tensor_parallel_size = 8

cache_config = MagicMock()
cache_config.block_size = 16

mock_vllm_config.model_config = model_config
mock_vllm_config.parallel_config = parallel_config
mock_vllm_config.cache_config = cache_config
from vllm_torchtpu.layers.vllm.quantization.deepseek_v4_fp8 import VllmDeepseekV4Fp8Config
quant_dict = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128]
}
VllmDeepseekV4Fp8Config.expert_dtype = property(lambda self: "fp4")
mock_vllm_config.quant_config = VllmDeepseekV4Fp8Config.from_config(quant_dict)

import vllm.config.vllm as vllm_config_mod
vllm_config_mod.get_current_vllm_config = MagicMock(return_value=mock_vllm_config)

import vllm.distributed.parallel_state as ps
ps._TP = MagicMock()
ps._TP.world_size = 8
ps._TP.rank = 0

from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
orig_init = MergedColumnParallelLinear.__init__
def new_init(self, *args, **kwargs):
    print(f"[INTERCEPT] MergedColumnParallelLinear.__init__ args={args} kwargs={kwargs}", flush=True)
    orig_init(self, *args, **kwargs)
MergedColumnParallelLinear.__init__ = new_init

orig_row_init = RowParallelLinear.__init__
def new_row_init(self, *args, **kwargs):
    print(f"[INTERCEPT] RowParallelLinear.__init__ args={args} kwargs={kwargs}", flush=True)
    orig_row_init(self, *args, **kwargs)
RowParallelLinear.__init__ = new_row_init

from vllm.model_executor.layers.linear import ColumnParallelLinear
orig_col_init = ColumnParallelLinear.__init__
def new_col_init(self, *args, **kwargs):
    print(f"[INTERCEPT] ColumnParallelLinear.__init__ args={args} kwargs={kwargs}", flush=True)
    orig_col_init(self, *args, **kwargs)
ColumnParallelLinear.__init__ = new_col_init

# Mock model context
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context

from vllm.config.vllm import set_current_vllm_config

print("Instantiating VllmDeepseekV4MLAAttention...")
with set_current_vllm_config(mock_vllm_config):
    with set_vllm_model_wrapper_context(mesh=None):
        layer = VllmDeepseekV4MLAAttention(mock_vllm_config, prefix="model.layers.0.self_attn")

print("\nNamed parameters of VllmDeepseekV4MLAAttention:")
for name, param in layer.named_parameters():
    print(f"  {name}: shape={list(param.shape)} dtype={param.dtype}")

print("\nClasses of MLA layers:")
print(f"  fused_wqa_wkv: {type(layer.fused_wqa_wkv)}")
print(f"  wo_a: {type(layer.wo_a)}")
print(f"  wo_b: {type(layer.wo_b)}")
