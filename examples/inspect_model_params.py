import torch
import torch.library
from vllm.config import ModelConfig, LoadConfig, VllmConfig, SchedulerConfig
from vllm.model_executor.model_loader import get_model

class MockCUDAStream:
    pass
torch.cuda.Stream = MockCUDAStream

# Mock parallel state groups
class MockParallelGroup:
    @property
    def is_first_rank(self): return True
    @property
    def is_last_rank(self): return True
    @property
    def rank_in_group(self): return 0
    @property
    def rank(self): return 0
    @property
    def world_size(self): return 1
    
    def all_reduce(self, x, *args, **kwargs):
        return x
    def all_gather(self, x, *args, **kwargs):
        return [x]
    def broadcast(self, x, *args, **kwargs):
        return x
    def barrier(self, *args, **kwargs):
        pass

import vllm.distributed as dist
import vllm.distributed.parallel_state as ps

mock_group = MockParallelGroup()
ps._TP = mock_group
ps._PP = mock_group
ps._DP = mock_group
ps._PCP = mock_group
ps._SP = mock_group
ps._EP = mock_group

dist.get_pp_group = lambda: mock_group
dist.get_tp_group = lambda: mock_group
dist.get_dp_group = lambda: mock_group
dist.get_pcp_group = lambda: mock_group
dist.get_sp_group = lambda: mock_group
dist.get_ep_group = lambda: mock_group
dist.get_tensor_model_parallel_rank = lambda: 0
dist.get_tensor_model_parallel_world_size = lambda: 1

ps.get_pp_group = lambda: mock_group
ps.get_tp_group = lambda: mock_group
ps.get_dp_group = lambda: mock_group
ps.get_pcp_group = lambda: mock_group

# Mock capability overrides
class MockDeviceCapability:
    major = 9
    minor = 0
    def to_int(self): return 90
    def __getitem__(self, idx):
        if idx == 0: return 9
        if idx == 1: return 0
        raise IndexError
    def __len__(self): return 2

from vllm.platforms import current_platform
current_platform.get_device_capability = lambda *args: MockDeviceCapability()
torch.cpu._is_amx_tile_supported = lambda: True

# Register custom operator fallbacks to avoid missing op crashes during init
if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "convert_weight_packed"):
    @torch.library.custom_op("_C::convert_weight_packed", mutates_args=())
    def convert_weight_packed_fallback(weight: torch.Tensor) -> torch.Tensor:
        return weight

if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "fp8_scaled_mm_cpu"):
    @torch.library.custom_op("_C::fp8_scaled_mm_cpu", mutates_args=())
    def fp8_scaled_mm_cpu_fallback(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_group_shape: list[int],
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        is_vnni: bool,
    ) -> torch.Tensor:
        return torch.empty(1)

# Enable deepseek_v4_fp8 config mapping
import vllm.models.deepseek_v4.nvidia.model as nvidia_model
import vllm.models.deepseek_v4.amd.model as amd_model
for model_module in [nvidia_model, amd_model]:
    if hasattr(model_module, "_make_deepseek_v4_weights_mapper"):
        orig_make_mapper = model_module._make_deepseek_v4_weights_mapper
        def patched_make_mapper(expert_dtype, orig_make_mapper=orig_make_mapper):
            mapper = orig_make_mapper(expert_dtype)
            mapper.orig_to_new_substr[".hc_head.hc_"] = ".hc_head_"
            mapper.orig_to_new_substr["hc_head.hc_"] = "hc_head_"
            mapper.orig_to_new_substr[".attn_hc."] = ".hc_attn_"
            mapper.orig_to_new_substr[".ffn_hc."] = ".hc_ffn_"
            mapper.orig_to_new_substr[".input_layernorm."] = ".attn_norm."
            mapper.orig_to_new_substr[".post_attention_layernorm."] = ".ffn_norm."
            mapper.orig_to_new_substr[".self_attn."] = ".attn."
            mapper.orig_to_new_substr[".mlp."] = ".ffn."
            return mapper
        model_module._make_deepseek_v4_weights_mapper = patched_make_mapper

# Build VllmConfig
model_dir = "/home/patemotter_google_com/workspace/dsv4/torchtpu-vllm/examples/dsv4-flash-4layer-fp8"
from vllm.config import DeviceConfig, CacheConfig
vllm_config = VllmConfig(
    model_config=ModelConfig(
        model=model_dir,
        tokenizer=model_dir,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
    ),
    cache_config=CacheConfig(block_size=32, cache_dtype="fp8"),
    device_config=DeviceConfig(device="cpu"),
)
vllm_config.scheduler_config.max_num_seqs = 4
vllm_config.scheduler_config.max_model_len = 512

vllm_config.parallel_config.world_size = 1

model = get_model(vllm_config=vllm_config)
print("=== Model Parameters Shapes ===")
for name, param in model.named_parameters():
    if "shared_experts.down_proj" in name or "experts.w13" in name or "shared_experts.gate_proj" in name:
        print(f"  {name}: shape={list(param.shape)}")
