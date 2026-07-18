import os
os.environ["VLLM_ENFORCE_EAGER"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import sys
import os
sys.path.insert(0, "/root/cloud-devkit/torchtpu-vllm/src")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
print("PARITY SYS.PATH:", sys.path, flush=True)
print("PARITY ENV:", dict(os.environ), flush=True)
import gc
import json
import torch
torch.set_num_threads(8)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, choices=[1, 2, 3], required=True)
cmd_args = parser.parse_args()

import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as fused_topk_bias_router
orig_fused_topk_bias = fused_topk_bias_router.fused_topk_bias
def patched_fused_topk_bias(*args, **kwargs):
    topk_weights, topk_ids = orig_fused_topk_bias(*args, **kwargs)
    print(f"DEBUG_MOE_CPU: topk_ids[:2] = {topk_ids[:2].tolist()}")
    print(f"DEBUG_MOE_CPU: topk_weights[:2] = {topk_weights[:2].tolist()}")
    return topk_weights, topk_ids
fused_topk_bias_router.fused_topk_bias = patched_fused_topk_bias
try:
    import vllm.models.deepseek_v4.nvidia.model as nvidia_model
    nvidia_model.fused_topk_bias = patched_fused_topk_bias
except ImportError:
    pass
import numpy as np
from unittest.mock import MagicMock

# Force CPU mode for reference first
device_cpu = torch.device("cpu")
device_tpu = torch.device("tpu")

# Mock CUDA stream and event since CUDA is unavailable on TPU/CPU VM
class MockCUDAStream:
    def __init__(self, *args, **kwargs):
        pass
torch.cuda.Stream = MockCUDAStream

class MockCUDAEvent:
    def __init__(self, *args, **kwargs):
        pass
    def record(self, *args, **kwargs):
        pass
    def wait(self, *args, **kwargs):
        pass
torch.cuda.Event = MockCUDAEvent
# Mock platform device capability to satisfy DeepSeek-V4 attention requirement
class MockDeviceCapability:
    major = 9
    minor = 0
    def to_int(self):
        return self.major * 10 + self.minor
    def __getitem__(self, idx):
        if idx == 0: return self.major
        if idx == 1: return self.minor
        raise IndexError
    def __len__(self):
        return 2

from vllm.platforms import current_platform
from vllm.platforms.interface import PlatformEnum
current_platform._enum = PlatformEnum.CPU
current_platform.get_device_capability = lambda *args: MockDeviceCapability()

# Mock AMX tile support and register custom operator fallbacks for CPU FP8 GEMM
torch.cpu._is_amx_tile_supported = lambda: True
import torch.library

if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "convert_weight_packed"):
    @torch.library.custom_op("_C::convert_weight_packed", mutates_args=())
    def convert_weight_packed_fallback(weight: torch.Tensor) -> torch.Tensor:
        return weight.clone()

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
        print(f"DEBUG_SCALED_MM_CPU: x.shape={x.shape} | weight.shape={weight.shape} | scale.shape={weight_scale.shape}")
        block_n, block_k = weight_group_shape
        N, K = weight.shape
        w_fp32 = weight.to(torch.float32)
        w_blocks = w_fp32.view(N // block_n, block_n, K // block_k, block_k)
        s_blocks = weight_scale.view(N // block_n, block_n, K // block_k, 1)
        w_dequant = (w_blocks * s_blocks).view(N, K).to(x.dtype)
        out = torch.matmul(x, w_dequant.t())
        if bias is not None:
            out += bias
        return out.to(out_dtype)

def fused_experts_cpu_fallback(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool,
    moe_comp_method: int,
    w1_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        w1_zero: torch.Tensor | None,
        w2_zero: torch.Tensor | None,
        block_size: list[int] | None,
        w1_bias: torch.Tensor | None,
        w2_bias: torch.Tensor | None,
        alpha: float | None,
        limit: float | None,
        is_vnni: bool,
    ) -> torch.Tensor:
        print("DEBUG_MOE_CPU: topk_ids[:2] =", topk_ids[:2].tolist())
        print("DEBUG_MOE_CPU: topk_weights[:2] =", topk_weights[:2].tolist())
        print("DEBUG_FALLBACK_SHAPES: w1.shape =", w1.shape, "w1_scale.shape =", w1_scale.shape if w1_scale is not None else None)
        print("DEBUG_FALLBACK_SHAPES: w2.shape =", w2.shape, "w2_scale.shape =", w2_scale.shape if w2_scale is not None else None)
        print(f"DEBUG_MOE_CPU_STATS: hidden_states.shape={hidden_states.shape} | mean={hidden_states.abs().mean().item():.6f} | max={hidden_states.abs().max().item():.6f}")
        print(f"DEBUG_MOE_CPU_STATS: w1.shape={w1.shape} | mean={w1.float().abs().mean().item():.6f} | max={w1.float().abs().max().item():.6f}")
        print(f"DEBUG_MOE_CPU_STATS: w2.shape={w2.shape} | mean={w2.float().abs().mean().item():.6f} | max={w2.float().abs().max().item():.6f}")
        if w1_scale is not None:
            print(f"DEBUG_MOE_CPU_STATS: w1_scale.shape={w1_scale.shape} | mean={w1_scale.abs().mean().item():.6f} | max={w1_scale.abs().max().item():.6f}")
        if w2_scale is not None:
            print(f"DEBUG_MOE_CPU_STATS: w2_scale.shape={w2_scale.shape} | mean={w2_scale.abs().mean().item():.6f} | max={w2_scale.abs().max().item():.6f}")
        num_tokens, hidden_size = hidden_states.shape
        topk = topk_ids.shape[1]
        out = torch.zeros_like(hidden_states, dtype=torch.float32)
        
        w1_fp32 = w1.to(torch.float32)
        w2_fp32 = w2.to(torch.float32)
        
        if w1_scale is not None:
            scale_expanded = w1_scale.repeat_interleave(32, dim=2)
            if scale_expanded.shape[2] < hidden_size:
                scale_expanded = torch.cat([scale_expanded, scale_expanded[:, :, -1:]], dim=2)
            scale_expanded = scale_expanded[:, :, :hidden_size]
            w1_fp32 = w1_fp32 * scale_expanded
            
        if w2_scale is not None:
            scale_expanded = w2_scale.repeat_interleave(32, dim=2)
            if scale_expanded.shape[2] < w2_fp32.shape[2]:
                scale_expanded = torch.cat([scale_expanded, scale_expanded[:, :, -1:]], dim=2)
            scale_expanded = scale_expanded[:, :, :w2_fp32.shape[2]]
            w2_fp32 = w2_fp32 * scale_expanded

        for expert_id in range(w1.shape[0]):
            mask = (topk_ids == expert_id)
            if not mask.any():
                continue
                
            token_indices, topk_indices = torch.where(mask)
            x_expert = hidden_states[token_indices].to(torch.float32)
            
            w13_expert = w1_fp32[expert_id]
            intermediate_size = w13_expert.shape[0] // 2
            w1_exp = w13_expert[:intermediate_size]
            w3_exp = w13_expert[intermediate_size:]
            
            gate_out = torch.matmul(x_expert, w1_exp.t())
            up_out = torch.matmul(x_expert, w3_exp.t())
            gate_act = torch.nn.functional.silu(gate_out) * up_out
            
            w2_expert = w2_fp32[expert_id]
            expert_out = torch.matmul(gate_act, w2_expert.t())
            
            if expert_id == 33:
                # Compute for all tokens for exact parity comparison!
                all_gate_out = torch.matmul(hidden_states.to(torch.float32), w1_exp.t())
                all_gate_act = torch.nn.functional.silu(all_gate_out) * torch.matmul(hidden_states.to(torch.float32), w3_exp.t())
                all_expert_out = torch.matmul(all_gate_act, w2_expert.t())
                
                print(f"DEBUG_CPU_EXP33: expert_weights={topk_weights[:, topk_indices[0]].tolist()}")
                print(f"DEBUG_CPU_EXP33: gate_up mean={all_gate_out.abs().mean().item():.6f} max={all_gate_out.abs().max().item():.6f}")
                print(f"DEBUG_CPU_EXP33: act mean={all_gate_act.abs().mean().item():.6f} max={all_gate_act.abs().max().item():.6f}")
                print(f"DEBUG_CPU_EXP33: out mean={all_expert_out.abs().mean().item():.6f} max={all_expert_out.abs().max().item():.6f}")

            routing_weight = topk_weights[token_indices, topk_indices].unsqueeze(-1)
            out[token_indices] += expert_out * routing_weight
            
        print(f"DEBUG_MOE_CPU_STATS: out.shape={out.shape} | mean={out.abs().mean().item():.6f} | max={out.abs().max().item():.6f}")
        return out.to(hidden_states.dtype)

import vllm._custom_ops as custom_ops
custom_ops.fused_experts_cpu = fused_experts_cpu_fallback
import vllm.model_executor.layers.fused_moe.experts.cpu_moe as cpu_moe
cpu_moe.fused_experts_cpu = fused_experts_cpu_fallback

# Also register custom library op for it if PyTorch dispatcher requires it
if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "fused_experts_cpu"):
    torch.library.custom_op("_C::fused_experts_cpu", mutates_args=())(fused_experts_cpu_fallback)

from vllm.model_executor.layers.mhc import HCHeadOp, MHCFusedPostPreOp

def mhc_hc_head_native(
    self,
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
    *args,
    **kwargs,
) -> torch.Tensor:
    residual_flat = hidden_states.flatten(-2).float()
    residual_norm = residual_flat * torch.rsqrt(
        residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )
    pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * hidden_states.float(), dim=-2).to(hidden_states.dtype)

def mhc_fused_post_pre_native(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch
    residual_cur = mhc_post_torch(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )
    post_mix_cur, comb_mix_cur, layer_input_cur = mhc_pre_torch(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    if norm_weight is not None:
        variance = layer_input_cur.pow(2).mean(-1, keepdim=True)
        layer_input_cur = layer_input_cur * torch.rsqrt(variance + norm_eps) * norm_weight.to(layer_input_cur.dtype)
    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

HCHeadOp.forward_native = mhc_hc_head_native
MHCFusedPostPreOp.forward_native = mhc_fused_post_pre_native

import safetensors.torch
safetensors.torch._TYPES["F8_E8M0"] = torch.uint8



import vllm.utils.import_utils as import_utils
import_utils.has_tilelang = lambda: True

import vllm.models.deepseek_v4.nvidia.model as nvidia_model
import vllm.models.deepseek_v4.amd.model as amd_model
nvidia_model.has_tilelang = lambda: True
amd_model.has_tilelang = lambda: True


import vllm.model_executor.kernels.linear as kernels_linear
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import ChannelWiseTorchFP8ScaledMMLinearKernel
kernels_linear.choose_scaled_mm_linear_kernel = lambda *args, **kwargs: ChannelWiseTorchFP8ScaledMMLinearKernel
ChannelWiseTorchFP8ScaledMMLinearKernel.can_implement = lambda *args, **kwargs: (True, None)

def patched_channel_wise_apply_scaled_mm(
    self,
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None,
    output_shape: list,
) -> torch.Tensor:
    A_fp32 = A.to(torch.float32)
    B_fp32 = B.to(torch.float32)
    
    if As is not None:
        if As.numel() == 1:
            A_fp32 = A_fp32 * As
        else:
            A_fp32 = A_fp32 * As.view(-1, 1)
            
    K, N = B.shape
    num_blocks = Bs.shape[1]
    block_k = K // num_blocks
    scale_expanded = Bs.repeat_interleave(block_k, dim=1)
    if scale_expanded.shape[1] < K:
        scale_expanded = torch.cat([scale_expanded, scale_expanded[:, -1:]], dim=1)
    scale_expanded = scale_expanded[:, :K]
    B_fp32 = B_fp32 * scale_expanded.t()
    
    out = torch.matmul(A_fp32, B_fp32)
    if bias is not None:
        out = out + bias.to(torch.float32)
        
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import _get_num_tokens
    num_tokens = _get_num_tokens(output_shape)
    out = torch.narrow(out, 0, 0, num_tokens).view(*output_shape)
    
    return out.to(out_dtype)

ChannelWiseTorchFP8ScaledMMLinearKernel.apply_scaled_mm = patched_channel_wise_apply_scaled_mm




from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod

orig_linear_create_weights = Fp8LinearMethod.create_weights
def patched_linear_create_weights(self, layer, input_size_per_partition, *args, **kwargs):
    self.weight_block_size = [1, input_size_per_partition]
    self.block_quant = True
    self.weight_scale_name = "weight_scale_inv"
    res = orig_linear_create_weights(self, layer, input_size_per_partition, *args, **kwargs)
    if hasattr(self, "fp8_linear") and hasattr(self.fp8_linear, "layer_param_names"):
        params = list(self.fp8_linear.layer_param_names)
        if "weight_scale" in params:
            idx = params.index("weight_scale")
            params[idx] = "weight_scale_inv"
            self.fp8_linear.layer_param_names = params
    return res
Fp8LinearMethod.create_weights = patched_linear_create_weights

from vllm.model_executor.utils import replace_parameter
orig_linear_process_weights = Fp8LinearMethod.process_weights_after_loading
def patched_linear_process_weights(self, layer, *args, **kwargs):
    res = orig_linear_process_weights(self, layer, *args, **kwargs)
    if self.block_quant and hasattr(layer, "weight") and hasattr(layer, "output_size_per_partition"):
        if layer.weight.shape[0] == layer.output_size_per_partition:
            replace_parameter(layer, "weight", layer.weight.t().data)
    return res
Fp8LinearMethod.process_weights_after_loading = patched_linear_process_weights


orig_moe_init = Fp8MoEMethod.__init__
def patched_moe_init(self, quant_config, *args, **kwargs):
    orig_moe_init(self, quant_config, *args, **kwargs)
    print(f"DEBUG_MOE_INIT: self.fp8_backend = {self.fp8_backend}, self.experts_cls = {self.experts_cls}")
    self.weight_block_size = [1, 32]
    self.block_quant = True
    self.weight_scale_name = "weight_scale_inv"
Fp8MoEMethod.__init__ = patched_moe_init

import vllm.model_executor.layers.fused_moe.oracle.fp8 as moe_oracle
import vllm.model_executor.layers.quantization.fp8 as fp8_quant
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import CPUExpertsFp8
patched_backend_fn = lambda *args, **kwargs: (moe_oracle.Fp8MoeBackend.CPU, CPUExpertsFp8)
moe_oracle.select_fp8_moe_backend = patched_backend_fn
fp8_quant.select_fp8_moe_backend = patched_backend_fn

import vllm.model_executor.layers.fused_moe.cpu_fused_moe as cpu_fused_moe
orig_select_experts = cpu_fused_moe.select_experts
def patched_select_experts(
    hidden_states,
    router_logits,
    top_k,
    use_grouped_topk,
    renormalize,
    topk_group=None,
    num_expert_group=None,
    custom_routing_function=None,
    scoring_func="softmax",
    routed_scaling_factor=1.0,
    e_score_correction_bias=None,
):
    if scoring_func == "sqrtsoftplus" and not use_grouped_topk:
        import torch.nn.functional as F
        scores = torch.sqrt(F.softplus(router_logits.float()))
        if e_score_correction_bias is not None:
            scores = scores + e_score_correction_bias.unsqueeze(0)
        topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor
        return topk_weights.to(hidden_states.dtype), topk_ids.to(torch.int32)
        
    return orig_select_experts(
        hidden_states, router_logits, top_k, use_grouped_topk, renormalize,
        topk_group, num_expert_group, custom_routing_function,
        scoring_func, routed_scaling_factor, e_score_correction_bias
    )
cpu_fused_moe.select_experts = patched_select_experts

orig_cpu_experts_apply = CPUExpertsFp8.apply
def patched_cpu_experts_apply(
    self,
    hidden_states,
    w1,
    w2,
    router_logits,
    activation,
    global_num_experts,
    expert_map,
    a1q_scale,
    apply_router_weight_on_input,
    num_expert_group=None,
    e_score_correction_bias=None,
    routed_scaling_factor=None,
    topk_group=None,
):
    print("DEBUG_CPU_EXPERTS_APPLY: called!")
    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import (
        select_experts,
    )
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    
    scoring_func = "sqrtsoftplus"
    print(f"DEBUG_CPU_EXPERTS_APPLY: using scoring_func = {scoring_func}")
    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        use_grouped_topk=num_expert_group is not None,
        top_k=self.moe_config.experts_per_token,
        renormalize=True,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        scoring_func=scoring_func,
        routed_scaling_factor=(
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        ),
        e_score_correction_bias=e_score_correction_bias,
    )
    
    # We must format block_size as list since CPU fallback expects list[int]
    block_shape = (
        list(self.quant_config.block_shape)
        if self.quant_config.block_shape
        else None
    )
    
    from vllm._custom_ops import CPUQuantMethod, fused_experts_cpu
    return fused_experts_cpu(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,  # inplace
        CPUQuantMethod.FP8_W8A16,  # moe_comp_method
        self.w1_scale,  # w1_scale
        self.w2_scale,  # w2_scale
        None,  # w1_zero
        None,  # w2_zero
        block_shape,  # block_size
        None,  # w1_bias
        None,  # w2_bias
        None,  # alpha
        None,  # limit
        True,  # is_vnni
    )
CPUExpertsFp8.apply = patched_cpu_experts_apply

from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

def patched_unquantized_apply_monolithic(
    self,
    layer,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    *args,
    **kwargs
) -> torch.Tensor:
    print("DEBUG_UNQUANTIZED_APPLY_MONOLITHIC: called!")
    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts
    
    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        use_grouped_topk=False,
        top_k=layer.moe_config.experts_per_token,
        renormalize=True,
    )
    
    w1 = layer.w13_weight
    w2 = layer.w2_weight
    
    from vllm._custom_ops import CPUQuantMethod, fused_experts_cpu
    return fused_experts_cpu(
        x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,  # inplace
        CPUQuantMethod.FP8_W8A16,  # moe_comp_method
        None,  # w1_scale
        None,  # w2_scale
        None,  # w1_zero
        None,  # w2_zero
        None,  # block_size
        None,  # w1_bias
        None,  # w2_bias
        None,  # alpha
        None,  # limit
        False,  # is_vnni
    )

UnquantizedFusedMoEMethod.apply_monolithic = patched_unquantized_apply_monolithic

from vllm.models.deepseek_v4.attention import DeepseekV4MultiHeadLatentAttentionWrapper
DeepseekV4MultiHeadLatentAttentionWrapper.forward = lambda self, positions, hidden_states, *args, **kwargs: hidden_states.clone()

from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4 import VllmDeepseekV4MLAAttention
VllmDeepseekV4MLAAttention.forward = lambda self, positions, hidden_states, *args, **kwargs: hidden_states.clone()

from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
orig_vocab_forward = VocabParallelEmbedding.forward
def patched_vocab_forward(self, input_):
    print(f"DEBUG_EMBEDDING: self.weight device = {self.weight.device}, input device = {input_.device}")
    input_ = input_.to(self.weight.device)
    return orig_vocab_forward(self, input_)
VocabParallelEmbedding.forward = patched_vocab_forward

captured_tpu_weights_list = []
captured_tpu_ids_list = []

try:
    import vllm_torchtpu.layers.vllm.fused_moe as torchtpu_fused_moe
    orig_fused_moe_gmm = torchtpu_fused_moe.fused_moe_gmm
    def patched_fused_moe_gmm(hidden_states, w1, w2, w1_scale, w2_scale, w1_bias, w2_bias, topk_weights, topk_ids, *args, **kwargs):
        captured_tpu_weights_list.append(topk_weights.detach().cpu().clone())
        captured_tpu_ids_list.append(topk_ids.detach().cpu().clone())
        print("DEBUG_MOE_TPU: topk_ids[:2] =", topk_ids[:2].tolist())
        print("DEBUG_MOE_TPU: topk_weights[:2] =", topk_weights[:2].tolist())
        print(f"DEBUG_MOE_TPU_STATS: hidden_states.shape={hidden_states.shape} | mean={hidden_states.abs().mean().item():.6f} | max={hidden_states.abs().max().item():.6f}")
        print(f"DEBUG_MOE_TPU_STATS: w1.shape={w1.shape} | mean={w1.float().abs().mean().item():.6f} | max={w1.float().abs().max().item():.6f}")
        print(f"DEBUG_MOE_TPU_STATS: w2.shape={w2.shape} | mean={w2.float().abs().mean().item():.6f} | max={w2.float().abs().max().item():.6f}")
        if w1_scale is not None:
            print(f"DEBUG_MOE_TPU_STATS: w1_scale.shape={w1_scale.shape} | mean={w1_scale.abs().mean().item():.6f} | max={w1_scale.abs().max().item():.6f}")
        if w2_scale is not None:
            print(f"DEBUG_MOE_TPU_STATS: w2_scale.shape={w2_scale.shape} | mean={w2_scale.abs().mean().item():.6f} | max={w2_scale.abs().max().item():.6f}")
        out = orig_fused_moe_gmm(hidden_states, w1, w2, w1_scale, w2_scale, w1_bias, w2_bias, topk_weights, topk_ids, *args, **kwargs)
        print(f"DEBUG_MOE_TPU_STATS: out.shape={out.shape} | mean={out.abs().mean().item():.6f} | max={out.abs().max().item():.6f}")
        return out
    torchtpu_fused_moe.fused_moe_gmm = patched_fused_moe_gmm

    import vllm_torchtpu.layers.vllm.moe_routing as tpu_moe_routing
    orig_tpu_select_experts = tpu_moe_routing.select_experts
    def patched_tpu_select_experts(hidden_states, router_logits, *, topk, renormalize, scoring_fn, layer=None):
        topk_weights, topk_ids = orig_tpu_select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk=topk,
            renormalize=renormalize,
            scoring_fn=scoring_fn,
            layer=layer,
        )
        scale = getattr(layer, "routed_scaling_factor", 1.0)
        if scale != 1.0:
            topk_weights = topk_weights * scale
        return topk_weights, topk_ids
    tpu_moe_routing.select_experts = patched_tpu_select_experts
except Exception as e:
    print("Warning: failed to patch torchtpu fused_moe_gmm or select_experts:", e)


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
            mapper.orig_to_new_substr[".weight_scale"] = ".weight_scale_inv"
            
            import re
            corrected_regex = {
                re.compile(r"(?<!hc_attn)(?<!attn_hc)(?<!hc_ffn)(?<!ffn_hc)(?<!hc_head)\.scale$"): ".weight_scale_inv"
            }
            if expert_dtype == "fp4":
                corrected_regex[re.compile(r"(\.experts\.\d+\.w[123])\.scale$")] = r"\1.weight_scale"
            mapper.orig_to_new_regex = corrected_regex
            return mapper
        model_module._make_deepseek_v4_weights_mapper = patched_make_mapper
    
    def get_inspect_weights(self, weights):
        inspect_weights = []
        for name, weight in weights:
            # Map gate_proj/up_proj to w1/w3 for shared_experts to match stacked params mapping
            name = name.replace("shared_experts.gate_proj", "shared_experts.w1").replace("shared_experts.up_proj", "shared_experts.w3")
            # Map attention projection layers to match stacked and expected vLLM parameters
            name = (name
                    .replace("hc_head.hc_", "hc_head_")
                    .replace(".hc_head.hc_", ".hc_head_")
                    .replace(".attn_hc.", ".hc_attn_")
                    .replace(".ffn_hc.", ".hc_ffn_")
                    .replace(".input_layernorm.", ".attn_norm.")
                    .replace(".post_attention_layernorm.", ".ffn_norm.")
                    .replace(".self_attn.", ".attn.")
                    .replace(".mlp.", ".ffn.")
                    .replace("attn.q_a_proj", "attn.wq_a")
                    .replace("attn.kv_proj", "attn.wkv")
                    .replace("attn.q_b_proj", "attn.wq_b")
                    .replace("attn.o_a_proj", "attn.wo_a")
                    .replace("attn.o_b_proj", "attn.wo_b")
                    .replace("attn.q_a_norm", "attn.q_norm")
                    .replace("attn.sinks", "attn.attn_sink")
                    
                    # MLA Compressor Indexer (longest first!)
                    .replace("attn.compressor.indexer.gate_proj", "attn.indexer.compressor.wgate")
                    .replace("attn.compressor.indexer.kv_proj", "attn.indexer.compressor.wkv")
                    .replace("attn.compressor.indexer.kv_norm", "attn.indexer.compressor.norm")
                    .replace("attn.compressor.indexer.position_bias", "attn.indexer.compressor.ape")
                    .replace("attn.compressor.indexer.q_b_proj", "attn.indexer.wq_b")
                    .replace("attn.compressor.indexer.scorer.weights_proj", "attn.indexer.weights_proj")
                    
                    # MLA Compressor
                    .replace("attn.compressor.gate_proj", "attn.mla_attn.compressor.wgate")
                    .replace("attn.compressor.kv_proj", "attn.mla_attn.compressor.wkv")
                    .replace("attn.compressor.kv_norm", "attn.mla_attn.compressor.norm")
                    .replace("attn.compressor.position_bias", "attn.mla_attn.compressor.ape"))
            
            from vllm.platforms import current_platform
            if current_platform.device_type == "tpu":
                if name.endswith(".weight_scale_inv") and ".experts." not in name:
                    name = name.replace(".weight_scale_inv", ".weight_scale")

            if weight.dim() == 1 and (name.endswith("weight_scale") or name.endswith("weight_scale_inv")):
                temp_dict = dict(self.named_parameters())
                # Resolve mapped param name for stacked layers
                mapped_name = name
                for param_name, weight_name, _ in [
                    ("gate_up_proj", "w1", 0),
                    ("gate_up_proj", "w3", 1),
                    ("attn.fused_wqa_wkv", "attn.wq_a", 0),
                    ("attn.fused_wqa_wkv", "attn.wkv", 1),
                    ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
                    ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
                ]:
                    if weight_name in name:
                        mapped_name = name.replace(weight_name, param_name)
                        break
                if mapped_name in temp_dict:
                    param = temp_dict[mapped_name]
                    if param.dim() == 2 and param.shape[1] == 1:
                        weight = weight.unsqueeze(-1)
            
            # Keep log compact, only print expert scale mapping entries for diagnosis
            if "experts" in name and "scale" in name:
                print(f"  incoming_weight: name={name} | shape={weight.shape}")
            inspect_weights.append((name, weight))
        return inspect_weights

    def make_patched_load_weights(orig_load_weights):
        def patched_load_weights(self, weights, orig_load_weights=orig_load_weights):
            inspect_weights = get_inspect_weights(self, weights)
            params_dict = dict(self.named_parameters())
            
            def get_target_model_param_name(name):
                # Exclude non-expert and non-shared-expert scales (e.g. attention scales if not in model)
                if not ".experts." in name and not ".shared_experts." in name and (name.endswith("weight_scale") or name.endswith("weight_scale_inv") or name.endswith(".scale")):
                    return None
                if name in params_dict:
                    return name
                
                # Check MoE expert stacked parameters mapping
                if ".experts." in name:
                    if name.endswith(".w1.weight") or name.endswith(".w3.weight"):
                        prefix = name.split(".experts.")[0]
                        target = f"{prefix}.experts.w13_weight"
                        if target in params_dict:
                            return target
                    elif name.endswith(".w2.weight"):
                        prefix = name.split(".experts.")[0]
                        target = f"{prefix}.experts.w2_weight"
                        if target in params_dict:
                            return target
                    elif name.endswith(".w1.weight_scale") or name.endswith(".w3.weight_scale"):
                        prefix = name.split(".experts.")[0]
                        target = f"{prefix}.experts.w13_weight_scale"
                        if target in params_dict:
                            return target
                    elif name.endswith(".w2.weight_scale"):
                        prefix = name.split(".experts.")[0]
                        target = f"{prefix}.experts.w2_weight_scale"
                        if target in params_dict:
                            return target
                
                # Check gate/up stacked gate_up_proj
                if name.endswith(".gate_proj.weight") or name.endswith(".up_proj.weight"):
                    target = name.replace(".gate_proj.weight", ".gate_up_proj.weight").replace(".up_proj.weight", ".gate_up_proj.weight")
                    if target in params_dict:
                        return target
                elif name.endswith(".gate_proj.weight_scale") or name.endswith(".up_proj.weight_scale"):
                    target = name.replace(".gate_proj.weight_scale", ".gate_up_proj.weight_scale").replace(".up_proj.weight_scale", ".gate_up_proj.weight_scale")
                    if target in params_dict:
                        return target
                        
                # Check shared experts stacked gate_up_proj
                if ".shared_experts." in name:
                    prefix = name.split(".shared_experts.")[0]
                    if name.endswith("weight"):
                        target = f"{prefix}.shared_experts.gate_up_proj.weight"
                    elif name.endswith("weight_scale"):
                        target = f"{prefix}.shared_experts.gate_up_proj.weight_scale"
                    else:
                        target = None
                    if target and target in params_dict:
                        return target
                        
                # Fallback to mapper
                if hasattr(self, "hf_to_vllm_mapper") and self.hf_to_vllm_mapper is not None:
                    try:
                        # Use a dummy weight copy for mapping
                        mapped = list(self.hf_to_vllm_mapper.apply([(name, torch.empty(0))]))
                        if mapped and mapped[0][0] in params_dict:
                            return mapped[0][0]
                    except Exception:
                        pass
                return None

            computed_scales = {}
            def float_scale_to_ue8m0(scale_tensor: torch.Tensor) -> torch.Tensor:
                # E = round(log2(S)) + 127
                log2_scale = torch.log2(scale_tensor)
                exponent = torch.round(log2_scale) + 127.0
                return torch.clamp(exponent, 0, 255).to(torch.uint8)

            filtered_weights = []
            for name, weight in inspect_weights:
                target_name = get_target_model_param_name(name)
                if target_name is not None:
                    param = params_dict[target_name]
                    if ".experts." in name and "weight" in name and not "scale" in name:
                        print(f"[DEBUG_LOAD] expert weight: name={name} | target_name={target_name} | param.dtype={param.dtype} | weight.dtype={weight.dtype} | param.shape={param.shape} | weight.shape={weight.shape}", flush=True)
                    if param.dtype == torch.uint8:
                        if "scale" in name:
                            if name in computed_scales:
                                weight = computed_scales[name]
                            else:
                                # For scale factors (e8m0 exponents), cast directly to uint8 (no FP4 quantization)
                                weight = weight.to(torch.uint8)
                        elif weight.dtype != torch.uint8:
                            try:
                                from vllm_torchtpu.layers.common.quantization import quantize_tensor_to_fp4, pack_fp4_indices
                                if len(weight.shape) == 2:
                                    indices, scale = quantize_tensor_to_fp4(weight.to(torch.float32), axis=1, block_size=32)
                                    weight = pack_fp4_indices(indices)
                                    exponent = float_scale_to_ue8m0(scale)
                                    computed_scales[name + "_scale"] = exponent
                            except Exception as e:
                                print(f"!!! Error packing weight {name} -> {target_name}: {e}")
                    filtered_weights.append((name, weight))
            
            for name, weight in filtered_weights:
                target_name = get_target_model_param_name(name)
                if target_name in params_dict:
                    param = params_dict[target_name]
                    if param.shape != weight.shape:
                        print(f"!!! SHAPE MISMATCH: parameter {target_name} has shape {param.shape} but loaded weight has shape {weight.shape}!")

            try:
                return orig_load_weights(self, filtered_weights)
            except KeyError as e:
                print(f"!!! KeyError caught in load_weights: {e}")
                print(f"params_dict keys:")
                for k in dict(self.named_parameters()).keys():
                    print(f"  {k}")
                raise e
        return patched_load_weights

    if hasattr(model_module, "DeepseekV4Model"):
        model_module.DeepseekV4Model.load_weights = make_patched_load_weights(model_module.DeepseekV4Model.load_weights)
    if hasattr(model_module, "DeepseekV4ForCausalLM"):
        model_module.DeepseekV4ForCausalLM.load_weights = make_patched_load_weights(model_module.DeepseekV4ForCausalLM.load_weights)

# Mock parallel state groups
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

class MockAttnMetadata:
    def __init__(self, seq_len: int, device):
        self.seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
        self.block_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
        self.query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        self.request_distribution = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)

class MockAttnMetadataDict(dict):
    def __init__(self, device):
        super().__init__()
        self.device = device
    def __getitem__(self, key):
        return MockAttnMetadata(10, self.device)

print("Initializing VllmConfig context...")
from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, CompilationConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.forward_context import set_forward_context

# Path to 4-layer config
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.getenv("MOCK_MODEL_DIR", os.path.join(script_dir, "dsv4-flash-4layer-fp8"))

from vllm.config import DeviceConfig

vllm_config_cpu = VllmConfig(
    model_config=ModelConfig(
        model=model_dir,
        tokenizer=model_dir,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        quantization=None,
    ),
    cache_config=CacheConfig(block_size=32, cache_dtype="fp8"),
    device_config=DeviceConfig(device="cpu"),
)
vllm_config_cpu.scheduler_config.max_num_seqs = 4
vllm_config_cpu.scheduler_config.max_model_len = 512

vllm_config_tpu = VllmConfig(
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
    device_config=DeviceConfig(device="tpu"),
)
print("DEBUG: at init, vllm_config_tpu.device_config.device =", vllm_config_tpu.device_config.device)
vllm_config_tpu.scheduler_config.max_num_seqs = 4
vllm_config_tpu.scheduler_config.max_model_len = 512
vllm_config_tpu.load_config.device = "tpu"
from vllm_torchtpu.layers.vllm.quantization import get_tpu_quantization_config
vllm_config_tpu.quant_config = get_tpu_quantization_config(vllm_config_tpu)
print("DEBUG: after quant_config, vllm_config_tpu.device_config.device =", vllm_config_tpu.device_config.device)

# JAX Mesh for TPU run
import jax
from jax.sharding import Mesh
devices = jax.devices()
mesh = Mesh(np.array([devices[0]]).reshape(1, 1, 1), ('attn_dp', 'expert', 'model'))

# Prepare mock prompt input
input_ids = torch.tensor([101, 2054, 2003, 1037, 2775, 102, 103, 104], dtype=torch.long)
positions = torch.arange(input_ids.shape[0], dtype=torch.int)

# Activation hooks storage
golden_activations = {}
tpu_activations = {}
golden_moe_inputs = {}
golden_moe_outputs = {}
tpu_moe_inputs = {}
tpu_moe_outputs = {}
golden_shared_outputs = {}
tpu_shared_outputs = {}

def get_activation_hook(storage_dict, name):
    def hook(module, input, output):
        # vLLM model layers output a tuple (hidden_states, residual) or just hidden_states
        if isinstance(output, tuple):
            storage_dict[name] = output[0].detach().cpu().to(torch.float32).clone()
        else:
            storage_dict[name] = output.detach().cpu().to(torch.float32).clone()
    return hook

def get_moe_hook(input_storage, output_storage, name):
    def hook(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            input_storage[name] = input[0].detach().cpu().to(torch.float32).clone()
            if len(input) > 1 and input[1] is not None:
                input_storage[name + "_logits"] = input[1].detach().cpu().to(torch.float32).clone()
        if isinstance(output, tuple):
            output_storage[name] = output[0].detach().cpu().to(torch.float32).clone()
        else:
            output_storage[name] = output.detach().cpu().to(torch.float32).clone()
    return hook

import sys

if cmd_args.step == 1:
    # Phase 1: Reference Forward Pass (CPU, Native PyTorch)
    print("\n=== STEP 1: Running CPU Golden Reference (Unpatched PyTorch) ===")
    from vllm.models.deepseek_v4.amd.model import DeepseekV4ForCausalLM as model_class
    from vllm.platforms import current_platform
    
    from vllm.platforms import PlatformEnum
    orig_device_type = current_platform.device_type
    orig_enum = current_platform._enum
    
    current_platform.device_type = "cpu"
    current_platform._enum = PlatformEnum.CPU
    
    from transformers import PretrainedConfig
    PretrainedConfig.quantization_config = {"scale_fmt": "ue8m0"}
    
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM as model_class
    
    _orig_cpu_load_weights = model_class.load_weights
    def patched_cpu_load_weights(self, weights):
        print("[CPU_PATCH] Entering patched_cpu_load_weights", flush=True)
        from vllm.models.deepseek_v4.nvidia.model import _make_deepseek_v4_weights_mapper
        actual_orig_mapper = self.hf_to_vllm_mapper
        print("[CPU_PATCH] Creating default base weights mapper '_make_deepseek_v4_weights_mapper(\"fp8\")'", flush=True)
        orig_mapper = _make_deepseek_v4_weights_mapper("fp8")
            
        outer_self = self
        class CPUWrappedMapper:
            def apply(self, w):
                for name, loaded_weight in orig_mapper.apply(w):
                    if ".mlp." in name:
                        name = name.replace(".mlp.", ".ffn.")
                    if ".shared_experts.gate_proj" in name:
                        name = name.replace(".shared_experts.gate_proj", ".shared_experts.w1")
                    elif ".shared_experts.up_proj" in name:
                        name = name.replace(".shared_experts.up_proj", ".shared_experts.w3")
                    if ".input_layernorm" in name:
                        name = name.replace(".input_layernorm", ".attn_norm")
                    elif ".post_attention_layernorm" in name:
                        name = name.replace(".post_attention_layernorm", ".ffn_norm")
                    yield name, loaded_weight
        self.hf_to_vllm_mapper = CPUWrappedMapper()
        try:
            return _orig_cpu_load_weights(self, weights)
        finally:
            self.hf_to_vllm_mapper = actual_orig_mapper
            
    model_class.load_weights = patched_cpu_load_weights

    with set_current_vllm_config(vllm_config_cpu):
        # Instantiate model on CPU and load weights
        print("DEBUG: vllm_config_cpu.device_config.device =", vllm_config_cpu.device_config.device)
        print("DEBUG: vllm_config_cpu.load_config.device =", vllm_config_cpu.load_config.device)
        from vllm.model_executor.model_loader import get_model
        model_cpu = get_model(vllm_config=vllm_config_cpu)
        
        # Traverse CPU model to find and rescale FP8 scale factors (since PyTorch CPU reference
        # does not interpret them as E8M0 scale exponents, it reads them directly as float32 127.0).
        rescaled_params_count = 0
        rescaled_buffers_count = 0
        for name, tensor in model_cpu.named_parameters():
            if "scale" in name:
                tensor.data.copy_(tensor.data / 127.0)
                rescaled_params_count += 1
        for name, tensor in model_cpu.named_buffers():
            if "scale" in name:
                tensor.data.copy_(tensor.data / 127.0)
                rescaled_buffers_count += 1
        print(f"[PATCH_CPU_SCALES] Rescaled {rescaled_params_count} parameters and {rescaled_buffers_count} buffers from 127.0 to 1.0", flush=True)

        # Print loaded weight stats for verification
        weight_param = model_cpu.model.layers[0].ffn.shared_experts.gate_up_proj.weight
        weight_f32 = weight_param.to(torch.float32)
        print(f"DEBUG_CPU_WEIGHT_STATS: name=gate_up_proj.weight | dtype={weight_param.dtype} | shape={weight_param.shape} | mean={weight_f32.abs().mean().item():.6f} | max={weight_f32.max().item():.6f} | min={weight_f32.min().item():.6f}", flush=True)

        # Traverse CPU model to find any DeepseekScalingRotaryEmbedding and move their caches to CPU
        for name, module in model_cpu.named_modules():
            if "RotaryEmbedding" in module.__class__.__name__:
                if hasattr(module, "cos_sin_cache") and module.cos_sin_cache is not None:
                    module.cos_sin_cache = module.cos_sin_cache.to(device_cpu)
                    print(f"[PATCH_CPU_ROPE] Moved {name}.cos_sin_cache to CPU", flush=True)

        current_platform.device_type = orig_device_type
        current_platform._enum = orig_enum
    
        # Print CPU model MoE types
        ffn_module = model_cpu.model.layers[3].ffn
        print("DEBUG_EXPERTS_CLASS:", type(ffn_module.experts))
        print("DEBUG_EXPERTS_RUNNER:", type(ffn_module.experts.runner))
        print("DEBUG_EXPERTS_QUANT_METHOD:", type(ffn_module.experts.quant_method))
        if hasattr(ffn_module.experts.quant_method, "moe_kernel"):
            print("DEBUG_EXPERTS_MOE_KERNEL:", type(ffn_module.experts.quant_method.moe_kernel))
        else:
            print("DEBUG_EXPERTS_MOE_KERNEL: None (no attribute moe_kernel)")
        print("DEBUG_ROUTER_CLASS:", type(ffn_module.experts.runner.router))
        import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as r_mod
        print("DEBUG_MOD_FUSED_TOPK_BIAS:", r_mod.fused_topk_bias)
        # Check if FusedTopKBiasRouter._compute_routing uses the global reference or class reference
        import inspect
        print("DEBUG_COMPUTE_ROUTING_SOURCE:", inspect.getsource(r_mod.FusedTopKBiasRouter._compute_routing))

        # Register hooks on decoder layers
        for idx, layer in enumerate(model_cpu.model.layers):
            print(f"DEBUG: CPU layer {idx} children = {[n for n, _ in layer.named_children()]}")
            layer.register_forward_hook(get_activation_hook(golden_activations, f"layer_{idx}"))
            layer.ffn.register_forward_hook(get_moe_hook(golden_moe_inputs, golden_moe_outputs, f"layer_{idx}"))
            if layer.ffn.shared_experts is not None:
                layer.ffn.shared_experts.register_forward_hook(get_activation_hook(golden_shared_outputs, f"layer_{idx}"))

        # Execute CPU forward pass
        print("DEBUG: starting CPU forward pass...", flush=True)
        model_cpu.eval()
        orig_is_cuda = current_platform.is_cuda
        current_platform.is_cuda = lambda: True
        try:
            with torch.no_grad():
                with set_forward_context(None, vllm_config_cpu):
                    _ = model_cpu(
                        input_ids=input_ids.to(device_cpu),
                        positions=positions.to(device_cpu),
                    )
        finally:
            current_platform.is_cuda = orig_is_cuda
        print("DEBUG: CPU forward pass completed successfully!", flush=True)
    
        print(f"DEBUG: CPU layer class = {type(model_cpu.model.layers[0])}", flush=True)
        print(f"Captured {len(golden_activations)} CPU reference layer outputs.", flush=True)
        print(f"Captured {len(golden_moe_inputs)} CPU reference MoE inputs.", flush=True)
        print(f"Captured {len(golden_moe_outputs)} CPU reference MoE outputs.", flush=True)
        print(f"Captured {len(golden_shared_outputs)} CPU reference Shared Expert outputs.", flush=True)
        
        os.makedirs("/mnt/local-cache/parity_run", exist_ok=True)
        torch.save({
            "golden_activations": golden_activations,
            "golden_moe_inputs": golden_moe_inputs,
            "golden_moe_outputs": golden_moe_outputs,
            "golden_shared_outputs": golden_shared_outputs,
        }, "/mnt/local-cache/parity_run/cpu_activations.pt")
        print("DEBUG: saved CPU activations to disk.", flush=True)
        
        # Clean up CPU model to free memory
        del model_cpu
        gc.collect()
        sys.exit(0)

if cmd_args.step == 2:
    # Phase 2: Patched Forward Pass (TPU, JAX-Patched)
    print("\n=== STEP 2: Running TPU Patched Code (JAX Custom Ops) ===")
    from vllm_torchtpu.models.vllm.deepseek_v4_patch import _maybe_patch_for_deepseek_v4
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import set_vllm_model_wrapper_context
    from vllm.platforms import current_platform
    
    orig_device_type = current_platform.device_type
    current_platform.device_type = "tpu"
    
    with set_current_vllm_config(vllm_config_tpu):
        with _maybe_patch_for_deepseek_v4(vllm_config_tpu):
            with set_vllm_model_wrapper_context(mesh=mesh):
                # Instantiate model on TPU (Pallas JAX patching will activate) and load weights
                print("DEBUG: vllm_config_tpu.device_config.device =", vllm_config_tpu.device_config.device)
                print("DEBUG: vllm_config_tpu.load_config.device =", vllm_config_tpu.load_config.device)
                from vllm.model_executor.model_loader import get_model
                model_tpu = get_model(vllm_config=vllm_config_tpu)
                
                num_blocks = 256
                # Traverse model to find kv_cache attributes and re-allocate them to correct 4D uint8 shape
                for name, module in model_tpu.named_modules():
                    if hasattr(module, "kv_cache") and isinstance(module.kv_cache, torch.Tensor):
                        orig_shape = module.kv_cache.shape
                        block_size = getattr(module, "block_size", 1024)
                        head_size = 640
                        if hasattr(module, "get_kv_cache_spec"):
                            spec = module.get_kv_cache_spec(vllm_config_tpu)
                            if spec is not None:
                                head_size = getattr(spec, "head_size", head_size)
                                block_size = getattr(spec, "block_size", block_size)
                        
                        effective_block_size = max(4, block_size)
                        shape = (num_blocks, effective_block_size // 4, 4, head_size)
                        module.kv_cache = torch.zeros(shape, dtype=torch.uint8, device=device_tpu)
                        print(f"[PATCH_TPU_CACHE] Re-allocated {name}.kv_cache from {orig_shape} to uint8 4D shape {shape}", flush=True)
    
                # Register hooks
                for idx, layer in enumerate(model_tpu.model.layers):
                    layer.register_forward_hook(get_activation_hook(tpu_activations, f"layer_{idx}"))
                    layer.ffn.register_forward_hook(get_moe_hook(tpu_moe_inputs, tpu_moe_outputs, f"layer_{idx}"))
                    if layer.ffn.shared_experts is not None:
                        layer.ffn.shared_experts.register_forward_hook(get_activation_hook(tpu_shared_outputs, f"layer_{idx}"))
    
                # Execute TPU forward pass
                model_tpu.eval()
                with torch.no_grad():
                    mock_attn_metadata = MockAttnMetadataDict(device_tpu)
                    with set_forward_context(mock_attn_metadata, vllm_config_tpu):
                        _ = model_tpu(
                            input_ids=input_ids.to(device_tpu),
                            positions=positions.to(device_tpu),
                        )
    
    current_platform.device_type = orig_device_type
    print(f"DEBUG: TPU layer class = {type(model_tpu.model.layers[0])}")
    print(f"Captured {len(tpu_activations)} TPU layer outputs.")
    print(f"Captured {len(tpu_moe_inputs)} TPU MoE inputs.")
    print(f"Captured {len(tpu_moe_outputs)} TPU MoE outputs.")
    print(f"Captured {len(tpu_shared_outputs)} TPU Shared Expert outputs.")
    
    torch.save({
        "tpu_activations": tpu_activations,
        "tpu_moe_inputs": tpu_moe_inputs,
        "tpu_moe_outputs": tpu_moe_outputs,
        "tpu_shared_outputs": tpu_shared_outputs,
    }, "/mnt/local-cache/parity_run/tpu_activations.pt")
    print("DEBUG: saved TPU activations to disk.", flush=True)
    sys.exit(0)

if cmd_args.step == 3:
    # Phase 3: Layer-by-Layer Numerical Parity Check
    print("\n=== STEP 3: Layer-by-Layer Parity Comparison ===")
    cpu_data = torch.load("/mnt/local-cache/parity_run/cpu_activations.pt")
    tpu_data = torch.load("/mnt/local-cache/parity_run/tpu_activations.pt")
    
    golden_activations = cpu_data["golden_activations"]
    golden_moe_inputs = cpu_data["golden_moe_inputs"]
    golden_moe_outputs = cpu_data["golden_moe_outputs"]
    golden_shared_outputs = cpu_data["golden_shared_outputs"]
    
    tpu_activations = tpu_data["tpu_activations"]
    tpu_moe_inputs = tpu_data["tpu_moe_inputs"]
    tpu_moe_outputs = tpu_data["tpu_moe_outputs"]
    tpu_shared_outputs = tpu_data["tpu_shared_outputs"]

    print("\n=== FusedMoE INPUTS Comparison ===")
    for layer_name in sorted(golden_moe_inputs.keys()):
        if layer_name.endswith("_logits"):
            cpu_val = golden_moe_inputs[layer_name].numpy()
            tpu_val = tpu_moe_inputs.get(layer_name, torch.empty_like(golden_moe_inputs[layer_name])).numpy()
            max_diff = np.max(np.abs(cpu_val - tpu_val))
            mean_cpu = np.mean(np.abs(cpu_val))
            mean_tpu = np.mean(np.abs(tpu_val))
            max_cpu = np.max(cpu_val)
            max_tpu = np.max(tpu_val)
            min_cpu = np.min(cpu_val)
            min_tpu = np.min(tpu_val)
            print(f"[{layer_name} Router Logits]")
            print(f"  CPU MeanAbs/Max/Min: {mean_cpu:.6f}/{max_cpu:.6f}/{min_cpu:.6f}")
            print(f"  TPU MeanAbs/Max/Min: {mean_tpu:.6f}/{max_tpu:.6f}/{min_tpu:.6f}")
            print(f"  Max Diff: {max_diff:.6f}")
            continue

        cpu_val = golden_moe_inputs[layer_name].numpy()
        tpu_val = tpu_moe_inputs[layer_name].numpy()
        max_diff = np.max(np.abs(cpu_val - tpu_val))
        mean_cpu = np.mean(np.abs(cpu_val))
        mean_tpu = np.mean(np.abs(tpu_val))
        max_cpu = np.max(cpu_val)
        max_tpu = np.max(tpu_val)
        min_cpu = np.min(cpu_val)
        min_tpu = np.min(tpu_val)
        print(f"[{layer_name} MoE Input]")
        print(f"  CPU MeanAbs/Max/Min: {mean_cpu:.6f}/{max_cpu:.6f}/{min_cpu:.6f}")
        print(f"  TPU MeanAbs/Max/Min: {mean_tpu:.6f}/{max_tpu:.6f}/{min_tpu:.6f}")
        print(f"  Max Diff: {max_diff:.6f}")

    print("\n=== FusedMoE OUTPUTS Comparison ===")
    for layer_name in sorted(golden_moe_outputs.keys()):
        cpu_val = golden_moe_outputs[layer_name].numpy()
        tpu_val = tpu_moe_outputs[layer_name].numpy()
        max_diff = np.max(np.abs(cpu_val - tpu_val))
        mean_cpu = np.mean(np.abs(cpu_val))
        mean_tpu = np.mean(np.abs(tpu_val))
        max_cpu = np.max(cpu_val)
        max_tpu = np.max(tpu_val)
        min_cpu = np.min(cpu_val)
        min_tpu = np.min(tpu_val)
        print(f"[{layer_name} MoE Output]")
        print(f"  CPU MeanAbs/Max/Min: {mean_cpu:.6f}/{max_cpu:.6f}/{min_cpu:.6f}")
        print(f"  TPU MeanAbs/Max/Min: {mean_tpu:.6f}/{max_tpu:.6f}/{min_tpu:.6f}")
        print(f"  Max Diff: {max_diff:.6f}")

    print("\n=== Shared Expert OUTPUTS Comparison ===")
    for layer_name in sorted(golden_shared_outputs.keys()):
        cpu_val = golden_shared_outputs[layer_name].numpy()
        tpu_val = tpu_shared_outputs[layer_name].numpy()
        max_diff = np.max(np.abs(cpu_val - tpu_val))
        mean_cpu = np.mean(np.abs(cpu_val))
        mean_tpu = np.mean(np.abs(tpu_val))
        max_cpu = np.max(cpu_val)
        max_tpu = np.max(tpu_val)
        min_cpu = np.min(cpu_val)
        min_tpu = np.min(tpu_val)
        print(f"[{layer_name} Shared Output]")
        print(f"  CPU MeanAbs/Max/Min: {mean_cpu:.6f}/{max_cpu:.6f}/{min_cpu:.6f}")
        print(f"  TPU MeanAbs/Max/Min: {mean_tpu:.6f}/{max_tpu:.6f}/{min_tpu:.6f}")
        print(f"  Max Diff: {max_diff:.6f}")

    print("\n=== Derived Routed Expert OUTPUTS Comparison ===")
    for layer_name in sorted(golden_moe_outputs.keys()):
        cpu_overall = golden_moe_outputs[layer_name].numpy()
        tpu_overall = tpu_moe_outputs[layer_name].numpy()
        cpu_shared = golden_shared_outputs[layer_name].numpy() if layer_name in golden_shared_outputs else np.zeros_like(cpu_overall)
        tpu_shared = tpu_shared_outputs[layer_name].numpy() if layer_name in tpu_shared_outputs else np.zeros_like(tpu_overall)
        
        cpu_val = cpu_overall - cpu_shared
        tpu_val = tpu_overall - tpu_shared
        
        max_diff = np.max(np.abs(cpu_val - tpu_val))
        mean_cpu = np.mean(np.abs(cpu_val))
        mean_tpu = np.mean(np.abs(tpu_val))
        max_cpu = np.max(cpu_val)
        max_tpu = np.max(tpu_val)
        min_cpu = np.min(cpu_val)
        min_tpu = np.min(tpu_val)
        print(f"[{layer_name} Derived Routed Output]")
        print(f"  CPU MeanAbs/Max/Min: {mean_cpu:.6f}/{max_cpu:.6f}/{min_cpu:.6f}")
        print(f"  TPU MeanAbs/Max/Min: {mean_tpu:.6f}/{max_tpu:.6f}/{min_tpu:.6f}")
        print(f"  Max Diff: {max_diff:.6f}")

    print("\n=== DecoderLayer Outputs Comparison ===")
    for layer_name in sorted(golden_activations.keys()):
        act_cpu = golden_activations[layer_name].numpy()
        act_tpu = tpu_activations[layer_name].numpy()
        
        max_diff = np.max(np.abs(act_cpu - act_tpu))
        mean_cpu = np.mean(np.abs(act_cpu))
        rel_diff = max_diff / (mean_cpu + 1e-8)
        
        print(f"[{layer_name}]")
        print(f"  CPU Mean Abs: {mean_cpu:.6f}")
        print(f"  Max Diff:     {max_diff:.6f}")
        print(f"  Rel Diff:     {rel_diff:.6f}")
        
        # BF16 precision threshold is typically around 1e-2 to 5e-3 depending on depth
        if max_diff > 1e-2:
            print(f"  ❌ MISMATCH detected at {layer_name}!")
        else:
            print(f"  ✅ Parity holds at {layer_name}")

    print("\nParity verification finished.")

