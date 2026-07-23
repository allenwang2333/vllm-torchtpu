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
"""Context manager to patch DeepSeek V4 model registry and classes during loading."""

from contextlib import contextmanager
import torch
from vllm.config import VllmConfig
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

import sys
import os
print("[PATCH] Top-level import of deepseek_v4_patch.py", file=sys.stderr, flush=True)

# Register collectives in Dynamo allow_in_graph to bypass tracing errors
try:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "allow_in_graph"):
        if hasattr(torch.ops, "_c10d_functional"):
            registered_ops = []
            for attr_name in dir(torch.ops._c10d_functional):
                try:
                    attr = getattr(torch.ops._c10d_functional, attr_name)
                    if callable(attr):
                        torch.compiler.allow_in_graph(attr)
                        registered_ops.append(attr_name)
                except Exception:
                    pass
            print(f"[PATCH] Registered functional collectives in Dynamo: {registered_ops}", file=sys.stderr, flush=True)
except Exception as e:
    print(f"[PATCH] Failed to register collectives in Dynamo: {e}", file=sys.stderr, flush=True)


def e8m0_to_fp32(u8: torch.Tensor) -> torch.Tensor:
    exponents = u8.to(torch.int32) - 127
    ones = torch.ones_like(u8, dtype=torch.float32)
    return torch.ldexp(ones, exponents)

def re_quantize_attention_weight(weight_fp8: torch.Tensor, scale_block_fp8: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if scale_block_fp8.dtype == torch.float8_e8m0fnu:
        scale_block_fp8 = scale_block_fp8.view(torch.uint8)
    scale_block_fp32 = e8m0_to_fp32(scale_block_fp8)
    scale_expanded = scale_block_fp32.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    weight_fp32 = weight_fp8.to(torch.float32) * scale_expanded
    
    max_fp8_value = 448.0
    row_max = torch.max(torch.abs(weight_fp32), dim=1, keepdim=True).values
    scale_per_channel = torch.clamp(row_max / max_fp8_value, min=1e-12)
    weight_quantized = torch.clamp(weight_fp32 / scale_per_channel, -max_fp8_value, max_fp8_value).to(torch.float8_e4m3fn)
    return weight_quantized, scale_per_channel

def attention_requantizer_generator(weights_iter):
    buffered = {}
    
    def is_fp8_projection_to_requantize(name: str) -> bool:
        if ".attn." in name:
            for proj in ["wq_a", "wq_b", "wo_a", "wo_b", "wkv", "indexer.wq_b"]:
                if f".attn.{proj}." in name:
                    return True
        if ".shared_experts." in name:
            for proj in ["w1", "w2", "w3"]:
                if f".shared_experts.{proj}." in name:
                    return True
        return False

    for name, loaded_weight in weights_iter:
        if is_fp8_projection_to_requantize(name):
            if name.endswith(".weight"):
                base_key = name[:-7]
                target_type = "weight"
            elif name.endswith(".scale"):
                base_key = name[:-6]
                target_type = "scale"
            elif name.endswith(".weight_scale"):
                base_key = name[:-13]
                target_type = "scale"
            elif name.endswith(".weight_scale_inv"):
                base_key = name[:-17]
                target_type = "scale"
            else:
                yield name, loaded_weight
                continue
                
            entry = buffered.setdefault(base_key, {})
            entry[target_type] = loaded_weight
            
            if "weight" in entry and "scale" in entry:
                w_new, s_new = re_quantize_attention_weight(entry["weight"], entry["scale"])
                yield f"{base_key}.weight", w_new
                yield f"{base_key}.scale", s_new
                del buffered[base_key]
        else:
            yield name, loaded_weight
            
    if buffered:
        print(f"[PATCH ERROR] Remaining unmatched buffered layers in weight loader: {list(buffered.keys())}", file=sys.stderr, flush=True)



@contextmanager
def _maybe_patch_for_deepseek_v4(vllm_config: VllmConfig):
    architectures = getattr(vllm_config.model_config.hf_config,
                            "architectures", None) or []
    print(f"[PATCH] Patcher checking architectures: {architectures}", file=sys.stderr, flush=True)
    is_ds_v4 = any("DeepseekV4ForCausalLM" in arch for arch in architectures)
    if not is_ds_v4:
        yield
        return

    logger.info("Applying DeepSeek V4 TPU model patches...")

    # Bypasses Dynamo json serialization bug during metrics logging
    try:
        import torch._dynamo.utils
        torch._dynamo.utils._get_dynamo_config_for_logging = lambda: {}
    except ImportError:
        pass

    # Patch vllm.distributed.parallel_state to reflect the TPU sharding strategy (TP vs EP)
    import vllm.distributed.parallel_state as ps
    import vllm.model_executor.layers.fused_moe.config as fmc
    from vllm_torchtpu.layers.common.sharding import ShardingConfigManager

    sharding_config = ShardingConfigManager.get(vllm_config)
    tp_size = max(1, sharding_config.tp_size)
    ep_size = max(1, sharding_config.expert_size)

    _orig_get_tp_world_size = ps.get_tensor_model_parallel_world_size
    _orig_get_tp_rank = ps.get_tensor_model_parallel_rank
    _orig_make = fmc.FusedMoEParallelConfig.make

    ps.get_tensor_model_parallel_world_size = lambda: tp_size
    ps.get_tensor_model_parallel_rank = lambda: torch.distributed.get_rank() % tp_size

    def patched_make(*args, **kwargs):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else -1
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if ep_size > 1:
            moe_tp_size = max(1, world_size // ep_size)
            ret_rank = rank // moe_tp_size if rank >= 0 else 0
            print(f"[DEBUG_MAKE] Rank {rank} (world_size={world_size}, dist_init={dist.is_initialized()}) | ep_size={ep_size} (moe_tp_size={moe_tp_size}) -> returning ep_rank={ret_rank}", flush=True)
            return fmc.FusedMoEParallelConfig(
                tp_size=1,
                tp_rank=0,
                pcp_size=1,
                pcp_rank=0,
                dp_size=1,
                dp_rank=0,
                ep_size=ep_size,
                ep_rank=ret_rank,
                sp_size=1,
                use_ep=True,
                all2all_backend="allgather_reducescatter",
                enable_eplb=False,
            )
        return _orig_make(*args, **kwargs)

    fmc.FusedMoEParallelConfig.make = patched_make

    # Patch MoERunner._apply_quant_method to inspect the execution path and quant method class
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    _orig_apply_quant_method = MoERunner._apply_quant_method

    def patched_apply_quant_method(self, layer, hidden_states, router_logits, shared_experts_input, input_ids=None):
        layer_idx = getattr(layer, "layer_idx", -1)
        if layer_idx in (0, 3, 4):
            hs_cpu = hidden_states.cpu().float()
            rl_cpu = router_logits.cpu().float()
            print(f"[DEBUG_PATCH] layer_{layer_idx} input hidden_states: shape={list(hs_cpu.shape)} dtype={hidden_states.dtype} | mean={hs_cpu.mean().item():.6f} std={hs_cpu.std().item():.6f} min={hs_cpu.min().item():.6f} max={hs_cpu.max().item():.6f}", flush=True)
            print(f"[DEBUG_PATCH] layer_{layer_idx} input router_logits: shape={list(rl_cpu.shape)} dtype={router_logits.dtype} | mean={rl_cpu.mean().item():.6f} std={rl_cpu.std().item():.6f} min={rl_cpu.min().item():.6f} max={rl_cpu.max().item():.6f}", flush=True)
        return _orig_apply_quant_method(self, layer, hidden_states, router_logits, shared_experts_input, input_ids)

    MoERunner._apply_quant_method = patched_apply_quant_method

    # 1. Patch the model registry to resolve to the AMD (ROCm) variant.
    # The AMD variant is more extensible and allows us to swap attention/MHC classes.
    import vllm.models.deepseek_v4 as ds_v4
    def patch_load_weights_for_class(cls):
        _orig_load_weights = cls.load_weights
        
        def patched_load_weights(self, weights):
            print(f"[PATCH] Entering patched_load_weights on {self.__class__.__name__}", file=sys.stderr, flush=True)
            params_keys = sorted(list(dict(self.named_parameters()).keys()))
            print(f"[PATCH] Model parameters count: {len(params_keys)}", file=sys.stderr, flush=True)
            for k in params_keys:
                if "layers.0.attn.wo_b" in k:
                    param = dict(self.named_parameters())[k]
                    print(f"[PATCH]   Param: {k} | Shape: {list(param.shape)} | Type: {param.__class__.__name__}", file=sys.stderr, flush=True)
            self.model.has_tilelang = False
            for idx, l in enumerate(self.model.layers):
                l.layer_idx = idx
                l.has_tilelang = False
                if hasattr(l, "ffn"):
                    l.ffn.layer_idx = idx
                    if hasattr(l.ffn, "experts"):
                        l.ffn.experts.layer_idx = idx
            quant_config = getattr(vllm_config, "quant_config", None)
            expert_dtype = getattr(quant_config, "expert_dtype", "fp8")
            is_fp4_experts = (expert_dtype == "fp4")
            orig_mapper = self.hf_to_vllm_mapper
            
            outer_self = self
            class WrappedMapper:
                def apply(self, w):
                    for name, loaded_weight in orig_mapper.apply(w):
                        if "layers.0.attn" in name:
                            print(f"[PATCH] Mapping: {name} shape={list(loaded_weight.shape)} dtype={loaded_weight.dtype}", file=sys.stderr, flush=True)
                        if not is_fp4_experts and ".experts." in name and name.endswith("weight_scale"):
                            name = name + "_inv"
                        if "hc_head.hc_" in name:
                            name = name.replace("hc_head.hc_", "hc_head_")
                            
                        # Map shared experts from HF (gate_proj/up_proj/down_proj) to vLLM (w1/w3/w2)
                        # so they are matched by stacked_params_mapping and loaded by the normal loader
                        if ".shared_experts.gate_proj" in name:
                            name = name.replace(".shared_experts.gate_proj", ".shared_experts.w1")
                        elif ".shared_experts.up_proj" in name:
                            name = name.replace(".shared_experts.up_proj", ".shared_experts.w3")
                            
                        if "layers." in name:
                            try:
                                layer_idx = int(name.split("layers.")[1].split(".")[0])
                                if layer_idx >= outer_self.config.num_hidden_layers:
                                    continue
                            except (ValueError, IndexError):
                                pass
                        # Map MHC layer parameters from HF (nested) to vLLM (flat)
                        if ".attn_hc.base" in name:
                            name = name.replace(".attn_hc.base", ".hc_attn_base")
                        elif ".attn_hc.fn" in name:
                            name = name.replace(".attn_hc.fn", ".hc_attn_fn")
                        elif ".attn_hc.weight_scale_inv" in name:
                            name = name.replace(".attn_hc.weight_scale_inv", ".hc_attn_scale")
                        elif ".ffn_hc.base" in name:
                            name = name.replace(".ffn_hc.base", ".hc_ffn_base")
                        elif ".ffn_hc.fn" in name:
                            name = name.replace(".ffn_hc.fn", ".hc_ffn_fn")
                        elif ".ffn_hc.weight_scale_inv" in name:
                            name = name.replace(".ffn_hc.weight_scale_inv", ".hc_ffn_scale")
                            
                        # Map layernorms
                        if ".input_layernorm" in name:
                            name = name.replace(".input_layernorm", ".attn_norm")
                        elif ".post_attention_layernorm" in name:
                            name = name.replace(".post_attention_layernorm", ".ffn_norm")
                        
                        # Map mlp to ffn
                        if ".mlp." in name:
                            name = name.replace(".mlp.", ".ffn.")
                            
                        # Map attention parameters
                        # 1. Indexer Compressor mappings (HF -> vLLM)
                        if ".self_attn.compressor.indexer.gate_proj" in name:
                            name = name.replace(".self_attn.compressor.indexer.gate_proj", ".attn.indexer.compressor.wgate")
                        elif ".self_attn.compressor.indexer.kv_proj" in name:
                            name = name.replace(".self_attn.compressor.indexer.kv_proj", ".attn.indexer.compressor.wkv")
                        elif ".self_attn.compressor.indexer.kv_norm" in name:
                            name = name.replace(".self_attn.compressor.indexer.kv_norm", ".attn.indexer.compressor.norm")
                        elif ".self_attn.compressor.indexer.position_bias" in name:
                            name = name.replace(".self_attn.compressor.indexer.position_bias", ".attn.indexer.compressor.ape")
                            
                        # 2. Indexer main mappings (HF -> vLLM)
                        elif ".self_attn.compressor.indexer.q_b_proj" in name:
                            name = name.replace(".self_attn.compressor.indexer.q_b_proj", ".attn.indexer.wq_b")
                        elif ".self_attn.compressor.indexer.scorer.weights_proj" in name:
                            name = name.replace(".self_attn.compressor.indexer.scorer.weights_proj", ".attn.indexer.weights_proj")
                            
                        # 3. Attention Compressor mappings (HF -> vLLM)
                        elif ".self_attn.compressor.gate_proj" in name:
                            name = name.replace(".self_attn.compressor.gate_proj", ".attn.mla_attn.compressor.wgate")
                        elif ".self_attn.compressor.kv_proj" in name:
                            name = name.replace(".self_attn.compressor.kv_proj", ".attn.mla_attn.compressor.wkv")
                        elif ".self_attn.compressor.kv_norm" in name:
                            name = name.replace(".self_attn.compressor.kv_norm", ".attn.mla_attn.compressor.norm")
                        elif ".self_attn.compressor.position_bias" in name:
                            name = name.replace(".self_attn.compressor.position_bias", ".attn.mla_attn.compressor.ape")
                        elif ".self_attn.q_a_proj" in name:
                            name = name.replace(".self_attn.q_a_proj", ".attn.wq_a")
                        elif ".self_attn.q_a_norm" in name:
                            name = name.replace(".self_attn.q_a_norm", ".attn.q_norm")
                        elif ".self_attn.q_b_proj" in name:
                            name = name.replace(".self_attn.q_b_proj", ".attn.wq_b")
                        elif ".self_attn.kv_proj" in name:
                            name = name.replace(".self_attn.kv_proj", ".attn.wkv")
                        elif ".self_attn.kv_norm" in name:
                            name = name.replace(".self_attn.kv_norm", ".attn.kv_norm")
                        elif ".self_attn.o_a_proj" in name:
                            name = name.replace(".self_attn.o_a_proj", ".attn.wo_a")
                        elif ".self_attn.o_b_proj" in name:
                            name = name.replace(".self_attn.o_b_proj", ".attn.wo_b")
                        elif ".self_attn.sinks" in name:
                            name = name.replace(".self_attn.sinks", ".attn.attn_sink")

                        if ".experts." not in name and name.endswith("weight_scale_inv"):
                            name = name.replace("weight_scale_inv", "weight_scale")

                        yield name, loaded_weight
                        
            # Check if all expert layers have been pre-quantized and cached on disk.
            # If so, we can skip loading the heavy expert weights from safetensors to save CPU memory and time.
            has_all_cached = False
            cache_dir = os.environ.get("VLLM_FP4_CACHE_DIR", "")
            if cache_dir and os.path.exists(cache_dir):
                import re
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
                cached_files = os.listdir(cache_dir)
                # Find cached files for this rank
                pattern = re.compile(rf"rank_{rank}_layer_(\d+).*\.pt")
                layers_found = set()
                for f in cached_files:
                    m = pattern.match(f)
                    if m:
                        layers_found.add(int(m.group(1)))
                # Check layers 0 to num_layers-1
                has_all_cached = True
                for i in range(self.config.num_hidden_layers):
                    if hasattr(self.model.layers[i], "ffn") and hasattr(self.model.layers[i].ffn, "experts"):
                        if i not in layers_found:
                            has_all_cached = False
                            break
                            
            if has_all_cached:
                logger.warning("[PATCH] All expert weights are cached. Skipping loading expert weights from safetensors checkpoint!")
                def filter_expert_weights(weights_iter):
                    for name, loaded_weight in weights_iter:
                        if ".experts." in name:
                            continue
                        yield name, loaded_weight
                weights = filter_expert_weights(weights)
            else:
                logger.warning("[PATCH] Cached expert weights not found or incomplete. Loading expert weights from safetensors checkpoint to generate cache.")

            self.hf_to_vllm_mapper = WrappedMapper()
            try:
                print(f"[PATCH] Model state_dict keys related to layers.0.attn: {[k for k in self.model.state_dict().keys() if 'layers.0.attn.' in k]}", file=sys.stderr, flush=True)
                # Update TP status of all parameters since new ones (like weight_scale)
                # are registered during weight creation which runs AFTER linear layer __init__.
                for name, module in self.model.named_modules():
                    if hasattr(module, "update_param_tp_status"):
                        module.update_param_tp_status()
                weights = attention_requantizer_generator(weights)
                def debug_weights_generator(weights_iter):
                    for name, loaded_weight in weights_iter:
                        if "layers.0.attn" in name:
                            print(f"[PATCH] Passing to load_weights: {name} shape={list(loaded_weight.shape)} dtype={loaded_weight.dtype}", file=sys.stderr, flush=True)
                        yield name, loaded_weight
                res = _orig_load_weights(self, debug_weights_generator(weights))
                if hasattr(self.model, "layers") and len(self.model.layers) > 0 and hasattr(self.model.layers[0], "attn") and hasattr(self.model.layers[0].attn, "wo_b") and hasattr(self.model.layers[0].attn.wo_b, "weight"):
                    print(f"[PATCH] POST_LOAD: layers.0.attn.wo_b.weight Shape: {list(self.model.layers[0].attn.wo_b.weight.shape)} | Type: {self.model.layers[0].attn.wo_b.weight.__class__.__name__}", file=sys.stderr, flush=True)

                
                # Manually invoke process_weights_after_loading on all FusedMoE layers
                # because the standard vLLM loader checks hasattr(module, "quant_method")
                # which returns False on FusedMoE layers (since it uses _quant_method instead)
                for idx, l in enumerate(self.model.layers):
                    if hasattr(l, "ffn") and hasattr(l.ffn, "experts"):
                        moe_layer = l.ffn.experts
                        if hasattr(moe_layer, "_quant_method"):
                            if hasattr(moe_layer._quant_method, "process_weights_after_loading"):
                                print(f"[PATCH] Manually invoking process_weights_after_loading on layer {idx} FusedMoE", flush=True)
                                moe_layer._quant_method.process_weights_after_loading(moe_layer)
                


                import torch.distributed as dist
                if dist.is_initialized():
                    logger.info("Aligning workers at barrier after _orig_load_weights")
                    dist.barrier()

                from torch_tpu._internal import sync
                print("[DEBUG_LOAD] Calling sync.synchronize(wait=True) and gc.collect() to finalize parameter materialization and free HBM...", flush=True)
                sync.synchronize(wait=True)
                import gc
                gc.collect()

                return res
            finally:
                self.hf_to_vllm_mapper = orig_mapper

        cls.load_weights = patched_load_weights

    # Patch custom Triton operations with PyTorch equivalents on TPU
    try:
        import vllm.models.deepseek_v4.attention as ds_attn
        import vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm as fused_qk_ops
        import vllm.models.deepseek_v4.common.ops.fused_indexer_q as fused_indexer_ops
        import vllm.models.deepseek_v4.common.ops as common_ops

        def fused_q_kv_rmsnorm_pytorch(
            qr: torch.Tensor,
            kv: torch.Tensor,
            q_weight: torch.Tensor,
            kv_weight: torch.Tensor,
            eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Standard PyTorch equivalent of GPU Triton RMSNorm kernel
            qr_out = qr * torch.rsqrt(qr.pow(2).mean(-1, keepdim=True) + eps) * q_weight
            kv_out = kv * torch.rsqrt(kv.pow(2).mean(-1, keepdim=True) + eps) * kv_weight
            return qr_out, kv_out

        def fused_indexer_q_rope_quant_pytorch(
            positions: torch.Tensor,
            index_q: torch.Tensor,
            index_q_cos_sin_cache: torch.Tensor,
            index_weights: torch.Tensor,
            index_weights_softmax_scale: float,
            index_weights_head_scale: float,
            use_fp4: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert not use_fp4, "FP4 path not supported on TPU"
            num_tokens = positions.shape[0]
            num_index_q_heads = index_q.shape[1]
            index_q_head_dim = index_q.shape[2]

            cos_sin = index_q_cos_sin_cache[positions]
            rot_dim = cos_sin.shape[-1]
            half_rot_dim = rot_dim // 2
            cos = cos_sin[:, :half_rot_dim]
            sin = cos_sin[:, half_rot_dim:]

            nope_dim = index_q_head_dim - rot_dim
            q_nope = index_q[..., :nope_dim]
            q_rot = index_q[..., nope_dim:]

            q_rot_reshaped = q_rot.view(num_tokens, num_index_q_heads, half_rot_dim, 2)
            x_even = q_rot_reshaped[..., 0]
            x_odd = q_rot_reshaped[..., 1]

            cos_expanded = cos.unsqueeze(1)
            sin_expanded = sin.unsqueeze(1)

            r_even = x_even * cos_expanded - x_odd * sin_expanded
            r_odd = x_odd * cos_expanded + x_even * sin_expanded

            r_even = r_even.to(torch.bfloat16).to(torch.float32)
            r_odd = r_odd.to(torch.bfloat16).to(torch.float32)

            r_stacked = torch.stack([r_even, r_odd], dim=-1)
            q_rot_out = r_stacked.view(num_tokens, num_index_q_heads, rot_dim)

            q_rot_final = torch.cat([q_nope, q_rot_out], dim=-1)

            amax = torch.max(torch.abs(q_rot_final), dim=-1).values
            scale = torch.clamp(amax, min=1e-4) / 448.0
            scale = torch.exp2(torch.ceil(torch.log2(scale))).unsqueeze(-1)

            index_q_fp8 = torch.clamp(q_rot_final / scale, -448.0, 448.0).to(torch.float8_e4m3fn)

            scale_squeezed = scale.squeeze(-1)
            index_weights_out = index_weights * scale_squeezed * index_weights_softmax_scale * index_weights_head_scale

            return index_q_fp8, index_weights_out

        # Overwrite all imports/locations of both functions
        ds_attn.fused_q_kv_rmsnorm = fused_q_kv_rmsnorm_pytorch
        fused_qk_ops.fused_q_kv_rmsnorm = fused_q_kv_rmsnorm_pytorch
        common_ops.fused_q_kv_rmsnorm = fused_q_kv_rmsnorm_pytorch

        ds_attn.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_pytorch
        fused_indexer_ops.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_pytorch
        common_ops.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_pytorch

        print("[PATCH] Successfully monkey-patched fused_q_kv_rmsnorm and fused_indexer_q_rope_quant with PyTorch equivalents.", flush=True)
    except Exception as e:
        print(f"[PATCH] Failed to patch Triton custom ops: {e}", file=sys.stderr, flush=True)

    # Capture and patch the standard CUDA/CPU variant from ds_v4
    try:
        _CudaDeepseekV4ForCausalLM = ds_v4.DeepseekV4ForCausalLM
        patch_load_weights_for_class(_CudaDeepseekV4ForCausalLM)
    except Exception as e:
        print(f"Failed to patch CUDA variant: {e}", file=sys.stderr)

    # Patch the AMD (ROCm) variant and register it
    _AmdDeepseekV4ForCausalLM = None
    try:
        from vllm.models.deepseek_v4.amd.model import DeepseekV4ForCausalLM as _AmdDeepseekV4ForCausalLM
        ds_v4.DeepseekV4ForCausalLM = _AmdDeepseekV4ForCausalLM
        patch_load_weights_for_class(_AmdDeepseekV4ForCausalLM)
    except Exception as e:
        print(f"Failed to patch AMD variant: {e}", file=sys.stderr)

    if _AmdDeepseekV4ForCausalLM is not None:
        _orig_amd_forward = _AmdDeepseekV4ForCausalLM.forward

        def patched_amd_forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors=None,
            inputs_embeds=None,
            *args,
            **kwargs
        ):
            if input_ids is not None:
                print(f"[DEBUG_FORWARD_INPUT] pid={os.getpid()} input_ids shape={list(input_ids.shape)} values={input_ids.tolist()}", flush=True)
            if os.environ.get("VLLM_BYPASS_TPU_INIT") == "1":
                import torchax
                import torchax.tensor as ta_tensor
                import jax
                with torchax.default_env() as env:
                    def to_torchax(t):
                        if isinstance(t, torch.Tensor) and not hasattr(t, "_elem"):
                            orig_dtype = t.dtype
                            # Check if the dtype requires float32 casting for numpy compatibility
                            if orig_dtype in (torch.bfloat16, getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
                                np_arr = t.detach().cpu().to(torch.float32).numpy()
                                jax_arr = jax.device_put(np_arr)
                                if orig_dtype == torch.bfloat16:
                                    jax_arr = jax_arr.astype(jax.numpy.bfloat16)
                                elif orig_dtype == getattr(torch, "float8_e4m3fn", None):
                                    jax_arr = jax_arr.astype(jax.numpy.float8_e4m3fn)
                                elif orig_dtype == getattr(torch, "float8_e5m2", None):
                                    jax_arr = jax_arr.astype(jax.numpy.float8_e5m2)
                            else:
                                jax_arr = jax.device_put(t.detach().cpu().numpy())
                            return ta_tensor.Tensor(jax_arr, env)
                        return t

                    # Convert all model parameters and buffers to torchax.Tensor
                    def convert_module_params(module):
                        for name in list(module._parameters.keys()):
                            param = module._parameters[name]
                            if param is not None and not hasattr(param, "_elem"):
                                wrapper = to_torchax(param)
                                del module._parameters[name]
                                setattr(module, name, wrapper)

                        for name in list(module._buffers.keys()):
                            buf = module._buffers[name]
                            if buf is not None and not hasattr(buf, "_elem"):
                                wrapper = to_torchax(buf)
                                del module._buffers[name]
                                setattr(module, name, wrapper)

                        for child in module.children():
                            convert_module_params(child)

                    convert_module_params(self)

                    input_ids_ta = to_torchax(input_ids)
                    positions_ta = to_torchax(positions)
                    inputs_embeds_ta = to_torchax(inputs_embeds)
                    args_ta = tuple(to_torchax(a) for a in args)
                    kwargs_ta = {k: to_torchax(v) for k, v in kwargs.items()}

                    res = _orig_amd_forward(
                        self,
                        input_ids_ta,
                        positions_ta,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds_ta,
                        *args_ta,
                        **kwargs_ta
                    )

                    def to_torch(t):
                        if hasattr(t, "torch"):
                            return t.torch()
                        return t
                    return jax.tree_util.tree_map(to_torch, res)
            return _orig_amd_forward(
                self,
                input_ids,
                positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                *args,
                **kwargs
            )


        _AmdDeepseekV4ForCausalLM.forward = patched_amd_forward

    # 1.5. Patch DeepseekV4Model to ensure _mtp_hidden_buffer is always allocated (prevents NoneType subscriptable errors in Dynamo)
    from vllm.models.deepseek_v4.amd.model import DeepseekV4Model
    _orig_model_init = DeepseekV4Model.__init__

    def patched_model_init(self, *args, **kwargs):
        _orig_model_init(self, *args, **kwargs)
        if self._mtp_hidden_buffer is None:
            self._mtp_hidden_buffer = torch.empty(
                self.vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=self.vllm_config.model_config.dtype,
                device=self.device,
            )

    DeepseekV4Model.__init__ = patched_model_init






    # 2. Clear cached resolved architectures so the AMD model is picked up.
    from vllm.model_executor.model_loader import utils as _ml_utils
    from vllm.model_executor.models.registry import _try_load_model_cls
    _ml_utils._MODEL_ARCH_BY_HASH.clear()
    _try_load_model_cls.cache_clear()

    # 3. Swap the DeepseekV4ROCMAiterMLAAttention class with our TPU implementation.
    from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4 import \
        patch_deepseek_v4_mla_cls
    patch_deepseek_v4_mla_cls()

    # 4. Mock CUDA device capability, Stream, and Event allocation since they are unavailable on TPU.
    from vllm.platforms import current_platform
    class DummyDeviceCapability:
        major = 9
        minor = 0

    _orig_get_device_capability = current_platform.get_device_capability
    current_platform.get_device_capability = lambda *args, **kwargs: DummyDeviceCapability()

    class DummyCudaEvent:
        def __init__(self, *args, **kwargs):
            pass
        def record(self, *args, **kwargs):
            pass
        def wait(self, *args, **kwargs):
            pass
        def query(self):
            return True
        def synchronize(self):
            pass

    _orig_cuda_stream = torch.cuda.Stream
    _orig_cuda_event = torch.cuda.Event
    torch.cuda.Stream = lambda *args, **kwargs: None
    torch.cuda.Event = DummyCudaEvent

    # 5. Patch HCHeadOp and MHCFusedPostPreOp forward_native with pure PyTorch implementations.
    from vllm.model_executor.layers.mhc import HCHeadOp, MHCFusedPostPreOp

    def mhc_hc_head_native(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        # hidden_states: shape (..., hc_mult, hidden_size)
        # hc_fn: shape (hc_mult, hc_mult * hidden_size)
        # hc_scale: shape (1,)
        # hc_base: shape (hc_mult,)
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
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
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
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    import safetensors.torch
    safetensors.torch._TYPES["F8_E8M0"] = torch.uint8

    _orig_hc_head_forward_native = HCHeadOp.forward_native
    _orig_mhc_fused_post_pre_forward_native = MHCFusedPostPreOp.forward_native
    HCHeadOp.forward_native = mhc_hc_head_native
    MHCFusedPostPreOp.forward_native = mhc_fused_post_pre_native

    # 6. Patch bind_kv_cache to bypass TPU platform NotImplementedError
    import vllm.v1.worker.utils as worker_utils
    import vllm_torchtpu.runner.tpu_runner as tpu_runner
    _orig_bind_kv_cache = worker_utils.bind_kv_cache

    def patched_bind_kv_cache(
        kv_caches,
        forward_context,
        runner_kv_caches,
        num_attn_module=1,
    ) -> None:
        from collections import defaultdict
        from vllm.model_executor.models.utils import extract_layer_index
        
        assert len(runner_kv_caches) == 0
        index2name = defaultdict(list)
        for layer_name in kv_caches:
            index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

        for layer_index in sorted(index2name.keys()):
            layer_names = index2name[layer_index]
            for layer_name in layer_names:
                runner_kv_caches.append(kv_caches[layer_name])

        for layer_name, kv_cache in kv_caches.items():
            forward_context[layer_name].kv_cache = kv_cache

    worker_utils.bind_kv_cache = patched_bind_kv_cache
    tpu_runner.bind_kv_cache = patched_bind_kv_cache
    import vllm.v1.core.kv_cache_utils as kv_utils
    _orig_get_kv_cache_groups = kv_utils.get_kv_cache_groups
    def patched_get_kv_cache_groups(vllm_config, kv_cache_spec):
        print(f"[DEBUG_GROUPS_INPUT] kv_cache_spec keys: {list(kv_cache_spec.keys())}", flush=True)
        for k, v in kv_cache_spec.items():
            print(f"  {k}: {v.__class__.__name__} window={getattr(v, 'sliding_window', None)} compress={getattr(v, 'compress_ratio', None)}", flush=True)
        res = _orig_get_kv_cache_groups(vllm_config, kv_cache_spec)
        print(f"[DEBUG_GROUPS_OUTPUT] returned {len(res)} groups:", flush=True)
        for idx, g in enumerate(res):
            print(f"  Group {idx}: {g.kv_cache_spec.__class__.__name__} layers={g.layer_names}", flush=True)
        return res
    kv_utils.get_kv_cache_groups = patched_get_kv_cache_groups

    try:
        yield
    finally:
        logger.info("DeepSeek V4 TPU model patches kept active for compilation and worker execution.")


from typing import Iterable

def patched_decoder_load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    import typing
    from typing import Iterable, Callable
    from vllm.model_executor.models.utils import is_pp_missing_parameter
    from vllm.model_executor.parameter import default_weight_loader
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank

    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("gate_up_proj", "w1", 0),
        ("gate_up_proj", "w3", 1),
        ("attn.fused_wqa_wkv", "attn.wq_a", 0),
        ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
        ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
    ]
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    # TP for attention
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    n_head = self.config.num_attention_heads
    n_local_head = n_head // tp_size
    head_rank_start = n_local_head * tp_rank
    head_rank_end = n_local_head * (tp_rank + 1)

    # Pre-compute expert mapping ONCE.
    expert_mapping = self.get_expert_mapping()

    for name, loaded_weight in weights:
        # Bypass loading MTP draft keys in the target model.
        if "mtp." in name:
            continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if ".experts." in name:
                continue
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)

            if is_pp_missing_parameter(name, self):
                break
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(name)
            break
        else:
            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for mapping in expert_mapping:
                    param_name, weight_name, expert_id, expert_shard_id = mapping
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                loaded_params.add(name_mapped)
                continue
            elif "attn_sink" in name:
                if is_pp_missing_parameter(name, self):
                    continue
                narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                n = narrow_weight.shape[0]
                params_dict[name][:n].copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif "attn.wq_b.weight" in name:  # Fixed check!
                param = params_dict["attn.wq_b.weight"]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                shard_size = param.data.shape[0]
                loaded_weight_shard = loaded_weight.narrow(
                    0, tp_rank * shard_size, shard_size
                )
                weight_loader(param, loaded_weight_shard)
                loaded_params.add("attn.wq_b.weight")
                continue
            elif "attn.wo_a.weight" in name:  # Fixed check!
                param = params_dict["attn.wo_a.weight"]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                shard_size = param.data.shape[0]
                loaded_weight_shard = loaded_weight.narrow(
                    0, tp_rank * shard_size, shard_size
                )
                weight_loader(param, loaded_weight_shard)
                loaded_params.add("attn.wo_a.weight")
                continue
            elif "attn.wo_b.weight" in name:  # Fixed check!
                param = params_dict["attn.wo_b.weight"]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                shard_size = param.data.shape[1]
                loaded_weight_shard = loaded_weight.narrow(
                    1, tp_rank * shard_size, shard_size
                )
                weight_loader(param, loaded_weight_shard)
                loaded_params.add("attn.wo_b.weight")
                continue
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue

    return loaded_params

