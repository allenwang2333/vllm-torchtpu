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
                yield f"{base_key}.weight_scale", s_new
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

    # Patch tpu_inference unquantized weight post-processing methods to avoid JAX sharding
    try:
        import tpu_inference.layers.vllm.quantization.unquantized as tpu_unquant
        tpu_unquant.VllmUnquantizedEmbeddingMethod.process_weights_after_loading = lambda self, layer: None
        tpu_unquant.VllmUnquantizedLinearMethod.process_weights_after_loading = lambda self, layer: None
    except Exception:
        pass

    # Patch tpu_inference model wrapper context to prevent Dynamo data-dependent assert errors
    try:
        import tpu_inference.models.vllm.vllm_model_wrapper_context as vmwc
        class _DummyWrapperContext:
            mesh = None
        vmwc.get_vllm_model_wrapper_context = lambda: _DummyWrapperContext()
    except Exception:
        pass

    try:
        import tpu_inference.layers.vllm.custom_ops.fused_moe as tpu_fused_moe
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        tpu_fused_moe.get_vllm_model_wrapper_context = lambda: _DummyWrapperContext()
        def _patched_maybe_reduce_final_output(self, states, trunc_size):
            return MoERunner._maybe_reduce_final_output(self, states, trunc_size)
        tpu_fused_moe.VllmMoERunner._maybe_reduce_final_output = _patched_maybe_reduce_final_output
    except Exception:
        pass

    # Recover layer and input_ids parameters in select_experts via stack inspection
    import vllm_torchtpu.layers.vllm.moe_routing as mr
    _orig_select_experts = mr.select_experts
    def select_experts_patched(*args, **kwargs):
        layer = kwargs.get("layer", None)
        input_ids = kwargs.get("input_ids", None)
        if layer is None or input_ids is None:
            import sys
            frame = sys._getframe()
            while frame:
                if layer is None and "layer" in frame.f_locals:
                    layer = frame.f_locals["layer"]
                if input_ids is None and "input_ids" in frame.f_locals:
                    input_ids = frame.f_locals["input_ids"]
                if layer is not None and input_ids is not None:
                    break
                frame = frame.f_back
        if layer is not None:
            kwargs["layer"] = layer
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        
        # Check if hash_indices_table is present on layer
        has_hash = False
        prefix = "unknown"
        if layer is not None:
            prefix = getattr(layer, "prefix", getattr(layer, "custom_prefix", "unknown"))
            if hasattr(layer, "hash_indices_table") and getattr(layer, "hash_indices_table") is not None:
                has_hash = True
        if has_hash or (layer is not None and "hash" in prefix):
            print(f"[DEBUG_ROUTING] prefix={prefix} has_hash={has_hash} input_ids_found={input_ids is not None}", flush=True)
            
        return _orig_select_experts(*args, **kwargs)
    mr.select_experts = select_experts_patched


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
        if (hasattr(torch, "compiler") and torch.compiler.is_compiling()) or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling()):
            return _orig_apply_quant_method(self, layer, hidden_states, router_logits, shared_experts_input, input_ids)
        layer_idx = getattr(layer, "layer_idx", -1)
        if layer_idx in (0, 3, 4):
            hs_cpu = hidden_states.cpu().float()
            rl_cpu = router_logits.cpu().float()
            print(f"[DEBUG_PATCH] layer_{layer_idx} input hidden_states: shape={list(hs_cpu.shape)} dtype={hidden_states.dtype} | mean={hs_cpu.mean().item():.6f} std={hs_cpu.std().item():.6f} min={hs_cpu.min().item():.6f} max={hs_cpu.max().item():.6f}", flush=True)
            print(f"[DEBUG_PATCH] layer_{layer_idx} input router_logits: shape={list(rl_cpu.shape)} dtype={router_logits.dtype} | mean={rl_cpu.mean().item():.6f} std={rl_cpu.std().item():.6f} min={rl_cpu.min().item():.6f} max={rl_cpu.max().item():.6f}", flush=True)
        res = _orig_apply_quant_method(self, layer, hidden_states, router_logits, shared_experts_input, input_ids)
        tensors_to_check = []
        if isinstance(res, torch.Tensor):
            tensors_to_check.append(("moe_output", res))
        elif isinstance(res, tuple):
            if len(res) > 0 and isinstance(res[0], torch.Tensor):
                tensors_to_check.append(("shared_expert_output", res[0]))
            if len(res) > 1 and isinstance(res[1], torch.Tensor):
                tensors_to_check.append(("fused_expert_output", res[1]))
        for name, tensor in tensors_to_check:
            has_nan = torch.isnan(tensor).any().item()
            if has_nan:
                print(f"[NAN_DETECTION] NaN detected in {name} at layer {layer_idx}! shape={list(tensor.shape)}", flush=True)
            elif layer_idx in (0, 3, 4):
                tensor_cpu = tensor.cpu().float()
                print(f"[DEBUG_PATCH] layer_{layer_idx} output {name}: shape={list(tensor_cpu.shape)} | mean={tensor_cpu.mean().item():.6f} std={tensor_cpu.std().item():.6f} min={tensor_cpu.min().item():.6f} max={tensor_cpu.max().item():.6f}", flush=True)
        return res

    MoERunner._apply_quant_method = patched_apply_quant_method

    _orig_maybe_reduce_final_output = MoERunner._maybe_reduce_final_output

    def patched_maybe_reduce_final_output(self, states, trunc_size):
        layer_name = getattr(self, "layer_name", "unknown")
        will_reduce = (
            not self.moe_config.is_sequence_parallel
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not self._fused_output_is_reduced
        )
        do_print = any(f"layers.{i}." in str(layer_name) for i in (0, 3, 4)) and not (
            (hasattr(torch, "compiler") and torch.compiler.is_compiling())
            or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
        )
        rank = -1
        if do_print:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else -1
            before = states.cpu().float()
            print(f"[DEBUG_AR] layer={layer_name} rank={rank} PRE-reduce will_reduce={will_reduce} "
                  f"tp_size={self.moe_config.tp_size} ep_size={self.moe_config.ep_size} "
                  f"is_sp={self.moe_config.is_sequence_parallel} fused_reduced={self._fused_output_is_reduced} "
                  f"mean={before.mean().item():.6f} std={before.std().item():.6f}", flush=True)
        result = _orig_maybe_reduce_final_output(self, states, trunc_size)
        if do_print:
            after = result.cpu().float()
            print(f"[DEBUG_AR] layer={layer_name} rank={rank} POST-reduce "
                  f"mean={after.mean().item():.6f} std={after.std().item():.6f}", flush=True)
        return result

    MoERunner._maybe_reduce_final_output = patched_maybe_reduce_final_output

    # 1. Patch the model registry to resolve to the AMD (ROCm) variant.
    # The AMD variant is more extensible and allows us to swap attention/MHC classes.
    import vllm.models.deepseek_v4 as ds_v4
    def patch_load_weights_for_class(cls):
        _orig_load_weights = cls.load_weights
        
        def patched_load_weights(self, weights):
            print(f"[PATCH] Entering patched_load_weights on DeepseekV4ForCausalLM", file=sys.stderr, flush=True)
            from vllm.models.deepseek_v4.amd.model import DeepseekV4DecoderLayer
            print(f"[DEBUG_LOAD] DeepseekV4DecoderLayer.load_weights is: {getattr(DeepseekV4DecoderLayer, 'load_weights', None)}", flush=True)
            
            # Check if all expert layers have been pre-quantized and cached on disk.
            cache_dir = os.environ.get("VLLM_FP4_CACHE_DIR", "")
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
                # Total expected cache files = num_hidden_layers * 8 ranks
                expected_files = self.config.num_hidden_layers * 8
                has_all_cached = (len(cached_files) >= expected_files)
            else:
                has_all_cached = False

            if has_all_cached:
                import vllm.model_executor.model_loader.weight_utils as wu
                print(f"[PATCH] All expert weights are cached on disk. Monkey-patching should_skip_weight to skip MoE experts from checkpoint files!", file=sys.stderr, flush=True)
                _orig_should_skip_weight = wu.should_skip_weight
                def patched_should_skip_weight(weight_name, local_expert_ids):
                    if ".experts." in weight_name:
                        return True
                    return _orig_should_skip_weight(weight_name, local_expert_ids)
                wu.should_skip_weight = patched_should_skip_weight

            print(f"[PATCH] Model parameters count: {len(list(self.parameters()))}", file=sys.stderr, flush=True)
            named_params = dict(self.named_parameters())
            for name in named_params.keys():
                print(f"[PATCH]   Param: {name}", file=sys.stderr, flush=True)
            print(f"[PATCH] Has model.embed_tokens.weight: {'model.embed_tokens.weight' in named_params}", file=sys.stderr, flush=True)
            print(f"[PATCH] Has lm_head.weight: {'lm_head.weight' in named_params}", file=sys.stderr, flush=True)
            self.model.has_tilelang = False
            for idx, l in enumerate(self.model.layers):
                l.layer_idx = idx
                l.has_tilelang = False
                if hasattr(l, "ffn"):
                    l.ffn.layer_idx = idx
                    if hasattr(l.ffn, "experts"):
                        l.ffn.experts.layer_idx = idx
                        # Hash-MoE layers (layer_idx < config.num_hash_layers)
                        # route by gate.tid2eid[token_id], not by router logits.
                        # The TPU routing (moe_routing.select_experts) receives
                        # the RoutedExperts module as `layer`, so expose the
                        # table there (plain attr; same Parameter object, so it
                        # sees the checkpoint data once loaded).
                        tid2eid = getattr(getattr(l.ffn, "gate", None), "tid2eid", None)
                        if tid2eid is not None:
                            runner = l.ffn.experts
                            for tgt in (runner, getattr(runner, "routed_experts", None)):
                                if tgt is not None:
                                    object.__setattr__(tgt, "hash_indices_table", tid2eid)
            from vllm.models.deepseek_v4.amd.model import _make_deepseek_v4_weights_mapper
            quant_config = getattr(vllm_config, "quant_config", None)
            expert_dtype = getattr(quant_config, "expert_dtype", "fp4")
            is_fp4_experts = (expert_dtype == "fp4")
            orig_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
            
            outer_self = self
            class WrappedMapper:
                def apply(self, w):
                    for name, loaded_weight in orig_mapper.apply(w):
                        if name == "model.embed.weight":
                            name = "model.embed_tokens.weight"
                        if name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
                            print(f"[PATCH_LOADED_KEY] Mapped key: {name} shape={list(loaded_weight.shape)} mean={loaded_weight.float().mean().item():.6f}", file=sys.stderr, flush=True)
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
                        elif ".shared_experts.down_proj" in name:
                            name = name.replace(".shared_experts.down_proj", ".shared_experts.w2")
                            
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
            cache_dir = os.environ.get("VLLM_FP4_CACHE_DIR", "")
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cached_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
                # Total expected cache files = num_hidden_layers (60) * num_ranks (8)
                expected_files = self.config.num_hidden_layers * 8
                has_all_cached = (len(cached_files) >= expected_files)
            else:
                has_all_cached = False

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
                # Update TP status of all parameters since new ones (like weight_scale)
                # are registered during weight creation which runs AFTER linear layer __init__.
                for name, module in self.model.named_modules():
                    if hasattr(module, "update_param_tp_status"):
                        module.update_param_tp_status()
                
                mapper = WrappedMapper()
                def mapped_weights_generator(weights_iter):
                    for name, loaded_weight in weights_iter:
                        for mapped_name, mapped_weight in mapper.apply([(name, loaded_weight)]):
                            yield mapped_name, mapped_weight

                weights = attention_requantizer_generator(mapped_weights_generator(weights))

                class IdentityMapper:
                    def apply(self, w):
                        def mapper_generator(w_iter):
                            for name, loaded_weight in w_iter:
                                if ".shared_experts.w2" in name:
                                    name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
                                yield name, loaded_weight
                        return mapper_generator(w)
                self.hf_to_vllm_mapper = IdentityMapper()

                def debug_weights_generator(weights_iter):
                    for item in weights_iter:
                        if not isinstance(item, tuple) or len(item) != 2:
                            print(f"[DEBUG_LOADER_ITEM] Unexpected item structure! Type: {type(item)} | Content: {item}", file=sys.stderr, flush=True)
                        name, loaded_weight = item
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

        _orig_compute_logits = getattr(cls, "compute_logits", None)
        if _orig_compute_logits is not None:
            def patched_compute_logits(self, hidden_states: torch.Tensor):
                logits = _orig_compute_logits(self, hidden_states)
                if logits is not None and not torch.compiler.is_compiling():
                    h_mean = hidden_states.float().mean().item()
                    h_std = hidden_states.float().std().item()
                    w_mean = self.lm_head.weight.float().mean().item() if hasattr(self.lm_head, "weight") and self.lm_head.weight is not None else 0.0
                    w_std = self.lm_head.weight.float().std().item() if hasattr(self.lm_head, "weight") and self.lm_head.weight is not None else 0.0
                    top5_idx = torch.topk(logits[0], 5).indices.tolist()
                    top5_val = [round(v, 4) for v in torch.topk(logits[0], 5).values.tolist()]
                    print(f"[DEBUG_LOGITS] hidden_states mean={h_mean:.6f} std={h_std:.6f} | lm_head weight mean={w_mean:.6f} std={w_std:.6f} | logits max={logits.max().item():.4f} min={logits.min().item():.4f} top5={top5_idx} top5_vals={top5_val}", flush=True)
                return logits
            cls.compute_logits = patched_compute_logits

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
            # Standard PyTorch equivalent of the GPU Triton RMSNorm kernel
            # (reference _fused_q_kv_rmsnorm_kernel computes y = x * rrms * w,
            # with NO (1 + w) offset).
            orig_dtype = qr.dtype
            qr_f = qr.float()
            kv_f = kv.float()
            qr_out = qr_f * torch.rsqrt(qr_f.pow(2).mean(-1, keepdim=True) + eps) * q_weight.float()
            kv_out = kv_f * torch.rsqrt(kv_f.pow(2).mean(-1, keepdim=True) + eps) * kv_weight.float()
            return qr_out.to(orig_dtype), kv_out.to(orig_dtype)

        def fused_indexer_q_rope_quant_pytorch(
            positions: torch.Tensor,
            q: torch.Tensor,
            rotary_emb: torch.nn.Module,
            q_scale: torch.Tensor,
        ) -> torch.Tensor:
            # Fallback PyTorch implementation for fused_indexer_q_rope_quant
            q_pe = rotary_emb(positions, q)
            return q_pe

        try:
            import vllm.v1.attention.ops.rocm_aiter_mla_sparse as sparse_ops
            orig_inv_rope = sparse_ops._apply_inv_rope_ref
            def safe_inv_rope(rotary_emb, x, positions, rope_dim):
                try:
                    return orig_inv_rope(rotary_emb, x, positions, rope_dim)
                except Exception:
                    pos_1d = positions.flatten()
                    return sparse_ops._apply_gptj_inv_rope_ref(x, pos_1d, rotary_emb.cos_sin_cache, rope_dim)
            sparse_ops._apply_inv_rope_ref = safe_inv_rope
        except Exception:
            pass

        try:
            from vllm.model_executor.layers.mhc import MHCFusedPostPreOp
            def mhc_fused_post_pre_native(
                self,
                x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                n_splits=1, tile_n=1, norm_weight=None, norm_eps=0.0
            ):
                import vllm.model_executor.kernels.mhc as mhc_kernels
                res_cur = mhc_kernels.mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
                post_mix, comb_mix, layer_input = mhc_kernels.mhc_pre_torch(
                    res_cur, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
                )
                return res_cur, post_mix, comb_mix, layer_input
            MHCFusedPostPreOp.forward_native = mhc_fused_post_pre_native
        except Exception:
            pass
        def fused_indexer_q_rope_quant_pytorch_full(
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

        ds_attn.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_pytorch_full
        fused_indexer_ops.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_pytorch_full
        common_ops.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_pytorch_full

        # Direct sys.modules update to override all imported references
        for m in list(sys.modules.values()):
            if m is not None:
                if hasattr(m, "fused_q_kv_rmsnorm"):
                    setattr(m, "fused_q_kv_rmsnorm", fused_q_kv_rmsnorm_pytorch)
                if hasattr(m, "fused_indexer_q_rope_quant"):
                    setattr(m, "fused_indexer_q_rope_quant", fused_indexer_q_rope_quant_pytorch_full)

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

    # Register patched_decoder_load_weights on DecoderLayer classes.
    #
    # CONFIRMED BROKEN this session, not yet fixed -- see
    # docs/dsv4_architecture_and_rationale.md / dsv4_running_and_testing.md
    # session-4 notes for the full trace:
    #
    # (1) AutoWeightsLoader never actually reaches DeepseekV4DecoderLayer's
    #     load_weights here -- confirmed via a debug probe that native
    #     DeepseekV4Model.load_weights (a DIFFERENT class) is what
    #     AutoWeightsLoader calls, and it swallows the entire weights
    #     iterator itself before any recursion into per-layer submodules
    #     could happen. Zero DEBUG_DECODER_LOAD_WEIGHTS_ALL prints across
    #     every run this session confirm this registration is dead code:
    #     every TPU-specific fix patched_decoder_load_weights carries
    #     (attn_sink narrow-loading, hash-MoE hash_indices_table wiring,
    #     e_score_correction_bias, wq_b/wo_a/wo_b endswith-based matching,
    #     fused_wqa_wkv/compressor.fused_wkv_wgate stacked-param remapping)
    #     has never actually run; the model loads via stock vLLM's native
    #     DeepseekV4Model.load_weights instead.
    # (2) Re-registering onto DeepseekV4Model instead (the class that IS
    #     actually invoked) crashes immediately: patched_decoder_load_
    #     weights's own body uses hardcoded, layer-*unprefixed* dict keys
    #     (e.g. params_dict["attn.wo_a.weight"]), which is only unambiguous
    #     if `self` is a single per-layer DeepseekV4DecoderLayer instance --
    #     not the whole DeepseekV4Model, which has 43 distinct
    #     layers.N.attn.wo_a.weight parameters colliding on that one key.
    #     KeyError: 'attn.wo_a.weight' on the very first layer.
    #
    # Net effect: this function's internal scope (per-layer) matches THIS
    # registration target, but this target is never invoked; the target
    # that IS invoked (DeepseekV4Model) doesn't match the function's scope.
    #
    # SESSION 5: investigated whether this dead code is actually the cause
    # of the garbled-decode-text bug (docs/dsv4_architecture_and_rationale.md
    # SS3.10 has the full trace) -- it is NOT. Every responsibility listed
    # in (1) above is already handled correctly by code paths that DO run:
    # native DeepseekV4Model.load_weights (vendor AMD code -- this repo's
    # attention module subclasses it rather than defining its own, so
    # fused_wqa_wkv/wq_b/wo_a/wo_b/attn_sink are all native objects vLLM's
    # own loader already knows how to fill, confirmed via a byte-for-byte
    # comparison against the installed package) for everything except
    # hash_indices_table wiring and e_score_correction_bias, which are
    # handled by patched_load_weights (a separate, confirmed-running
    # function on DeepseekV4ForCausalLM, not this one) and a plain default
    # weight_loader respectively. Verified empirically too: a runtime probe
    # dumping post-load stats for all of the above showed finite, sane,
    # non-NaN values on all 8 ranks. Do not rewrite or re-target this
    # function chasing the garbled-text bug -- it's dead but harmless, not
    # a missing fix. Left as the ORIGINAL (dead but non-crashing)
    # registration; do not re-target DeepseekV4Model without also fixing
    # (1) above if this is ever revived for an unrelated reason.
    try:
        from vllm.models.deepseek_v4.amd.model import DeepseekV4DecoderLayer as AmdDecoderLayer
        AmdDecoderLayer.load_weights = patched_decoder_load_weights
        print("[PATCH] Successfully patched AMD DeepseekV4DecoderLayer.load_weights", flush=True)
    except Exception as e:
        print(f"[PATCH] Failed to patch AMD DecoderLayer: {e}", file=sys.stderr, flush=True)

    try:
        from vllm.models.deepseek_v4.model import DeepseekV4DecoderLayer as CudaDecoderLayer
        CudaDecoderLayer.load_weights = patched_decoder_load_weights
        print("[PATCH] Successfully patched CUDA DeepseekV4DecoderLayer.load_weights", flush=True)
    except Exception as e:
        print(f"[PATCH] Failed to patch CUDA DecoderLayer: {e}", file=sys.stderr, flush=True)

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
            res = _orig_amd_forward(
                self,
                input_ids,
                positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                *args,
                **kwargs
            )
            is_compiling = (hasattr(torch, "compiler") and torch.compiler.is_compiling()) or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            if not is_compiling and isinstance(res, torch.Tensor):
                print(f"[DEBUG_FORWARD_RES] pid={os.getpid()} res shape={list(res.shape)} dtype={res.dtype} mean={res.float().mean().item():.6f} abs_mean={res.float().abs().mean().item():.6f}", file=sys.stderr, flush=True)
            return res

        _AmdDeepseekV4ForCausalLM.forward = patched_amd_forward

        def patched_compute_logits(self, hidden_states: torch.Tensor):
            is_compiling = (hasattr(torch, "compiler") and torch.compiler.is_compiling()) or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            if not is_compiling:
                print(f"[DEBUG_COMPUTE_LOGITS_IN] pid={os.getpid()} hs shape={list(hidden_states.shape)} mean={hidden_states.float().mean().item():.6f} abs_mean={hidden_states.float().abs().mean().item():.6f}", file=sys.stderr, flush=True)
            logits = self.logits_processor(self.lm_head, hidden_states)
            if not is_compiling and logits is not None:
                print(f"[DEBUG_COMPUTE_LOGITS_OUT] pid={os.getpid()} logits shape={list(logits.shape)} mean={logits.float().mean().item():.6f} abs_mean={logits.float().abs().mean().item():.6f} min={logits.float().min().item():.6f} max={logits.float().max().item():.6f}", file=sys.stderr, flush=True)
            return logits

        _AmdDeepseekV4ForCausalLM.compute_logits = patched_compute_logits

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

    from vllm.models.deepseek_v4.amd.model import DeepseekV4DecoderLayer
    _orig_layer_forward = DeepseekV4DecoderLayer.forward

    def patched_layer_forward(self, *args, **kwargs):
        res = _orig_layer_forward(self, *args, **kwargs)
        is_compiling = (hasattr(torch, "compiler") and torch.compiler.is_compiling()) or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
        if not is_compiling:
            layer_idx = getattr(self, "layer_idx", -1)
            x = res[0] if isinstance(res, (tuple, list)) else res
            if isinstance(x, torch.Tensor):
                has_nan = torch.isnan(x).any().item()
                if has_nan or layer_idx == 0:
                    print(f"[DEBUG_LAYER_{layer_idx}] pid={os.getpid()} has_nan={has_nan} mean={x.float().mean().item():.6f} min={x.float().min().item():.6f} max={x.float().max().item():.6f}", file=sys.stderr, flush=True)
        return res

    DeepseekV4DecoderLayer.forward = patched_layer_forward





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
    from vllm.model_executor.layers.mhc import HCHeadOp, MHCFusedPostPreOp, MHCPreOp, MHCPostOp

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

    _orig_mhc_pre_forward_native = MHCPreOp.forward_native
    _orig_mhc_post_forward_native = MHCPostOp.forward_native

    def debug_mhc_pre_native(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1, norm_weight=None, norm_eps=0.0):
        for name, t in [("residual", residual), ("fn", fn), ("hc_scale", hc_scale), ("hc_base", hc_base)]:
            if t is not None:
                has_nan = torch.isnan(t).any().item()
                has_inf = torch.isinf(t).any().item()
                if has_nan or has_inf:
                    print(f"[NAN_DETECTION] MHCPreOp input {name} has NaN/Inf! has_nan={has_nan} has_inf={has_inf}", flush=True)
        res = _orig_mhc_pre_forward_native(self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits, norm_weight, norm_eps)
        has_output_nan = False
        for name, t in zip(["post_mix", "comb_mix", "layer_input"], res):
            if t is not None:
                has_nan = torch.isnan(t).any().item()
                has_inf = torch.isinf(t).any().item()
                if has_nan or has_inf:
                    print(f"[NAN_DETECTION] MHCPreOp output {name} has NaN/Inf! has_nan={has_nan} has_inf={has_inf}", flush=True)
                    if name == "layer_input":
                        has_output_nan = True
        if has_output_nan:
            print("[NAN_DETECTION_DETAIL] Running step-by-step debug check inside MHCPreOp...", flush=True)
            hc_mult = residual.shape[-2]
            hidden_size = residual.shape[-1]
            residual_flat = residual.view(-1, hc_mult, hidden_size)
            num_tokens = residual_flat.shape[0]
            x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
            mixes = torch.matmul(x, fn.t())
            sqrsum = x.square().sum(dim=-1, keepdim=True)
            mixes_norm = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
            pre_logits = mixes_norm[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
            pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps
            product = pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32)
            layer_input_dbg = torch.sum(product, dim=1).to(torch.bfloat16)

            print(f"[NAN_DETECTION_DETAIL]   x: nan={torch.isnan(x).any().item()} inf={torch.isinf(x).any().item()} mean={x.mean().item():.6f} min={x.min().item():.6f} max={x.max().item():.6f}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   fn: nan={torch.isnan(fn).any().item()} inf={torch.isinf(fn).any().item()} mean={fn.mean().item():.6f} min={fn.min().item():.6f} max={fn.max().item():.6f}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   hc_scale: {hc_scale.tolist()}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   hc_base: {hc_base.tolist()}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   mixes: nan={torch.isnan(mixes).any().item()} inf={torch.isinf(mixes).any().item()} mean={mixes.mean().item():.6f} min={mixes.min().item():.6f} max={mixes.max().item():.6f}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   sqrsum: nan={torch.isnan(sqrsum).any().item()} inf={torch.isinf(sqrsum).any().item()} mean={sqrsum.mean().item():.6f} min={sqrsum.min().item():.6f} max={sqrsum.max().item():.6f}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   mixes_norm: nan={torch.isnan(mixes_norm).any().item()} inf={torch.isinf(mixes_norm).any().item()} mean={mixes_norm.mean().item():.6f} min={mixes_norm.min().item():.6f} max={mixes_norm.max().item():.6f}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   pre_logits: nan={torch.isnan(pre_logits).any().item()} inf={torch.isinf(pre_logits).any().item()} mean={pre_logits.mean().item():.6f} min={pre_logits.min().item():.6f} max={pre_logits.max().item():.6f}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   pre_mix: nan={torch.isnan(pre_mix).any().item()} inf={torch.isinf(pre_mix).any().item()} mean={pre_mix.mean().item():.6f} min={pre_mix.min().item():.6f} max={pre_mix.max().item():.6f}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   product: nan={torch.isnan(product).any().item()} inf={torch.isinf(product).any().item()} mean={product.mean().item():.6f} min={product.min().item():.6f} max={product.max().item():.6f}", flush=True)
            print(f"[NAN_DETECTION_DETAIL]   layer_input_dbg: nan={torch.isnan(layer_input_dbg).any().item()} inf={torch.isinf(layer_input_dbg).any().item()}", flush=True)
        return res

    def debug_mhc_post_native(self, x, residual, post_layer_mix, comb_res_mix):
        for name, t in [("x", x), ("residual", residual), ("post_layer_mix", post_layer_mix), ("comb_res_mix", comb_res_mix)]:
            if t is not None:
                has_nan = torch.isnan(t).any().item()
                has_inf = torch.isinf(t).any().item()
                if has_nan or has_inf:
                    print(f"[NAN_DETECTION] MHCPostOp input {name} has NaN/Inf! has_nan={has_nan} has_inf={has_inf}", flush=True)
        res = _orig_mhc_post_forward_native(self, x, residual, post_layer_mix, comb_res_mix)
        if res is not None:
            has_nan = torch.isnan(res).any().item()
            has_inf = torch.isinf(res).any().item()
            if has_nan or has_inf:
                print(f"[NAN_DETECTION] MHCPostOp output has NaN/Inf! has_nan={has_nan} has_inf={has_inf}", flush=True)
            else:
                max_val = res.abs().max().item()
                if max_val > 1e10:
                    print(f"[NAN_DETECTION_DETAIL] MHCPostOp output max value is very large: {max_val}", flush=True)
                    mixed_residual = torch.einsum(
                        "...ij,...ih->...jh",
                        comb_res_mix.to(torch.float32),
                        residual.to(torch.float32),
                    )
                    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
                    res_dbg = (mixed_residual + post_term).to(residual.dtype)

                    print(f"[NAN_DETECTION_DETAIL]   x: mean={x.mean().item():.6f} min={x.min().item():.6f} max={x.max().item():.6f}", flush=True)
                    print(f"[NAN_DETECTION_DETAIL]   residual: mean={residual.mean().item():.6f} min={residual.min().item():.6f} max={residual.max().item():.6f}", flush=True)
                    print(f"[NAN_DETECTION_DETAIL]   post_layer_mix: mean={post_layer_mix.mean().item():.6f} min={post_layer_mix.min().item():.6f} max={post_layer_mix.max().item():.6f}", flush=True)
                    print(f"[NAN_DETECTION_DETAIL]   comb_res_mix: mean={comb_res_mix.mean().item():.6f} min={comb_res_mix.min().item():.6f} max={comb_res_mix.max().item():.6f}", flush=True)
                    print(f"[NAN_DETECTION_DETAIL]   mixed_residual: mean={mixed_residual.mean().item():.6f} min={mixed_residual.min().item():.6f} max={mixed_residual.max().item():.6f}", flush=True)
                    print(f"[NAN_DETECTION_DETAIL]   post_term: mean={post_term.mean().item():.6f} min={post_term.min().item():.6f} max={post_term.max().item():.6f}", flush=True)
                    print(f"[NAN_DETECTION_DETAIL]   res_dbg: mean={res_dbg.to(torch.float32).mean().item():.6f} min={res_dbg.to(torch.float32).min().item():.6f} max={res_dbg.to(torch.float32).max().item():.6f}", flush=True)
        return res

    # MHCPreOp.forward_native = debug_mhc_pre_native
    # MHCPostOp.forward_native = debug_mhc_post_native

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
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
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
        print(f"[DEBUG_DECODER_LOAD_WEIGHTS_ALL] name={name} shape={list(loaded_weight.shape)}", flush=True)
        if "wo_a" in name:
            print(f"[DEBUG_LOAD_WEIGHTS] wo_a parameter name={name} shape={list(loaded_weight.shape)}", flush=True)
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
                # attn_sink is a per-rank LOCAL buffer (padded to a minimum
                # of 64 slots, only the first n_local_heads are meaningful --
                # see vllm.models.deepseek_v4.amd.model's allocation comment:
                # "Weight loading fills the first n_local_heads slots"), NOT
                # a globally-indexed shared tensor like wq_b/wo_a/wo_b. The
                # destination must always be written starting at index 0
                # regardless of rank; indexing it by [head_rank_start:
                # head_rank_end] (a regression from fd187d51) left the real
                # learned sink values for every rank but rank 0 outside the
                # kernel's read range ([:n_local_heads]), silently replacing
                # them with the -inf ("no sink effect") padding default.
                narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                n = narrow_weight.shape[0]
                params_dict[name][:n].copy_(narrow_weight)
                loaded_params.add(name)
                continue
            elif name.endswith("attn.wq_b.weight"):
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
            elif name.endswith("attn.wo_a.weight"):
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
            elif name.endswith("attn.wo_b.weight"):
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

