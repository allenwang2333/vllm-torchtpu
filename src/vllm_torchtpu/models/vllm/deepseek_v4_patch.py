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
        if ep_size > 1:
            return fmc.FusedMoEParallelConfig(
                tp_size=1,
                tp_rank=0,
                pcp_size=1,
                pcp_rank=0,
                dp_size=1,
                dp_rank=0,
                ep_size=ep_size,
                ep_rank=torch.distributed.get_rank() // tp_size,
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
                def debug_weights_generator(weights_iter):
                    for name, loaded_weight in weights_iter:
                        if "layers.0.attn" in name:
                            print(f"[PATCH] Passing to load_weights: {name} shape={list(loaded_weight.shape)} dtype={loaded_weight.dtype}", file=sys.stderr, flush=True)
                        yield name, loaded_weight
                res = _orig_load_weights(self, debug_weights_generator(weights))
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
                            jax_arr = jax.device_put(t.detach().cpu().numpy())
                            return ta_tensor.Tensor(jax_arr, env)
                        return t

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
