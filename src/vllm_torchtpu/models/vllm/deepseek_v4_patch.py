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

import os  # noqa: E402

# Register collectives in Dynamo allow_in_graph to bypass tracing errors
try:
    if hasattr(torch, "compiler") and hasattr(torch.compiler,
                                              "allow_in_graph"):
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
            logger.info(
                f"Registered functional collectives in Dynamo: {registered_ops}"
            )
except Exception as e:
    logger.warning(f"Failed to register collectives in Dynamo: {e}")

from vllm_torchtpu.models.vllm.deepseek_v4_requant import \
    attention_requantizer_generator  # noqa: E402


from typing import Iterable  # noqa: E402


def patched_decoder_load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    import typing
    from typing import Callable

    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
    from vllm.model_executor.model_loader.weight_utils import \
        default_weight_loader
    from vllm.model_executor.models.utils import is_pp_missing_parameter

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
                if ("weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for mapping in expert_mapping:
                    param_name, weight_name, expert_id, expert_shard_id = mapping
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(Callable[..., bool],
                                                param.weight_loader)
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
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                shard_size = param.data.shape[0]
                loaded_weight_shard = loaded_weight.narrow(
                    0, tp_rank * shard_size, shard_size)
                weight_loader(param, loaded_weight_shard)
                loaded_params.add("attn.wq_b.weight")
                continue
            elif name.endswith("attn.wo_a.weight"):
                param = params_dict["attn.wo_a.weight"]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                shard_size = param.data.shape[0]
                loaded_weight_shard = loaded_weight.narrow(
                    0, tp_rank * shard_size, shard_size)
                weight_loader(param, loaded_weight_shard)
                loaded_params.add("attn.wo_a.weight")
                continue
            elif name.endswith("attn.wo_b.weight"):
                param = params_dict["attn.wo_b.weight"]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                shard_size = param.data.shape[1]
                loaded_weight_shard = loaded_weight.narrow(
                    1, tp_rank * shard_size, shard_size)
                weight_loader(param, loaded_weight_shard)
                loaded_params.add("attn.wo_b.weight")
                continue
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
                continue

    return loaded_params


@contextmanager
def _maybe_patch_for_deepseek_v4(vllm_config: VllmConfig):
    architectures = getattr(vllm_config.model_config.hf_config,
                            "architectures", None) or []
    logger.info(f"Patcher checking architectures: {architectures}")

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
        _has_hash = False
        _prefix = "unknown"
        if layer is not None:
            _prefix = getattr(layer, "prefix",
                              getattr(layer, "custom_prefix", "unknown"))
            if hasattr(layer, "hash_indices_table") and getattr(
                    layer, "hash_indices_table") is not None:
                _has_hash = True

        return _orig_select_experts(*args, **kwargs)

    mr.select_experts = select_experts_patched

    # Patch vllm.distributed.parallel_state to reflect the TPU sharding strategy (TP vs EP)
    import vllm.distributed.parallel_state as ps
    import vllm.model_executor.layers.fused_moe.config as fmc

    parallel_config = vllm_config.parallel_config
    tp_size = max(1, parallel_config.tensor_parallel_size)
    ep_size = (parallel_config.world_size if getattr(
        parallel_config, "enable_expert_parallel", False) else 1)

    _orig_get_tp_world_size = ps.get_tensor_model_parallel_world_size
    _orig_get_tp_rank = ps.get_tensor_model_parallel_rank
    _orig_make = fmc.FusedMoEParallelConfig.make

    ps.get_tensor_model_parallel_world_size = lambda: tp_size
    ps.get_tensor_model_parallel_rank = lambda: torch.distributed.get_rank(
    ) % tp_size

    def patched_make(*args, **kwargs):
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else -1
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if ep_size > 1:
            moe_tp_size = max(1, world_size // ep_size)
            ret_rank = rank // moe_tp_size if rank >= 0 else 0

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
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import \
        MoERunner
    _orig_apply_quant_method = MoERunner._apply_quant_method

    def patched_apply_quant_method(self, *args, **kwargs):
        # NOTE (vLLM 0.25.1 vs 0.22.1): _apply_quant_method signature varies.
        # 0.22.1: self, layer, hidden_states, router_logits, shared_experts_input, input_ids=None
        # 0.25.1: self, hidden_states, router_logits, shared_experts_input, input_ids=None
        if (hasattr(torch, "compiler") and torch.compiler.is_compiling()) or (
                hasattr(torch, "_dynamo") and torch._dynamo.is_compiling()):
            return _orig_apply_quant_method(self, *args, **kwargs)

        return _orig_apply_quant_method(self, *args, **kwargs)

    MoERunner._apply_quant_method = patched_apply_quant_method

    _orig_maybe_reduce_final_output = MoERunner._maybe_reduce_final_output

    def patched_maybe_reduce_final_output(self, states, trunc_size):
        return _orig_maybe_reduce_final_output(self, states, trunc_size)

    MoERunner._maybe_reduce_final_output = patched_maybe_reduce_final_output

    # 1. Patch the model registry to resolve to the AMD (ROCm) variant.
    # The AMD variant is more extensible and allows us to swap attention/MHC classes.
    import vllm.models.deepseek_v4 as ds_v4

    def patch_load_weights_for_class(cls):
        _orig_load_weights = cls.load_weights

        def patched_load_weights(self, weights):

            # Check if all expert layers have been pre-quantized and cached on disk.
            cache_dir = os.environ.get("VLLM_FP4_CACHE_DIR", "")
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cached_files = [
                    f for f in os.listdir(cache_dir) if f.endswith(".pt")
                ]
                # Total expected cache files = (num_hidden_layers - first_k_dense_replace) * tp_size
                from vllm.distributed import \
                    get_tensor_model_parallel_world_size
                first_k_dense = getattr(self.config, "first_k_dense_replace",
                                        1)
                num_moe_layers = self.config.num_hidden_layers - first_k_dense
                tp_size = get_tensor_model_parallel_world_size()
                expected_files = num_moe_layers * tp_size
                has_all_cached = (len(cached_files) >= expected_files)
            else:
                has_all_cached = False

            if has_all_cached:
                import vllm.model_executor.model_loader.weight_utils as wu
                logger.info(
                    "All expert weights are cached on disk. Skipping MoE experts from checkpoint files."
                )
                _orig_should_skip_weight = wu.should_skip_weight

                def patched_should_skip_weight(weight_name, local_expert_ids):
                    if ".experts." in weight_name:
                        return True
                    return _orig_should_skip_weight(weight_name,
                                                    local_expert_ids)

                wu.should_skip_weight = patched_should_skip_weight

            self.model.has_tilelang = False
            for idx, l in enumerate(self.model.layers):  # noqa: E741
                l.layer_idx = idx
                l.has_tilelang = False
                if hasattr(l, "ffn"):
                    l.ffn.layer_idx = idx
                    if hasattr(l.ffn, "experts"):
                        l.ffn.experts.layer_idx = idx
                        # layer_idx is read off the RoutedExperts submodule
                        # (not just the owning MoERunner) by quant methods'
                        # process_weights_after_loading, e.g. for the
                        # VLLM_FP4_CACHE_DIR pre-quantized-weight cache path
                        # in mxfp4.py -- set it there too.
                        routed_experts_for_idx = getattr(
                            l.ffn.experts, "routed_experts", None)
                        if routed_experts_for_idx is not None:
                            routed_experts_for_idx.layer_idx = idx
                        # Hash-MoE layers (layer_idx < config.num_hash_layers)
                        # route by gate.tid2eid[token_id], not by router logits.
                        # The TPU routing (moe_routing.select_experts) receives
                        # the RoutedExperts module as `layer`, so expose the
                        # table there (plain attr; same Parameter object, so it
                        # sees the checkpoint data once loaded).
                        tid2eid = getattr(getattr(l.ffn, "gate", None),
                                          "tid2eid", None)
                        if tid2eid is not None:
                            runner = l.ffn.experts
                            for tgt in (runner,
                                        getattr(runner, "routed_experts",
                                                None)):
                                if tgt is not None:
                                    object.__setattr__(tgt,
                                                       "hash_indices_table",
                                                       tid2eid)
            from vllm.models.deepseek_v4.amd.model import \
                _make_deepseek_v4_weights_mapper
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
                        pass
                        if not is_fp4_experts and ".experts." in name and name.endswith(
                                "weight_scale"):
                            name = name + "_inv"
                        if "hc_head.hc_" in name:
                            name = name.replace("hc_head.hc_", "hc_head_")

                        # Map shared experts from HF (gate_proj/up_proj/down_proj) to vLLM (w1/w3/w2)
                        # so they are matched by stacked_params_mapping and loaded by the normal loader
                        if ".shared_experts.gate_proj" in name:
                            name = name.replace(".shared_experts.gate_proj",
                                                ".shared_experts.w1")
                        elif ".shared_experts.up_proj" in name:
                            name = name.replace(".shared_experts.up_proj",
                                                ".shared_experts.w3")
                        elif ".shared_experts.down_proj" in name:
                            name = name.replace(".shared_experts.down_proj",
                                                ".shared_experts.w2")

                        if "layers." in name:
                            try:
                                layer_idx = int(
                                    name.split("layers.")[1].split(".")[0])
                                if layer_idx >= outer_self.config.num_hidden_layers:
                                    continue
                            except (ValueError, IndexError):
                                pass
                        # Map MHC layer parameters from HF (nested) to vLLM (flat)
                        if ".attn_hc.base" in name:
                            name = name.replace(".attn_hc.base",
                                                ".hc_attn_base")
                        elif ".attn_hc.fn" in name:
                            name = name.replace(".attn_hc.fn", ".hc_attn_fn")
                        elif ".attn_hc.weight_scale_inv" in name:
                            name = name.replace(".attn_hc.weight_scale_inv",
                                                ".hc_attn_scale")
                        elif ".ffn_hc.base" in name:
                            name = name.replace(".ffn_hc.base", ".hc_ffn_base")
                        elif ".ffn_hc.fn" in name:
                            name = name.replace(".ffn_hc.fn", ".hc_ffn_fn")
                        elif ".ffn_hc.weight_scale_inv" in name:
                            name = name.replace(".ffn_hc.weight_scale_inv",
                                                ".hc_ffn_scale")

                        # Map layernorms
                        if ".input_layernorm" in name:
                            name = name.replace(".input_layernorm",
                                                ".attn_norm")
                        elif ".post_attention_layernorm" in name:
                            name = name.replace(".post_attention_layernorm",
                                                ".ffn_norm")

                        # Map mlp to ffn
                        if ".mlp." in name:
                            name = name.replace(".mlp.", ".ffn.")

                        # Map attention parameters
                        # 1. Indexer Compressor mappings (HF -> vLLM)
                        if ".self_attn.compressor.indexer.gate_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.gate_proj",
                                ".attn.indexer.compressor.wgate")
                        elif ".self_attn.compressor.indexer.kv_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.kv_proj",
                                ".attn.indexer.compressor.wkv")
                        elif ".self_attn.compressor.indexer.kv_norm" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.kv_norm",
                                ".attn.indexer.compressor.norm")
                        elif ".self_attn.compressor.indexer.position_bias" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.position_bias",
                                ".attn.indexer.compressor.ape")

                        # 2. Indexer main mappings (HF -> vLLM)
                        elif ".self_attn.compressor.indexer.q_b_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.q_b_proj",
                                ".attn.indexer.wq_b")
                        elif ".self_attn.compressor.indexer.scorer.weights_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.indexer.scorer.weights_proj",
                                ".attn.indexer.weights_proj")

                        # 3. Attention Compressor mappings (HF -> vLLM)
                        elif ".self_attn.compressor.gate_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.gate_proj",
                                ".attn.compressor.wgate")
                        elif ".self_attn.compressor.kv_proj" in name:
                            name = name.replace(
                                ".self_attn.compressor.kv_proj",
                                ".attn.compressor.wkv")
                        elif ".self_attn.compressor.kv_norm" in name:
                            name = name.replace(
                                ".self_attn.compressor.kv_norm",
                                ".attn.compressor.norm")
                        elif ".self_attn.compressor.position_bias" in name:
                            name = name.replace(
                                ".self_attn.compressor.position_bias",
                                ".attn.compressor.ape")
                        elif ".self_attn.q_a_proj" in name:
                            name = name.replace(".self_attn.q_a_proj",
                                                ".attn.wq_a")
                        elif ".self_attn.q_a_norm" in name:
                            name = name.replace(".self_attn.q_a_norm",
                                                ".attn.q_norm")
                        elif ".self_attn.q_b_proj" in name:
                            name = name.replace(".self_attn.q_b_proj",
                                                ".attn.wq_b")
                        elif ".self_attn.kv_proj" in name:
                            name = name.replace(".self_attn.kv_proj",
                                                ".attn.wkv")
                        elif ".self_attn.kv_norm" in name:
                            name = name.replace(".self_attn.kv_norm",
                                                ".attn.kv_norm")
                        elif ".self_attn.o_a_proj" in name:
                            name = name.replace(".self_attn.o_a_proj",
                                                ".attn.wo_a")
                        elif ".self_attn.o_b_proj" in name:
                            name = name.replace(".self_attn.o_b_proj",
                                                ".attn.wo_b")
                        elif ".self_attn.sinks" in name:
                            name = name.replace(".self_attn.sinks",
                                                ".attn.attn_sink")

                        if ".experts." not in name and name.endswith(
                                "weight_scale_inv"):
                            name = name.replace("weight_scale_inv",
                                                "weight_scale")

                        yield name, loaded_weight

            # Check if all expert layers have been pre-quantized and cached on disk.
            # If so, we can skip loading the heavy expert weights from safetensors to save CPU memory and time.
            cache_dir = os.environ.get("VLLM_FP4_CACHE_DIR", "")
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                cached_files = [
                    f for f in os.listdir(cache_dir) if f.endswith(".pt")
                ]
                # Total expected cache files = (num_hidden_layers - first_k_dense_replace) * tp_size
                from vllm.distributed import \
                    get_tensor_model_parallel_world_size
                first_k_dense = getattr(self.config, "first_k_dense_replace",
                                        1)
                num_moe_layers = self.config.num_hidden_layers - first_k_dense
                tp_size = get_tensor_model_parallel_world_size()
                expected_files = num_moe_layers * tp_size
                has_all_cached = (len(cached_files) >= expected_files)
            else:
                has_all_cached = False

            if has_all_cached:
                logger.warning(
                    "[PATCH] All expert weights are cached. Skipping loading expert weights from safetensors checkpoint!"
                )

                def filter_expert_weights(weights_iter):
                    for name, loaded_weight in weights_iter:
                        if ".experts." in name:
                            continue
                        yield name, loaded_weight

                weights = filter_expert_weights(weights)
            else:
                logger.warning(
                    "[PATCH] Cached expert weights not found or incomplete. Loading expert weights from safetensors checkpoint to generate cache."
                )

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
                        for mapped_name, mapped_weight in mapper.apply([
                            (name, loaded_weight)
                        ]):
                            yield mapped_name, mapped_weight

                weights = attention_requantizer_generator(
                    mapped_weights_generator(weights))

                class IdentityMapper:

                    def apply(self, w):

                        def mapper_generator(w_iter):
                            for name, loaded_weight in w_iter:
                                if ".shared_experts.w2" in name:
                                    name = name.replace(
                                        ".shared_experts.w2",
                                        ".shared_experts.down_proj")
                                yield name, loaded_weight

                        return mapper_generator(w)

                self.hf_to_vllm_mapper = IdentityMapper()

                res = _orig_load_weights(self, weights)

                # NOTE (vLLM 0.25.1): process_weights_after_loading for MoE
                # layers used to need a manual call here, because the
                # standard vLLM loader's `hasattr(module, "quant_method")`
                # check returned False for FusedMoE layers (which exposed
                # their quant method as `_quant_method` instead). That is no
                # longer true: FusedMoE is now a factory function returning a
                # MoERunner, and the actual weight-holding submodule
                # (RoutedExperts, at e.g. `layer.ffn.experts.routed_experts`)
                # sets `self.quant_method` directly in its own __init__
                # (vllm/model_executor/layers/fused_moe/routed_experts.py).
                # vLLM's own generic `process_weights_after_loading()` walker
                # (model_loader/utils.py, called right after this function
                # returns) now finds and calls it automatically via
                # `model.named_modules()`. A manual call here would double-
                # process every MoE layer's weights -- the second call reads
                # the *already-processed* (packed/quantized/on-TPU) tensors
                # as if they were still raw, corrupting them or crashing with
                # a shape mismatch.

                import torch.distributed as dist
                if dist.is_initialized():
                    logger.info(
                        "Aligning workers at barrier after _orig_load_weights")
                    dist.barrier()

                from torch_tpu._internal import sync
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
        logger.error(f"Failed to patch CUDA variant: {e}")

    _AmdDeepseekV4ForCausalLM = None
    try:
        from vllm.models.deepseek_v4.amd.model import \
            DeepseekV4ForCausalLM as _AmdDeepseekV4ForCausalLM
        ds_v4.DeepseekV4ForCausalLM = _AmdDeepseekV4ForCausalLM
        patch_load_weights_for_class(_AmdDeepseekV4ForCausalLM)
    except Exception as e:
        logger.error(f"Failed to patch AMD variant: {e}")

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
        from vllm.models.deepseek_v4.amd.model import \
            DeepseekV4DecoderLayer as AmdDecoderLayer
        AmdDecoderLayer.load_weights = patched_decoder_load_weights
        logger.info(
            "Successfully patched AMD DeepseekV4DecoderLayer.load_weights")
    except Exception as e:
        logger.error(f"Failed to patch AMD DecoderLayer: {e}")

    try:
        try:
            from vllm.models.deepseek_v4.nvidia.model import \
                DeepseekV4DecoderLayer as CudaDecoderLayer
        except ImportError:
            from vllm.models.deepseek_v4.model import \
                DeepseekV4DecoderLayer as CudaDecoderLayer
        CudaDecoderLayer.load_weights = patched_decoder_load_weights
        logger.info(
            "Successfully patched CUDA/NVIDIA DeepseekV4DecoderLayer.load_weights"
        )
    except Exception as e:
        logger.error(f"Failed to patch CUDA/NVIDIA DecoderLayer: {e}")

    if _AmdDeepseekV4ForCausalLM is not None:
        _orig_amd_forward = _AmdDeepseekV4ForCausalLM.forward

        def patched_amd_forward(self,
                                input_ids: torch.Tensor,
                                positions: torch.Tensor,
                                intermediate_tensors=None,
                                inputs_embeds=None,
                                *args,
                                **kwargs):
            pass
            if os.environ.get("VLLM_BYPASS_TPU_INIT") == "1":
                import jax  # type: ignore[import-not-found]
                import torchax  # type: ignore[import-not-found]
                import torchax.tensor as ta_tensor  # type: ignore[import-not-found]
                with torchax.default_env() as env:

                    def to_torchax(t):
                        if isinstance(
                                t, torch.Tensor) and not hasattr(t, "_elem"):
                            orig_dtype = t.dtype
                            # Check if the dtype requires float32 casting for numpy compatibility
                            if orig_dtype in (torch.bfloat16,
                                              getattr(torch, "float8_e4m3fn",
                                                      None),
                                              getattr(torch, "float8_e5m2",
                                                      None)):
                                np_arr = t.detach().cpu().to(
                                    torch.float32).numpy()
                                jax_arr = jax.device_put(np_arr)
                                if orig_dtype == torch.bfloat16:
                                    jax_arr = jax_arr.astype(
                                        jax.numpy.bfloat16)
                                elif orig_dtype == getattr(
                                        torch, "float8_e4m3fn", None):
                                    jax_arr = jax_arr.astype(
                                        jax.numpy.float8_e4m3fn)
                                elif orig_dtype == getattr(
                                        torch, "float8_e5m2", None):
                                    jax_arr = jax_arr.astype(
                                        jax.numpy.float8_e5m2)
                            else:
                                jax_arr = jax.device_put(
                                    t.detach().cpu().numpy())
                            return ta_tensor.Tensor(jax_arr, env)
                        return t

                    # Convert all model parameters and buffers to torchax.Tensor
                    def convert_module_params(module):
                        for name in list(module._parameters.keys()):
                            param = module._parameters[name]
                            if param is not None and not hasattr(
                                    param, "_elem"):
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
                        **kwargs_ta)

                    def to_torch(t):
                        if hasattr(t, "torch"):
                            return t.torch()
                        return t

                    return jax.tree_util.tree_map(to_torch, res)
            res = _orig_amd_forward(self,
                                    input_ids,
                                    positions,
                                    intermediate_tensors=intermediate_tensors,
                                    inputs_embeds=inputs_embeds,
                                    *args,
                                    **kwargs)
            return res

        _AmdDeepseekV4ForCausalLM.forward = patched_amd_forward

        def patched_compute_logits(self, hidden_states: torch.Tensor):
            return self.logits_processor(self.lm_head, hidden_states)

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
        layer_idx = getattr(self, "layer_idx", -1)
        x = res[0] if isinstance(res, (tuple, list)) else res
        if os.getenv("DSV4_DUMP_HIDDEN_STATES") == "1":
            if isinstance(res, (tuple, list)) and len(res) == 4:
                if any(r is None for r in res):
                    x_post = res[0]
                else:
                    import vllm.model_executor.kernels.mhc as mhc_kernels
                    x_post = mhc_kernels.mhc_post_torch(
                        res[0], res[1], res[2], res[3])
            else:
                x_post = x

            if isinstance(x_post, torch.Tensor):
                rank = torch.distributed.get_rank(
                ) if torch.distributed.is_initialized() else 0
                if rank == 0:
                    if not hasattr(self, "_step_idx"):
                        self._step_idx = 0
                    import numpy as np
                    tensor_cpu = x_post.detach().cpu().to(
                        torch.float32).numpy()
                    os.makedirs(
                        "/home/patemotter_google_com/workspace/scratch/hidden_states/vllm_torchtpu",
                        exist_ok=True)
                    filepath = f"/home/patemotter_google_com/workspace/scratch/hidden_states/vllm_torchtpu/layer{layer_idx:02d}_step{self._step_idx:03d}.npy"
                    np.save(filepath, tensor_cpu)

                    if layer_idx == 0:
                        in_x = args[0] if len(args) > 0 else kwargs.get(
                            "hidden_states", None)
                        if isinstance(in_x, torch.Tensor):
                            in_cpu = in_x.detach().cpu().to(
                                torch.float32).numpy()
                            in_filepath = f"/home/patemotter_google_com/workspace/scratch/hidden_states/vllm_torchtpu/input_step{self._step_idx:03d}.npy"
                            np.save(in_filepath, in_cpu)

                    self._step_idx += 1

            # Layer-substitution hook (see docs/dsv4/00_NEXT_PHASE_PROMPT.md
            # 3.1). Replaces this layer's output with either the
            # tpu-inference reference dump or noise, so a sweep over
            # DSV4_SUBSTITUTE_LAYER=0..42 can localise the first diverging
            # layer by brute force -- cheap in TPU runs, near-zero reasoning.
            #
            #   DSV4_SUBSTITUTE_LAYER  int, which layer to override
            #   DSV4_SUBSTITUTE_MODE   "reference" (default) | "noise"
            #   DSV4_SUBSTITUTE_DIR    reference dump dir (reference mode)
            #
            # "noise" is the control: it must visibly degrade generation. If it
            # does not, the hook is not taking effect and every negative result
            # from the sweep is worthless.
            _sub = os.getenv("DSV4_SUBSTITUTE_LAYER")
            # Only compute the substitution target when actually substituting.
            # This block used to run on every layer of every forward, which
            # cost an extra mhc_post_torch and emitted 11352 [DEBUG_NULL_STREAM]
            # lines in a single 32-token run.
            x_post = None
            if _sub is not None and layer_idx == int(_sub):
                if isinstance(res, (tuple, list)) and len(res) == 4:
                    if any(r is None for r in res):
                        # Normal for most layers: residual/post_mix/res_mix are
                        # only populated in the fused-MHC path. res[0] is
                        # already the [tokens, hc_mult, hidden] stream tensor
                        # (empirically (64, 4, 4096)), which is what the
                        # reference dumps, so it is the right target.
                        x_post = res[0]
                    else:
                        import vllm.model_executor.kernels.mhc as mhc_kernels
                        x_post = mhc_kernels.mhc_post_torch(
                            res[0], res[1], res[2], res[3])
                else:
                    x_post = x

            if (_sub is not None and isinstance(x_post, torch.Tensor)
                    and layer_idx == int(_sub)):
                _mode = os.getenv("DSV4_SUBSTITUTE_MODE", "reference")
                _n = getattr(self, "_sub_step_idx", 0)
                self._sub_step_idx = _n + 1
                _new = None
                if _mode == "shuffle":
                    # PREFERRED CONTROL. Permute the real tensor along the
                    # token axis: identical value distribution, wrong content.
                    # Generation must degrade, but every value stays
                    # in-distribution so downstream Pallas kernels behave.
                    _t = x_post.detach()
                    _perm = torch.randperm(_t.shape[0], device=_t.device)
                    _new = _t[_perm].clone()
                elif _mode == "noise":
                    # Gaussian replacement. NOTE: with the stream tuple
                    # correctly preserved, this propagates through the full
                    # Hyper-Connections machinery and has been observed to trip
                    # a TPU UserFatal (RuntimeUnexpectedCoreHalt) in a
                    # downstream Pallas kernel -- the values are far
                    # out-of-distribution. A crash does demonstrate causal
                    # effect, but you cannot compare generated text against it,
                    # so prefer mode=shuffle for the control.
                    #
                    # NB: torch.Tensor.std() is not implemented on TorchTPU
                    # (aten::std.correction). Compute it manually, the same
                    # way check_nan() in deepseek_v4_attention.py does.
                    _f = x_post.detach().float()
                    _mu = _f.mean()
                    _sd = (_f - _mu).pow(2).mean().sqrt()
                    _new = (torch.randn_like(_f) * _sd + _mu).to(x_post.dtype)
                else:
                    import numpy as np
                    _dir = os.getenv(
                        "DSV4_SUBSTITUTE_DIR",
                        "/home/patemotter_google_com/workspace/scratch/"
                        "hidden_states/tpu_inference")
                    _p = f"{_dir}/layer{layer_idx:02d}_step{_n:03d}.npy"
                    if os.path.exists(_p):
                        _a = np.load(_p)
                        seq_len_jax = _a.shape[0]
                        seq_len_live = x_post.shape[0]
                        if seq_len_jax != seq_len_live and _a.ndim == 3 and x_post.ndim == 3:
                            _a_sliced = _a[0:seq_len_live]
                            if tuple(_a_sliced.shape) == tuple(x_post.shape):
                                _a = _a_sliced
                        if tuple(_a.shape) == tuple(x_post.shape):
                            _new = torch.from_numpy(_a).to(
                                device=x_post.device, dtype=x_post.dtype)
                        else:
                            logger.warning(
                                f"[SUBSTITUTE] SHAPE MISMATCH layer={layer_idx}"
                                f" step={_n} ref={tuple(_a.shape)} "
                                f"live={tuple(x_post.shape)} -- NOT substituted"
                            )
                    else:
                        logger.warning(
                            f"[SUBSTITUTE] missing {_p} -- NOT substituted")
                if _new is not None:
                    logger.info(
                        f"[SUBSTITUTE] layer={layer_idx} step={_n} "
                        f"mode={_mode} shape={tuple(x_post.shape)} "
                        f"absmax {x_post.detach().float().abs().max().item():.4g}"
                        f" -> {_new.detach().float().abs().max().item():.4g}")

                    # Preserve the remaining stream state. The caller threads
                    # all four values through EVERY subsequent layer:
                    #   residual, post_mix, res_mix = None, None, None
                    #   for layer in ...:
                    #       hidden_states, residual, post_mix, res_mix = layer(...)
                    # so returning (_new, None, None, None) resets the
                    # Hyper-Connections state for layers L+1..42 -- they would
                    # behave as if they were the first layer. That confounds
                    # "replace layer L's output" with "wipe the HC streams",
                    # making sweep results uninterpretable.
                    #
                    # NOTE: the noise control does NOT catch this. Generation
                    # still degrades, so the control passes while the
                    # experiment is invalid. See 2.1 rule 5.
                    if isinstance(res, tuple):
                        res = (_new, ) + tuple(res[1:])
                    elif isinstance(res, list):
                        res = [_new] + list(res[1:])
                    else:
                        res = _new
        return res

    DeepseekV4DecoderLayer.forward = patched_layer_forward

    # 1.6. Sub-layer instrumentation (layer 0 only): dump hidden states at
    # each of the 6 checkpoints inside _forward_unfused_post_pre, to localize
    # the Layer-0 divergence found by the corrected hidden-state bisection
    # (docs/bisection_progress_report.md) to attention vs. HC-mixing vs. FFN.
    _orig_forward_unfused_post_pre = DeepseekV4DecoderLayer._forward_unfused_post_pre

    def patched_forward_unfused_post_pre(self,
                                         x,
                                         positions,
                                         input_ids,
                                         post_mix=None,
                                         res_mix=None,
                                         residual=None):
        layer_idx = getattr(self, "layer_idx", -1)
        is_compiling = (hasattr(torch, "compiler")
                        and torch.compiler.is_compiling()) or (hasattr(
                            torch, "_dynamo") and torch._dynamo.is_compiling())
        # Layer 2 (compress_ratio=4, CSA) added to the original layer-0-only
        # instrumentation to bisect the decode-step blowup found between
        # layer 1's and layer 2's output (see bisection_progress_report.md
        # Iteration 7) -- historically also a recurring trouble spot per
        # earlier sessions' notes.
        dump = (os.getenv("DSV4_DUMP_HIDDEN_STATES") == "1"
                and layer_idx in (0, 1, 2) and not is_compiling)
        if not dump:
            return _orig_forward_unfused_post_pre(self, x, positions,
                                                  input_ids, post_mix, res_mix,
                                                  residual)

        rank = torch.distributed.get_rank(
        ) if torch.distributed.is_initialized() else 0
        if rank != 0:
            return _orig_forward_unfused_post_pre(self, x, positions,
                                                  input_ids, post_mix, res_mix,
                                                  residual)

        if not hasattr(self, "_step_idx"):
            self._step_idx = 0
        step = self._step_idx

        import numpy as np
        outdir = "/home/patemotter_google_com/workspace/scratch/hidden_states/vllm_torchtpu"
        os.makedirs(outdir, exist_ok=True)
        layer_suffix = f"_L{layer_idx}" if layer_idx != 0 else ""

        def save(name, t):
            np.save(f"{outdir}/sub_{name}{layer_suffix}_step{step:03d}.npy",
                    t.detach().cpu().to(torch.float32).numpy())

        residual_ = x
        x1, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale,
                                     self.hc_attn_base)
        save("01_hc_pre_attn", x1)
        x2 = self.attn_norm(x1)
        x3 = self.attn(positions, x2, None)
        save("02_post_attn", x3)
        x4 = self.hc_post(x3, residual_, post, comb)
        save("03_hc_post_attn", x4)

        residual_ = x4
        x5, post2, comb2 = self.hc_pre(x4, self.hc_ffn_fn, self.hc_ffn_scale,
                                       self.hc_ffn_base)
        save("04_hc_pre_ffn", x5)
        x6 = self.ffn_norm(x5)
        x7 = self.ffn(x6, input_ids)
        save("05_post_ffn", x7)
        x8 = self.hc_post(x7, residual_, post2, comb2)
        save("06_hc_post_ffn", x8)
        return x8, None, None, None

    DeepseekV4DecoderLayer._forward_unfused_post_pre = patched_forward_unfused_post_pre

    # 2. Clear cached resolved architectures so the AMD model is picked up.
    from vllm.model_executor.model_loader import utils as _ml_utils
    from vllm.model_executor.models.registry import _try_load_model_cls
    _ml_utils._MODEL_ARCH_BY_HASH.clear()
    _try_load_model_cls.cache_clear()

    # 3. Swap the DeepseekV4ROCMAiterMLAAttention class with our TPU implementation.
    from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_attention import \
        patch_deepseek_v4_mla_cls
    patch_deepseek_v4_mla_cls()

    # 4. Mock CUDA device capability, Stream, and Event allocation since they are unavailable on TPU.
    from vllm_torchtpu.models.vllm.deepseek_v4_stubs import patch_cuda_stubs
    _orig_get_device_capability, _orig_cuda_stream, _orig_cuda_event = (
        patch_cuda_stubs())

    # 5. Patch HCHeadOp forward_native with a pure PyTorch implementation.
    from vllm.model_executor.layers.mhc import HCHeadOp

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
        #
        # hc_fn/hc_scale/hc_base are declared dtype=torch.float32 in vLLM's
        # own model definition (models/deepseek_v4/amd/model.py) -- true in
        # both this repo and the reference's mirrored copy -- so the linear
        # projection against hc_fn must run in fp32; casting the normalized
        # activations down to bf16 first (matching the reference's
        # VllmHCHeadOp.forward_tpu) raises `mm(): expected the two arguments
        # to have the same dtype, got bfloat16 vs float32` against these
        # real fp32 weights. That reference code is unreachable here anyway
        # (this platform's `_enum = PlatformEnum.OOT`, so CustomOp dispatch
        # never selects forward_tpu -- see dispatch_forward in
        # vllm/model_executor/custom_op.py), so treat it as unvalidated
        # rather than authoritative for this op.
        residual_flat = hidden_states.flatten(-2).float()
        residual_norm = residual_flat * torch.rsqrt(
            residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps)
        pre_mix = torch.nn.functional.linear(residual_norm, hc_fn)
        pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
        return torch.sum(pre_mix.unsqueeze(-1) * hidden_states.float(),
                         dim=-2).to(hidden_states.dtype)

    import safetensors.torch  # pyrefly: ignore
    safetensors.torch._TYPES["F8_E8M0"] = torch.uint8

    HCHeadOp.forward_native = mhc_hc_head_native

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
            index2name[extract_layer_index(layer_name,
                                           num_attn_module)].append(layer_name)

        for layer_index in sorted(index2name.keys()):
            layer_names = index2name[layer_index]
            for layer_name in layer_names:
                runner_kv_caches.append(kv_caches[layer_name])

        for layer_name, kv_cache in kv_caches.items():
            forward_context[layer_name].kv_cache = kv_cache

    worker_utils.bind_kv_cache = patched_bind_kv_cache
    tpu_runner.bind_kv_cache = patched_bind_kv_cache

    try:
        yield
    finally:
        logger.info(
            "DeepSeek V4 TPU model patches kept active for compilation and worker execution."
        )
