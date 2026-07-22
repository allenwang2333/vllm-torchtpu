"""Process-wide defaults required before torch_tpu autoloads."""

import os
import sys

# Dynamically alias tpu_inference imports to vllm_torchtpu for backward compatibility
class ModuleAliasRedirector:
    def find_spec(self, fullname, path, target=None):
        if fullname == "tpu_inference" or fullname.startswith("tpu_inference."):
            alias_name = fullname.replace("tpu_inference", "vllm_torchtpu", 1)
            try:
                __import__(alias_name)
                sys.modules[fullname] = sys.modules[alias_name]
                return sys.modules[alias_name].__spec__
            except ImportError:
                pass
        return None

sys.meta_path.insert(0, ModuleAliasRedirector())

if os.environ.get("VLLM_BYPASS_TPU_INIT") == "1":
    print(f"[SITECUSTOMIZE_JAX_MODE] Short-circuiting sitecustomize for JAX run in pid={os.getpid()}", file=sys.stderr, flush=True)
    try:
        import torch
        torch._dynamo.config.disable = True
        print(f"[SITECUSTOMIZE_PATCH] Disabled PyTorch Dynamo globally in pid={os.getpid()}", file=sys.stderr, flush=True)
        try:
            import vllm.models.deepseek_v4.attention as att_mod
            if not hasattr(att_mod, "DeepseekV4Attention") and hasattr(att_mod, "DeepseekV4MLAAttention"):
                att_mod.DeepseekV4Attention = att_mod.DeepseekV4MLAAttention
        except ImportError:
            pass
        try:
            from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
            def patched_wrapper_call(self, *args, **kwargs):
                return self._call_with_optional_nvtx_range(self.forward, *args, **kwargs)
            TorchCompileWithNoGuardsWrapper.__call__ = patched_wrapper_call
            print(f"[SITECUSTOMIZE_PATCH] Bypassed TorchCompileWithNoGuardsWrapper globally in pid={os.getpid()}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[SITECUSTOMIZE_PATCH] Failed to patch TorchCompileWithNoGuardsWrapper: {e}", file=sys.stderr, flush=True)
        try:
            from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
            def patched_verify_and_split_kv_cache_groups(self):
                attention_groups = []
                for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
                    manager_cls = self.single_type_managers[i].__class__
                    spec = g.kv_cache_spec
                    for existing_spec, group_ids, existing_cls in attention_groups:
                        if existing_spec == spec:
                            assert manager_cls is existing_cls
                            group_ids.append(i)
                            break
                    else:
                        attention_groups.append((spec, [i], manager_cls))
                self.attention_groups = attention_groups
            HybridKVCacheCoordinator.verify_and_split_kv_cache_groups = patched_verify_and_split_kv_cache_groups
            print(f"[SITECUSTOMIZE_PATCH] Patched HybridKVCacheCoordinator globally in pid={os.getpid()}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[SITECUSTOMIZE_PATCH] Failed to patch HybridKVCacheCoordinator: {e}", file=sys.stderr, flush=True)
        try:
            from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper
            _orig_wrapper_load_weights = VllmModelWrapper.load_weights
            class DummyZeroLenDict(dict):
                def __len__(self):
                    return 0
                def __bool__(self):
                    return False
            def patched_wrapper_load_weights(self, *args, **kwargs):
                res = _orig_wrapper_load_weights(self, *args, **kwargs)
                dummy_dict = DummyZeroLenDict(self.vllm_config.compilation_config.static_forward_context)
                self.vllm_config.compilation_config.static_forward_context = dummy_dict
                print(f"[SITECUSTOMIZE_PATCH] Replaced static_forward_context with DummyZeroLenDict on vllm_config globally in pid={os.getpid()}", file=sys.stderr, flush=True)
                return res
            VllmModelWrapper.load_weights = patched_wrapper_load_weights
            print(f"[SITECUSTOMIZE_PATCH] Patched VllmModelWrapper.load_weights globally in pid={os.getpid()}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[SITECUSTOMIZE_PATCH] Failed to patch VllmModelWrapper: {e}", file=sys.stderr, flush=True)
        try:
            from tpu_inference.layers.vllm.custom_ops.deepseek_v4_attention import VllmDeepseekV4MLAAttention
            from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
            if AttentionLayerBase not in VllmDeepseekV4MLAAttention.__bases__:
                VllmDeepseekV4MLAAttention.__bases__ = VllmDeepseekV4MLAAttention.__bases__ + (AttentionLayerBase,)
            print(f"[SITECUSTOMIZE_PATCH] Added AttentionLayerBase to VllmDeepseekV4MLAAttention bases globally in pid={os.getpid()}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[SITECUSTOMIZE_PATCH] Failed to add AttentionLayerBase base class: {e}", file=sys.stderr, flush=True)
        from tpu_inference.layers.vllm.quantization.deepseek_v4_fp8 import VllmDeepseekV4Fp8Config
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from tpu_inference.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod

        _orig_get_quant_method = VllmDeepseekV4Fp8Config.get_quant_method

        def patched_get_quant_method(self, layer, prefix):
            print(f"[SITECUSTOMIZE_PATCH] get_quant_method called for: {layer.__class__.__name__} | {prefix}", file=sys.stderr, flush=True)
            if isinstance(layer, FusedMoE):
                print(f"[SITECUSTOMIZE_PATCH] Redirecting FusedMoE to VllmMxfp4MoEMethod!", file=sys.stderr, flush=True)
                layer.prefix = prefix
                moe_config = self.get_moe_config(layer)
                if self.expert_dtype == "fp4":
                    return VllmMxfp4MoEMethod(moe_config, self.mesh)
                else:
                    from tpu_inference.layers.vllm.quantization.fp8 import VllmFp8MoEMethod
                    layer.moe_config = moe_config
                    return VllmFp8MoEMethod(self, layer, self.mesh)
            return _orig_get_quant_method(self, layer, prefix)

        VllmDeepseekV4Fp8Config.get_quant_method = patched_get_quant_method
        print("[SITECUSTOMIZE_PATCH] Successfully patched VllmDeepseekV4Fp8Config.get_quant_method for JAX.", file=sys.stderr, flush=True)

        # Helper to pre-expand 1D checkpoint scales to match 2D blockwise class parameters
        def pre_expand_scale_tensor(param, loaded_weight):
            cls_name = param.__class__.__name__
            if len(loaded_weight.shape) == 0:
                loaded_weight = loaded_weight.reshape(1)
            param_shape = param.data.shape
            if len(param_shape) == 2 and len(loaded_weight.shape) == 1:
                # Expand 1D scale to match the block size dimension of the parameter [partition_out_features, num_blocks]
                loaded_weight = loaded_weight.unsqueeze(1).expand(loaded_weight.shape[0], param_shape[1])
            print(f"[DEBUG_PRE_EXPAND] param_type: {cls_name} | param_shape: {param_shape} | expanded_shape: {loaded_weight.shape}", file=sys.stderr, flush=True)
            return loaded_weight

        # Patch RowParallelLinear weight_loader_v2
        from vllm.model_executor.layers.linear import RowParallelLinear
        _orig_row_parallel_weight_loader_v2 = RowParallelLinear.weight_loader_v2

        def patched_row_parallel_weight_loader_v2(self, param, loaded_weight):
            cls_name = param.__class__.__name__
            if "ScaleParameter" in cls_name or "scale" in cls_name.lower():
                loaded_weight = pre_expand_scale_tensor(param, loaded_weight)
            return _orig_row_parallel_weight_loader_v2(self, param, loaded_weight)

        RowParallelLinear.weight_loader_v2 = patched_row_parallel_weight_loader_v2
        print("[SITECUSTOMIZE_PATCH] Successfully patched RowParallelLinear.weight_loader_v2 for JAX.", file=sys.stderr, flush=True)

        # Patch ColumnParallelLinear weight_loader_v2
        from vllm.model_executor.layers.linear import ColumnParallelLinear
        _orig_column_parallel_weight_loader_v2 = ColumnParallelLinear.weight_loader_v2

        def patched_column_parallel_weight_loader_v2(self, param, loaded_weight):
            cls_name = param.__class__.__name__
            if "ScaleParameter" in cls_name or "scale" in cls_name.lower():
                loaded_weight = pre_expand_scale_tensor(param, loaded_weight)
            return _orig_column_parallel_weight_loader_v2(self, param, loaded_weight)

        ColumnParallelLinear.weight_loader_v2 = patched_column_parallel_weight_loader_v2
        print("[SITECUSTOMIZE_PATCH] Successfully patched ColumnParallelLinear.weight_loader_v2 for JAX.", file=sys.stderr, flush=True)

        # Patch MergedColumnParallelLinear weight_loader_v2
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear
        _orig_merged_column_parallel_weight_loader_v2 = MergedColumnParallelLinear.weight_loader_v2

        def patched_merged_column_parallel_weight_loader_v2(self, param, loaded_weight, *args, **kwargs):
            cls_name = param.__class__.__name__
            if "ScaleParameter" in cls_name or "scale" in cls_name.lower():
                loaded_weight = pre_expand_scale_tensor(param, loaded_weight)
            return _orig_merged_column_parallel_weight_loader_v2(self, param, loaded_weight, *args, **kwargs)

        MergedColumnParallelLinear.weight_loader_v2 = patched_merged_column_parallel_weight_loader_v2
        print("[SITECUSTOMIZE_PATCH] Successfully patched MergedColumnParallelLinear.weight_loader_v2 for JAX.", file=sys.stderr, flush=True)

        # Patch QKVParallelLinear weight_loader_v2
        from vllm.model_executor.layers.linear import QKVParallelLinear
        _orig_qkv_parallel_weight_loader_v2 = QKVParallelLinear.weight_loader_v2

        def patched_qkv_parallel_weight_loader_v2(self, param, loaded_weight, *args, **kwargs):
            cls_name = param.__class__.__name__
            if "ScaleParameter" in cls_name or "scale" in cls_name.lower():
                loaded_weight = pre_expand_scale_tensor(param, loaded_weight)
            return _orig_qkv_parallel_weight_loader_v2(self, param, loaded_weight, *args, **kwargs)

        QKVParallelLinear.weight_loader_v2 = patched_qkv_parallel_weight_loader_v2
        print("[SITECUSTOMIZE_PATCH] Successfully patched QKVParallelLinear.weight_loader_v2 for JAX.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[SITECUSTOMIZE_PATCH] Failed to patch: {e}", file=sys.stderr, flush=True)
else:
    # PyTorch-specific initialization
    os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                          "false")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

    print(f"[SITECUSTOMIZE_SYS_MODULES] pid={os.getpid()} argv={sys.argv} vllm modules: {[k for k in sys.modules if k.startswith('vllm')]}", file=sys.stderr, flush=True)

    # Register LazyPatcher process-wide at Python startup.
    import vllm_torchtpu

    # Compatibility alias for renamed DeepseekV4Attention class in newer vLLM versions
    try:
        import vllm.models.deepseek_v4.attention as att_mod
        if not hasattr(att_mod, "DeepseekV4Attention") and hasattr(att_mod, "DeepseekV4MLAAttention"):
            att_mod.DeepseekV4Attention = att_mod.DeepseekV4MLAAttention
    except ImportError:
        pass


