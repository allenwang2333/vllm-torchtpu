"""Process-wide defaults required before torch_tpu autoloads."""

import os
import sys

if os.environ.get("VLLM_BYPASS_TPU_INIT") == "1":
    print(f"[SITECUSTOMIZE_JAX_MODE] Short-circuiting sitecustomize for JAX run in pid={os.getpid()}", file=sys.stderr, flush=True)
    try:
        from tpu_inference.layers.vllm.quantization.deepseek_v4_fp8 import VllmDeepseekV4Fp8Config
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from tpu_inference.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod

        _orig_get_quant_method = VllmDeepseekV4Fp8Config.get_quant_method

        def patched_get_quant_method(self, layer, prefix):
            print(f"[SITECUSTOMIZE_PATCH] get_quant_method called for: {layer.__class__.__name__} | {prefix}", file=sys.stderr, flush=True)
            if isinstance(layer, FusedMoE):
                print(f"[SITECUSTOMIZE_PATCH] Redirecting FusedMoE to VllmMxfp4MoEMethod!", file=sys.stderr, flush=True)
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
