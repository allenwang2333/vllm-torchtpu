"""Process-wide defaults required before torch_tpu autoloads."""

import os
import sys

if os.environ.get("VLLM_BYPASS_TPU_INIT") == "1":
    print(f"[SITECUSTOMIZE_JAX_MODE] Short-circuiting sitecustomize for JAX run in pid={os.getpid()}", file=sys.stderr, flush=True)
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
