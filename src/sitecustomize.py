"""Process-wide defaults required before torch_tpu autoloads."""

import os
import sys

# Dynamically alias tpu_inference imports to vllm_torchtpu for backward compatibility
class ModuleAliasRedirector:
    def find_spec(self, fullname, path, target=None):
        if fullname == "tpu_inference" or fullname.startswith("tpu_inference."):
            alias_name = fullname.replace("tpu_inference", "vllm_torchtpu", 1)
            try:
                import importlib
                mod = importlib.import_module(alias_name)
                sys.modules[fullname] = mod
                return mod.__spec__
            except ImportError:
                pass
        return None

sys.meta_path.insert(0, ModuleAliasRedirector())

if os.environ.get("VLLM_BYPASS_TPU_INIT") == "1":
    print(f"[SITECUSTOMIZE_JAX_MODE] Short-circuiting sitecustomize for JAX run in pid={os.getpid()}", file=sys.stderr, flush=True)
    try:
        import torchax
        torchax.enable_globally()
        print("[SITECUSTOMIZE_JAX_MODE] Successfully enabled torchax globally.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[SITECUSTOMIZE_JAX_MODE] Failed to enable torchax: {e}", file=sys.stderr, flush=True)
    
    # Still import vllm_torchtpu so process-wide hooks / aliases load
    try:
        import vllm_torchtpu
    except Exception:
        pass
    sys.exit(0) if "run_jax_verify.py" not in "".join(sys.argv) and "EngineCore" not in "".join(sys.argv) else None

# Apply monkey-patches to prevent device registration conflicts process-wide
try:
    import torch
    import torch.utils
    _orig_rename = getattr(torch.utils, "rename_privateuse1_backend", None)
    if _orig_rename is not None:
        def patched_rename(name):
            try:
                _orig_rename(name)
            except RuntimeError as e:
                if "already been set" in str(e):
                    pass
                else:
                    raise e
        torch.utils.rename_privateuse1_backend = patched_rename

    _orig_register = getattr(torch, "_register_device_module", None)
    if _orig_register is not None:
        def patched_register(device_type, module):
            try:
                _orig_register(device_type, module)
            except RuntimeError as e:
                if "already be" in str(e) or "Expected one of" in str(e) or "already registered" in str(e):
                    pass
                else:
                    raise e
        torch._register_device_module = patched_register
except ImportError:
    pass

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
