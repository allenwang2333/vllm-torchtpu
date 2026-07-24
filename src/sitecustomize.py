"""Process-wide defaults required before torch_tpu autoloads."""

import os
import sys

try:
    import triton
    triton.next_power_of_2 = lambda x: 1 << (int(x) - 1).bit_length()
except Exception:
    pass

# Detect if we are running pytest unit tests on CPU, and configure JAX/Pallas env variables
# BEFORE any package (including vllm_torchtpu) imports jax.
if any("pytest" in arg for arg in sys.argv) and "--use-tpu" not in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["PALLAS_INTERPRET"] = "1"
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8").strip()
    print("[SITECUSTOMIZE] CPU unit test run detected. Configured JAX/Pallas for CPU interpret mode with 8 virtual devices.", file=sys.stderr, flush=True)
    try:
        import jax
        from jax._src.config import pallas_tpu_interpret_mode_context_manager
        from jax._src.pallas.mosaic.interpret.params import InterpretParams
        pallas_tpu_interpret_mode_context_manager.set_global(InterpretParams())
        print("[SITECUSTOMIZE] Successfully enabled Pallas CPU interpret mode globally.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[SITECUSTOMIZE] Failed to enable Pallas CPU interpret mode: {e}", file=sys.stderr, flush=True)

if any("verify_dsv4_equivalence" in arg for arg in sys.argv):
    try:
        import vllm.distributed.parallel_state as ps
        ps.get_tensor_model_parallel_world_size = lambda: 1
        ps.get_tensor_model_parallel_rank = lambda: 0
        import vllm.distributed as dist
        dist.get_tensor_model_parallel_world_size = lambda: 1
        dist.get_tensor_model_parallel_rank = lambda: 0
        
        # Mock CUDA Event and Stream for TPU VM compatibility
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
        class DummyCudaStream:
            def __init__(self, *args, **kwargs):
                pass
        import torch
        torch.cuda.Event = DummyCudaEvent
        torch.cuda.Stream = DummyCudaStream
        
        print("[SITECUSTOMIZE] Equivalence verification run detected. Monkey-patched vllm parallel state and cuda events globally.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[SITECUSTOMIZE] Failed to monkey-patch parallel state / cuda: {e}", file=sys.stderr, flush=True)

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

try:
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    if not hasattr(ds_v4_amd_model, "DeepseekV4Attention"):
        if hasattr(ds_v4_amd_model, "DeepseekV4ROCMAiterMLAAttention"):
            ds_v4_amd_model.DeepseekV4Attention = ds_v4_amd_model.DeepseekV4ROCMAiterMLAAttention
        elif hasattr(ds_v4_amd_model, "DeepseekV4MLAAttention"):
            ds_v4_amd_model.DeepseekV4Attention = ds_v4_amd_model.DeepseekV4MLAAttention
except ImportError:
    pass
