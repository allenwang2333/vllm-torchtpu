"""Process-wide defaults required before torch_tpu autoloads."""

import os

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

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

import os
import sys
print(f"[SITECUSTOMIZE_SYS_MODULES] pid={os.getpid()} argv={sys.argv} vllm modules: {[k for k in sys.modules if k.startswith('vllm')]}", file=sys.stderr, flush=True)

# Register LazyPatcher process-wide at Python startup.
import vllm_torchtpu
