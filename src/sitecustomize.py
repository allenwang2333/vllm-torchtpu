"""Process-wide defaults required before torch_tpu autoloads."""

import os

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
