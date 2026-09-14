"""Process-wide defaults required before torch_tpu autoloads."""

import os

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Auto-apply TPU omni runtime patches, but avoid polluting stdout during
# package manager / build-system probe scripts (e.g. uv, pip, setuptools).
import sys
_cmd = " ".join(sys.argv)
if not any(token in _cmd for token in ("markers", "get_requires_for_build", "prepare_metadata_for_build", "dist_info")):
    try:
        from vllm_torchtpu.omni.patches import apply_omni_patches
        apply_omni_patches()
    except Exception:
        pass


