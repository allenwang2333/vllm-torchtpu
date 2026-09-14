"""Process-wide defaults required before torch_tpu autoloads."""

import os
import sys

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Fix GKE TPU single-host bounds and worker hostname environment issues:
# GKE TPU nodes often set TPU_HOST_BOUNDS="1,2,1" and TPU_WORKER_HOSTNAMES to a warning string,
# which causes libtpu to assume a broken multi-host configuration and fail initialization.
if os.environ.get("TPU_WORKER_HOSTNAMES", "").startswith("WARNING"):
    del os.environ["TPU_WORKER_HOSTNAMES"]
if os.environ.get("TPU_HOST_BOUNDS") in ("1,2,1", "1,2,2", "2,2,1"):
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")

class _OmniPatchFinder:
    """Lazily triggers vLLM-Omni TPU patches only when omni modules are imported."""

    _installed = False

    def find_spec(self, fullname, path=None, target=None):
        if not self._installed and (
            fullname == "vllm_omni"
            or fullname.startswith("vllm_omni.")
            or fullname == "vllm_torchtpu.omni"
            or fullname.startswith("vllm_torchtpu.omni.")
        ):
            self._installed = True
            try:
                from vllm_torchtpu.omni.patches import apply_omni_patches

                apply_omni_patches()
            except Exception:
                pass
        return None


sys.meta_path.insert(0, _OmniPatchFinder())


