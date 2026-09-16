"""Process-wide defaults required before torch_tpu autoloads."""

import os

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Bisect-only hypothesis test: disable torch_tpu's reassociate_norm_weights
# pass (torch_tpu b256d993) on the otherwise unchanged bad stack.
if os.environ.get("ROLE") == "decode":
    os.environ.setdefault("TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS", "0")
