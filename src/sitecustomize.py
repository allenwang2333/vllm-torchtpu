"""Process-wide defaults required before torch_tpu autoloads."""

import os

# torch_tpu reads this during import. If unset, it marks c10d functional
# collectives as Dynamo graph breaks, which prevents vLLM TP graphs from
# compiling. This file is imported by Python before vLLM imports torch.
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS",
                      "false")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")

# Bisect-only workaround: torch-tpu wheels built between torch_tpu b256d993 and
# e1c0da37 read an absl flag before flags are parsed; mark flags parsed at
# interpreter start so those wheels can be exercised.
try:
    import absl.flags as _absl_flags

    _absl_flags.FLAGS.mark_as_parsed()
except Exception:  # noqa: BLE001
    pass
