# SPDX-License-Identifier: Apache-2.0
"""Torch bridge that donates only the feature-owned layer flag vector."""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.hidden_nan_flags import record_hidden_nan

_record = pallas.jax_op(
    "pallas::hidden_nan_flags", record_hidden_nan, donate_argnums=(0,)
)


def _fake_record(flags, armed, hidden, layer_slot):
    return torch.empty_like(flags)


_record.register_fake(_fake_record)


def record_layer_nan(
    flags: torch.Tensor,
    armed: torch.Tensor,
    hidden: torch.Tensor,
    layer_slot: int,
) -> None:
    slot = torch.scalar_tensor(layer_slot, dtype=torch.int32, device=hidden.device)
    flags.copy_(_record(flags, armed, hidden, slot))
