# SPDX-License-Identifier: Apache-2.0
"""Donate only metadata; snapshots retain versions for delayed CPU consumption."""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.hidden_nan_flags import record_hidden_nan

_record = pallas.jax_op(
    "pallas::hidden_nan_flags", record_hidden_nan, donate_argnums=(0,)
)


def _fake_record(
    state, saved_hidden, saved_tokens, armed, hidden, input_ids, input_rows, layer_slot
):
    return tuple(
        torch.empty_like(tensor) for tensor in (state, saved_hidden, saved_tokens)
    )


_record.register_fake(_fake_record)


def record_layer_nan(
    state: torch.Tensor,
    saved_hidden: torch.Tensor,
    saved_tokens: torch.Tensor,
    armed: torch.Tensor,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    input_rows: torch.Tensor,
    layer_slot: int,
) -> None:
    slot = torch.scalar_tensor(layer_slot, dtype=torch.int32, device=hidden.device)
    result = _record(
        state, saved_hidden, saved_tokens, armed, hidden, input_ids, input_rows, slot
    )
    for destination, source in zip((state, saved_hidden, saved_tokens), result):
        destination.copy_(source)
