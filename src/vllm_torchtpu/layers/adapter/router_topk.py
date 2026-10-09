# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch bridge for the sort-free MoE router top-k.

On TPU the selection runs as the Pallas kernel in ``kernels/router_topk.py``,
elsewhere as the identical plain-torch algorithm in
``layers/core/rowmax_topk.py``.
"""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.router_topk import router_topk, router_topk_with_raw
from vllm_torchtpu.layers.core.rowmax_topk import rowmax_topk

# Above 256 rows, the Pallas custom-call barrier forces the [rows, experts]
# score tensor(s) through HBM between the gate epilogue and renormalization;
# plain-torch rowmax_topk lets XLA fuse the entire router in registers.
PALLAS_MAX_ROWS = 256


def _fake_router_topk(scores: torch.Tensor, topk: int):
    rows = scores.shape[0]
    return (
        torch.empty((rows, topk), dtype=torch.float32, device=scores.device),
        torch.empty((rows, topk), dtype=torch.int32, device=scores.device),
    )


def _fake_router_topk_with_raw(
    scores: torch.Tensor, raw_scores: torch.Tensor, topk: int
):
    rows = scores.shape[0]
    return (
        torch.empty((rows, topk), dtype=torch.float32, device=scores.device),
        torch.empty((rows, topk), dtype=torch.int32, device=scores.device),
    )


# Registered at import, not lazily: this op is called from inside the MoE
# forward, and Dynamo cannot trace a lock-guarded registration there.
_op = pallas.jax_op("pallas::moe_router_topk", router_topk)
_op.register_fake(_fake_router_topk)

_op_with_raw = pallas.jax_op(
    "pallas::moe_router_topk_with_raw", router_topk_with_raw
)
_op_with_raw.register_fake(_fake_router_topk_with_raw)


def rowmax_select(
    scores: torch.Tensor,
    topk: int,
    raw_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort-free ``topk`` over the expert axis of ``[rows, experts]`` scores."""
    if scores.device.type != "tpu":
        return rowmax_topk(scores, topk, raw_scores=raw_scores)
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    scores_f32 = scores.float()
    raw_f32 = raw_scores.float() if raw_scores is not None else None
    if scores_f32.shape[0] > PALLAS_MAX_ROWS:
        return rowmax_topk(scores_f32, topk, raw_scores=raw_f32)
    if raw_f32 is not None:
        return _op_with_raw(scores_f32, raw_f32, topk)
    return _op(scores_f32, topk)
