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
"""DCP attention for MLA: the head gather and the log-sum-exp merge that
bracket the sharded attention call. The MLA counterpart of `cp_attention.py`."""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu.distributed.dcp import get_dcp_group


def all_gather_heads(q: torch.Tensor) -> torch.Tensor:
    """Widen a rank's query head slice to the whole DCP group's heads:
    `[num_tokens, h, dim] -> [num_tokens, h * dcp_size, dim]`."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return q
    return group.all_gather(q.contiguous(), dim=1)


def merge_lse_partials_scatter_heads(output: torch.Tensor,
                                     lse: torch.Tensor) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers' as
    `o = sum_r softmax_r(lse) * o_r`, then scatter the head axis back."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    weighted = _weight_by_lse(output, lse, group)
    return group.reduce_scatter(weighted, dim=1).to(output.dtype)


def _weight_by_lse(output: torch.Tensor, lse: torch.Tensor,
                   group: Any) -> torch.Tensor:
    """This rank's `softmax_r(lse) * o_r` term, in f32."""
    lse_f32 = lse.to(torch.float32).contiguous()
    # [world_size, num_tokens, num_heads]; no further denominator reduction.
    gathered = group.all_gather(lse_f32.unsqueeze(0), dim=0)
    if not isinstance(gathered, torch.Tensor):
        gathered = torch.cat(list(gathered), dim=0)

    finite = torch.isfinite(gathered)
    shift = torch.where(finite, gathered,
                        torch.full_like(gathered, float("-inf"))).amax(dim=0)
    # No contributor on any rank would give inf - inf; pin the shift to 0.
    empty = ~torch.isfinite(shift)
    shift = torch.where(empty, torch.zeros_like(shift), shift)

    weights = torch.where(finite, torch.exp(gathered - shift.unsqueeze(0)),
                          torch.zeros_like(gathered))
    denom = weights.sum(dim=0)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    mine = weights[int(group.rank_in_group)] / safe_denom
    return (output.to(torch.float32) * mine.unsqueeze(-1)).contiguous()
