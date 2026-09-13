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
"""Torch half of the DCP indexer merge: the candidate all-gather feeding
`streamidx_topk_dcp_merge`, and the ownership filter that consumes it."""

from __future__ import annotations

import torch

from vllm_torchtpu.distributed.dcp import get_dcp_group


def all_gather_candidates(x: torch.Tensor) -> torch.Tensor:
    """`[num_tokens, k] -> [num_tokens, dcp_size * k]`. The column order is the
    collective's business, not ours. Unchanged without a DCP group."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return x
    return group.all_gather(x.contiguous(), dim=1)


def cp_global_to_local(global_idx: torch.Tensor, dcp_size: int,
                       interleave_c: int) -> torch.Tensor:
    """Inverse of `streamindex_topk.cp_local_to_global`. Needs no rank: once
    the owner is fixed the arithmetic is the same everywhere."""
    if dcp_size == 1:
        return global_idx
    cycle_c = dcp_size * interleave_c
    return (
        torch.div(global_idx, cycle_c, rounding_mode="floor") * interleave_c +
        torch.remainder(global_idx, interleave_c))


def cp_owner_rank(global_idx: torch.Tensor, dcp_size: int,
                  interleave_c: int) -> torch.Tensor:
    """Rank holding `global_idx`, read out of the value. Mirrors
    `streamindex_topk.cp_owner_rank`."""
    if dcp_size == 1:
        return torch.zeros_like(global_idx)
    return torch.remainder(
        torch.div(global_idx, interleave_c, rounding_mode="floor"), dcp_size)


def _left_pack(mine: torch.Tensor, local: torch.Tensor,
               width: int) -> torch.Tensor:
    """Pack the `mine` entries of `local` into a prefix, `-1` padded."""
    num_rows = mine.shape[0]
    mine_i = mine.to(torch.int32)
    slot = torch.cumsum(mine_i, dim=1, dtype=torch.int32) - mine_i
    dst = torch.where(mine, slot, torch.full_like(slot, width)).to(torch.int64)
    src = torch.where(mine, local, torch.full_like(local, -1))
    out = torch.full((num_rows, width + 1),
                     -1,
                     dtype=torch.int32,
                     device=local.device)
    out.scatter_(1, dst, src)
    return out[:, :width]


def select_owned_winners(
    slots: torch.Tensor,
    idxs: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    interleave_c: int,
    gather=all_gather_candidates,
) -> torch.Tensor:
    """Keep the winners this rank owns, left-packed as local indices and `-1`
    padded. Ownership is read out of the index value, so gather order is free."""
    merged = torch.gather(gather(idxs), 1,
                          torch.clamp(slots, min=0).to(torch.int64))
    merged = torch.where(slots >= 0, merged, torch.full_like(merged, -1))
    mine = (merged >= 0) & (cp_owner_rank(merged, dcp_size, interleave_c)
                            == dcp_rank)
    local = cp_global_to_local(torch.clamp(merged, min=0), dcp_size,
                               interleave_c).to(torch.int32)
    return _left_pack(mine, local, slots.shape[1])
