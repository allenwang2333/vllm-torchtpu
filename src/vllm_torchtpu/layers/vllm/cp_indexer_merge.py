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
    """`[num_tokens, k] -> [num_tokens, dcp_size * k]` in source-rank order,
    so column `c` belongs to rank `c // k`. Unchanged without a DCP group."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return x
    return group.all_gather(x.contiguous(), dim=1)


def cp_global_to_local(global_idx: torch.Tensor, dcp_size: int,
                       interleave_c: int) -> torch.Tensor:
    """Inverse of `streamindex_topk.cp_local_to_global`. Needs no rank: once
    the owner is fixed the arithmetic is the same everywhere, which is what
    lets the merge address a winner into its owner's cache from any rank."""
    if dcp_size == 1:
        return global_idx
    cycle_c = dcp_size * interleave_c
    return (
        torch.div(global_idx, cycle_c, rounding_mode="floor") * interleave_c +
        torch.remainder(global_idx, interleave_c))


def select_owned_winners(
    slots: torch.Tensor,
    idxs: torch.Tensor,
    *,
    k: int,
    dcp_size: int,
    dcp_rank: int,
    interleave_c: int,
) -> torch.Tensor:
    """Keep the winners this rank owns, left-packed as local indices and `-1`
    padded. Ownership is `slots // k`, so the gathered `idxs` are never read."""
    num_rows, width = slots.shape
    owner = torch.div(slots, k, rounding_mode="floor")
    mine = (slots >= 0) & (owner == dcp_rank)

    col = torch.clamp(slots - dcp_rank * k, 0, k - 1).to(torch.int64)
    global_idx = torch.gather(idxs, 1, col)
    mine = mine & (global_idx >= 0)
    local = cp_global_to_local(torch.clamp(global_idx, min=0), dcp_size,
                               interleave_c).to(torch.int32)

    mine_i = mine.to(torch.int32)
    slot = torch.cumsum(mine_i, dim=1, dtype=torch.int32) - mine_i
    dst = torch.where(mine, slot, torch.full_like(slot, width)).to(torch.int64)
    src = torch.where(mine, local, torch.full_like(local, -1))
    out = torch.full((num_rows, width + 1),
                     -1,
                     dtype=torch.int32,
                     device=slots.device)
    out.scatter_(1, dst, src)
    return out[:, :width]
