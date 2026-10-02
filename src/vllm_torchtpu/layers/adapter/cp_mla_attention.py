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
"""DCP (Decode Context Parallelism) attention for MLA: the head gather and
the log-sum-exp merge that bracket the sharded attention call.

The MLA counterpart of `cp_attention.py`.

Every collective runs on tensors already in torch's hands, outside
the exported op.

Each DCP rank stores a shard of the KV cache, so it can only produce a
*partial* attention result.

  1. `all_gather_heads` widens this rank's query head slice to the whole
     group's heads, so every rank's heads are attended against every rank's
     KV shard.
  2. the kernel runs against this rank's shard alone, returning a partial
     output and its log-sum-exp.
  3. `merge_lse_partials_scatter_heads` folds the partials into the exact
     whole-context result and narrows the head axis back to this rank's own.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import get_dcp_group

# TPU_DCP_COLLECTIVE_IMPL=a2a cut-offs, from the workload-microbenchmark collective component on
# 2x2x2 (16 TensorCores, 2 hosts). XLA's all-to-all is one all-pairs exchange (~8-9 us when small),
# its all-gather / reduce-scatter a multi-step schedule (~15-17 us small; ~48-52 us for a 2 MiB
# 3-D reduce-scatter). A gather as an all-to-all sends `world` copies of the input, so it only
# wins while that is small: q_pe (32 KiB) and lse (64 KiB) 17 -> 9.5 us, q_nope (256 KiB) no
# change. The reduce-scatter as all-to-all + local sum moves the same bytes: 52 -> 30 us at 2 MiB,
# and loses at large sizes (the mixed step's 512 MiB).
_A2A_GATHER_MAX_BYTES = 128 * 1024
_A2A_SCATTER_MAX_BYTES = 4 * 1024 * 1024


def _a2a(x: torch.Tensor, group: Any) -> torch.Tensor:
    """all_to_all_single over dim 0: chunk j goes to rank j, chunk i of the result came from
    rank i."""
    import torch.distributed._functional_collectives as fc

    return fc.all_to_all_single(x, None, None, group.device_group)


def _gather_as_a2a(x: torch.Tensor, group: Any) -> torch.Tensor | None:
    """`[world, *x.shape]`, rank i's `x` at index i, or None when the all-to-all would be
    larger than the gather's latency floor is worth."""
    if envs.TPU_DCP_COLLECTIVE_IMPL != "a2a":
        return None
    world = int(group.world_size)
    if world * x.numel() * x.element_size() > _A2A_GATHER_MAX_BYTES:
        return None
    sent = x.unsqueeze(0).expand(world, *x.shape).contiguous()
    return _a2a(sent.view(world * x.shape[0], *x.shape[1:]), group).view(world, *x.shape)


def all_gather_heads(q: torch.Tensor) -> torch.Tensor:
    """Widen a rank's query head slice to the whole DCP group's heads.

    DCP doesn't add a new parallel dimension—it borrows from TP. It
    reallocates dcp ways from heads to sequence positions, reducing
    head sharding from tp to tp // dcp.

    Args:
      q: `[num_tokens, tp_local_num_heads, dim]`, this rank's head slice.

    Returns:
      `[num_tokens, tp_local_num_heads * dcp_size, dim]`, identical
      on every rank. Returned unchanged without a DCP group.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return q
    stacked = _gather_as_a2a(q.contiguous(), group)
    if stacked is not None:
        # [world, T, h, d] -> [T, world * h, d], rank-major heads like all_gather(dim=1)
        t, h, d = q.shape
        return stacked.permute(1, 0, 2, 3).reshape(t, -1, d)
    return group.all_gather(q.contiguous(), dim=1)


def merge_lse_partials_scatter_heads(
    output: torch.Tensor, lse: torch.Tensor
) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers'.

    With per-rank normalized outputs `o_r` and `lse_r = m_r + log(l_r)`, the
    exact whole-context result is `o = sum_r softmax_r(lse) * o_r`. Correctness
    does not depend on the order the ranks are combined in.

    Two collectives:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         Leaves every rank holding every `lse_r`.
      2. reduce-scatter the weighted outputs

    Args:
      output: `[num_tokens, tp_local_num_heads * dcp_size, kv_lora_rank]`,
        this rank's partial.
      lse: `[num_tokens, tp_local_num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.

    Returns:
      `[num_tokens, tp_local_num_heads, kv_lora_rank]` in `output.dtype` --
      this rank's head slice of the whole-context result.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    weighted = _weight_by_lse(output, lse, group)
    world = int(group.world_size)
    if (
        envs.TPU_DCP_COLLECTIVE_IMPL == "a2a"
        and weighted.numel() * weighted.element_size() <= _A2A_SCATTER_MAX_BYTES
    ):
        # Send each rank its own heads ([T, world*h, D] -> [world, T, h, D], chunk j for rank
        # j), then sum the world contributions locally: reduce_scatter(dim=1) as one exchange.
        t, heads, d = weighted.shape
        h = heads // world
        sent = weighted.view(t, world, h, d).permute(1, 0, 2, 3).contiguous()
        got = _a2a(sent.view(world * t, h, d), group).view(world, t, h, d)
        return got.sum(dim=0).to(output.dtype)
    return group.reduce_scatter(weighted, dim=1).to(output.dtype)


def _weight_by_lse(output: torch.Tensor, lse: torch.Tensor, group: Any) -> torch.Tensor:
    """This rank's `softmax_r(lse) * o_r` term, in f32."""
    lse_f32 = lse.to(torch.float32).contiguous()
    # [world_size, num_tokens, num_heads]: small, and it removes the need for
    # any further reduction over the denominator.
    gathered = _gather_as_a2a(lse_f32, group)
    if gathered is None:
        gathered = group.all_gather(lse_f32.unsqueeze(0), dim=0)
    if not isinstance(gathered, torch.Tensor):
        gathered = torch.cat(list(gathered), dim=0)

    finite = torch.isfinite(gathered)
    shift = torch.where(
        finite, gathered, torch.full_like(gathered, float("-inf"))
    ).amax(dim=0)
    # A token with no contributor on any rank would give inf - inf; pin the
    # shift to 0 there and let the zero denominator carry the result to zero.
    empty = ~torch.isfinite(shift)
    shift = torch.where(empty, torch.zeros_like(shift), shift)

    weights = torch.where(
        finite, torch.exp(gathered - shift.unsqueeze(0)), torch.zeros_like(gathered)
    )
    denom = weights.sum(dim=0)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    # This rank's row of `gathered` is `lse_f32` itself. Weighting from the
    # local copy instead of indexing by rank keeps the rank out of the traced
    # program, so every DCP rank compiles the same executable.
    mine = (
        torch.where(
            torch.isfinite(lse_f32),
            torch.exp(lse_f32 - shift),
            torch.zeros_like(lse_f32),
        )
        / safe_denom
    )
    return (output.to(torch.float32) * mine.unsqueeze(-1)).contiguous()
