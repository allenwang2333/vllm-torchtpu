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
# TPU_DCP_COLLECTIVE_IMPL=kernel: the one-shot Pallas collectives (kernels/collectives/oneshot.py)
# where they win on the same 2x2x2 sweep -- every rank DMAs straight into every peer after one
# barrier, ~6 us flat: gather 15-17 -> 6 us up to 512 KiB of output, reduce-scatter 12-48 ->
# 6-17 us up to 1 MiB of input. Above that the kernel sends n x the bytes of a ring and loses; the
# reduce-scatter then takes the all-to-all path (2 MiB partials: kernel 32 us, a2a 30 us).
_KERNEL_GATHER_MAX_BYTES = 512 * 1024
_KERNEL_SCATTER_MAX_BYTES = 1024 * 1024
_KERNEL_OPS: dict[str, Any] = {}


def prepare_dcp_collectives() -> None:
    """Build the one-shot kernel ops for the DCP group, once, outside any trace (building the
    mesh runs a CPU collective). A no-op unless TPU_DCP_COLLECTIVE_IMPL=kernel."""
    if envs.TPU_DCP_COLLECTIVE_IMPL != "kernel" or _KERNEL_OPS:
        return
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return
    from vllm_torchtpu.kernels.collectives import oneshot

    for op in ("all_gather", "reduce_scatter", "all_to_all"):
        _KERNEL_OPS[op] = oneshot.torch_op(op, group, "dcp")


def _gather_as_kernel(x: torch.Tensor, group: Any) -> torch.Tensor | None:
    """`[world, *x.shape]` through the one-shot all-gather, or None outside its range."""
    world = int(group.world_size)
    op = _KERNEL_OPS.get("all_gather")
    if op is None or world * x.numel() * x.element_size() > _KERNEL_GATHER_MAX_BYTES:
        return None
    return op(x.reshape(x.shape[0], -1)).view(world, *x.shape)


def _static(*_xs: torch.Tensor) -> bool:
    """Whether the size cut-offs below may run. They are Python branches on the token count, so
    they are only taken inside the backbone warmup (`shape_variants.in_warmup`), which gives
    each side of a token-count branch its own trace. Every other trace -- the MTP drafter is
    traced once with a symbolic token count and compiled for all buckets -- would freeze the
    branch at its first bucket (ShapeSpecializationError for the rest), so it keeps the stock
    collectives."""
    from vllm_torchtpu.compilation import shape_variants

    return shape_variants.in_warmup()


def _a2a(x: torch.Tensor, group: Any) -> torch.Tensor:
    """all_to_all_single over dim 0: chunk j goes to rank j, chunk i of the result came from
    rank i."""
    import torch.distributed._functional_collectives as fc

    return fc.all_to_all_single(x, None, None, group.device_group)


def _gather_as_a2a(x: torch.Tensor, group: Any) -> torch.Tensor | None:
    """`[world, *x.shape]`, rank i's `x` at index i, or None when the all-to-all would be
    larger than the gather's latency floor is worth."""
    if envs.TPU_DCP_COLLECTIVE_IMPL not in ("a2a", "kernel") or not _static(x):
        return None
    if envs.TPU_DCP_COLLECTIVE_IMPL == "kernel":
        stacked = _gather_as_kernel(x, group)
        if stacked is not None:
            return stacked
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


def all_gather_heads_packed(*parts: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """`all_gather_heads` of several same-head-count tensors in one collective.

    Concatenates `parts` on the last axis, runs a single head all-gather, and
    splits the result back. Every element lands exactly where a separate
    `all_gather_heads` per part would have put it -- the gather is pure data
    movement -- but the group pays one collective launch and one
    rank-major -> token-major relayout instead of one per part.

    Args:
      *parts: `[num_tokens, tp_local_num_heads, dim_i]`, same leading dims.

    Returns:
      One `[num_tokens, tp_local_num_heads * dcp_size, dim_i]` per part.
    """
    if len(parts) == 1:
        return (all_gather_heads(parts[0]),)
    widths = [int(p.shape[-1]) for p in parts]
    gathered = all_gather_heads(torch.cat(parts, dim=-1))
    return tuple(gathered.split(widths, dim=-1))


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

    if (
        envs.TPU_DCP_COLLECTIVE_IMPL in ("a2a", "kernel")
        and _static(output, lse)
        and output.numel() * output.element_size() <= _A2A_SCATTER_MAX_BYTES
    ):
        return _merge_at_receiver(output, lse, group)
    weighted = _weight_by_lse(output, lse, group)
    world = int(group.world_size)
    if not _static(weighted):
        return group.reduce_scatter(weighted, dim=1).to(output.dtype)
    nbytes = weighted.numel() * weighted.element_size()
    op = _KERNEL_OPS.get("reduce_scatter")
    if op is not None and nbytes <= _KERNEL_SCATTER_MAX_BYTES:
        t, heads, d = weighted.shape
        h = heads // world
        sent = weighted.view(t, world, h, d).permute(1, 0, 2, 3).contiguous()
        got = op(sent.view(world * t, h * d))           # this rank's [t, h * d]
        return got.view(t, h, d).to(output.dtype)
    if (
        envs.TPU_DCP_COLLECTIVE_IMPL in ("a2a", "kernel")
        and nbytes <= _A2A_SCATTER_MAX_BYTES
    ):
        # Send each rank its own heads ([T, world*h, D] -> [world, T, h, D], chunk j for rank
        # j), then sum the world contributions locally: reduce_scatter(dim=1) as one exchange.
        t, heads, d = weighted.shape
        h = heads // world
        sent = weighted.view(t, world, h, d).permute(1, 0, 2, 3).contiguous()
        got = _a2a(sent.view(world * t, h, d), group).view(world, t, h, d)
        return got.sum(dim=0).to(output.dtype)
    return group.reduce_scatter(weighted, dim=1).to(output.dtype)


def _merge_at_receiver(output: torch.Tensor, lse: torch.Tensor, group: Any) -> torch.Tensor:
    """The merge with the weighting moved to the receiving rank: no lse all-gather, no f32
    weighted product on the wire.

    Each rank sends rank j the raw partial for j's heads in the kernel's own dtype (bf16) --
    half the bytes of the f32 weighted reduce-scatter -- laid out [world * T, h, D] so the
    exchange keeps the partial's lane-aligned layout, and, in a second tiny all-to-all, the
    matching f32 lse. (Packing the lse into two extra lanes of the partial made rows 514 wide:
    the receive side then paid relayouts and a strided reduce, +35 us a layer on 2x2x2.) The
    receiver holds every rank's (partial, lse) for its heads and computes exactly what
    `_weight_by_lse` + the reduce-scatter compute: sum_r softmax_r(lse) * o_r in f32 (only the
    order of the f32 sum differs).
    """
    world = int(group.world_size)
    t, heads, d = output.shape
    h = heads // world
    sent = output.view(t, world, h, d).permute(1, 0, 2, 3).contiguous()
    part = _a2a(sent.view(world * t, h, d), group).view(world, t, h, d).to(torch.float32)
    lse_sent = lse.to(torch.float32).view(t, world, h).permute(1, 0, 2).contiguous()
    lse_sent = lse_sent.view(world, t * h)
    op = _KERNEL_OPS.get("all_to_all")
    lse_all = op(lse_sent) if op is not None else _a2a(lse_sent, group)
    lse_all = lse_all.view(world, t, h)

    finite = torch.isfinite(lse_all)
    shift = torch.where(finite, lse_all, torch.full_like(lse_all, float("-inf"))).amax(dim=0)
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    weights = torch.where(finite, torch.exp(lse_all - shift.unsqueeze(0)),
                          torch.zeros_like(lse_all))
    denom = weights.sum(dim=0)
    weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom)).unsqueeze(0)
    return (part * weights.unsqueeze(-1)).sum(dim=0).to(output.dtype)


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
