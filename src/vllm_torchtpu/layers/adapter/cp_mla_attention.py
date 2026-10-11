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
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs


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
    return group.all_gather(q.contiguous(), dim=1)


def all_gather_q_nope_pe_heads(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused all-gather of `(ql_nope, q_pe)` across DCP ranks in a single call."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return ql_nope, q_pe
    q_dcp = group.all_gather(torch.cat([ql_nope, q_pe], dim=-1), dim=1)
    return (
        q_dcp[..., :kv_lora_rank].contiguous(),
        q_dcp[..., kv_lora_rank:].contiguous(),
    )


def merge_lse_partials_scatter_heads(
    output: torch.Tensor, lse: torch.Tensor, is_backbone: bool = True
) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers'.

    With per-rank normalized outputs `o_r` and `lse_r = m_r + log(l_r)`, the
    exact whole-context result is `o = sum_r softmax_r(lse) * o_r`. Correctness
    does not depend on the order the ranks are combined in.

    Two collectives:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         Leaves every rank holding every `lse_r`.
      2. reduce-scatter the weighted outputs

    With TPU_DCP_LSE_MERGE_A2A set, the backbone's steps of at most
    TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS tokens, and every step of a caller that is
    not traced per bucket (the MTP drafter), instead take
    `merge_lse_partials_all_to_all`, which gives the same result.

    Opt-in variants of the reduce-scatter, same weights as `cp_lse_ag_out_rs`:
    TPU_DCP_MERGE_CHUNKED splits a large backbone bucket into token chunks
    (`_merge_chunked`), TPU_DCP_MERGE_HEAD_MAJOR weights the transposed
    partial, and TPU_DCP_LSE_MERGE_RS_BF16 (experimental) reduce-scatters in
    bf16. TPU_DCP_RS_OUT_PRECISION (default on) reduce-scatters the backbone's
    weighted terms in the output's precision on every reduce-scatter path,
    chunked included. With none of the opt-in variants set the merge is exactly
    `cp_lse_ag_out_rs`.

    Args:
      output: `[num_tokens, tp_local_num_heads * dcp_size, kv_lora_rank]`,
        this rank's partial.
      lse: `[num_tokens, tp_local_num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.
      is_backbone: True if the caller is the backbone, the only model that is
        traced separately per token bucket. Only then may the merge compare
        the token count to TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS (or
        TPU_DCP_MERGE_CHUNKED_MIN_TOKENS), since that comparison is fixed at
        trace time. Pass False from the MTP drafter, which is traced once for
        every bucket: with TPU_DCP_LSE_MERGE_A2A it takes the all-to-all at
        every size, otherwise the unsplit reduce-scatter.

    Returns:
      `[num_tokens, tp_local_num_heads, kv_lora_rank]` in `output.dtype` --
      this rank's head slice of the whole-context result.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    # This `if` runs while tracing, so its answer is baked into the graph.
    # Only the backbone is traced once per bucket group, so only it can pick
    # by size: the all-to-all wins at decode sizes, the reduce-scatter above
    # because it sums in the network. The MTP drafter has one graph for every
    # bucket; comparing its token count would pin that graph to the first
    # size traced, so it takes the exchange everywhere, which costs it only
    # its rare large buckets. `is_backbone` is checked before the token count,
    # so the drafter never compares it.
    if envs.TPU_DCP_LSE_MERGE_A2A and (
        not is_backbone or output.shape[0] <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
    ):
        return merge_lse_partials_all_to_all(output, lse, group)

    # TPU_DCP_RS_OUT_PRECISION (backbone only): reduce-scatter the weighted
    # terms in the output's precision. TPU_DCP_LSE_MERGE_RS_BF16 forces it.
    rs_low = envs.TPU_DCP_LSE_MERGE_RS_BF16 or (
        is_backbone and envs.TPU_DCP_RS_OUT_PRECISION
    )
    chunks = _merge_chunks(output.shape[0]) if is_backbone else 1
    if chunks > 1:
        return _merge_chunked(output, lse, group, chunks, rs_low)
    if envs.TPU_DCP_MERGE_HEAD_MAJOR or envs.TPU_DCP_LSE_MERGE_RS_BF16:
        return _reduce_weighted(output, _lse_weights(lse, group), group, rs_low)
    rs_dtype = output.dtype if rs_low else None
    return cp_lse_ag_out_rs(output, lse, group, rs_dtype=rs_dtype)


def _merge_chunks(num_tokens: int) -> int:
    """How many token chunks `TPU_DCP_MERGE_CHUNKED` splits this bucket's
    merge into: 1 (unsplit) unless the bucket reaches
    `TPU_DCP_MERGE_CHUNKED_MIN_TOKENS` and splits into equal chunks whose
    length is a multiple of 8 (the f32 sublane tile)."""
    chunks = envs.TPU_DCP_MERGE_CHUNKED
    if (
        chunks <= 1
        or num_tokens < envs.TPU_DCP_MERGE_CHUNKED_MIN_TOKENS
        or num_tokens % (8 * chunks)
    ):
        return 1
    return chunks


def _merge_chunked(
    output: torch.Tensor, lse: torch.Tensor, group: Any, chunks: int, rs_low: bool
) -> torch.Tensor:
    """The reduce-scatter merge over `chunks` independent token chunks.

    Every token's merge depends on its own row alone, so splitting the token
    axis changes no per-row arithmetic: each chunk gathers its own rows' lse
    (or slices one whole-step gather, `TPU_DCP_MERGE_CHUNKED_LSE_ONCE`), weights
    them and reduce-scatters them exactly as the unsplit merge does. What
    changes is the schedule. The f32 reduce-scatter is a SparseCore-offloaded
    async collective while the lse gather and the weight multiply are
    TensorCore work, so chunk `k`'s reduce-scatter can run under chunk
    `k + 1`'s gather and multiply instead of after all of them.
    """
    step = output.shape[0] // chunks
    mine = _lse_weights(lse, group) if envs.TPU_DCP_MERGE_CHUNKED_LSE_ONCE else None
    pieces = []
    for c in range(chunks):
        rows = slice(c * step, (c + 1) * step)
        weights = mine[rows] if mine is not None else _lse_weights(lse[rows], group)
        pieces.append(_reduce_weighted(output[rows], weights, group, rs_low))
    return torch.cat(pieces, dim=0)


def _lse_weights(lse: torch.Tensor, group: Any) -> torch.Tensor:
    """This rank's `[T, H]` f32 merge weights `exp(lse_r - logsumexp_r lse)`,
    computed exactly as `cp_lse_ag_out_rs` computes them."""
    lse = lse.to(torch.float32)
    lses = group.all_gather(lse.unsqueeze(0).contiguous(), dim=0)
    if not isinstance(lses, torch.Tensor):
        lses = torch.cat(list(lses), dim=0)
    return torch.exp(lse - torch.logsumexp(lses, dim=0))


def _reduce_weighted(
    output: torch.Tensor, mine: torch.Tensor, group: Any, rs_low: bool
) -> torch.Tensor:
    """Weight this rank's partial by its `[T, H]` f32 merge weights and
    reduce-scatter the heads: `[T, H, V]` -> this rank's `[T, H / D, V]`.

    As in `cp_lse_ag_out_rs`, a zero-weight partial is zeroed with `where`
    rather than multiplied (it may be NaN)."""
    if envs.TPU_DCP_MERGE_HEAD_MAJOR:
        # The reduce-scatter scatters its leading axis, so a `[T, H, V]` term
        # is transposed to `[H, T, V]` before it (an f32 relayout of the whole
        # term, offloaded to SparseCore on the mixed step). Weighting the
        # transposed partial instead writes the term head-major in the
        # multiply itself; the elementwise products are the same.
        weight = mine.transpose(0, 1).unsqueeze(-1)
        weighted = output.transpose(0, 1).to(torch.float32) * weight
        weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
        weighted = weighted.contiguous()
        if rs_low:
            weighted = weighted.to(output.dtype)
        return (
            group.reduce_scatter(weighted, dim=0)
            .to(output.dtype)
            .transpose(0, 1)
            .contiguous()
        )
    weight = mine.unsqueeze(-1)
    weighted = output.to(torch.float32) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
    weighted = weighted.contiguous()
    if rs_low:
        weighted = weighted.to(output.dtype)
    return group.reduce_scatter(weighted, dim=1).to(output.dtype)


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block `d` of `x`'s leading axis goes to rank `d`; block `r` of the
    result came from rank `r`."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    return funcol.wait_tensor(out)


def merge_lse_partials_all_to_all(
    output: torch.Tensor, lse: torch.Tensor, group: Any
) -> torch.Tensor:
    """Same result as the reduce-scatter merge, via all-to-all.

    Each rank sends head block `d` of its partial and lse to rank `d`, then
    merges its own heads locally in f32. Faster for small token counts.
    """
    world = int(group.world_size)
    num_tokens, num_heads, vdim = output.shape
    local = num_heads // world
    # Destination-major: [world, T, local, V] and [world, T, local].
    o_send = output.reshape(num_tokens, world, local, vdim).transpose(0, 1)
    l_send = lse.to(torch.float32).reshape(num_tokens, world, local).transpose(0, 1)
    o_recv = _all_to_all(o_send, group)
    l_recv = _all_to_all(l_send, group)
    return _merge_received(o_recv, l_recv).to(output.dtype)


def _merge_received(o_recv: torch.Tensor, l_recv: torch.Tensor) -> torch.Tensor:
    """`[R, T, h, V]` partials and `[R, T, h]` lse -> the f32 `[T, h, V]` merge."""
    finite = torch.isfinite(l_recv)
    shift = torch.where(finite, l_recv, torch.full_like(l_recv, float("-inf"))).amax(
        dim=0
    )
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    weights = torch.where(
        finite, torch.exp(l_recv - shift.unsqueeze(0)), torch.zeros_like(l_recv)
    )
    denom = weights.sum(dim=0)
    weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom))
    # The compiler lowers multiply-then-sum to a dot that rounds f32 operands to
    # bf16. Splitting the weights into a bf16 part and a residual keeps it exact.
    o_f32 = o_recv.to(torch.float32)
    w_hi = weights.to(torch.bfloat16).to(torch.float32)
    w_lo = weights - w_hi
    return (o_f32 * w_hi.unsqueeze(-1)).sum(dim=0) + (o_f32 * w_lo.unsqueeze(-1)).sum(
        dim=0
    )
