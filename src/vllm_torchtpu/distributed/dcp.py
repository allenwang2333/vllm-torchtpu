# SPDX-License-Identifier: Apache-2.0
"""Torch-level DCP (decode context parallel) helpers.

The twin of `pcp.py`, for the axis that shards KV *positions* rather than
queries. Its one substantive job is the log-sum-exp merge that turns the
per-rank partial attention outputs from
`attention_interface.sparse_mla_attention_dcp` back into the whole-context
result.

The merge lives here, outside the exported JAX op, on purpose: the torch->JAX
export path constrains how collectives inside `pallas.jax_op` are launched
across worker processes, and the merge needs none of that -- it is two plain
process-group calls on tensors that are already in torch's hands.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_torchtpu.distributed.pcp import (PcpGroupLayout,
                                           get_or_create_pcp_mesh,
                                           get_pcp_group_layout)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def get_dcp_group() -> Any | None:
    """Return vLLM's native DCP group, or None before distributed init."""
    try:
        from vllm.distributed.parallel_state import get_dcp_group as _get_group

        return _get_group()
    except (AssertionError, ImportError):
        return None


def get_dcp_rank() -> int:
    group = get_dcp_group()
    return 0 if group is None else int(group.rank_in_group)


def get_dcp_world_size() -> int:
    group = get_dcp_group()
    return 1 if group is None else int(group.world_size)


def get_dcp_group_layout() -> PcpGroupLayout:
    """The DCP group's rank order and the TPU device each rank sits on."""
    return get_pcp_group_layout(get_dcp_group())


def get_or_create_dcp_mesh(axis_name: str | None = None) -> Any:
    """Build the op-local JAX mesh for the DCP axis.

    Shares `pcp.py`'s implementation because none of it is PCP-specific -- the
    mesh is just this group's devices in rank order. Under `tp=N, dcp=N` the
    DCP group covers the same devices as the TP group, and the mesh cache is
    keyed on the axis name as well as the device ids, so a run with both axes
    live keeps two distinct meshes rather than aliasing them.

    The axis name defaults to the kernels' `DCP_AXIS_NAME`, imported here
    rather than at module scope so that the torch-only merge above stays
    importable without pulling in JAX -- the same reason `pcp.py` defers its
    own JAX imports.
    """
    if axis_name is None:
        from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
            DCP_AXIS_NAME
        axis_name = DCP_AXIS_NAME
    return get_or_create_pcp_mesh(axis_name=axis_name, group=get_dcp_group())


def merge_lse_partials(output: torch.Tensor,
                       lse: torch.Tensor) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers'.

    With per-rank normalized outputs `o_r` and `lse_r = m_r + log(l_r)`, the
    exact whole-context result is `o = sum_r softmax_r(lse) * o_r`. Correctness
    does not depend on the order the ranks are combined in -- the partial state
    `(m, l, l*o)` is a commutative monoid -- which is exactly why DCP needs one
    reduction rather than a ring: rotating queries around 16 ranks would move
    ~3 MB per token, and this moves ~64 KB.

    Two collectives, deliberately lopsided:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         256 B per token per rank at 64 heads -- so gathering them outright is
         cheap, and it leaves *every* rank holding every `lse_r`.
      2. all-reduce the weighted outputs, `[num_tokens, num_heads, v_dim]`, the
         one big tensor.

    The point of (1) is that it makes the softmax denominator a local
    computation: with all the `lse_r` in hand each rank derives the shared
    maximum and the full sum itself, so there is no second reduction over the
    large tensor and no numerically fragile `exp` of an unshifted logit.

    Available improvement, not taken here: under `tp=N, dcp=N` every rank
    computes all heads but `o_proj` only consumes its own head slice, so step 2
    could be a reduce-scatter over the head axis -- half the bytes of an
    all-reduce, landing exactly the shape `o_proj` wants. That changes this
    function's output contract, so it belongs with the wiring that can honour
    it rather than here.

    Args:
      output: `[num_tokens, num_heads, v_dim]`, this rank's partial output.
      lse: `[num_tokens, num_heads]`, the matching log-sum-exps. Entries are
        `-inf` for tokens this rank owns none of the top-k for; such a rank was
        still fed a dummy key (the kernel requires `kv_len >= 1`) and its
        output must not be weighted in.

    Returns:
      `[num_tokens, num_heads, v_dim]` in `output.dtype`, identical on every
      rank. Tokens no rank contributed to -- padding rows -- come back zeroed
      rather than NaN.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    lse_f32 = lse.to(torch.float32).contiguous()
    # [world_size, num_tokens, num_heads]: small, and it removes the need for
    # any further reduction over the denominator.
    gathered = group.all_gather(lse_f32.unsqueeze(0), dim=0)
    if not isinstance(gathered, torch.Tensor):
        gathered = torch.cat(list(gathered), dim=0)

    finite = torch.isfinite(gathered)
    shift = torch.where(finite, gathered,
                        torch.full_like(gathered, float("-inf"))).amax(dim=0)
    # A token with no contributor on any rank would give inf - inf; pin the
    # shift to 0 there and let the zero denominator carry the result to zero.
    empty = ~torch.isfinite(shift)
    shift = torch.where(empty, torch.zeros_like(shift), shift)

    weights = torch.where(finite, torch.exp(gathered - shift.unsqueeze(0)),
                          torch.zeros_like(gathered))
    denom = weights.sum(dim=0)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    mine = weights[get_dcp_rank()] / safe_denom
    weighted = (output.to(torch.float32) * mine.unsqueeze(-1)).contiguous()
    return group.all_reduce(weighted).to(output.dtype)
