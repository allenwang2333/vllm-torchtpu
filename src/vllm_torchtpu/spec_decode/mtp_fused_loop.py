# SPDX-License-Identifier: Apache-2.0
"""An MTP draft's loop steps 1..K-1 as one compiled program.

Each loop step used to be two host dispatches -- the draft forward and the
lm-head + argmax -- plus a metadata build, so K=3 paid for four programs and
two builds between step 0 and the verify. The steps differ only in values the
device can derive itself: positions and attention seq_lens advance by one, and
the input ids are the previous step's argmax. This program unrolls the steps
and advances both on device, so the host builds step 1's metadata once and
dispatches once.

seq_lens reaches the attention kernel through the forward context, not as an
argument, so the program is handed the same tensor the step-1 metadata holds
and bumps it in place between steps; the traced attention then reads the
bumped value. The caller must pass a tensor built for this call alone.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read.
"""

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.spec_decode.utils import VocabParallelArgmax, vocab_parallel_argmax


class MtpDraftModel(Protocol):
    """What the loop calls on the MTP draft. The draft is an nn.Module, but
    nn.Module types its attributes as Tensor | Module, so name the calls."""

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


def _greedy_draft_token(
    mtp: MtpDraftModel,
    hidden_states: torch.Tensor,
    argmax_info: VocabParallelArgmax | None,
) -> torch.Tensor:
    """lm-head + greedy argmax, as `EagleProposer._draft_propose_token`. With
    `argmax_info`, `compute_logits` returns this rank's vocab shard (see
    `install_vocab_parallel_draft_argmax`) and the argmax reduces per shard."""
    logits = mtp.compute_logits(hidden_states)
    if argmax_info is not None:
        return vocab_parallel_argmax(logits, argmax_info)
    return logits.argmax(dim=-1).to(torch.int32)


def _unwrap_draft_out(out: Any) -> tuple[torch.Tensor, torch.Tensor]:
    # Same normalisation as EagleProposer._unwrap_model_out.
    if isinstance(out, (list, tuple)) and len(out) == 1:
        out = out[0]
    if isinstance(out, (list, tuple)):
        last_hidden, hidden_states = out
    else:
        last_hidden = hidden_states = out
    return last_hidden, hidden_states


class _MtpFusedLoop(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: MtpDraftModel,
        num_steps: int,
        argmax_info: VocabParallelArgmax | None = None,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps
        self.argmax_info = argmax_info

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """`positions` and `seq_lens` are step 1's. Returns the K-1 draft
        tokens, one [p] int32 tensor per step."""
        tokens = []
        for i in range(self.num_steps):
            if i > 0:
                positions = positions + 1
                seq_lens.add_(1)
            out = type(self.mtp).forward(
                self.mtp,
                input_ids,
                positions,
                hidden_states,
                intermediate_tensors,
                inputs_embeds,
                spec_step_idx=i + 1,
            )
            last_hidden, hidden_states = _unwrap_draft_out(out)
            input_ids = _greedy_draft_token(self.mtp, last_hidden, self.argmax_info)
            tokens.append(input_ids)
        return tuple(tokens)


# The tail's tensor arguments, every one [p, ...] for the chunk's loop bucket p.
# `seq_lens` is not listed: it is [kernel_num_reqs], fixed for the program.
MTP_FUSED_TAIL_DYNAMIC_ARG_DIMS: dict[str, int | list[int] | dict[int, str]] = {
    "last_hidden": 0,
    "hidden_states": 0,
    "positions": 0,
    "gather_indices": 0,
}


class _MtpIndexShareFusedTail(nn.Module):
    """Everything an IndexShare MTP draft does after its step-0 forward, for
    one chunk, as one program: compact step 0's top-k rows to one per request,
    the step-0 lm-head + argmax, then loop steps 1..K-1 as in `_MtpFusedLoop`.

    IndexShare is why the loop steps could not fuse before: step 0's indexer
    writes one top-k row per token into a buffer the loop steps read by
    request, and the proposer gathered, snapshotted and restored those rows
    between programs on the host. Here the gather is the program's first op
    and the loop steps read the buffer right after it, so nothing can land in
    between and there is nothing to snapshot. The caller runs this right after
    the chunk's own step-0 forward (chunk-major), and traces it with every MTP
    layer's `skip_topk` set, so the loop steps reuse the rows instead of
    running the indexer.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: MtpDraftModel,
        num_steps: int,
        compact_topk_indices: Callable[[torch.Tensor], None],
        argmax_info: VocabParallelArgmax | None = None,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps
        self.compact_topk_indices = compact_topk_indices
        self.argmax_info = argmax_info

    def forward(
        self,
        last_hidden: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        gather_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """`last_hidden` / `hidden_states` / `positions` are step 0's
        per-request carries, `gather_indices` the (clamped) step-0 token row of
        each request and `seq_lens` step 1's. Returns the [p, K] draft tokens."""
        self.compact_topk_indices(gather_indices)
        input_ids = _greedy_draft_token(self.mtp, last_hidden, self.argmax_info)
        tokens = [input_ids]
        for i in range(self.num_steps):
            positions = positions + 1
            if i > 0:
                seq_lens.add_(1)
            out = type(self.mtp).forward(
                self.mtp,
                input_ids,
                positions,
                hidden_states,
                None,
                None,
                spec_step_idx=i + 1,
            )
            last_hidden, hidden_states = _unwrap_draft_out(out)
            input_ids = _greedy_draft_token(self.mtp, last_hidden, self.argmax_info)
            tokens.append(input_ids)
        return torch.stack(tokens, dim=1)


def build_mtp_fused_loop(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    num_steps: int,
    dynamic_arg_dims: dict,
    argmax_info: VocabParallelArgmax | None = None,
) -> nn.Module:
    """Compile `_MtpFusedLoop` with the draft's own dynamic dims (Qwen3.5's
    MTP marks positions -1, DeepSeek's 0), so both programs specialise the
    token dimension the same way."""
    cls = support_torch_compile(dynamic_arg_dims=dict(dynamic_arg_dims))(
        type("MtpFusedLoop", (_MtpFusedLoop,), {})
    )
    return cls(
        vllm_config=vllm_config,
        prefix=prefix,
        mtp=mtp,
        num_steps=num_steps,
        argmax_info=argmax_info,
    )


def build_mtp_index_share_fused_tail(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    num_steps: int,
    compact_topk_indices: Callable[[torch.Tensor], None],
    argmax_info: VocabParallelArgmax | None = None,
) -> nn.Module:
    """Compile `_MtpIndexShareFusedTail`. Its token dimension is the loop
    bucket p on every argument, dim 0 as in DeepSeek's MTP."""
    cls = support_torch_compile(dynamic_arg_dims=MTP_FUSED_TAIL_DYNAMIC_ARG_DIMS)(
        type("MtpIndexShareFusedTail", (_MtpIndexShareFusedTail,), {})
    )
    return cls(
        vllm_config=vllm_config,
        prefix=prefix,
        mtp=mtp,
        num_steps=num_steps,
        compact_topk_indices=compact_topk_indices,
        argmax_info=argmax_info,
    )
