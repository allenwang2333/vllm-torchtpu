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

from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.sequence import IntermediateTensors


class MtpDraftModel(Protocol):
    """What the loop calls on the MTP draft. The draft is an nn.Module, but
    nn.Module types its attributes as Tensor | Module, so name the calls."""

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


def _resolve_fast_lm_head(
    mtp: Any,
) -> tuple[torch.Tensor, Any, int] | None:
    lm_head = getattr(mtp, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None or not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return None
    if getattr(lm_head, "bias", None) is not None:
        return None
    quant_method = getattr(lm_head, "quant_method", None)
    if quant_method is not None and type(quant_method).__name__ not in (
        "UnquantizedEmbeddingMethod",
        "UnquantizedLinearMethod",
    ):
        return None
    logits_proc = getattr(mtp, "logits_processor", None)
    if (
        getattr(logits_proc, "softcap", None) is not None
        or getattr(logits_proc, "scale", 1.0) != 1.0
    ):
        return None
    tp_group = get_tp_group()
    tp_size = getattr(tp_group, "world_size", 1)
    rank = getattr(tp_group, "rank_in_group", 0)
    shard_vocab = weight.shape[0]
    rank_offset = rank * shard_vocab
    org_vocab_size = getattr(lm_head, "org_vocab_size", shard_vocab * tp_size)
    valid_cols = min(shard_vocab, max(0, org_vocab_size - rank_offset))
    effective_weight = weight[:valid_cols] if valid_cols < shard_vocab else weight
    return (
        effective_weight,
        tp_group if tp_size > 1 else None,
        rank_offset,
    )


class _MtpFusedLoop(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: MtpDraftModel,
        num_steps: int,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps
        self._fast_lm_head = _resolve_fast_lm_head(mtp)

    def _next_token_ids(self, last_hidden: torch.Tensor) -> torch.Tensor:
        fast = self._fast_lm_head
        if fast is None:
            next_ids = self.mtp.compute_logits(last_hidden).argmax(dim=-1)
            return (
                next_ids if next_ids.dtype == torch.int32 else next_ids.to(torch.int32)
            )
        weight, tp_group, rank_offset = fast
        shard_logits = F.linear(last_hidden, weight)
        if tp_group is None:
            next_ids = shard_logits.argmax(dim=-1)
            return (
                next_ids if next_ids.dtype == torch.int32 else next_ids.to(torch.int32)
            )
        local_max, local_idx = shard_logits.max(dim=-1)
        local_idx_i32 = (
            local_idx if local_idx.dtype == torch.int32 else local_idx.to(torch.int32)
        ) + rank_offset
        packed = torch.stack(
            (local_max.to(torch.float32), local_idx_i32.to(torch.float32)), dim=-1
        )
        gathered = tp_group.all_gather(packed.unsqueeze(0), dim=0)
        best_rank = gathered[..., 0].argmax(dim=0, keepdim=True)
        return gathered[..., 1].gather(0, best_rank).squeeze(0).to(torch.int32)

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
        mtp = self.mtp
        forward_fn = type(mtp).forward
        tokens = []
        for i in range(self.num_steps):
            if i > 0:
                positions = positions + 1
                seq_lens.add_(1)
            out = forward_fn(
                mtp,
                input_ids,
                positions,
                hidden_states,
                intermediate_tensors,
                inputs_embeds,
                spec_step_idx=i + 1,
            )
            # Same normalisation as EagleProposer._unwrap_model_out.
            if isinstance(out, (list, tuple)) and len(out) == 1:
                out = out[0]
            if isinstance(out, (list, tuple)):
                last_hidden, hidden_states = out
            else:
                last_hidden = hidden_states = out
            input_ids = self._next_token_ids(last_hidden)
            tokens.append(input_ids)
        return tuple(tokens)


_FUSED_LOOP_CLS_CACHE: dict[tuple[tuple[str, Any], ...], type[_MtpFusedLoop]] = {}


def build_mtp_fused_loop(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    num_steps: int,
    dynamic_arg_dims: dict,
) -> nn.Module:
    """Compile `_MtpFusedLoop` with the draft's own dynamic dims (Qwen3.5's
    MTP marks positions -1, DeepSeek's 0), so both programs specialise the
    token dimension the same way."""
    key = tuple(sorted(dynamic_arg_dims.items()))
    cls = _FUSED_LOOP_CLS_CACHE.get(key)
    if cls is None:
        cls = support_torch_compile(dynamic_arg_dims=dict(dynamic_arg_dims))(
            type("MtpFusedLoop", (_MtpFusedLoop,), {})
        )
        _FUSED_LOOP_CLS_CACHE[key] = cls
    return cls(vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps)
