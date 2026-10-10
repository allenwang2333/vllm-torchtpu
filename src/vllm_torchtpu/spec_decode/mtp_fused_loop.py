# SPDX-License-Identifier: Apache-2.0
"""Compiled single-graph multi-step MTP draft loop for Qwen3.5 speculative decoding.

Chains multiple MTP draft steps inside a single compiled XLA computation to
eliminate per-draft-step host dispatch overhead.
"""

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors


class _MtpFusedLoop(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: nn.Module,
        num_steps: int,
        greedy_head_owner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps
        self.greedy_head_owner = greedy_head_owner

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`positions` and `seq_lens` are step 1's; `hidden_states` is step 0's
        gathered carry [p, H]. Proposes step 0's token from `hidden_states` and
        runs steps 1..K-1, returning the stacked [p, K] int32 draft tokens."""
        if self.greedy_head_owner is not None:
            input_ids = self.greedy_head_owner.logits_processor.get_top_tokens(
                self.greedy_head_owner.lm_head, hidden_states
            ).to(torch.int32)
        else:
            input_ids = (
                self.mtp.compute_logits(hidden_states).argmax(dim=-1).to(torch.int32)
            )
        tokens = [input_ids]
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
            # Same normalisation as EagleProposer._unwrap_model_out.
            if isinstance(out, (list, tuple)) and len(out) == 1:
                out = out[0]
            if isinstance(out, (list, tuple)):
                last_hidden, hidden_states = out
            else:
                last_hidden = hidden_states = out
            if self.greedy_head_owner is not None:
                input_ids = self.greedy_head_owner.logits_processor.get_top_tokens(
                    self.greedy_head_owner.lm_head, last_hidden
                ).to(torch.int32)
            else:
                input_ids = (
                    self.mtp.compute_logits(last_hidden).argmax(dim=-1).to(torch.int32)
                )
            tokens.append(input_ids)
        return torch.stack(tokens, dim=1)


def build_mtp_fused_loop(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: nn.Module,
    num_steps: int,
    dynamic_arg_dims: dict,
    greedy_head_owner: nn.Module | None = None,
) -> nn.Module:
    """Compile `_MtpFusedLoop` with the draft's own dynamic dims (Qwen3.5's
    MTP marks positions -1, DeepSeek's 0), so both programs specialise the
    token dimension the same way."""
    loop_dims = {
        k: v for k, v in dynamic_arg_dims.items() if k != "input_ids"
    }
    cls = support_torch_compile(dynamic_arg_dims=loop_dims)(
        type("MtpFusedLoop", (_MtpFusedLoop,), {})
    )
    return cls(
        vllm_config=vllm_config,
        prefix=prefix,
        mtp=mtp,
        num_steps=num_steps,
        greedy_head_owner=greedy_head_owner,
    )
