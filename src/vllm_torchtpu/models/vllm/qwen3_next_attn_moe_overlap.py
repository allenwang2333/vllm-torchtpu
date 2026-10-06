# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3Next and Qwen3.5 attention-to-MoE collective overlap monkey-patch."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextSparseMoeBlock,
)

from vllm_torchtpu import envs

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False
_orig_qwen3_next_decoder_init = Qwen3NextDecoderLayer.__init__
_orig_qwen3_5_decoder_init = Qwen3_5DecoderLayer.__init__
_orig_decoder_forward = Qwen3NextDecoderLayer.forward
_orig_moe_forward = Qwen3NextSparseMoeBlock.forward


def _configure_decoder_layer(self: Any, prefix: str = "") -> None:
    if getattr(self, "_tpu_attn_moe_overlap_configured", False):
        return
    self._tpu_attn_moe_overlap_configured = True

    if (
        "mtp" in prefix
        or not isinstance(getattr(self, "mlp", None), Qwen3NextSparseMoeBlock)
        or get_tensor_model_parallel_world_size() <= 1
    ):
        self._tpu_attn_moe_overlap = False
        return

    assert not getattr(self.mlp, "replicate_shared_expert", False)

    proj = (
        self.linear_attn.out_proj
        if self.layer_type == "linear_attention"
        else self.self_attn.o_proj
    )
    assert getattr(proj, "reduce_results", False) is True
    proj.reduce_results = False

    self.mlp.experts.gate = None
    self.mlp.experts.moe_config.skip_final_all_reduce = True
    assert self.mlp.experts.moe_config.skip_final_all_reduce is True

    if not hasattr(self.mlp.experts, "maybe_all_reduce_tensor_model_parallel"):
        self.mlp.experts.maybe_all_reduce_tensor_model_parallel = lambda x: (
            tensor_model_parallel_all_reduce(x)
        )

    self.mlp._tpu_attn_moe_overlap = True
    self._tpu_attn_moe_overlap = True


def _patched_qwen3_next_decoder_init(
    self: Any,
    vllm_config: Any,
    layer_type: str,
    prefix: str = "",
) -> None:
    _orig_qwen3_next_decoder_init(self, vllm_config, layer_type, prefix=prefix)
    _configure_decoder_layer(self, prefix)


def _patched_qwen3_5_decoder_init(
    self: Any,
    vllm_config: Any,
    layer_type: str,
    prefix: str = "",
) -> None:
    _orig_qwen3_5_decoder_init(self, vllm_config, layer_type, prefix=prefix)
    _configure_decoder_layer(self, prefix)


def _patched_moe_forward(
    self: Qwen3NextSparseMoeBlock,
    hidden_states: torch.Tensor,
    already_sequence_parallel: bool = False,
    router_logits: torch.Tensor | None = None,
    fold_rows: torch.Tensor | None = None,
    ffn_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if not getattr(self, "_tpu_attn_moe_overlap", False):
        return _orig_moe_forward(
            self,
            hidden_states,
            already_sequence_parallel=already_sequence_parallel,
        )

    orig_shape = hidden_states.shape
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.view(-1, hidden_dim)
    if router_logits is None:
        gate_out = self.gate(hidden_states)
        router_logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )
    if (
        getattr(self, "replicate_shared_expert", False)
        and getattr(self, "shared_expert", None) is not None
    ):
        replicated_shared_output = self.shared_expert(hidden_states)
        final_hidden_states = final_hidden_states + replicated_shared_output
    if ffn_scale is not None:
        final_hidden_states = final_hidden_states * ffn_scale
    if fold_rows is not None:
        final_hidden_states = final_hidden_states + fold_rows
    if self.tp_size > 1:
        if hasattr(self.experts, "maybe_all_reduce_tensor_model_parallel"):
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )
        else:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states.view(orig_shape)


def _patched_decoder_forward(
    self: Qwen3NextDecoderLayer,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not getattr(self, "_tpu_attn_moe_overlap", False):
        return _orig_decoder_forward(
            self,
            hidden_states,
            residual,
            positions,
            **kwargs,
        )

    full_num_tokens = positions.shape[-1]
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(hidden_states=hidden_states)
    elif self.layer_type == "full_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )
    else:
        raise ValueError(f"Invalid layer_type: {self.layer_type}")

    min_tokens = max(envs.TPU_ATTN_MOE_OVERLAP_MIN_TOKENS, 16)
    if full_num_tokens < min_tokens:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        if self.layer_scale:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        if self.layer_scale:
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        return hidden_states, residual

    if self.layer_scale:
        hidden_states = hidden_states * (
            self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
        )

    tp_size = self.mlp.tp_size
    a_partial = hidden_states
    x_loc = tensor_model_parallel_reduce_scatter(
        a_partial + residual.to(a_partial.dtype) * (1.0 / tp_size), 0
    )
    h_loc = self.post_attention_layernorm(x_loc)
    gate_out = self.mlp.gate(h_loc)
    logits_loc = gate_out[0] if isinstance(gate_out, tuple) else gate_out
    logits = tensor_model_parallel_all_gather(logits_loc, 0).narrow(
        0, 0, full_num_tokens
    )
    h = tensor_model_parallel_all_gather(h_loc, 0).narrow(0, 0, full_num_tokens)
    ffn_scale = (self.ffn_layer_scale.to(h.dtype)[0] + 1) if self.layer_scale else None
    out = self.mlp(
        h,
        router_logits=logits,
        fold_rows=a_partial,
        ffn_scale=ffn_scale,
    )
    return out, residual


def apply_qwen3_next_attn_moe_overlap_patch() -> None:
    """Monkey-patch Qwen3Next and Qwen3.5 decoder layer for attention-MoE overlap."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    tpu_moe_sp = getattr(envs, "TPU_MOE_SEQUENCE_PARALLEL", False) or (
        os.environ.get("TPU_MOE_SEQUENCE_PARALLEL", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )
    use_moe_fused_ep = getattr(envs, "USE_MOE_FUSED_EP_KERNEL", False) or (
        os.environ.get("USE_MOE_FUSED_EP_KERNEL", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )
    if tpu_moe_sp or use_moe_fused_ep:
        raise ValueError(
            "TPU_ATTN_MOE_OVERLAP=1 is only supported on the default TP MoE path "
            "(TPU_MOE_SEQUENCE_PARALLEL=0, USE_MOE_FUSED_EP_KERNEL=0)."
        )

    Qwen3NextDecoderLayer.__init__ = _patched_qwen3_next_decoder_init
    Qwen3_5DecoderLayer.__init__ = _patched_qwen3_5_decoder_init
    Qwen3NextDecoderLayer.forward = _patched_decoder_forward
    Qwen3NextSparseMoeBlock.forward = _patched_moe_forward

    _PATCH_APPLIED = True
