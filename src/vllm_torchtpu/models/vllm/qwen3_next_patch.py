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
"""Trace-stable sequence-parallel forward for Qwen3Next and Qwen3.5 MoE.

With `TPU_MOE_SEQUENCE_PARALLEL=1`, steps of at least `TPU_SP_MIN_ROWS` rows
shard the tokens over TP between attention and MoE: attention reduce-scatters
instead of all-reducing, the router, shared expert and experts run on T/tp
rows, and the next layer all-gathers (as fp8 rows under `TPU_SP_FP8_GATHER`).

MTP draft step 0 can gather the last-token rows right after attention
(`TPU_MTP_STEP0_POST_ATTN_GATHER`), so the gate, o_proj, post-attention norm,
MoE and final norm run on one row per request instead of every verify row.
"""

import contextlib
import functools
from itertools import islice
from typing import Any

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models import qwen3_next, qwen3_next_mtp
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter import linear_common
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

UPSTREAM_DECODER_LAYER_FORWARD = qwen3_next.Qwen3NextDecoderLayer.forward
UPSTREAM_MODEL_FORWARD = qwen3_next.Qwen3NextModel.forward
UPSTREAM_MTP_FORWARD = qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward
UPSTREAM_ATTN_FORWARD = qwen3_next.Qwen3NextAttention.forward

_step0_gather_indices: torch.Tensor | None = None


@contextlib.contextmanager
def step0_post_attn_gather_context(indices: torch.Tensor | None):
    """Gather last-token rows immediately after self_attn.attn(q, k, v) on MTP step 0."""
    global _step0_gather_indices
    prev = _step0_gather_indices
    _step0_gather_indices = indices
    try:
        yield
    finally:
        _step0_gather_indices = prev


def is_step0_post_attn_gather_installed(draft_model: Any) -> bool:
    """Return True if `draft_model` executes the step-0 post-attn gather path."""
    if draft_model is None:
        return False
    inner = getattr(draft_model, "model", draft_model)
    if getattr(inner, "_supports_step0_post_attn_gather", False) or getattr(
        draft_model, "_supports_step0_post_attn_gather", False
    ):
        return True
    return (
        qwen3_next.Qwen3NextAttention.forward is patched_qwen3_next_attn_forward
        and qwen3_next.Qwen3NextDecoderLayer.forward is patched_decoder_layer_forward
        and getattr(type(inner), "forward", None) is patched_mtp_forward
    )


def _forward_qwen3_next_attn_from_qkv(
    attn_layer: Any,
    qkv: torch.Tensor,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    q, k, v, gate = attn_layer._project_qkv_gate(qkv, positions)
    attn_output = attn_layer.attn(q, k, v)
    gather_idx = _step0_gather_indices
    if gather_idx is not None:
        attn_output = torch.index_select(attn_output, 0, gather_idx)
        if gate is not None:
            gate = torch.index_select(gate, 0, gather_idx)
    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = attn_layer.o_proj(attn_output)
    return output


def patched_qwen3_next_attn_forward(
    self: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if prequant is None:
        qkv, _ = self.qkv_proj(hidden_states)
    else:
        proj = self.qkv_proj
        bias = None if proj.skip_bias_add else proj.bias
        qkv = proj.quant_method.apply_prequant(proj, prequant[0], prequant[1], bias)
    return _forward_qwen3_next_attn_from_qkv(self, qkv, positions)


def _sp_chunk(x: torch.Tensor, full_num_tokens: int) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return x
    tp_rank = get_tensor_model_parallel_rank()
    chunk = full_num_tokens // tp_size
    return x.narrow(0, tp_rank * chunk, chunk)


def _all_gather_hidden_and_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    full_num_tokens: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if residual is None:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states.narrow(0, 0, full_num_tokens)
        return hidden_states, None

    combined_states = torch.cat([hidden_states, residual], dim=-1)
    combined_states = tensor_model_parallel_all_gather(combined_states, 0)
    combined_states = combined_states.narrow(0, 0, full_num_tokens)
    hidden_states, residual = combined_states.split([hidden_size, hidden_size], dim=-1)
    return hidden_states, residual


def patched_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor | None = None,
    **kwargs: object,
):
    from vllm_torchtpu.layers.adapter import sequence_parallel as _sp

    sp = False
    if self.use_attn_reduce_scatter_for_moe and positions.shape[-1] >= _sp.min_rows():
        sp = True
    input_is_sequence_parallel = sp and residual is not None

    if hasattr(self, "_tpu_sp_rank"):
        positions = positions + (self._tpu_sp_rank * 0)[0]

    prequant = None
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if input_is_sequence_parallel and _sp.fp8_gather():
        x_q, x_s = linear_common.quantize_rows(hidden_states)
        x_q = tensor_model_parallel_all_gather(x_q, 0)
        x_s = tensor_model_parallel_all_gather(x_s, 0)
        prequant = (x_q, x_s)
        hidden_states = x_q
    elif input_is_sequence_parallel:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            prequant=prequant,
        )
    elif self.layer_type == "full_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            prequant=prequant,
        )
        gather_idx = _step0_gather_indices
        if gather_idx is not None:
            residual = torch.index_select(residual, 0, gather_idx)
            sp = False
    else:
        raise ValueError("Invalid layer_type")

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        else:
            hidden_states = hidden_states * (
                self.attn_layer_scale.to(hidden_states.dtype) + 1
            )

    if self.use_attn_reduce_scatter_for_moe:
        if sp:
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            if not input_is_sequence_parallel:
                tp = get_tensor_model_parallel_world_size()
                if hasattr(self, "_tpu_sp_rank"):
                    residual = (
                        residual.reshape(tp, -1, residual.shape[-1])
                        .index_select(0, self._tpu_sp_rank)
                        .squeeze(0)
                    )
                else:
                    residual = _sp_chunk(residual, positions.shape[-1])
        else:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if self.use_attn_reduce_scatter_for_moe:
        hidden_states = self.mlp(
            hidden_states,
            already_sequence_parallel=True,
        )
    else:
        hidden_states = self.mlp(hidden_states)

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        else:
            assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                f"shape must be the same {len(hidden_states.shape)}, "
                f"{len(self.ffn_layer_scale.shape)}"
            )
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype) + 1
            )

    return hidden_states, residual


def patched_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    full_num_tokens = positions.shape[-1]
    from vllm_torchtpu.layers.adapter import sequence_parallel as _sp

    sp = positions.shape[-1] >= _sp.min_rows()

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        if (
            sp
            and residual is not None
            and not layer.use_attn_reduce_scatter_for_moe
        ):
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        if (layer_idx + 1) in self.aux_hidden_state_layers and sp:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        self._maybe_add_hidden_state(
            aux_hidden_states, layer_idx + 1, hidden_states, residual
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
    if sp and self.layers[self.end_layer - 1].use_attn_reduce_scatter_for_moe:
        hidden_states, residual = _all_gather_hidden_and_residual(
            hidden_states,
            residual,
            full_num_tokens,
            self.config.hidden_size,
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def patched_mtp_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor | IntermediateTensors:
    if get_pp_group().is_first_rank:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    current_step_idx = spec_step_idx % self.num_mtp_layers
    mtp_layer = self.layers[current_step_idx]
    from vllm_torchtpu.layers.adapter import sequence_parallel as _sp

    sp = (
        mtp_layer.use_attn_reduce_scatter_for_moe
        and _step0_gather_indices is None
        and positions.shape[-1] >= _sp.min_rows()
    )

    hidden_states, residual = mtp_layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    if sp:
        full_num_tokens = positions.shape[-1]
        hidden_states, residual = _all_gather_hidden_and_residual(
            hidden_states,
            residual,
            full_num_tokens,
            self.config.hidden_size,
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


def maybe_patch_qwen3_next(*args: object, **kwargs: object) -> bool | None:
    """Install the trace-stable sequence-parallel forwards for Qwen3Next / Qwen3.5 MoE."""
    from vllm_torchtpu.layers.adapter import sequence_parallel as _sp

    if not _sp.enabled():
        return False

    if _sp.fp8_gather():
        linear_common.prebuild_prequant_ops()

    def _with_rank_buffer(original_init):
        @functools.wraps(original_init)
        def __init__(self, *a, **kw):
            original_init(self, *a, **kw)
            device = self.input_layernorm.weight.device
            self._tpu_sp_rank = torch.tensor(
                [get_tensor_model_parallel_rank()],
                dtype=torch.int32,
                device=device,
            )

        __init__._tpu_sp_rank_init = True
        return __init__

    from vllm.model_executor.models import qwen3_5 as _q35
    from vllm.model_executor.models import qwen3_5_mtp as _q35_mtp

    for layer_cls in (qwen3_next.Qwen3NextDecoderLayer, _q35.Qwen3_5DecoderLayer):
        if "__init__" in vars(layer_cls) and not getattr(
            layer_cls.__init__, "_tpu_sp_rank_init", False
        ):
            layer_cls.__init__ = _with_rank_buffer(layer_cls.__init__)

    # The decoder forward hands the projections the fp8-gathered rows.
    qwen3_next.Qwen3NextAttention.forward = patched_qwen3_next_attn_forward
    qwen3_next.Qwen3NextDecoderLayer.forward = patched_decoder_layer_forward
    qwen3_next.Qwen3NextModel.forward = patched_model_forward
    qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward = patched_mtp_forward
    _q35_mtp.Qwen3_5MultiTokenPredictor.forward = patched_mtp_forward
    logger.info(
        "Applied TPU patch: Qwen3Next row-gated sequence-parallel forward (>= %d rows).",
        _sp.min_rows(),
    )
    return None
