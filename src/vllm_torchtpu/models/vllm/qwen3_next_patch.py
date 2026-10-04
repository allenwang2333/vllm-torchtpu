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
"""Trace-stable sequence-parallel forwards for Qwen3-Next and Qwen3.5 MoE.

Copies of the upstream vLLM forwards with three changes, so torch.compile
serves every bucket under VLLM_TPU_MOE_SEQUENCE_PARALLEL (including with
USE_MOE_FUSED_EP_KERNEL=1):
- torch._check(num_tokens % tp == 0) states that buckets divide by TP.
- narrow(0, 0, num_tokens) replaces [:num_tokens] after each all-gather,
  because a slice by a symbolic length lowers to a clamped Min that the
  shape-variant guard refuses.
- The pad before the reduce-scatter is dropped; its size came from a
  post-collective shape and added another ceil(./tp) per layer.
"""

from itertools import islice
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models import qwen3_next
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

linear_common: Any = None

UPSTREAM_ATTENTION_FORWARD = qwen3_next.Qwen3NextAttention.forward
UPSTREAM_DECODER_LAYER_FORWARD = qwen3_next.Qwen3NextDecoderLayer.forward
UPSTREAM_MODEL_FORWARD = qwen3_next.Qwen3NextModel.forward
UPSTREAM_MLP_INIT = qwen3_next.Qwen3NextMLP.__init__
_QWEN3_NEXT_MOE_MODEL_TYPES = frozenset(
    {"qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text"}
)


def _get_linear_common() -> Any:
    global linear_common
    if linear_common is None:
        from vllm_torchtpu.layers.adapter import linear_common as _lc

        linear_common = _lc
    return linear_common


def _check_tokens_divide_by_tp(num_tokens: int) -> None:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        torch._check(num_tokens % tp_size == 0)


def _narrow_to_tokens(
    x: torch.Tensor, full_num_tokens: int | torch.SymInt
) -> torch.Tensor:
    # Under torch.compile, _check_tokens_divide_by_tp has already established
    # full_num_tokens % tp_size == 0, so tensor_model_parallel_all_gather on
    # [full_num_tokens // tp_size, ...] already has leading dimension
    # full_num_tokens. Returning x directly avoids threading the symbolic token
    # count across split_graph submodule boundaries.
    if isinstance(full_num_tokens, torch.SymInt):
        return x
    if x.shape[0] == full_num_tokens:
        return x
    return x.narrow(0, 0, full_num_tokens)


def _should_use_sp(
    full_num_tokens: int | torch.SymInt, *, is_mtp: bool = False
) -> bool:
    """Whether this step should shard activations [T/TP, H] across TP ranks."""
    min_sp_tokens = envs.VLLM_TPU_SP_MIN_TOKENS
    if not min_sp_tokens and envs.USE_MOE_FUSED_TP_AR:
        from vllm_torchtpu.layers.adapter.fused_moe_tp_ar import MAX_DECODE_TOKENS

        min_sp_tokens = MAX_DECODE_TOKENS + 1
    if min_sp_tokens <= 2:
        return True
    from vllm_torchtpu.compilation import shape_variants

    # Submodules compiled outside shape_variants.warmup (such as the 1-layer
    # MTP drafter) keep a single token-independent graph without SP collectives.
    if is_mtp or (torch.compiler.is_compiling() and shape_variants._session is None):
        return False
    return bool(full_num_tokens >= min_sp_tokens)


def patched_mlp_init(
    self,
    *args: Any,
    **kwargs: Any,
) -> None:
    is_sequence_parallel = kwargs.get("is_sequence_parallel", False)
    prefix = kwargs.get("prefix", "")
    if (
        is_sequence_parallel
        and envs.USE_MOE_FUSED_TP_AR
        and prefix.endswith(".shared_expert")
    ):
        from vllm.config import get_current_vllm_config

        vllm_cfg = get_current_vllm_config()
        if not vllm_cfg.parallel_config.enable_expert_parallel:
            # Keep the shared expert TP-sharded so arm_fused_tp_ar can fuse its
            # [H, 2I/TP] and [I/TP, H] slices into the routed MoE kernel.
            kwargs["is_sequence_parallel"] = False
    UPSTREAM_MLP_INIT(self, *args, **kwargs)



def patched_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if prequant is None:
        return UPSTREAM_ATTENTION_FORWARD(
            self, positions=positions, hidden_states=hidden_states
        )
    qkv = _get_linear_common().project_prequant(self.qkv_proj, hidden_states, prequant)
    q, k, v, gate = self._project_qkv_gate(qkv, positions)
    attn_output = self.attn(q, k, v)
    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = self.o_proj(attn_output)
    return output


def patched_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    **kwargs: object,
):
    full_num_tokens = positions.shape[-1]

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    attn_kwargs: dict[str, object] = {}
    use_sp = False
    if self.use_attn_reduce_scatter_for_moe:
        use_sp = _should_use_sp(
            full_num_tokens, is_mtp=getattr(self, "_is_mtp_layer", False)
        )
    if use_sp:
        from vllm_torchtpu.compilation import shape_variants

        min_fp8_tokens = envs.VLLM_TPU_SP_FP8_GATHER_MIN_TOKENS or (
            16 * get_tensor_model_parallel_world_size() ** 2 + 1
        )
        # Token-threshold branching requires per-bucket re-tracing
        # (shape_variants.warmup), which only wraps the main backbone
        # (_precompile_backbone). Submodules compiled outside that scope
        # (e.g. MTP draft forward) must keep a single token-independent graph.
        can_specialize_on_tokens = (
            min_fp8_tokens <= 2
            or not torch.compiler.is_compiling()
            or shape_variants._session is not None
        )
        if (
            envs.VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER
            and can_specialize_on_tokens
            and full_num_tokens >= min_fp8_tokens
        ):
            x_q, x_s = _get_linear_common().quantize_rows(hidden_states)
            x_q = _narrow_to_tokens(
                tensor_model_parallel_all_gather(x_q, 0), full_num_tokens
            )
            x_s = _narrow_to_tokens(
                tensor_model_parallel_all_gather(x_s, 0), full_num_tokens
            )
            attn_kwargs["prequant"] = (x_q, x_s)
            hidden_states = x_q
        else:
            hidden_states = _narrow_to_tokens(
                tensor_model_parallel_all_gather(hidden_states, 0),
                full_num_tokens,
            )

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(hidden_states=hidden_states, **attn_kwargs)
    elif self.layer_type == "full_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            **attn_kwargs,
        )
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
        if use_sp:
            _check_tokens_divide_by_tp(full_num_tokens)
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
        else:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    if self.use_attn_reduce_scatter_for_moe:
        if envs.USE_MOE_FUSED_TP_AR:
            from vllm_torchtpu.layers.adapter import fused_moe_tp_ar

            prev_use_sp = fused_moe_tp_ar._CURRENT_LAYER_USE_SP
            fused_moe_tp_ar._CURRENT_LAYER_USE_SP = use_sp
            try:
                hidden_states = self.mlp(
                    hidden_states,
                    already_sequence_parallel=True,
                )
            finally:
                fused_moe_tp_ar._CURRENT_LAYER_USE_SP = prev_use_sp
        else:
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

    full_num_tokens = hidden_states.shape[0]
    if torch.compiler.is_compiling():
        positions = positions.reshape(*positions.shape[:-1], full_num_tokens)
    use_sp = False
    if self.use_sequence_parallel:
        _check_tokens_divide_by_tp(full_num_tokens)
        use_sp = _should_use_sp(full_num_tokens)
        if use_sp:
            hidden_states = qwen3_next.sequence_parallel_chunk(hidden_states)
            assert residual is None

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        self._maybe_add_hidden_state(
            aux_hidden_states, layer_idx + 1, hidden_states, residual
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
    hidden_states, _ = self.norm(hidden_states, residual)
    if self.use_sequence_parallel and use_sp:
        if aux_hidden_states:
            hidden_size = hidden_states.shape[-1]
            hidden_states = torch.cat([hidden_states, *aux_hidden_states], dim=-1)
            hidden_states = _narrow_to_tokens(
                tensor_model_parallel_all_gather(hidden_states, 0),
                full_num_tokens,
            )
            hidden_states, *aux_hidden_states = hidden_states.split(hidden_size, dim=-1)
        else:
            hidden_states = _narrow_to_tokens(
                tensor_model_parallel_all_gather(hidden_states, 0),
                full_num_tokens,
            )
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def patched_mtp_forward(
    self,
    input_ids: torch.Tensor | None,
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

    full_num_tokens = hidden_states.shape[0]
    if torch.compiler.is_compiling():
        positions = positions.reshape(*positions.shape[:-1], full_num_tokens)
    current_step_idx = spec_step_idx % self.num_mtp_layers
    mtp_layer = self.layers[current_step_idx]
    mtp_layer._is_mtp_layer = True
    use_sp = False
    if mtp_layer.use_attn_reduce_scatter_for_moe:
        _check_tokens_divide_by_tp(full_num_tokens)
        use_sp = _should_use_sp(full_num_tokens, is_mtp=True)
        if use_sp:
            hidden_states = qwen3_next.sequence_parallel_chunk(hidden_states)
            assert residual is None
    hidden_states, residual = mtp_layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    if mtp_layer.use_attn_reduce_scatter_for_moe and use_sp:
        hidden_states = _narrow_to_tokens(
            tensor_model_parallel_all_gather(hidden_states, 0), full_num_tokens
        )
    return hidden_states


def maybe_patch_qwen3_next(model_config: "ModelConfig | None" = None) -> bool | None:
    """Install trace-stable forwards when VLLM_TPU_MOE_SEQUENCE_PARALLEL is active."""
    if (
        envs.VLLM_TPU_MOE_SEQUENCE_PARALLEL == "off"
        or model_config is None
        or model_config.hf_config.model_type not in _QWEN3_NEXT_MOE_MODEL_TYPES
    ):
        return False
    from vllm.model_executor.models import qwen3_5_mtp, qwen3_next_mtp

    if envs.VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER:
        lc = _get_linear_common()
        lc._get_quantize_rows_op()
        lc._get_prequant_matmul_op()
        qwen3_next.Qwen3NextAttention.forward = patched_attention_forward
    if envs.USE_MOE_FUSED_TP_AR:
        qwen3_next.Qwen3NextMLP.__init__ = patched_mlp_init
    qwen3_next.Qwen3NextDecoderLayer.forward = patched_decoder_layer_forward
    qwen3_next.Qwen3NextModel.forward = patched_model_forward
    qwen3_5_mtp.Qwen3_5MultiTokenPredictor.forward = patched_mtp_forward
    qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward = patched_mtp_forward
    logger.info(
        "Applied TPU patch: Qwen3Next trace-stable sequence-parallel forward "
        "(fp8_gather=%s, fp8_min_tokens=%d, sp_min_tokens=%d, fused_tp_ar=%s).",
        envs.VLLM_TPU_SEQUENCE_PARALLEL_FP8_GATHER,
        envs.VLLM_TPU_SP_FP8_GATHER_MIN_TOKENS,
        envs.VLLM_TPU_SP_MIN_TOKENS,
        envs.USE_MOE_FUSED_TP_AR,
    )
    return None
