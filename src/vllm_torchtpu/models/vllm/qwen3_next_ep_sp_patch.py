# SPDX-License-Identifier: Apache-2.0
"""Trace-stable sequence-parallel forward for Qwen3Next and Qwen3.5 MoE.

These are copies of upstream vLLM forwards modified to ensure torch.compile
trace stability under Sequence Parallelism (TPU_MOE_SEQUENCE_PARALLEL=1):
- narrow() replaces slicing by symbolic tensor length (which lowers to dynamic
  clamped Min refused by TPU shape-variant checks).
- Dynamic sp_pad is eliminated because torch._check(full_num_tokens % tp == 0)
  guarantees divisibility.
- Slices after all-gather use narrow(0, 0, full_num_tokens).
"""

from itertools import islice
import os

import torch
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.models import qwen3_next, qwen3_next_mtp
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

UPSTREAM_DECODER_LAYER_FORWARD = qwen3_next.Qwen3NextDecoderLayer.forward
UPSTREAM_MODEL_FORWARD = qwen3_next.Qwen3NextModel.forward
UPSTREAM_MTP_FORWARD = qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward


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
    full_num_tokens = positions.shape[-1]
    tp_world_size = get_tensor_model_parallel_world_size()
    if tp_world_size > 1:
        torch._check(full_num_tokens % tp_world_size == 0)

    input_is_sequence_parallel = (
        self.use_attn_reduce_scatter_for_moe
        and hidden_states.shape[0] != full_num_tokens
    )

    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    if input_is_sequence_parallel:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states.narrow(0, 0, full_num_tokens)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(hidden_states=hidden_states)
    elif self.layer_type == "full_attention":
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
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
        hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
        if not input_is_sequence_parallel:
            residual = _sp_chunk(residual, full_num_tokens)

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
    tp_world_size = get_tensor_model_parallel_world_size()
    if tp_world_size > 1:
        torch._check(full_num_tokens % tp_world_size == 0)

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        if (
            hidden_states.shape[0] != full_num_tokens
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
        if (layer_idx + 1) in self.aux_hidden_state_layers and hidden_states.shape[
            0
        ] != full_num_tokens:
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
    if hidden_states.shape[0] != full_num_tokens:
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
    if mtp_layer.use_attn_reduce_scatter_for_moe:
        full_num_tokens = positions.shape[-1]
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size > 1:
            torch._check(full_num_tokens % tp_world_size == 0)
        hidden_states = _sp_chunk(hidden_states, full_num_tokens)
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
    if mtp_layer.use_attn_reduce_scatter_for_moe:
        full_num_tokens = positions.shape[-1]
        hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        hidden_states = hidden_states.narrow(0, 0, full_num_tokens)
    return hidden_states


def maybe_patch_qwen3_next(*args: object, **kwargs: object) -> bool | None:
    """Install trace-stable forwards when TPU_MOE_SEQUENCE_PARALLEL=1."""
    if not envs.TPU_MOE_SEQUENCE_PARALLEL:
        return False

    if getattr(envs, "TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS", 0) == 0:
        flags = [
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false",
        ]
        existing = os.environ.get("LIBTPU_INIT_ARGS", "")
        to_add = [f for f in flags if f not in existing]
        if to_add:
            os.environ["LIBTPU_INIT_ARGS"] = (existing + " " + " ".join(to_add)).strip()
            logger.info(
                "Appended TensorCore ICI flags to LIBTPU_INIT_ARGS for SP=1: %s",
                " ".join(to_add),
            )

    qwen3_next.Qwen3NextDecoderLayer.forward = patched_decoder_layer_forward
    qwen3_next.Qwen3NextModel.forward = patched_model_forward
    qwen3_next_mtp.Qwen3NextMultiTokenPredictor.forward = patched_mtp_forward
    logger.info("Applied TPU patch: Qwen3Next trace-stable sequence-parallel forward.")
    return None
