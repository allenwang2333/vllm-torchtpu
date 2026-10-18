# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""DeepseekV2Model forward with the segment megakernel's layer hand-offs.

TPU_SEGMENT_MEGAKERNEL (layers/adapter/segment_megakernel.py) runs MoE layer
N's tail and skip layer N+1's head as one Pallas call, so layer N hands layer
N+1 the new residual, q_c and the KV write rows instead of
(hidden_states, residual). The decoder loop is the only place both layers
meet, so this replaces vLLM's DeepseekV2Model.forward (GLM-5.2 is
GlmMoeDsaForCausalLM on it) with a copy of the v0.29.0 forward whose loop
steps each layer through `segment_megakernel.run_layer`. A layer that is not
armed, or a step the megakernel does not take, runs `layer(...)` exactly as
upstream, so the patch changes nothing until `segment_megakernel.arm` arms a
boundary.
"""

from itertools import islice

import torch
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.model_executor.models import deepseek_v2
from vllm.sequence import IntermediateTensors

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

UPSTREAM_MODEL_FORWARD = deepseek_v2.DeepseekV2Model.forward
# layers/adapter/segment_megakernel, bound when the patch is installed (it
# imports the TPU kernel stack, which a platform-activation import must not).
_SEGMENT = None


def patched_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError(
                    "Either input_ids or inputs_embeds must be provided "
                    "to DeepseekV2Model.forward"
                )
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # Compute llama 4 scaling once per forward pass if enabled
    llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = deepseek_v2._get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    aux_hidden_states = []
    handoff = None  # segment megakernel: layer N's output for layer N+1
    for idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        if handoff is None:
            # all gather if we need to use the whole states
            if (
                hidden_states.shape[0] != positions.shape[0]
                and not layer.use_sequence_parallel_moe
            ):
                combined_states = torch.cat([hidden_states, residual], dim=-1)
                combined_states = tensor_model_parallel_all_gather(combined_states, 0)
                combined_states = combined_states[: positions.shape[0]]
                hidden_states, residual = combined_states.split(
                    [self.hidden_size, self.hidden_size], dim=-1
                )
            if idx in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states + residual
                if aux_hidden_state.shape[0] != positions.shape[0]:
                    aux_hidden_state = tensor_model_parallel_all_gather(
                        aux_hidden_state, 0
                    )
                    aux_hidden_state = aux_hidden_state[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)
        # Arming refuses a hand-off into an aux layer, so the branch above
        # never needs hidden_states while one is pending.
        hidden_states, residual, handoff = _SEGMENT.run_layer(
            layer, positions, hidden_states, residual, llama_4_scaling, handoff
        )
    assert handoff is None, "the last layer cannot hand off to a next one"

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    if hidden_states.shape[0] != positions.shape[0]:
        combined_states = torch.cat([hidden_states, residual], dim=-1)
        combined_states = tensor_model_parallel_all_gather(combined_states, 0)
        combined_states = combined_states[: positions.shape[0]]
        hidden_states, residual = combined_states.split(
            [self.hidden_size, self.hidden_size], dim=-1
        )

    if self.end_layer in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states + residual)

    hidden_states, _ = self.norm(hidden_states, residual)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


def maybe_patch_deepseek_v2_segment(*args: object, **kwargs: object) -> None:
    """Install the hand-off loop when TPU_SEGMENT_MEGAKERNEL=1."""
    global _SEGMENT
    if not envs.TPU_SEGMENT_MEGAKERNEL:
        return None
    from vllm_torchtpu.layers.adapter import segment_megakernel

    _SEGMENT = segment_megakernel
    deepseek_v2.DeepseekV2Model.forward = patched_model_forward
    logger.info(
        "Applied TPU patch: DeepseekV2Model forward with segment megakernel hand-offs."
    )
    return None
