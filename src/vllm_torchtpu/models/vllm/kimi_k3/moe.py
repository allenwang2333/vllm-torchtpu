# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi MoE routed through vLLM's TPU Pallas fused-MoE backend."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_tensor_model_parallel_rank,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (GroupCoordinator, get_world_group,
                                             init_model_parallel_group)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.chip_topology import hierarchical_moe_split
from vllm_torchtpu.layers.vllm.moe_hierarchical import \
    hierarchical_moe_parallel_config, hierarchical_split_or_none

from .layers import KimiMLP, _linear_tp_override


_K3_HOST_TP_GROUP: GroupCoordinator | None = None
_K3_HIERARCHICAL_TP_GROUP: GroupCoordinator | None = None
_K3_PROJECTION_TP_GROUPS: dict[int, GroupCoordinator] = {}


def _get_k3_host_tp_group() -> GroupCoordinator:
    """Create four independent, host-local TP8 groups for a v7x-32 slice."""
    global _K3_HOST_TP_GROUP
    if _K3_HOST_TP_GROUP is not None:
        return _K3_HOST_TP_GROUP

    world = get_world_group()
    host_tp_size = 8
    if world.world_size != 32 or world.local_rank != world.rank % host_tp_size:
        raise ValueError(
            "TPU_K3_HOST_REPLICATED_SHARED_EXPERTS requires a 4-host v7x-32 "
            "deployment with eight contiguous global ranks per host")
    group_ranks = [
        list(range(start, start + host_tp_size))
        for start in range(0, world.world_size, host_tp_size)
    ]
    _K3_HOST_TP_GROUP = init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=world.local_rank,
        backend=world.torch_distributed_backend,
        group_name="k3_host_tp",
    )
    return _K3_HOST_TP_GROUP


def _get_k3_hierarchical_tp_group() -> GroupCoordinator:
    """Create the topology-defined TP groups used by hierarchical MoE."""
    global _K3_HIERARCHICAL_TP_GROUP
    if _K3_HIERARCHICAL_TP_GROUP is not None:
        return _K3_HIERARCHICAL_TP_GROUP

    world = get_world_group()
    groups: dict[int, list[tuple[int, int]]] = {}
    for rank in range(world.world_size):
        split = hierarchical_moe_split(
            world.world_size, rank, envs.TPU_MOE_HIERARCHICAL_TP_SIZE)
        if split is None:
            raise ValueError("hierarchical TP group requested without a "
                             "usable chip topology")
        _, ep_rank, _, tp_rank = split
        groups.setdefault(ep_rank, []).append((tp_rank, rank))
    group_ranks = [[rank for _, rank in sorted(members)]
                   for _, members in sorted(groups.items())]
    _K3_HIERARCHICAL_TP_GROUP = init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=world.local_rank,
        backend=world.torch_distributed_backend,
        group_name="k3_hierarchical_tp",
    )
    return _K3_HIERARCHICAL_TP_GROUP


def _get_k3_projection_tp_group(tp_size: int) -> GroupCoordinator:
    """Create topology-ordered groups for an independently sharded projection."""
    if tp_size == envs.TPU_MOE_HIERARCHICAL_TP_SIZE:
        return _get_k3_hierarchical_tp_group()
    if tp_size in _K3_PROJECTION_TP_GROUPS:
        return _K3_PROJECTION_TP_GROUPS[tp_size]

    world = get_world_group()
    groups: dict[int, list[tuple[int, int]]] = {}
    for rank in range(world.world_size):
        split = hierarchical_moe_split(world.world_size, rank, tp_size)
        if split is None:
            raise ValueError(f"projection TP{tp_size} requested without a "
                             "usable whole-chip topology")
        _, group_rank, _, tp_rank = split
        groups.setdefault(group_rank, []).append((tp_rank, rank))
    group_ranks = [[rank for _, rank in sorted(members)]
                   for _, members in sorted(groups.items())]
    group = init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=world.local_rank,
        backend=world.torch_distributed_backend,
        group_name=f"k3_projection_tp{tp_size}",
    )
    _K3_PROJECTION_TP_GROUPS[tp_size] = group
    return group


class KimiRoutedInputTransform(nn.Module):
    """Finish a row-sharded latent down projection within its chip group."""

    def __init__(self, down_proj: RowParallelLinear,
                 tp_group: GroupCoordinator) -> None:
        super().__init__()
        self.down_proj = down_proj
        self.tp_group = tp_group

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states

    def reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.tp_group.all_reduce(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.project(hidden_states))


class KimiRoutedOutputTransform(nn.Module):

    def __init__(
        self,
        norm: RMSNorm | None,
        up_proj: ReplicatedLinear | RowParallelLinear,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj
        self.output_scale = output_scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.up_proj(hidden_states)
        if self.output_scale != 1.0:
            hidden_states = hidden_states * self.output_scale
        return hidden_states


class KimiShardedUpMoERunner(MoERunner):
    """Finish a row-sharded latent up-projection in the shared-output AR.

    The latent expert output still needs its global reduction before RMSNorm.
    After that, every rank consumes its TP shard of the normalized latent and
    produces a hidden-width partial.  Keeping the shared expert partial local
    lets the existing second all-reduce finish both projections together.
    """

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        del fused_output_is_reduced
        return shared_output

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        del output_is_reduced
        states = tensor_model_parallel_all_reduce(states)
        return states[..., :trunc_size] if trunc_size is not None else states


class KimiFusedMoECombine(nn.Module):
    """Combine shared-expert and routed-latent partials in one all-reduce.

    All-reduce is linear and elementwise, so concatenating both partials,
    reducing once, and splitting yields the same sums as the separate
    pre-norm latent reduction and shared/routed combine it replaces.
    """

    def __init__(
        self,
        norm: RMSNorm,
        up_proj: ReplicatedLinear,
        hidden_size: int,
        expert_hidden_size: int,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj
        self.split_sizes = [hidden_size, expert_hidden_size]

    def forward(self, shared_partial: torch.Tensor,
                latent_partial: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([shared_partial, latent_partial], dim=-1)
        fused = tensor_model_parallel_all_reduce(fused)
        shared_sum, latent = fused.split(self.split_sizes, dim=-1)
        latent = self.norm(latent)
        routed, _ = self.up_proj(latent)
        return shared_sum + routed


class KimiMoE(nn.Module):
    """Standard or latent Kimi MoE backed by the fused TPU GMM custom op."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.num_experts is not None
        assert config.num_experts_per_token is not None
        assert config.moe_intermediate_size is not None

        hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        vllm_config = get_current_vllm_config_or_none()
        self.use_ep = (vllm_config is not None
                       and vllm_config.parallel_config.enable_expert_parallel)
        routed_expert_hidden_size = getattr(config,
                                            "routed_expert_hidden_size", None)
        expert_hidden_size = (routed_expert_hidden_size
                              if routed_expert_hidden_size is not None else
                              hidden_size)

        dp_size = 1
        pcp_size = 1
        if vllm_config is not None:
            parallel_config = vllm_config.parallel_config
            dp_size = parallel_config.data_parallel_size
            pcp_size = parallel_config.prefill_context_parallel_size
        shard_latent_up = (
            envs.TPU_K3_SHARD_LATENT_UP_PROJ
            and routed_expert_hidden_size is not None
            and getattr(config, "latent_moe_use_norm", False)
            and config.num_shared_experts > 0 and self.tp_size > 1
            and dp_size == 1 and pcp_size == 1
            and config.routed_scaling_factor == 1.0)
        hierarchical_split = hierarchical_split_or_none()
        shard_latent_down = (
            envs.TPU_K3_SHARD_LATENT_DOWN_PROJ
            and routed_expert_hidden_size is not None
            and hierarchical_split is not None and self.use_ep
            and self.tp_size > 1 and dp_size == 1 and pcp_size == 1)

        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts))

        latent_down_tp_size = self.tp_size
        latent_down_tp_rank = 0
        latent_down_tp_group = None
        if shard_latent_down:
            assert hierarchical_split is not None
            requested_down_tp = envs.TPU_K3_LATENT_DOWN_TP_SIZE
            if requested_down_tp:
                down_split = hierarchical_moe_split(
                    get_tensor_model_parallel_world_size(),
                    get_tensor_model_parallel_rank(), requested_down_tp)
                if down_split is None:
                    raise ValueError(
                        f"TPU_K3_LATENT_DOWN_TP_SIZE={requested_down_tp} is "
                        "not a valid whole-chip topology split")
            else:
                down_split = hierarchical_split
            _, _, latent_down_tp_size, latent_down_tp_rank = down_split
            latent_down_tp_group = _get_k3_projection_tp_group(
                latent_down_tp_size)
        down_proj_cls = (RowParallelLinear
                         if shard_latent_down else ReplicatedLinear)
        down_proj_kwargs = ({
            "input_is_parallel": False,
            "reduce_results": False,
        } if shard_latent_down else {})
        down_proj_context = (_linear_tp_override(latent_down_tp_rank,
                                                 latent_down_tp_size)
                             if shard_latent_down else nullcontext())
        with down_proj_context:
            down_proj = (down_proj_cls(
                hidden_size,
                expert_hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_down_proj",
                **down_proj_kwargs,
            ) if routed_expert_hidden_size is not None else None)
        # Keep the checkpoint-visible parameter path unchanged; only the
        # module handed to MoERunner wraps the projection with a subgroup AR.
        self.routed_expert_down_proj = down_proj
        routed_input_transform = (KimiRoutedInputTransform(
            down_proj, latent_down_tp_group) if shard_latent_down else down_proj)
        self.routed_input_transform = routed_input_transform
        self.overlap_latent_down_router = (
            envs.TPU_K3_OVERLAP_LATENT_DOWN_ROUTER
            and isinstance(routed_input_transform, KimiRoutedInputTransform))
        self.routed_expert_norm = (
            RMSNorm(expert_hidden_size, config.rms_norm_eps)
            if down_proj is not None
            and getattr(config, "latent_moe_use_norm", False) else None)
        latent_up_tp_size = self.tp_size
        latent_up_tp_rank = 0
        if shard_latent_up:
            split = hierarchical_split_or_none()
            if split is not None:
                _, _, latent_up_tp_size, latent_up_tp_rank = split
        up_proj_cls = RowParallelLinear if shard_latent_up else ReplicatedLinear
        up_proj_kwargs = ({
            "input_is_parallel": False,
            "reduce_results": False,
        } if shard_latent_up else {})
        up_proj_context = (_linear_tp_override(latent_up_tp_rank,
                                               latent_up_tp_size)
                           if shard_latent_up else nullcontext())
        with up_proj_context:
            self.routed_expert_up_proj = (up_proj_cls(
                expert_hidden_size,
                hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_up_proj",
                **up_proj_kwargs,
            ) if self.routed_expert_down_proj is not None else None)
        latent_up_scale = (latent_up_tp_size / self.tp_size
                           if shard_latent_up else 1.0)
        routed_output_transform = (KimiRoutedOutputTransform(
            self.routed_expert_norm, self.routed_expert_up_proj,
            latent_up_scale)
                                   if self.routed_expert_up_proj is not None
                                   else None)

        self.host_replicated_shared = (
            envs.TPU_K3_HOST_REPLICATED_SHARED_EXPERTS
            and config.num_shared_experts > 0 and self.tp_size == 32
            and dp_size == 1 and pcp_size == 1)
        self.early_shared_reduce = (
            envs.TPU_K3_EARLY_SHARED_ALL_REDUCE
            and config.num_shared_experts > 0 and self.tp_size > 1
            and dp_size == 1 and pcp_size == 1
            and not self.host_replicated_shared)
        shared_tp_group = (_get_k3_host_tp_group()
                           if self.host_replicated_shared else None)

        self.shared_experts = (KimiMLP(
            hidden_size,
            config.moe_intermediate_size * config.num_shared_experts,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_experts",
            reduce_results=self.early_shared_reduce,
            situ_beta=getattr(config, "activation_situ_beta", None),
            situ_linear_beta=getattr(config, "activation_situ_linear_beta",
                                     None),
            tp_group=shared_tp_group,
        ) if config.num_shared_experts > 0 else None)

        # Fused single-collective combine (TPU_K3_FUSED_MOE_COMBINE): the
        # latent norm forces a reduction before up_proj, and the shared
        # experts provide the second reduction to share. Unsupported
        # configs (DP/PCP > 1, scaled routing) keep the two-reduce path;
        # KimiMoE never passes is_sequence_parallel, so SP cannot reach it.
        use_fused_combine = (envs.TPU_K3_FUSED_MOE_COMBINE
                             and not shard_latent_up
                             and not self.host_replicated_shared
                             and not self.early_shared_reduce
                             and self.routed_expert_norm is not None
                             and self.shared_experts is not None
                             and self.tp_size > 1 and dp_size == 1
                             and pcp_size == 1
                             and config.routed_scaling_factor == 1.0)
        self.fused_combine = (KimiFusedMoECombine(
            self.routed_expert_norm,
            self.routed_expert_up_proj,
            hidden_size,
            expert_hidden_size,
        ) if use_fused_combine else None)

        padded_intermediate_size = config.moe_intermediate_size
        min_per_partition = getattr(config,
                                    "min_moe_intermediate_per_partition", 128)
        if (not self.use_ep and self.tp_size > 1 and
                padded_intermediate_size // self.tp_size < min_per_partition):
            padded_intermediate_size = min_per_partition * self.tp_size

        situ_beta = (getattr(config, "activation_situ_beta", None)
                     if config.hidden_act == "situ" else None)
        situ_linear_beta = (getattr(config, "activation_situ_linear_beta",
                                    None)
                            if config.hidden_act == "situ" else None)
        # Optionally build the routed weights with EP between physical chips
        # and TP between the two tensor cores on each chip.
        with hierarchical_moe_parallel_config():
            self.experts = FusedMoEFactory(
                shared_experts=(None if self.fused_combine is not None
                                or self.host_replicated_shared
                                or self.early_shared_reduce else
                                self.shared_experts),
                num_experts=config.num_experts,
                top_k=config.num_experts_per_token,
                hidden_size=expert_hidden_size,
                intermediate_size=padded_intermediate_size,
                activation=config.hidden_act,
                activation_situ_beta=situ_beta,
                activation_situ_linear_beta=situ_linear_beta,
                renormalize=config.moe_renormalize,
                quant_config=quant_config,
                use_grouped_topk=config.use_grouped_topk,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.moe_router_activation_func,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=config.routed_scaling_factor,
                routed_input_transform=routed_input_transform,
                routed_output_transform=(
                    None if self.fused_combine is not None else
                    routed_output_transform),
                reduce_results=self.fused_combine is None,
                runner_cls=(KimiShardedUpMoERunner
                            if shard_latent_up else None),
            )
        if self.fused_combine is not None:
            assert self.experts.moe_config.skip_final_all_reduce
        if padded_intermediate_size != config.moe_intermediate_size:
            routed_experts = self.experts.routed_experts
            w13_weight = getattr(routed_experts, "w13_weight", None)
            if w13_weight is None:
                w13_weight = routed_experts.w13_weight_packed
            w2_weight = getattr(routed_experts, "w2_weight", None)
            if w2_weight is None:
                w2_weight = routed_experts.w2_weight_packed
            w13_weight.data.zero_()
            w2_weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                config.moe_intermediate_size // self.tp_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        routed_hidden_states = hidden_states
        shared_experts_input = None
        if self.overlap_latent_down_router:
            assert isinstance(self.routed_input_transform,
                              KimiRoutedInputTransform)
            routed_partial = self.routed_input_transform.project(hidden_states)
            router_logits, _ = self.gate(hidden_states)
            routed_hidden_states = self.routed_input_transform.reduce(
                routed_partial)
            # MoERunner treats a non-None shared input as proof that the caller
            # pre-applied routed_input_transform. Shared experts still consume
            # the original hidden width while routed experts consume latent.
            shared_experts_input = hidden_states
        else:
            router_logits, _ = self.gate(hidden_states)
        if self.fused_combine is not None:
            # The MoE runner returns the un-reduced latent partial
            # (reduce_results=False); one concatenated all-reduce combines
            # it with the shared-expert partial.
            shared_partial = self.shared_experts(hidden_states)
            latent_partial = self.experts(
                hidden_states=routed_hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input)
            output = self.fused_combine(shared_partial, latent_partial)
        elif self.host_replicated_shared or self.early_shared_reduce:
            # The routed path performs its required global TP reduction. The
            # shared path is already complete after either its host-local TP8
            # reduce or its deliberately early global TP reduce.
            shared_output = self.shared_experts(hidden_states)
            routed_output = self.experts(
                hidden_states=routed_hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input)
            output = shared_output + routed_output
        else:
            output = self.experts(
                hidden_states=routed_hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input)
        return output.view(num_tokens, hidden_size)
