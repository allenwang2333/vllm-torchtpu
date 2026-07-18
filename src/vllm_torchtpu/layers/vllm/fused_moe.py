# Copyright 2025 Google LLC
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
"""Torch bridge for fused MoE based on fused_moe_func in fused_moe_gmm."""

import functools
from typing import Callable, Optional

import torch
import jax
from vllm_torchtpu.patches import pallas_patched as pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.common.fused_moe_gmm import fused_moe_func

_kernel_instance_counter = 0
_fused_moe_kernel_cache: dict[tuple[int, str, Optional[int], bool], Callable] = {}


def _allocate_kernel_instance_id() -> int:
    global _kernel_instance_counter
    kernel_instance_id = _kernel_instance_counter
    _kernel_instance_counter += 1
    return kernel_instance_id


def _fused_moe_sharded_jax(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: Optional[jax.Array],
    w2_scale: Optional[jax.Array],
    w1_bias: Optional[jax.Array],
    w2_bias: Optional[jax.Array],
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    input_partition_specs: tuple[tuple[str, ...], ...],
    output_partition_specs: tuple[tuple[str, ...], ...],
    experts_start: Optional[int],
    topk: int,
    activation: str,
    use_sparse_core: bool,
) -> jax.Array:
    from jax import shard_map

    def _local_fused_moe(hs, w1_, w2_, w1_s, w2_s, w1_b, w2_b, tw, ti):
        local_out = fused_moe_func(
            hs,
            w1_,
            w2_,
            w1_s,
            w2_s,
            w1_b,
            w2_b,
            tw,
            ti,
            experts_start=experts_start,
            topk=topk,
            activation=activation,
            use_ep=experts_start is not None,
            use_sparse_core=use_sparse_core,
            onehot_moe_permute_threshold=envs.ONEHOT_MOE_PERMUTE_THRESHOLD,
        )
        if experts_start is not None:
            return jax.lax.psum(local_out, axis_name="expert")
        return local_out

    out_specs = output_partition_specs[0]

    return shard_map(
        _local_fused_moe,
        mesh=mesh,
        in_specs=input_partition_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk_weights,
        topk_ids,
    )


def _build_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    experts_start: Optional[int],
    has_bias: bool,
):
    kernel_instance_id = _allocate_kernel_instance_id()
    op_name = f"pallas::fused_moe_kernel_{kernel_instance_id}"

    # TODO: once the pod's vLLM includes --moe-backend (added upstream in
    # vLLM PR #33807, 2026-03-01; absent in the 0.19.0 we run), register the
    # SparseCore vs plain variants through that selector + the MoE oracle
    # instead of this env flag -- the MoE analog of how #191 moved batched
    # RPA onto --attention-backend CUSTOM. Note MoEBackend has no CUSTOM slot
    # yet, so that also needs an upstream OOT-backend hook.
    use_sparse_core = envs.USE_MOE_SPARSE_CORE

    from jax.sharding import PartitionSpec as P
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import get_vllm_model_wrapper_context
    from vllm_torchtpu.layers.common.sharding import ShardingAxisName

    vllm_context = get_vllm_model_wrapper_context()
    mesh = vllm_context.mesh
    use_ep = experts_start is not None

    if use_ep:
        w1_spec = P("expert", None, None)
        w2_spec = P("expert", None, None)
        w1_scale_spec = P("expert", None, None)
        w2_scale_spec = P("expert", None, None)
        w1_bias_spec = P("expert", None, None) if has_bias else None
        w2_bias_spec = P("expert", None, None) if has_bias else None
    else:
        w1_spec = P(None, None, "model")
        w2_spec = P(None, "model", None)
        w1_scale_spec = P(None, None, None, "model")
        w2_scale_spec = P(None, "model", None, None)
        w1_bias_spec = P(None, None, "model") if has_bias else None
        w2_bias_spec = P(None, None, None) if has_bias else None

    data_axis = ShardingAxisName.MLP_DATA

    input_partition_specs = (
        P(data_axis, None),       # hidden_states
        w1_spec,                  # w1
        w2_spec,                  # w2
        w1_scale_spec,            # w1_scale
        w2_scale_spec,            # w2_scale
        w1_bias_spec,             # w1_bias
        w2_bias_spec,             # w2_bias
        P(data_axis, None),       # topk_weights
        P(data_axis, None),       # topk_ids
    )

    output_partition_specs = (P(data_axis, None),)

    wrapped_fn = functools.partial(
        _fused_moe_sharded_jax,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_partition_specs,
        experts_start=experts_start,
        topk=topk,
        activation=activation,
        use_sparse_core=use_sparse_core,
    )

    fused_moe_kernel_impl = pallas.jax_op(
        op_name,
        wrapped_fn,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_partition_specs,
    )


    # We must overwrite the default fake implementation as vLLM uses dynamic
    # dimensions for the hidden_states.
    def _fake_fused_moe(hidden_states, *args, **kwargs):
        return torch.empty_like(hidden_states)

    fused_moe_kernel_impl.register_fake(_fake_fused_moe)

    cache_key = (topk, activation, experts_start, has_bias)
    _fused_moe_kernel_cache[cache_key] = fused_moe_kernel_impl
    return fused_moe_kernel_impl


def _get_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    experts_start: Optional[int],
    has_bias: bool,
):
    cache_key = (topk, activation, experts_start, has_bias)
    kernel = _fused_moe_kernel_cache.get(cache_key)
    if kernel is not None:
        return kernel
    return _build_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        experts_start=experts_start,
        has_bias=has_bias,
    )


def prebuild_fused_moe_kernel(
    *,
    topk: int,
    activation: str,
    experts_start: Optional[int],
    has_bias: bool = False,
) -> None:
    """Prebuild and cache fused MoE custom op outside compile-time tracing."""
    _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        experts_start=experts_start,
        has_bias=has_bias,
    )


def fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_bias: Optional[torch.Tensor],
    w2_bias: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: Optional[int],
    topk: int,
    activation: str,
) -> torch.Tensor:
    """Fused MoE forward pass with precomputed routing.

    ``experts_start`` is the first global expert id owned by this shard under
    linear EP placement (or ``None`` for non-EP). It is a Python int derived at
    load time from ``ep_rank``, ``ep_size``, and ``global_num_experts``, baked
    into the JAX kernel closure as a compile-time constant -- so the global->
    local remap is a literal subtract fused into the routing loop. When
    present, the sparse-core parity path dispatches ragged gather/gather-reduce.
    """
    # Deleted debug GMM inputs prints to prevent OOM on item() syncs
    fused_moe = _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        experts_start=experts_start,
        has_bias=w1_bias is not None,
    )
    
    w1_cloned = w1.clone()
    w2_cloned = w2.clone()
    w1_scale_cloned = w1_scale.clone() if w1_scale is not None else None
    w2_scale_cloned = w2_scale.clone() if w2_scale is not None else None
    w1_bias_cloned = w1_bias.clone() if w1_bias is not None else None
    w2_bias_cloned = w2_bias.clone() if w2_bias is not None else None

    res = fused_moe(
        hidden_states,
        w1_cloned,
        w2_cloned,
        w1_scale_cloned,
        w2_scale_cloned,
        w1_bias_cloned,
        w2_bias_cloned,
        topk_weights,
        topk_ids,
    )
    # Deleted debug GMM outputs prints to prevent OOM on item() syncs
    return res
