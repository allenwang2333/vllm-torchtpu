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
"""Torch bridge for the fused expert-parallel MoE kernel v3.

Selected with MOE_FUSED_EP_KERNEL_IMPL=v3. Like v2 (`fused_moe_ep.py`) the
whole layer -- top-k, dispatch, both expert matmuls, combine -- runs as one
SPMD program over the EP mesh, taking and returning this rank's own tokens.

v3 serves data parallelism with expert parallelism only (TP=1, PCP=1), with
per-channel fp8 weights and a plain softmax top-k router. A deployment outside
that raises at weight load; a layer outside it (shape, routing, weights) is
declined on its own and keeps the grouped-matmul path.

Padded rows: with TPU_MOE_SKIP_PADDED_TOKENS the op takes the rank's valid-row
count and hands the kernel a NaN logit row for every row past it, which the
kernel's selector routes nowhere (`drop_nan_rows`); the input row is zeroed so
nothing of it travels, and the kernel returns zero for it.
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal.pallas import jax_op

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
)
from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, Config, fused_ep_moe
from vllm_torchtpu.kernels.fused_moe.v3 import layer as _v3_layer
from vllm_torchtpu.kernels.fused_moe.v3 import layout as _v3_layout
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.fused_moe_ep import (
    _max_node_tokens,
    _pcp_size,
    _squeeze_channel_scale,
    _squeeze_channel_scale_checked,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The FFN tile height (the kernel's `tile_rows`). Smaller tiles are far slower
# with the kernel's DMA strategy, so it is not a knob.
TILE_ROWS = 128
WEIGHT_FORMAT = "fp8"

# One op per distinct closure, as in v2: a per-layer op would re-export the
# same program once per MoE layer, and two `CustomOpDef`s under one qualname
# silently share a dispatch entry. The name must not be v2's
# `pallas::fused_moe_ep`: both modules can be imported in one process.
FUSED_MOE_EP_V3_OP_NAME = "pallas::fused_moe_ep_v3"
_OPS: dict[tuple, Any] = {}

# The attribute prebuild's op is recorded under, on the layer's quant method.
FUSED_MOE_EP_V3_OP_ATTR = "_fused_moe_ep_v3_op"

# This rank's mesh index, as a device tensor: the kernel takes it as data,
# because `lax.axis_index` lowers to a partition-id the export path refuses.
_RANK_BUFFER: Any = None


def mask_local_rows(x: jax.Array, gating: jax.Array, valid_tokens: jax.Array):
    """One rank's rows: row r is valid when r < valid_tokens[0]. An invalid
    row gets NaN logits and a zero input row."""
    local = jnp.arange(x.shape[0], dtype=jnp.int32)
    valid = (local < valid_tokens.astype(jnp.int32)[0])[:, None]
    gating = jnp.where(valid, gating, jnp.nan)
    x = jnp.where(valid, x, jnp.zeros_like(x))
    return x, gating


def mask_padded_rows(
    x: jax.Array, gating: jax.Array, valid_tokens: jax.Array, ep: int, mesh=None
):
    """Rows past each rank's valid prefix route nowhere and carry nothing.

    `x` and `gating` are the global arrays, `ep * t_local` rows in rank order;
    `valid_tokens` is int32 [ep]. With `mesh`, each rank masks its own rows in
    a shard_map: the global form's iota over the token axis partitions into a
    partition-id, which the torch bridge's export refuses.
    """
    if mesh is not None:
        spec = PartitionSpec(MESH_AXIS)
        return jax.shard_map(
            mask_local_rows,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=(spec, spec),
        )(x, gating, valid_tokens)
    tokens = x.shape[0]
    t_local = tokens // ep
    local = jnp.arange(tokens, dtype=jnp.int32) % t_local
    limit = jnp.repeat(valid_tokens.astype(jnp.int32), t_local)
    valid = (local < limit)[:, None]
    gating = jnp.where(valid, gating, jnp.nan)
    x = jnp.where(valid, x, jnp.zeros_like(x))
    return x, gating


def served_config(*, hidden: int, inter: int, experts_per_shard: int, top_k: int):
    """The kernel's Config for this shape, logged once."""
    config = Config.for_shape(
        hidden=hidden, inter=inter, experts_per_shard=experts_per_shard, top_k=top_k
    )
    logger.info_once(
        "Fused EP MoE v3 settings for hidden=%d inter=%d experts_per_shard=%d "
        "top_k=%d: tile_rows=%d %s",
        hidden,
        inter,
        experts_per_shard,
        top_k,
        TILE_ROWS,
        config,
    )
    return config


def run_moe_v3(
    mesh,
    topk: int,
    renormalize: bool,
    activation: str,
    *,
    hidden: int,
    inter: int,
    experts_per_shard: int,
):
    """The `run_moe` the op wraps: global arrays sharded on the EP axis in,
    one [tokens, hidden] array sharded like x out."""
    # The torch bridge exports the op as one jitted function: the layer's own
    # inner jit is what the export path's SPMD partitioner refuses, it wants a
    # sharding on every side-effecting custom call, and it refuses
    # `lax.axis_index`, so the shard's index goes in as data.
    _v3_layer.WRAP_IN_JIT = False
    _v3_layout.SIDE_EFFECTS = False
    _v3_layout.SHARD_INDEX_OPERAND = True
    if MESH_AXIS not in mesh.axis_names:
        raise ValueError(
            f"the v3 kernel splits the experts over the mesh axis named "
            f"{MESH_AXIS!r}; the adapter's mesh has {mesh.axis_names}"
        )
    ep = mesh.shape[MESH_AXIS]
    config = served_config(
        hidden=hidden, inter=inter, experts_per_shard=experts_per_shard, top_k=topk
    )

    def run_moe(x, w1, w2, w1_scale, w2_scale, gating, rank, valid_tokens):
        gating = gating.astype(jnp.float32)
        drop_nan_rows = valid_tokens is not None
        if drop_nan_rows:
            x, gating = mask_padded_rows(x, gating, valid_tokens, ep, mesh=mesh)
        return fused_ep_moe(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            gating,
            top_k=topk,
            renormalize=renormalize,
            mesh=mesh,
            tile_rows=TILE_ROWS,
            weight_format=WEIGHT_FORMAT,
            activation=activation,
            drop_nan_rows=drop_nan_rows,
            config=config,
            shard_index=rank,
            experts_per_shard=experts_per_shard,
        )

    return run_moe


def _build_op(
    mesh,
    topk: int,
    renormalize: bool,
    activation: str,
    skip_padded_tokens: bool,
    *,
    hidden: int,
    inter: int,
    local_experts: int,
):
    """The op for this closure, built once per distinct closure and reused."""
    key = (
        topk,
        renormalize,
        activation,
        skip_padded_tokens,
        hidden,
        inter,
        local_experts,
    )
    cached = _OPS.get(key)
    if cached is not None:
        return cached
    name = (
        FUSED_MOE_EP_V3_OP_NAME
        if not _OPS
        else f"{FUSED_MOE_EP_V3_OP_NAME}_{len(_OPS)}"
    )
    run_moe = run_moe_v3(
        mesh,
        topk,
        renormalize,
        activation,
        hidden=hidden,
        inter=inter,
        experts_per_shard=local_experts,
    )

    # Every operand is annotated jax.Array: torch_tpu infers static argnums
    # from the signature and refuses an unannotated one.
    def moe(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return run_moe(x, w1, w2, w1_scale, w2_scale, gating, rank, None)

    def moe_skip_padded(
        x: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        gating: jax.Array,
        rank: jax.Array,
        valid_tokens: jax.Array,
    ) -> jax.Array:
        return run_moe(x, w1, w2, w1_scale, w2_scale, gating, rank, valid_tokens)

    fn = moe_skip_padded if skip_padded_tokens else moe
    spec = PartitionSpec(EP_AXIS_NAME)
    # x, w1, w2, w1_scale, w2_scale, gating, rank [, valid_tokens] -- every
    # operand and the single output are sharded on axis 0 over the EP axis.
    op = jax_op(
        name,
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * (8 if skip_padded_tokens else 7),
        output_partition_specs=spec,
    )

    # The kernel returns the rows it was given, so the fake needs no export.
    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def fused_moe_ep_v3_supported(owner: Any) -> bool:
    """Whether `prebuild_fused_moe_ep_v3` armed the v3 op for this layer.

    `owner` is the layer's quant method, where the op was recorded.
    """
    return getattr(owner, FUSED_MOE_EP_V3_OP_ATTR, None) is not None


def fused_moe_ep_v3(
    owner: Any,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Run one MoE layer through the v3 kernel.

    Takes this rank's own tokens and returns this rank's own tokens, so the
    caller must not all-gather the input or reduce-scatter the output. Only
    call it when `fused_moe_ep_v3_supported(owner)` said yes.
    """
    op = getattr(owner, FUSED_MOE_EP_V3_OP_ATTR)
    args = (
        hidden_states,
        w1,
        w2,
        _squeeze_channel_scale_checked(w1_scale),
        _squeeze_channel_scale_checked(w2_scale),
        router_logits,
        _RANK_BUFFER,
    )
    if envs.TPU_MOE_SKIP_PADDED_TOKENS:
        valid_tokens = token_padding.get_local_valid_tokens()
        if valid_tokens is None:
            raise RuntimeError(
                "TPU_MOE_SKIP_PADDED_TOKENS is set but the runner did not "
                "install padding state"
            )
        return op(*args, valid_tokens)
    return op(*args)


def _check_deployment(layer) -> None:
    """Raise on a deployment v3 cannot serve at all.

    The operator selected v3 explicitly, so these are errors rather than a
    silent fallback to the grouped-matmul path.
    """
    from vllm.config import get_current_vllm_config

    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )

    # Read off the config, not `get_tp_group()`: a draft model's weight load
    # rewrites the group coordinator to width 1.
    tp = get_current_vllm_config().parallel_config.tensor_parallel_size
    if tp > 1:
        raise NotImplementedError(
            f"MOE_FUSED_EP_KERNEL_IMPL=v3 serves data parallelism with expert "
            f"parallelism only; tensor_parallel_size={tp} replicates every "
            "token on each TP rank, so the kernel would do tp_size times the "
            "expert work. Use --data-parallel-size with "
            "--tensor-parallel-size=1, or MOE_FUSED_EP_KERNEL_IMPL=v2."
        )
    pcp = _pcp_size(layer)
    if pcp > 1:
        raise NotImplementedError(
            f"MOE_FUSED_EP_KERNEL_IMPL=v3 does not support prefill context "
            f"parallelism (pcp_size={pcp}). Use MOE_FUSED_EP_KERNEL_IMPL=v2."
        )
    if not layer.moe_config.moe_parallel_config.use_ep:
        raise NotImplementedError(
            "MOE_FUSED_EP_KERNEL_IMPL=v3 addresses experts by EP shard; pass "
            "--enable-expert-parallel."
        )
    if enable_pipelined_collective_and_compute():
        raise NotImplementedError(
            "MOE_FUSED_EP_KERNEL_IMPL=v3 and MoE chunk pipelining "
            "(TPU_MOE_COLLECTION_CHUNK_SIZE) both claim the expert-parallel "
            "exchange; enable one of them."
        )


def prebuild_fused_moe_ep_v3(
    layer, topk: int, renormalize: bool, activation: str
) -> Any | None:
    """Resolve the mesh and build the v3 op for one layer.

    Called once per MoE layer -- the MTP draft's included -- from
    `process_weights_after_loading`: mapping EP ranks to devices is a
    collective and a `Mesh` cannot survive Dynamo's tracing. Raises on a
    deployment v3 cannot serve; returns None for a layer it declines, which
    keeps the grouped-matmul path.
    """
    global _RANK_BUFFER
    _check_deployment(layer)
    mesh = build_ep_mesh()
    if mesh is None:
        logger.info_once(
            "Fused EP MoE v3 requested but there is no EP group; the "
            "grouped-matmul path serves every step."
        )
        return None
    ep = mesh.shape[EP_AXIS_NAME]

    w1, w2 = layer.w13_weight, layer.w2_weight
    w1_scale, w2_scale = layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
    local_experts, hidden = w1.shape[0], w1.shape[1]
    inter = w1.shape[2] // 2

    max_node_tokens = _max_node_tokens(ep)
    if max_node_tokens < envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS:
        logger.info_once(
            "Fused EP MoE v3 not engaged: the largest node-wide batch is %d "
            "tokens, below MOE_FUSED_EP_KERNEL_MIN_TOKENS=%d.",
            max_node_tokens,
            envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS,
        )
        return None

    reason = (
        _unsupported_layer_reason(layer, local_experts, ep)
        or _unsupported_routing_reason(layer)
        or _unsupported_weights_reason(
            w1, w2, w1_scale, w2_scale, local_experts, hidden, inter, activation
        )
    )
    if reason is not None:
        logger.warning_once("Fused EP MoE v3 not engaged for a layer: %s", reason)
        return None

    # The mesh is rank-ordered (google-pytorch/torch_tpu#3522), so this rank's
    # mesh index is its EP rank and its expert block needs no reordering.
    _RANK_BUFFER = torch.tensor(
        [[ep_mesh_index()]], dtype=torch.int32, device=w1.device
    )
    skip_padded_tokens = bool(envs.TPU_MOE_SKIP_PADDED_TOKENS)
    op = _build_op(
        mesh,
        topk,
        bool(renormalize),
        activation,
        skip_padded_tokens,
        hidden=hidden,
        inter=inter,
        local_experts=local_experts,
    )
    logger.info_once(
        "Fused EP MoE v3 armed | hidden=%d inter=%d local_experts=%d ep=%d "
        "topk=%d tile_rows=%d skip_padded_tokens=%s",
        hidden,
        inter,
        local_experts,
        ep,
        topk,
        TILE_ROWS,
        skip_padded_tokens,
    )
    return op


def _unsupported_layer_reason(layer, local_experts: int, ep: int) -> str | None:
    """Why this layer's expert layout or biases cannot go through v3."""
    if layer.global_num_experts != local_experts * ep:
        return (
            f"{layer.global_num_experts} experts do not split evenly over "
            f"{ep} shards of {local_experts}"
        )
    if (
        getattr(layer, "w13_bias", None) is not None
        or getattr(layer, "w2_bias", None) is not None
    ):
        return "expert biases are not wired through the v3 op"
    return None


def _unsupported_routing_reason(layer) -> str | None:
    """Why this layer's routing is not the kernel's, or None.

    The kernel scores with a plain softmax and selects top-k itself, with
    `renormalize` as its only option. Stricter than v2's guard, which also
    serves sigmoid scoring, a score-correction bias, routed scaling and hash
    routing: none of those is computed here, and taking the path anyway would
    route tokens to different experts with different weights, silently.
    """
    if layer.custom_routing_function is not None:
        return "the layer supplies a custom_routing_function"
    if layer.scoring_func != "softmax":
        return f"scoring_func {layer.scoring_func!r}; the kernel scores with softmax"
    if layer.e_score_correction_bias is not None:
        return "the layer has an e_score_correction_bias"
    if layer.routed_scaling_factor != 1.0:
        return f"routed_scaling_factor {layer.routed_scaling_factor}"
    if (
        layer.use_grouped_topk
        and layer.num_expert_group > 1
        and layer.topk_group < layer.num_expert_group
    ):
        return "grouped top-k routing"
    if layer.apply_router_weight_on_input:
        return "the router weight is applied on the input"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "the layer routes by hash table"
    if moe_routing._SIMULATION_STRATEGY is not None:
        return (
            "VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set and the kernel "
            "does its own routing"
        )
    return None


def _unsupported_weights_reason(
    w1, w2, w1_scale, w2_scale, local_experts, hidden, inter, activation
) -> str | None:
    """Why this layer's weights or activation cannot use v3, or None."""
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
        return f"weight rank/expert mismatch {tuple(w1.shape)} {tuple(w2.shape)}"
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        return (
            f"the kernel's fp8 form takes float8_e4m3fn expert weights, got "
            f"{w1.dtype} / {w2.dtype}"
        )
    if w2.shape[1] != inter or w2.shape[2] != hidden:
        return f"w2 {tuple(w2.shape)} is not [E, {inter}, {hidden}]"
    if activation != "silu":
        return f"activation {activation!r}; the adapter wires silu only"
    if _squeeze_channel_scale(w1_scale, local_experts, 2 * inter) is None:
        return "w1_scale is not the per-output-channel form the kernel takes"
    if _squeeze_channel_scale(w2_scale, local_experts, hidden) is None:
        return "w2_scale is not the per-output-channel form the kernel takes"
    return None
