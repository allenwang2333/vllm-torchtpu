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
"""Torch bridge for the tokamax fused reduce-scatter MoE kernel.

`kernels/fused_moe_rs` is a copy of
`tokamax/_src/ops/experimental/fused_moe_rs` at openxla/tokamax `6303c14`,
carrying **three local deltas**, each marked `LOCAL DELTA vs tokamax 6303c14`
at its site in `kernels/fused_moe_rs/fused_moe_rs.py`. All three exist because
this tree reaches the kernel through torch_tpu's `jax.export` bridge, which
re-partitions the serialized module and rejects two constructs upstream is free
to use under plain `jax.jit`: (1) a `shard_map` boundary reshard, and (2)
`lax.axis_index` / a sharded jit-internal constant, both of which become
`partition-id`. `gmm_fused_rs_nodedup.py` and `gmm_v2_gather_scatter.py` are
untouched. It runs the whole EP MoE --
gather, GMM1, activation, GMM2, and the cross-chip exchange -- as one Pallas
program, so the reduce-scatter is the kernel's epilogue rather than an exposed
collective after it. That is the same argument `fused_moe_ep` makes for the v2
kernel, and the two present the identical torch-side contract: this rank's
tokens in, this rank's tokens out, no caller-side all-gather or reduce-scatter.
Sharing that contract is deliberate -- it lets both reuse every upstream
collective-ownership patch in `vllm_torchtpu/__init__.py` unchanged.

**Why this exists next to `fused_moe_ep` rather than inside it.** The two
kernels are different programs with different tilings; this one fuses the
reduce-scatter into its epilogue. It is a performance alternative, not a
capability one.

It is worth being explicit that the original justification for this path was
**wrong**, because it is the kind of claim that gets repeated. The reasoning
was: GLM-5.2-FP8's checkpoint is block-scaled [128, 128], the v2 kernel wants
one scale per output channel, therefore v2 structurally cannot serve GLM-5.2
and this kernel can. The premise does not survive contact with the loader.
`_process_fp8_moe_weights` dequantizes the checkpoint and **requantizes** it
before either adapter sees the layer -- per-channel by default, since
`MOE_REQUANTIZE_BLOCK_SIZE` is unset -- so what arrives here is per-channel fp8
in K-major layout, which is exactly what v2 accepts too. Measured on
job j-3c919167, the server logs
``w13_scale=[16, 1, 1, 4096] ... requant_block_size=per-channel``.
Whether this kernel beats v2 or the grouped-matmul path on GLM-5.2 is a
question for a benchmark, not for a weight layout.

**Routing is not fused here, and that is the point.** The kernel will select
its own top-k from a gating matrix, but its selector is a plain scoring
function plus `renormalize`, and GLM-5.2 routes with sigmoid scoring, a
noaux_tc score-correction bias and a routed scaling factor. Rather than
reproduce that inside the kernel -- where getting it wrong sends every token to
a different expert with nothing raised and nothing logged -- this path calls
vLLM's own `moe_routing.route`, exactly as the grouped-matmul path does, and
hands the result in through `fused_moe_func_rs`'s `topk_weights` /
`topk_indices` arguments. Routing is then equal to the baseline's by
construction rather than by test. Fusing the router is a later, separately
measurable change.

**Weight layout: there is nothing to do.** The kernel's GMM is K-major -- `w1`
is [E, hidden, 2*inter], `w2` is [E, inter, hidden] -- and its scale DMA
indexes four dimensions, [E, K_blocks, 1, N], with N in *elements*.
`_process_fp8_moe_weights` has already produced precisely that by the time
`prebuild_fused_moe_rs` runs, so `bind_weights_for_rs` only points the RS
attributes at the existing tensors. No transpose, no copy, no second buffer.

This module previously transposed and repeated here, on the assumption that it
was handed vLLM's on-disk [E, N, K] storage layout. It is not, and the
conversion was corrupting weights that were already correct.
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch

from jax.sharding import PartitionSpec

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (RS_DATA_AXIS_NAME,
                                               RS_EXPERT_AXIS_NAME,
                                               build_ep_rs_mesh,
                                               ep_mesh_index)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

FUSED_MOE_RS_OP_NAME = "pallas::fused_moe_rs"

# The attribute `prebuild_fused_moe_rs`'s result is recorded under, on the
# quant method that owns the layer -- per-layer state without a registry,
# because vLLM builds one quant method per MoE layer. Same device as
# `FUSED_MOE_EP_OP_ATTR` serves for the v2 path.
FUSED_MOE_RS_OP_ATTR = "_fused_moe_rs_op"

# Where `bind_weights_for_rs` points at the layer's already-K-major tensors.
# These are aliases, not copies.
RS_W1_ATTR = "_rs_w13_weight"
RS_W2_ATTR = "_rs_w2_weight"
RS_W1_SCALE_ATTR = "_rs_w13_weight_scale"
RS_W2_SCALE_ATTR = "_rs_w2_weight_scale"

# `fused_moe_rs.py:_assert_fused_rs_smem_safe` trips at this dispatch height:
# the kernel's per-row scalar-prefetch arrays are 4 * size_m bytes each and
# stop fitting SMEM. Mirrored here so the refusal is a log line at weight load
# rather than an assert inside a traced forward on the first real batch.
_FUSED_RS_MAX_SAFE_SIZE_M = 262144

# This rank's mesh index, as a device tensor. The kernel needs it as DATA:
# `lax.axis_index` lowers to `partition-id`, which the SPMD partitioner rejects
# when torch_tpu recompiles the exported module ("partitionId instruction is
# not supported for SPMD partitioning"). `fused_moe_ep._RANK_BUFFER` exists for
# exactly the same reason. It is the MESH index, not the EP rank -- the mesh is
# built in ascending device-id order; see `ep_mesh.ep_rank_order`.
_RS_RANK_BUFFER: Any = None

# Ops are built once per distinct closure and reused; a fresh `CustomOpDef`
# per layer would re-export the same program 75 times on GLM-5.2. Keyed on
# what the op closes over, because a single slot is last-writer-wins.
_OPS: dict[tuple, Any] = {}


def _op_name() -> str:
    """A distinct torch qualname per closure.

    Two `CustomOpDef`s registered under one name do not raise -- they share a
    dispatch entry, and the second closure silently serves the first.
    """
    return (FUSED_MOE_RS_OP_NAME
            if not _OPS else f"{FUSED_MOE_RS_OP_NAME}_{len(_OPS)}")


def _build_op(mesh, topk: int, activation: str):
    """The op for this closure, built once per distinct closure and reused."""
    from vllm_torchtpu.kernels.fused_moe_rs import fused_moe_func_rs

    key = (id(mesh), topk, activation)
    cached = _OPS.get(key)
    if cached is not None:
        return cached

    def moe(x: jax.Array, w1: jax.Array, w2: jax.Array, w1_scale: jax.Array,
            w2_scale: jax.Array, topk_weights: jax.Array,
            topk_indices: jax.Array, rank: jax.Array) -> jax.Array:
        # `renormalize` and `scoring_fn` are dead arguments on this path: both
        # are only read when the kernel computes its own top-k, and supplying
        # both `topk_weights` and `topk_indices` takes the branch that does
        # not. They are passed as the identity choice so a future reader does
        # not read them as the routing actually in effect.
        return fused_moe_func_rs(
            x,
            w1,
            w2,
            w1_scale,
            w2_scale,
            None,  # w1_bias -- GLM-5.2's experts are bias-free
            None,  # w2_bias
            None,  # gating_output, unused when routing is supplied
            topk,
            False,  # renormalize -- already applied by moe_routing.route
            mesh,
            activation,
            "softmax",
            topk_weights=topk_weights.astype(x.dtype),
            topk_indices=topk_indices.astype(jnp.int32),
            rank=rank.astype(jnp.int32),
        )

    # x and the routing tensors carry this rank's own tokens, so they are
    # sharded on the expert axis like the weights; the kernel's internal
    # shard_map reshards the hidden states to `P("data")` -- replicated over
    # the 16-wide expert axis -- which is the all-gather, and returns
    # `P(("data", "expert"))`, which is this rank's shard again.
    spec = PartitionSpec(RS_EXPERT_AXIS_NAME)
    op = sharded_jax_op(
        _op_name(),
        moe,
        mesh=mesh,
        input_partition_specs=(spec, ) * 8,
        output_partition_specs=spec,
    )

    # The kernel returns the rows it was given, so the output aval is known
    # without the real `jax.export` the stock shard-aware fake would run on
    # every Dynamo trace.
    def _fake(x, *_args, **_kwargs):
        return torch.empty_like(x)

    op.register_fake(_fake)
    _OPS[key] = op
    return op


def bind_weights_for_rs(layer) -> None:
    """Point the RS attributes at this layer's weights. No copy, no transpose.

    `process_weights_after_loading` has already run `_process_fp8_moe_weights`
    by the time this is called, and that does the whole conversion: it
    dequantizes the checkpoint, **requantizes** (per-channel by default --
    `MOE_REQUANTIZE_BLOCK_SIZE` is unset), and transposes into exactly the
    K-major layout the kernel's GMM wants. On GLM-5.2 it logs

        FP8 weights transposed for GMM kernel: w13=[16, 6144, 4096],
        w2=[16, 2048, 6144], w13_scale=[16, 1, 1, 4096],
        w2_scale=[16, 1, 1, 6144], requant_block_size=per-channel

    which is [E, hidden, 2*inter] / [E, inter, hidden] for the weights and
    [E, K_blocks, 1, N] for the scales -- the four-dimensional rank the
    kernel's scale DMA indexes, with K_blocks 1. So there is nothing to
    convert, and the previous transpose-and-repeat here was undoing work that
    had already been done correctly.

    Per-channel is the K_blocks == 1 case of the same contract, not a special
    one: `_recover_quant_block_size(size_k, 1)` short-circuits to `size_k`, so
    the kernel reads one quant block spanning all of K with a scale per N
    column, which is what per-channel means.
    """
    setattr(layer, RS_W1_ATTR, layer.w13_weight.data)
    setattr(layer, RS_W2_ATTR, layer.w2_weight.data)
    setattr(layer, RS_W1_SCALE_ATTR, layer.w13_weight_scale_inv.data)
    setattr(layer, RS_W2_SCALE_ATTR, layer.w2_weight_scale_inv.data)


def _weights_unsupported_reason(layer) -> str | None:
    """Why this layer's weights do not match the kernel's contract, or None.

    Checked against what `_process_fp8_moe_weights` produces, not against
    vLLM's on-disk storage layout -- nothing reaches this function without
    going through that conversion first.
    """
    for name in ("w13_weight", "w2_weight", "w13_weight_scale_inv",
                 "w2_weight_scale_inv"):
        if getattr(layer, name, None) is None:
            return f"the layer has no {name}; this path is fp8 only"

    for name, w, s in (("w13", layer.w13_weight, layer.w13_weight_scale_inv),
                       ("w2", layer.w2_weight, layer.w2_weight_scale_inv)):
        if w.ndim != 3:
            return f"{name}_weight {tuple(w.shape)} is not [E, K, N]"
        # Four dims with a unit axis 2 is what the kernel's scale DMA
        # indexes; three would raise `indices must not be longer than shape`
        # from inside Pallas on the first batch, which is a much worse place
        # to find out.
        if s.ndim != 4 or s.shape[2] != 1:
            return (f"{name} scale {tuple(s.shape)} is not the "
                    "[E, K_blocks, 1, N] this kernel's scale DMA indexes")
        size_k, size_n = w.shape[1], w.shape[2]
        if s.shape[0] != w.shape[0]:
            return (f"{name} scale has {s.shape[0]} experts against the "
                    f"weight's {w.shape[0]}")
        # The last axis is N in ELEMENTS, not in blocks -- one scale per
        # output column. A scale still expressed in N blocks has the right
        # rank and the wrong extent, and lands every scale on wrong columns.
        if s.shape[3] != size_n:
            return (f"{name} scale last axis is {s.shape[3]}, not N={size_n}; "
                    "the kernel indexes it in N elements, not N blocks")
        k_blocks = s.shape[1]
        if k_blocks == 0 or size_k % k_blocks != 0:
            return (f"{name} K={size_k} is not divisible by its {k_blocks} "
                    "scale blocks, so the kernel cannot recover the block size")
    return None


def fused_moe_rs_supported(owner: Any) -> bool:
    """Whether this quant method's layer was armed for the kernel."""
    return getattr(owner, FUSED_MOE_RS_OP_ATTR, None) is not None


def prebuild_fused_moe_rs(layer, topk: int, activation: str) -> Any | None:
    """Resolve the mesh and build the fused-RS op for one layer.

    Called once per layer from `process_weights_after_loading`. Returns the op
    to serve this layer with, or None on any refusal -- the caller records it,
    which is what makes arming per layer rather than per process.
    """
    if not envs.USE_MOE_FUSED_RS_KERNEL:
        return None
    # Both paths claim the same layer and the v2 arming runs first, so an
    # operator who set both would get v2 and a silent no-op here. Say so.
    if envs.USE_MOE_FUSED_EP_KERNEL:
        logger.warning_once(
            "Fused RS MoE not engaged: USE_MOE_FUSED_EP_KERNEL is also set "
            "and the v2 path arms first. Unset it to use the RS kernel.")
        return None

    # The experts have to actually be expert-parallel. vLLM builds an EP group
    # for every MoE model whether or not `--enable-expert-parallel` was passed,
    # so the mesh alone does not answer this: with the experts replicated each
    # rank holds the WHOLE set, and the kernel would read its shard as one
    # 1/ep block of a global set ep times too large.
    if not layer.moe_config.moe_parallel_config.use_ep:
        logger.info_once(
            "Fused RS MoE not engaged: the experts are replicated rather "
            "than expert-parallel.")
        return None

    tp = int(layer.moe_config.moe_parallel_config.tp_size)
    if tp > 1:
        # The kernel's mesh is the EP group alone. Under TP the token rows are
        # replicated across TP ranks and the output would need a TP reduction
        # the fused epilogue does not perform.
        logger.info_once(
            "Fused RS MoE not engaged: tp_size=%d, and this path's mesh is "
            "the EP group alone.", tp)
        return None

    mesh = build_ep_rs_mesh()
    if mesh is None:
        logger.info_once("Fused RS MoE requested but there is no EP group.")
        return None
    ep = mesh.shape[RS_EXPERT_AXIS_NAME]

    local_experts = layer.w13_weight.shape[0]
    # The kernel derives a token's owning chip from a uniform contiguous block
    # per shard (`group_offset` is an even arange). vLLM does not guarantee
    # that: `determine_expert_map` hands the first `global % ep` ranks one
    # extra expert, and EPLB gives a rank more still. Past the first uneven
    # boundary this would address the wrong shard, silently.
    if layer.global_num_experts != local_experts * ep:
        logger.warning_once(
            "Fused RS MoE not engaged: the kernel needs one equal contiguous "
            "expert block per shard, but %d global experts over ep=%d does "
            "not divide into this rank's %d.", layer.global_num_experts, ep,
            local_experts)
        return None

    reason = _weights_unsupported_reason(layer)
    if reason is not None:
        logger.warning_once("Fused RS MoE not engaged: %s", reason)
        return None

    # `size_m` is the dispatch height the kernel builds scalar-prefetch tables
    # over: every token the node presents, times top-k. The scheduler's cap is
    # the only place a token count is known outside a traced forward.
    max_node_tokens = _max_node_tokens(ep)
    size_m = max_node_tokens * topk
    if max_node_tokens and size_m >= _FUSED_RS_MAX_SAFE_SIZE_M:
        logger.warning_once(
            "Fused RS MoE not engaged: the largest node-wide batch is %d "
            "tokens, so size_m = %d x topk %d = %d would exceed the kernel's "
            "SMEM bound of %d. Reduce --max-num-batched-tokens or shard more.",
            max_node_tokens, max_node_tokens, topk, size_m,
            _FUSED_RS_MAX_SAFE_SIZE_M)
        return None

    # `w13_weight` is [E, hidden, 2*inter] by this point -- see
    # `bind_weights_for_rs` for who put it in that layout.
    hidden = layer.w13_weight.shape[1]
    global _RS_RANK_BUFFER
    _RS_RANK_BUFFER = torch.tensor([[ep_mesh_index()]],
                                   dtype=torch.int32,
                                   device=layer.w13_weight.device)
    bind_weights_for_rs(layer)
    op = _build_op(mesh, topk, activation)
    logger.info_once(
        "Fused RS MoE armed | hidden=%d local_experts=%d ep=%d topk=%d "
        "act=%s max_node_tokens=%d size_m=%d", hidden,
        local_experts, ep, topk, activation, max_node_tokens, size_m)
    return op


def _max_node_tokens(ep: int) -> int:
    """The largest node-wide token count a step can carry, or 0 if unknown.

    A DP scheduler cap is per independent batch, so every rank on the flattened
    EP mesh contributes another cap's worth of tokens.
    """
    from vllm.config import get_current_vllm_config

    try:
        cfg = get_current_vllm_config()
    except Exception:  # noqa: BLE001 - no config outside a served model
        return 0
    cap = cfg.scheduler_config.max_num_batched_tokens
    if not isinstance(cap, int):
        return 0
    return int(cap) * max(ep, 1)


def fused_moe_rs(
    owner: Any,
    layer: Any,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Run one MoE layer through the fused reduce-scatter kernel.

    Takes this rank's own tokens and returns this rank's own tokens: the
    all-gather and the reduce-scatter both happen inside the program, so the
    caller must do neither. `topk_ids` are GLOBAL expert ids -- the kernel
    derives each row's owning chip from them and there is no `experts_start`
    subtraction on this path.

    Only call this when `fused_moe_rs_supported(owner)` said yes.
    """
    op = getattr(owner, FUSED_MOE_RS_OP_ATTR)
    return op(
        hidden_states,
        getattr(layer, RS_W1_ATTR),
        getattr(layer, RS_W2_ATTR),
        getattr(layer, RS_W1_SCALE_ATTR),
        getattr(layer, RS_W2_SCALE_ATTR),
        topk_weights,
        topk_ids,
        _RS_RANK_BUFFER,
    )
