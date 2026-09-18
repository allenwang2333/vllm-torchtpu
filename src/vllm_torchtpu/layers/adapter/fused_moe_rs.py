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

`kernels/fused_moe_rs` is a verbatim copy of
`tokamax/_src/ops/experimental/fused_moe_rs`. It runs the whole EP MoE --
gather, GMM1, activation, GMM2, and the cross-chip exchange -- as one Pallas
program, so the reduce-scatter is the kernel's epilogue rather than an exposed
collective after it. That is the same argument `fused_moe_ep` makes for the v2
kernel, and the two present the identical torch-side contract: this rank's
tokens in, this rank's tokens out, no caller-side all-gather or reduce-scatter.
Sharing that contract is deliberate -- it lets both reuse every upstream
collective-ownership patch in `vllm_torchtpu/__init__.py` unchanged.

**Why this exists next to `fused_moe_ep` rather than inside it.** The v2 kernel
takes one fp8 scale per output channel; `_squeeze_channel_scale` is the
conversion, and it refuses anything with more than one block along K. This
kernel's GMM reads BLOCK-scaled weights natively, recovering the block size
from the scale's own shape. GLM-5.2-FP8 is block-scaled [128, 128], so it is a
model the v2 path structurally cannot serve and this one can.

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

**Weight layout.** The kernel's GMM is K-major: `w1` is [E, hidden, 2*inter]
and `w2` is [E, inter, hidden], where vLLM stores [E, 2*inter, hidden] and
[E, hidden, inter]. `relayout_weights_for_rs` transposes both, and their block
scales with them, once at weight load. Doing it per forward would be a
45 GB relayout per step on GLM-5.2.
"""

from typing import Any

import jax
import jax.numpy as jnp
import torch

from jax.sharding import PartitionSpec

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import (RS_DATA_AXIS_NAME,
                                               RS_EXPERT_AXIS_NAME,
                                               build_ep_rs_mesh)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

FUSED_MOE_RS_OP_NAME = "pallas::fused_moe_rs"

# The attribute `prebuild_fused_moe_rs`'s result is recorded under, on the
# quant method that owns the layer -- per-layer state without a registry,
# because vLLM builds one quant method per MoE layer. Same device as
# `FUSED_MOE_EP_OP_ATTR` serves for the v2 path.
FUSED_MOE_RS_OP_ATTR = "_fused_moe_rs_op"

# Where `relayout_weights_for_rs` parks the K-major copies.
RS_W1_ATTR = "_rs_w13_weight"
RS_W2_ATTR = "_rs_w2_weight"
RS_W1_SCALE_ATTR = "_rs_w13_weight_scale"
RS_W2_SCALE_ATTR = "_rs_w2_weight_scale"

# `fused_moe_rs.py:_assert_fused_rs_smem_safe` trips at this dispatch height:
# the kernel's per-row scalar-prefetch arrays are 4 * size_m bytes each and
# stop fitting SMEM. Mirrored here so the refusal is a log line at weight load
# rather than an assert inside a traced forward on the first real batch.
_FUSED_RS_MAX_SAFE_SIZE_M = 262144

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
            topk_indices: jax.Array) -> jax.Array:
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
        input_partition_specs=(spec, ) * 7,
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


def _transpose_last_two(t: torch.Tensor) -> torch.Tensor:
    """A contiguous [.., A, B] -> [.., B, A] copy.

    `.contiguous()` is not optional: the bridge hands the tensor's buffer to
    JAX, so a transposed view would be read back in the original order and
    every expert would compute against the wrong weight block.
    """
    return t.transpose(-2, -1).contiguous()


def relayout_weights_for_rs(layer) -> None:
    """Stage K-major copies of this layer's expert weights and block scales.

    vLLM stores `w13_weight` as [E, 2*inter, hidden] and `w2_weight` as
    [E, hidden, inter]; the kernel's GMM wants [E, hidden, 2*inter] and
    [E, inter, hidden]. The block scales transpose with them, which is what
    keeps `_recover_quant_block_size(w.shape[1], scale.shape[1])` -- how the
    kernel learns its block size, from the scale's shape alone -- recovering
    128 rather than a silently wrong stride.

    The originals are left in place. They are what the grouped-matmul fallback
    reads, and a layer this path refuses still has to run.
    """
    w1 = _transpose_last_two(layer.w13_weight.data)
    w2 = _transpose_last_two(layer.w2_weight.data)
    w1_scale = _transpose_last_two(layer.w13_weight_scale_inv.data)
    w2_scale = _transpose_last_two(layer.w2_weight_scale_inv.data)
    setattr(layer, RS_W1_ATTR, w1)
    setattr(layer, RS_W2_ATTR, w2)
    setattr(layer, RS_W1_SCALE_ATTR, w1_scale)
    setattr(layer, RS_W2_SCALE_ATTR, w2_scale)


def _relayout_unsupported_reason(layer) -> str | None:
    """Why this layer's weights cannot be relaid out for the kernel, or None."""
    for name in ("w13_weight", "w2_weight", "w13_weight_scale_inv",
                 "w2_weight_scale_inv"):
        if getattr(layer, name, None) is None:
            return f"the layer has no {name}; this path is fp8-block-scaled only"
    w1 = layer.w13_weight
    scale = layer.w13_weight_scale_inv
    if w1.ndim != 3 or scale.ndim != 3:
        return (f"w13_weight {tuple(w1.shape)} / scale {tuple(scale.shape)} "
                "are not the [E, N, K] and [E, N_blocks, K_blocks] this path "
                "transposes")
    # The kernel recovers its block size by dividing, so a K that the scale's
    # block count does not evenly cover rounds to a power of two that is not
    # the checkpoint's block size, and every scale lands on the wrong columns.
    size_k, k_blocks = w1.shape[-1], scale.shape[-1]
    if k_blocks == 0 or size_k % k_blocks != 0:
        return (f"w13 K={size_k} is not divisible by its {k_blocks} scale "
                "blocks, so the kernel cannot recover the block size")
    recovered = 1 << ((-(-size_k // k_blocks) - 1).bit_length())
    if recovered != size_k // k_blocks:
        return (f"w13 block size {size_k // k_blocks} is not a power of two; "
                f"the kernel would recover {recovered}")
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

    reason = _relayout_unsupported_reason(layer)
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

    relayout_weights_for_rs(layer)
    op = _build_op(mesh, topk, activation)
    logger.info_once(
        "Fused RS MoE armed | hidden=%d local_experts=%d ep=%d topk=%d "
        "act=%s max_node_tokens=%d size_m=%d", layer.w13_weight.shape[-1],
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
    )
