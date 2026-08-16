# SPDX-License-Identifier: Apache-2.0
"""Torch bridge for the fused expert-parallel MoE kernel (``USE_MOE_EP_KERNEL``).

Token ownership is the whole story here. The kernel wants the EP group's tokens
*sharded* one slice per rank, and returns each rank its own slice, already
combined across every expert. vLLM hands us tokens that are replicated across
the TP group (the MoE input comes out of a RowParallelLinear all-reduce), and
replicated across the DP group only if the DP all-gather ran -- which we
suppress (see ``VllmFp8MoEMethodTPU.supports_internal_mk``).

So each rank takes row slice ``tp_rank`` of ``tp_size`` from its DP-local batch,
runs the kernel, and scatters its result rows back into a zero-filled buffer of
the DP-local shape. vLLM's existing TP all-reduce then does double duty: it
sums the disjoint row slices *and* the TP-sharded shared-expert partials, at
exactly the size it already had. With ``tp_size == 1`` (the DP8_EP layout) none
of that runs: in, kernel, out.

Every rank-derived value crosses into the traced graph as tensor data, never as
a Python constant -- see ``moe_routing.py`` and fa8faaf5.

The kernel needs every rank to bring the *same* number of rows (a shard_map
shard is one shape). That holds because ``TPUModelRunner._dp_coordinated_step``
MAX-all-reduces the token count over the DP CPU group and pads to a shared
bucket before the forward -- the same invariant the DP all-gather relies on
today.
"""

from __future__ import annotations

import functools
import os

import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (EP_AXIS_NAME,
                                               get_or_create_ep_mesh,
                                               validate_ep_mesh)
from vllm_torchtpu.layers.common.fused_moe_ep import fused_ep_moe_func
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_op_cache: dict[tuple, object] = {}
_op_counter = 0

# tokens, gating, w1, w2, w1_scale, w2_scale -- all sharded on dim 0 (tokens by
# row, weights by expert).
_INPUT_PARTITION_SPECS = (PartitionSpec(EP_AXIS_NAME), ) * 6


def _replica_group() -> tuple[int, int]:
    """(size, rank) of the group across which MoE input tokens are replicated.

    Under expert parallelism vLLM sets the MoE's own ``tp_size`` to 1 and folds
    TP into the EP axis, so the group that replicates tokens is the model's TP
    group, not ``moe_config.tp_size``.
    """
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
    return get_tensor_model_parallel_world_size(
    ), get_tensor_model_parallel_rank()


def unsupported_reason(layer) -> str | None:
    """Why the fused EP kernel cannot serve ``layer``, or None if it can."""
    moe = layer.moe_config
    if not moe.moe_parallel_config.use_ep:
        return "expert parallelism is off"
    if moe.ep_size <= 1:
        return f"ep_size={moe.ep_size}"
    if getattr(moe, "is_sequence_parallel", False):
        return ("sequence parallelism reduces the MoE output itself, leaving "
                "no all-reduce to reassemble the row slices")
    if getattr(moe, "pcp_size", 1) > 1:
        return "PCP adds its own MoE dispatch/combine pair"
    if getattr(layer, "use_grouped_topk", False):
        return "grouped top-k routing is not implemented in the kernel"
    if getattr(layer, "hash_indices_table", None) is not None:
        return "hash routing bypasses the gate the kernel reads"
    if os.getenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY"):
        # The kernel routes internally, so the simulation override would be
        # silently ignored rather than applied.
        return "VLLM_MOE_ROUTING_SIMULATION_STRATEGY is set"
    if getattr(layer, "e_score_correction_bias", None) is not None:
        return "bias-corrected expert selection is not implemented in the kernel"
    if float(getattr(layer, "routed_scaling_factor", 1.0)) != 1.0:
        return "routed_scaling_factor != 1 is not implemented in the kernel"
    scoring = getattr(layer, "scoring_func", "softmax")
    if scoring not in ("softmax", "sigmoid"):
        return f"scoring_func={scoring!r}"
    activation = getattr(layer, "_tpu_activation_str", "silu")
    if activation not in ("silu", "gelu", "swigluoai"):
        return f"activation={activation!r}"
    if getattr(layer, "has_bias", False):
        return "expert biases are not wired through this path"
    replica_size, _ = _replica_group()
    dp_size = int(getattr(moe, "dp_size", 1))
    if dp_size * replica_size != moe.ep_size:
        return (f"ep_size={moe.ep_size} is not dp_size={dp_size} x "
                f"tp_size={replica_size}; token ownership is undefined")
    if moe.num_experts % moe.ep_size:
        return (f"num_experts={moe.num_experts} is not divisible by "
                f"ep_size={moe.ep_size}")
    return None


def is_enabled(layer) -> bool:
    """Read the flag once per layer, and log why it did or did not engage.

    Engagement is per layer, not per step -- ``USE_MOE_EP_KERNEL`` plus a layer
    the kernel can serve, the same condition as tpu-inference's
    ``select_moe_backend_from_fused_moe_config``. A layer that answers True runs
    the kernel on *every* step, which is what entitles it to claim
    ``supports_internal_mk`` and own the EP dispatch/combine outright.
    """
    if not envs.USE_MOE_EP_KERNEL:
        return False
    reason = unsupported_reason(layer)
    if reason is not None:
        logger.warning_once(
            "USE_MOE_EP_KERNEL=1 but the fused EP MoE kernel cannot serve this "
            "layer (%s); falling back to the GMM path.", reason)
        return False
    validate_ep_mesh(layer)
    # Warn, not info: this is slower than what it replaces. It has to be here,
    # at weight-load time -- Dynamo refuses a logger call inside a captured
    # graph ("logging.Logger method not supported for non-export cases").
    logger.warning_once(
        "USE_MOE_EP_KERNEL=1: the fused EP MoE kernel is enabled for this "
        "layer (ep_size=%d, top_k=%d). On Qwen3.5-397B geometry (H=4096, "
        "F=1024, E=512, top_k=10, EP=8, v7x) it measures 0.19x-0.05x of the "
        "production GMM path per MoE layer and -86%% end to end, because it "
        "takes decode steps too; those numbers are ours, at that shape. It is "
        "provided for parity with tpu-inference, not for serving. Unset the "
        "flag to use the GMM path.", layer.moe_config.ep_size,
        layer.moe_config.experts_per_token)
    return True


def _get_op(*, top_k: int, renormalize: bool, activation: str, scoring_fn: str,
            subc_quant_w1_sz: int | None, subc_quant_w2_sz: int | None):
    key = (top_k, renormalize, activation, scoring_fn, subc_quant_w1_sz,
           subc_quant_w2_sz)
    op = _op_cache.get(key)
    if op is not None:
        return op

    global _op_counter
    name = f"pallas::fused_ep_moe_kernel_{_op_counter}"
    _op_counter += 1
    mesh = get_or_create_ep_mesh()
    fn = functools.partial(
        fused_ep_moe_func,
        mesh=mesh,
        ep_axis_name=EP_AXIS_NAME,
        top_k=top_k,
        renormalize=renormalize,
        activation=activation,
        scoring_fn=scoring_fn,
        subc_quant_w1_sz=subc_quant_w1_sz,
        subc_quant_w2_sz=subc_quant_w2_sz,
    )
    op = pallas.jax_op(name,
                       fn,
                       mesh=mesh,
                       input_partition_specs=_INPUT_PARTITION_SPECS)

    # The default fake kernel re-exports to learn the output shape and refuses
    # symbolic dimensions; the output is always the token operand's shape.
    op.register_fake(lambda tokens, *_args: torch.empty_like(tokens))
    _op_cache[key] = op
    return op


def prebuild_fused_ep_moe_kernel(**kwargs) -> None:
    """Register the custom op outside compile-time tracing."""
    _get_op(**kwargs)


def register_replica_rank_buffer(layer, *, device: torch.device) -> None:
    """Register ``layer._ep_replica_rank`` as a persistent 0-d int32 buffer.

    Same reasoning as ``moe_routing.register_experts_start_buffer``: this rank's
    index within the token-replica (TP) group decides which rows it owns, and it
    must reach the compiled graph as module state, not as a Python int folded
    into a constant.
    """
    replica_size, replica_rank = _replica_group()
    buffer = (torch.tensor(replica_rank, dtype=torch.int32, device=device)
              if replica_size > 1 else None)
    layer.register_buffer("_ep_replica_rank", buffer, persistent=False)


# Smallest per-rank row count the kernel is happy with: its default block sizes
# derive `btc = min(local_rows // 2, 64)`, which must stay >= 1, and 8 keeps the
# rows on the sublane granularity. AOT-gated at 8 in
# tests/kernels/fused_moe/test_fused_ep_aot.py.
_MIN_LOCAL_ROWS = 8


def _padded_rows(num_rows: int, replica_size: int) -> int:
    """Row count to pad the DP-local batch to before slicing it per replica."""
    per_replica = -(-num_rows // replica_size)
    per_replica = max(_MIN_LOCAL_ROWS,
                      -(-per_replica // _MIN_LOCAL_ROWS) * _MIN_LOCAL_ROWS)
    return per_replica * replica_size


def _row_slice_index(num_rows: int, replica_size: int,
                     replica_rank: torch.Tensor) -> torch.Tensor:
    rows_per_replica = num_rows // replica_size
    return replica_rank * rows_per_replica + torch.arange(
        rows_per_replica, dtype=torch.int32, device=replica_rank.device)


def _scatter_replica_rows(local_rows: torch.Tensor, replica_size: int,
                          replica_rank: torch.Tensor) -> torch.Tensor:
    """``[S, H]`` -> ``[replica_size * S, H]``, zero outside this rank's block.

    A masked broadcast rather than a scatter: ``index_copy_`` demands an int64
    index, and ``narrow`` with a tensor start calls ``.item()`` (a lazy-tensor
    sync, and it would fold the rank back into a constant).
    """
    mask = (torch.arange(replica_size,
                         dtype=torch.int32,
                         device=local_rows.device) == replica_rank).to(
                             local_rows.dtype)
    blocks = local_rows.unsqueeze(0) * mask.view(replica_size, 1, 1)
    return blocks.reshape(-1, local_rows.shape[-1])


def fused_moe_ep(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    *,
    top_k: int,
    renormalize: bool,
    activation: str,
    scoring_fn: str,
    subc_quant_w1_sz: int | None,
    subc_quant_w2_sz: int | None,
    replica_rank: torch.Tensor | None,
) -> torch.Tensor:
    """Run the fused EP MoE for this rank's DP-local batch.

    ``hidden_states`` / ``router_logits`` are the DP-local rows; the returned
    tensor has the same row count and is zero outside this rank's slice when
    ``tp_size > 1`` (vLLM's TP all-reduce reassembles it).
    """
    op = _get_op(top_k=top_k,
                 renormalize=renormalize,
                 activation=activation,
                 scoring_fn=scoring_fn,
                 subc_quant_w1_sz=subc_quant_w1_sz,
                 subc_quant_w2_sz=subc_quant_w2_sz)

    replica_size, _ = _replica_group()
    num_rows = hidden_states.shape[0]
    pad_rows = _padded_rows(num_rows, replica_size) - num_rows
    if pad_rows:
        hidden_states = torch.nn.functional.pad(hidden_states,
                                                (0, 0, 0, pad_rows))
        router_logits = torch.nn.functional.pad(router_logits,
                                                (0, 0, 0, pad_rows))

    if replica_size > 1:
        assert replica_rank is not None, (
            "register_replica_rank_buffer() was not called for this layer")
        index = _row_slice_index(num_rows + pad_rows, replica_size,
                                 replica_rank)
        local_states = torch.index_select(hidden_states, 0, index)
        local_logits = torch.index_select(router_logits, 0, index)
    else:
        local_states, local_logits = hidden_states, router_logits

    local_out = op(local_states, local_logits, w1, w2, w1_scale, w2_scale)

    if replica_size > 1:
        out = _scatter_replica_rows(local_out, replica_size, replica_rank)
    else:
        out = local_out
    return out[:num_rows] if pad_rows else out
