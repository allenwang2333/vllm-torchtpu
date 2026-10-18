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

import enum
import functools
from collections.abc import Callable
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    fused_moe_func_ep_combine,
    quantize_to_native_fp4_kmajor,
    requant_unpack_kmajor,
    resolve_onehot_permute_threshold,
    unpack_fp4_to_e2m1,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_kernel_instance_counter = 0
_fused_moe_kernel_cache: dict[tuple[int, str, bool, Any, bool], Callable] = {}
_load_kmajor_fp4_op = None
_requant_kmajor_fp4_ops: dict[int, Callable] = {}
_quantize_native_fp4_kmajor_ops: dict[tuple[int, bool], Callable] = {}


def _resolve_activation_name(activation: Any) -> str:
    if isinstance(activation, enum.Enum):
        return activation.value
    return activation.value if hasattr(activation, "value") else str(activation)


def get_fused_moe_activation(activation, moe_config) -> str:
    """Encode vLLM's activation config for the Pallas GMM kernel."""
    activation = _resolve_activation_name(activation)
    if activation != "situ":
        return activation

    beta = moe_config.activation_situ_beta
    linear_beta = moe_config.activation_situ_linear_beta
    assert beta is not None
    return f"situ:{beta}:{'none' if linear_beta is None else linear_beta}"


class TpuMoEActivationMixin:
    """Holds the Pallas GMM kernel's activation string for one MoE layer.

    Written by process_weights_after_loading, read by apply_monolithic.
    vLLM builds one quant method per MoE layer, so the method is the right
    owner. Defines no __init__, so it does not disturb the upstream
    __init__s these methods deliberately bypass; mix it in ahead of the
    upstream class.
    """

    _tpu_activation_str: str | None = None

    def _resolve_tpu_activation(self, layer) -> str:
        """Encode this layer's activation for the kernel.

        Overridden where a method needs a different encoding -- the string
        is not uniformly recomputable from layer.activation.
        """
        return get_fused_moe_activation(layer.activation, layer.moe_config)

    def _set_tpu_activation(self, layer) -> str:
        """Resolve and store; returns it, since callers also use it locally."""
        self._tpu_activation_str = self._resolve_tpu_activation(layer)
        return self._tpu_activation_str


def load_kmajor_fp4(w_u8: torch.Tensor) -> torch.Tensor:
    """One-time load transform: packed uint8 ``[..., N, K//2]`` -> native
    fp4 ``torch.float4_e2m1fn_x2`` ``[..., K, N]`` (K-major).

    Unpacks (bitcast uint8 -> float4_e2m1fn) and transposes the contracting axis
    to gmm_v2's K-major layout ONCE at load. The result is stored as a native-fp4
    Parameter, so the forward hands it straight to gmm_v2 with no per-forward
    unpack or transpose (which keeps the 480B HBM footprint bounded). Requires
    torch_tpu's native ``torch.float4_e2m1fn_x2`` dtype
    (google-pytorch/torch_tpu#1560).
    """
    global _load_kmajor_fp4_op
    if _load_kmajor_fp4_op is None:
        _load_kmajor_fp4_op = pallas.jax_op(
            "pallas::nvfp4_load_kmajor", unpack_fp4_to_e2m1
        )
    return _load_kmajor_fp4_op(w_u8)


def requant_load_kmajor_fp4(
    w_u8: torch.Tensor, scale_f: torch.Tensor, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """W4A8 requant + K-major load in one JAX op (matches tpu-inference). Takes
    the packed uint8 weight ``[..., N, K//2]`` + its fused fp32 block-16 scale
    ``[..., N, K//16]``; returns native-fp4 ``torch.float4_e2m1fn_x2`` ``[...,
    K, N]`` requantized to ``block`` plus the fp32 kernel scale ``[...,
    K//block, 1, N]``. Doing the dequant+requant in JAX (vs torch) avoids
    materializing the dequantized weight on the host and rounds to fp4 with the
    same cast the kernel reads."""
    op = _requant_kmajor_fp4_ops.get(block)
    if op is None:
        op = pallas.jax_op(
            f"pallas::nvfp4_requant_kmajor_b{block}",
            functools.partial(requant_unpack_kmajor, block=block),
        )
        _requant_kmajor_fp4_ops[block] = op
    return op(w_u8, scale_f)


def quantize_native_fp4_kmajor(
    w: torch.Tensor, block: int, pack: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an already-dequantized, K-major float weight to native FP4.

    ``pack=False`` hands the weight back as ``torch.float4_e2m1fn_x2`` rather
    than packed uint8 -- the form the fused EP MoE kernel takes. The two are
    distinct traced programs with distinct op names, so the cache is keyed on
    both: one qualname serving two closures would silently dispatch to
    whichever registered first.
    """
    key = (block, pack)
    op = _quantize_native_fp4_kmajor_ops.get(key)
    if op is None:
        op = pallas.jax_op(
            f"pallas::nvfp4_quantize_native_kmajor_b{block}"
            f"{'' if pack else '_unpacked'}",
            functools.partial(quantize_to_native_fp4_kmajor, block=block, pack=pack),
        )
        _quantize_native_fp4_kmajor_ops[key] = op
    return op(w)


def _allocate_kernel_instance_id() -> int:
    global _kernel_instance_counter
    kernel_instance_id = _kernel_instance_counter
    _kernel_instance_counter += 1
    return kernel_instance_id


def _build_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    skip_padded_tokens: bool,
    rhs_quant_dtype=None,
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

    # Pass experts_start as a runtime tensor rather than static closure arg so every EP
    # rank compiles an identical program.
    # See fa8faaf5 for background on rank-uniform sharding in compiled Pallas graphs.
    wrapped_fn = functools.partial(
        fused_moe_func,
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
    )

    fused_moe_kernel_impl = pallas.jax_op(op_name, wrapped_fn)

    # We must overwrite the default fake implementation as vLLM uses dynamic
    # dimensions for the hidden_states.
    def _fake_fused_moe(hidden_states, *args, **kwargs):
        return torch.empty_like(hidden_states)

    fused_moe_kernel_impl.register_fake(_fake_fused_moe)

    cache_key = (topk, activation, use_ep, rhs_quant_dtype, skip_padded_tokens)
    _fused_moe_kernel_cache[cache_key] = fused_moe_kernel_impl
    return fused_moe_kernel_impl


def _get_fused_moe_custom_op(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
):
    if skip_padded_tokens is None:
        skip_padded_tokens = envs.TPU_MOE_SKIP_PADDED_TOKENS
    cache_key = (topk, activation, use_ep, rhs_quant_dtype, skip_padded_tokens)
    kernel = _fused_moe_kernel_cache.get(cache_key)
    if kernel is not None:
        return kernel
    return _build_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
    )


def prebuild_fused_moe_kernel(
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
) -> None:
    """Prebuild and cache fused MoE custom op outside compile-time tracing."""
    _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
    )


def fused_moe_gmm(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor | None,
    topk: int,
    activation: str,
    rhs_quant_dtype=None,
    skip_padded_tokens: bool | None = None,
) -> torch.Tensor:
    """Fused MoE forward pass with precomputed routing.

    ``experts_start`` is the first global expert id owned by this shard under
    linear EP placement (or ``None`` for non-EP) -- see
    ``moe_routing.get_experts_start_buffer``. It must be a 0-d int32 tensor
    (a persistent buffer registered once per layer), not a Python int: every
    EP rank owns a different value, and passing it as real tensor data (vs.
    binding it as a Python int into the compiled closure) keeps the compiled
    program identical across ranks. The kernel remaps global ids to local ids
    with an elementwise subtract and masks non-local experts. When present,
    the sparse-core parity path dispatches ragged gather/gather-reduce.
    """
    use_ep = experts_start is not None
    fused_moe = _get_fused_moe_custom_op(
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
    )
    if experts_start is None:
        # The compiled program for use_ep=False never reads this operand
        # (see the `if use_ep:` guard in fused_moe_func), but the custom op
        # still needs a concrete tensor to call with a fixed arity.
        experts_start = torch.zeros((), dtype=torch.int32, device=hidden_states.device)
    return fused_moe(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        experts_start,
    )


# Serve DP-attention + EP layers with gmm_v2_ffn_ep, which sends each routed
# row to its token's owner, in place of the combine + reduce-scatter.
GMM_EP_COMBINE = True
GMM_EP_COMBINE_OP_ATTR = "_gmm_ep_combine_op"


def gmm_ep_combine_supported(owner: Any) -> bool:
    """Whether `prebuild_gmm_ep_combine` armed this layer's quant method."""
    return getattr(owner, GMM_EP_COMBINE_OP_ATTR, None) is not None


def _gmm_ep_combine_refusal(layer) -> str | None:
    from vllm.distributed.parallel_state import get_dp_group

    from vllm_torchtpu.distributed.dcp import get_dcp_world_size
    from vllm_torchtpu.distributed.ep_mesh import get_ep_group
    from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
        enable_pipelined_collective_and_compute,
    )

    parallel = layer.moe_config.moe_parallel_config
    if not GMM_EP_COMBINE:
        return "disabled"
    if not parallel.use_ep or parallel.dp_size <= 1:
        return "not DP attention with expert parallelism"
    if parallel.tp_size != 1 or parallel.pcp_size != 1 or get_dcp_world_size() != 1:
        return "TP, PCP or DCP is on"
    if parallel.is_sequence_parallel:
        return "MoE sequence parallelism is on"
    if enable_pipelined_collective_and_compute():
        return "the chunked collective pipeline owns the dispatch"
    if getattr(layer, "w13_bias", None) is not None:
        return "expert biases are not wired through"
    ep_group = get_ep_group()
    if ep_group is None or tuple(ep_group.ranks) != tuple(get_dp_group().ranks):
        # The op reads the DP-gathered token block of rank r as EP rank r's.
        return "the EP group is not the DP group in the same order"
    if layer.global_num_experts != layer.w13_weight.shape[0] * ep_group.world_size:
        return "experts do not split into equal contiguous blocks"
    return None


def prebuild_gmm_ep_combine(
    layer, *, topk: int, activation: str, rhs_quant_dtype=None
) -> Any | None:
    """The EP-combine op for this layer, or None to keep the combine in vLLM.

    Called from process_weights_after_loading on every rank; refusals depend
    only on the parallel config, so every rank arms the same layers (the
    kernel's barrier would otherwise wait forever).
    """
    from vllm_torchtpu.distributed.ep_mesh import (
        EP_AXIS_NAME,
        build_ep_mesh,
        ep_rank_order,
    )

    reason = _gmm_ep_combine_refusal(layer)
    mesh = build_ep_mesh() if reason is None else None
    if reason is None and mesh is None:
        reason = "there is no EP mesh"
    if reason is not None:
        logger.info_once("gmm_v2_ffn_ep combine not engaged: %s", reason)
        return None
    order = ep_rank_order()
    assert order is not None
    mesh_index_of_rank = tuple(order.index(r) for r in range(len(order)))
    key = ("ep_combine", id(mesh), topk, activation, rhs_quant_dtype)
    op = _fused_moe_kernel_cache.get(key)
    if op is not None:
        return op

    local = functools.partial(
        fused_moe_func_ep_combine,
        topk=topk,
        activation=activation,
        mesh_index_of_rank=mesh_index_of_rank,
        axis_name=EP_AXIS_NAME,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=envs.TPU_MOE_SKIP_PADDED_TOKENS,
    )
    spec = PartitionSpec(EP_AXIS_NAME)

    # torch_tpu infers static arguments from annotations: every operand is a
    # jax.Array, sharded on axis 0 per rank.
    def fn(
        hidden_states: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        experts_start: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(spec,) * 8,
            out_specs=spec,
            check_vma=False,
        )(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            experts_start,
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::fused_moe_ep_combine_{_allocate_kernel_instance_id()}",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 8,
        output_partition_specs=spec,
    )
    ep = len(mesh_index_of_rank)

    def _fake(hidden_states, *_args, **_kwargs):
        return hidden_states.new_empty(
            (hidden_states.shape[0] // ep, hidden_states.shape[1])
        )

    op.register_fake(_fake)
    _fused_moe_kernel_cache[key] = op
    logger.info_once(
        "gmm_v2_ffn_ep combine armed | ep=%d mesh_index_of_rank=%s",
        ep,
        mesh_index_of_rank,
    )
    return op


def gmm_ep_combine(
    owner: Any,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    experts_start: torch.Tensor,
) -> torch.Tensor:
    """This rank's tokens of the DP-gathered `hidden_states`, summed over all
    ranks' experts: the caller must not reduce-scatter the result."""
    op = getattr(owner, GMM_EP_COMBINE_OP_ATTR)
    return op(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        experts_start.reshape(1),
    )
