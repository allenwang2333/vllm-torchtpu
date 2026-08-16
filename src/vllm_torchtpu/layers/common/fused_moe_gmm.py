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

import functools

import jax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.megablox.gmm_v2 import get_packing_factor, gmm_v2
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import \
    ragged_gather_reduce as ragged_gather_reduce_v1
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import \
    ragged_gather_reduce_v2
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import \
    ragged_gather_reduce as ragged_gather_reduce_v3
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2


def _select_ragged_gather_reduce(version: str):
    return {
        "v1": ragged_gather_reduce_v1,
        "v2": ragged_gather_reduce_v2,
        "v3": ragged_gather_reduce_v3,
    }[version]


# The server sets its environment before importing model code, so the selected
# implementation remains fixed for the process lifetime. The default is v2;
# set RAGGED_GATHER_REDUCE_VERSION=v3 to enable the destination-major kernel.
ragged_gather_reduce = _select_ragged_gather_reduce(
    envs.RAGGED_GATHER_REDUCE_VERSION)

# One-hot permute/combine cost grows O(routed_rows * num_tokens * hidden), so
# auto mode never enables it above this many routed rows, even where the
# SparseCore block is larger. TODO(xutingl): raise the cap after benchmarks.
_ONEHOT_AUTO_CAP = 64


def resolve_onehot_permute_threshold() -> int:
    """Routed-row count at/below which MoE permute+combine run as TensorCore
    one-hot matmuls instead of the SparseCore gather/reduce kernels.

    Below one SparseCore block ``ragged_gather_v2`` pads to a whole block and
    does the same SparseCore work regardless of occupancy, so auto routes such
    small batches to the one-hot path, capped at ``_ONEHOT_AUTO_CAP``.

    ``ONEHOT_MOE_PERMUTE_THRESHOLD`` overrides auto: 0 disables the one-hot
    path (kill switch), a positive value forces that threshold, and
    unset/empty/negative selects auto. Auto resolves to 0 on hosts without a
    SparseCore, leaving those paths unchanged.
    """
    explicit = envs.ONEHOT_MOE_PERMUTE_THRESHOLD
    if explicit is not None and explicit >= 0:
        return explicit
    try:
        sc_info = pltpu.get_tpu_info().sparse_core
    except ValueError:
        # get_tpu_info raises for unsupported device kinds (e.g. CPU hosts).
        return 0
    if sc_info is None:
        return 0
    block_rows = sc_info.num_lanes * sc_info.num_cores * sc_info.num_subcores
    threshold = min(block_rows - 1, _ONEHOT_AUTO_CAP)

    return threshold


def routed_buffer_capacity(
    num_tokens: int,
    topk: int,
    ep_size: int,
    *,
    alpha: float,
    beta: float = 7.0,
    c0: int = 16,
    quantum: int = 2048,
) -> int:
    """Rows to reserve in the per-rank routed buffer.

    Only ``num_tokens * topk / ep_size`` of the ``num_tokens * topk`` routed
    (token, expert) rows belong to this EP rank. Sizing the buffer to a
    capacity instead of the global count shrinks the permute, the GMM ``m``
    dimension and the XLA temp by ``ep_size``, at the cost of dropping rows if
    the rank's actual load exceeds the capacity.

    The slack is multiplicative *and* additive: the relative fluctuation of the
    rank total scales as ``1/sqrt(mean)``, so a constant ``alpha`` is wrong --
    the alpha needed for a 1e-12 drop bound runs from 2.15 at 32 tokens to
    1.046 at 16384. ``beta``/``c0`` price the sampling noise exactly; ``alpha``
    is the margin for everything the iid model gets wrong, and at the 16384
    bucket it buys almost exactly its face value: ``alpha - 1`` of systematic
    per-rank load skew, or correlated routing blocks of ~``((alpha-1)*mean /
    (7.3*sd))^2`` tokens. Tail derivation and the tolerance tables:
    ``$AMPLIO_ARTIFACT_DIR/p6/TAIL.md``.

    ``quantum`` is 2048 because that is the SparseCore gather's block, not
    because of the GMM: ``ragged_gather_v2`` rounds its output rows up to
    ``num_lanes * num_cores * num_subcores * max_row_subchunks`` (2048 on
    tpu7x) and slices back, so a capacity that is not a whole number of blocks
    materialises a full extra copy of the routed buffer -- 344 MB/layer at
    C=21504. 2048 is also a whole number of GMM m-tiles (512) and of lhs
    sublanes (16), so no consumer pads.
    """
    routed_rows = num_tokens * topk
    if ep_size <= 1:
        return routed_rows
    mean = routed_rows / ep_size
    target = alpha * mean + beta * (mean**0.5) + c0
    capacity = int(-(-target // quantum)) * quantum
    return min(capacity, routed_rows)


def unpack_fp4_to_e2m1(w_packed: jax.Array) -> jax.Array:
    """Unpack a uint8-packed E2M1 weight (2 fp4 per byte along the last axis)
    into ``float4_e2m1fn`` and move the contracting axis into GMM layout. Input
    is the checkpoint layout ``[..., N, K/2]`` (output-major, packed contracting
    dim last); output is ``[..., K, N]`` to match the ``[size_group, size_k,
    size_n]`` rhs gmm_v2 expects. Run this once at weight load so the forward
    pass hands native fp4 straight to the kernel with no per-forward unpack."""
    fp4 = jax.lax.bitcast_convert_type(w_packed, jnp.float4_e2m1fn)
    fp4 = fp4.reshape(*w_packed.shape[:-1], -1)  # [..., N, K]
    return jnp.swapaxes(fp4, -1, -2)  # [..., K, N]


def requant_unpack_kmajor(w_packed: jax.Array, scale_f: jax.Array,
                          block: int) -> tuple[jax.Array, jax.Array]:
    """W4A8 requantization in JAX: unpack the checkpoint block-16 fp4 weight,
    dequantize with its fused scale, requantize to ``block`` fp4, and lay out
    K-major for gmm_v2. Input is checkpoint layout ``[..., N, K/2]`` packed
    uint8 with fused fp32 block-16 ``scale_f`` ``[..., N, K/16]``; output is
    native fp4 ``[..., K, N]`` plus the fp32 kernel scale ``[..., K/block, 1,
    N]``. Requantizing in JAX (not torch) keeps the dequantized weight off the
    host and rounds to fp4 with the cast the kernel expects."""
    fp4 = jax.lax.bitcast_convert_type(w_packed, jnp.float4_e2m1fn)
    fp4 = fp4.reshape(*w_packed.shape[:-1], -1)  # [..., N, K]
    size_k = fp4.shape[-1]
    num_in_blocks = scale_f.shape[-1]
    dequant = (fp4.astype(jnp.float32).reshape(*fp4.shape[:-1], num_in_blocks,
                                               size_k // num_in_blocks) *
               scale_f[..., None]).reshape(*fp4.shape[:-1], size_k)
    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
    blocked = dequant.reshape(*dequant.shape[:-1], size_k // block, block)
    scale = jnp.max(jnp.abs(blocked), axis=-1, keepdims=True) * (1.0 / fp4_max)
    scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    requant = jnp.clip(blocked * scale_inv, -fp4_max,
                       fp4_max).astype(jnp.float4_e2m1fn)
    requant = requant.reshape(*dequant.shape[:-1], size_k)  # [..., N, K]
    scale = scale.squeeze(-1).astype(jnp.float32)  # [..., N, K/block]
    return (jnp.swapaxes(requant, -1,
                         -2), jnp.expand_dims(jnp.swapaxes(scale, -1, -2), -2))


def quantize_to_native_fp4_kmajor(w: jax.Array,
                                  block: int) -> tuple[jax.Array, jax.Array]:
    """Quantize K-major float weight ([..., K, N]) to packed uint8 e2m1 and per-block FP32 scale.
    Returns uint8 to safely cross the PyTorch/JAX bridge before unpacking in gmm_v2.

    TODO: Make a more generic version of this function; possibly combine with quantize_tensor_to_fp4
    """

    size_k = w.shape[-2]
    num_blocks = size_k // block
    blocked = w.astype(jnp.float32).reshape(*w.shape[:-2], num_blocks, block,
                                            w.shape[-1])
    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
    abs_max = jnp.max(jnp.abs(blocked), axis=-2, keepdims=True)
    scale = abs_max / fp4_max
    scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    quantized = jnp.clip(blocked * scale_inv, -fp4_max,
                         fp4_max).astype(jnp.float4_e2m1fn)
    quantized = quantized.reshape(*w.shape[:-2], size_k,
                                  w.shape[-1])  # [..., K, N]

    # Real e2m1 bit-packing (2 values/byte along K), matching gmm_v2's own
    # should_unpack/bitcast(quant_dtype) unpacking convention.
    pairs = quantized.reshape(*w.shape[:-2], size_k // 2, 2, w.shape[-1])
    pairs = jnp.swapaxes(pairs, -1, -2)  # [..., K/2, N, 2] -- pair axis last
    packed = jax.lax.bitcast_convert_type(pairs, jnp.uint8)  # [..., K/2, N]

    scale = scale.astype(jnp.float32)  # [..., num_blocks, 1, N]
    return packed, scale


def gmm_wrapper(lhs,
                rhs,
                rhs_scale,
                rhs_bias,
                group_sizes,
                group_offset,
                zero_initialize=False,
                fuse_act=None,
                preferred_element_type=None,
                rhs_quant_dtype=None):
    # fp4 weights: keep bf16 activations. Quantizing activations to fp8 collapses
    # fp4 accuracy (error compounds across MoE layers) with no decode speedup
    # (decode is weight-HBM-bound). fp8/int4 weights keep the default fp8 act.
    is_fp4_weight = (jnp.issubdtype(rhs.dtype, jnp.floating)
                     and jax.dtypes.itemsize_bits(rhs.dtype) == 4)
    return gmm_v2(
        lhs=lhs,
        rhs=rhs,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_sizes=group_sizes,
        group_offset=group_offset[0],
        zero_initialize=zero_initialize,
        fuse_act=fuse_act,
        preferred_element_type=preferred_element_type,
        maybe_quantize_lhs=not is_fp4_weight,
        rhs_quant_dtype=rhs_quant_dtype,
    )


def prepare_routed_gmm_inputs(
    hidden_states_local: jax.Array,
    topk_indices_local: jax.Array,
    topk_weights_local: jax.Array,
    *,
    local_num_experts: int,
    topk: int,
    use_ep: bool,
    use_sparse_core: bool,
    onehot_moe_permute_threshold: int = 0,
    skip_padded_tokens: bool = False,
    routed_capacity: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare the local-only routing layout using already-local expert ids.

    For non-EP runs `topk_indices_local` is always non-negative, so the
    valid-mask path computed below is a no-op (jnp.where on an all-True
    mask). For EP, the caller applies the expert-map remap upstream
    (inside this JIT) and marks non-local entries with a negative id which
    the mask filters out. Always running the masked branch keeps a single
    code path.
    """
    num_tokens_local = hidden_states_local.shape[0]
    topk_indices_flat = topk_indices_local.flatten()
    topk_weights_flat = topk_weights_local.flatten()
    token_indices_flat = jnp.arange(num_tokens_local,
                                    dtype=jnp.int32).repeat(topk)

    if skip_padded_tokens:
        # Padded tokens keep their selected expert ids, but carry zero
        # expert routing weights. Treat them as invalid to remove them
        # from GMM work.
        valid_mask = (topk_indices_flat >= 0) & (topk_weights_flat != 0)
    else:
        valid_mask = topk_indices_flat >= 0
    sort_keys = jnp.where(valid_mask, topk_indices_flat, local_num_experts)
    sorted_indices = jnp.argsort(sort_keys)
    argsort_revert_indices = jnp.argsort(sorted_indices)

    compacted = (routed_capacity is not None
                 and routed_capacity < sort_keys.size)
    if compacted:
        # Valid rows sort ahead of invalid ones, so a valid row's destination
        # `argsort_revert_indices` is already its index in the compacted
        # buffer; rows past the capacity are the overflow and are dropped by
        # clearing their mask bit. The sentinel keeps every index in range for
        # the bounds-check-free SparseCore gathers.
        kept_mask = valid_mask & (argsort_revert_indices < routed_capacity)
        argsort_revert_indices = jnp.where(kept_mask, argsort_revert_indices,
                                           0)
        valid_mask = kept_mask
        sorted_indices = sorted_indices[:routed_capacity]

    token_indices_sorted = token_indices_flat[sorted_indices]
    # Name both arms, so an op table distinguishes "the control ran the control"
    # from "the control silently ran the treatment".
    with jax.named_scope("moe_routed_capacity" if compacted else
                         "moe_routed_full"):
        topk_indices_for_count = jnp.where(valid_mask, topk_indices_flat, 0)
        group_sizes_local = (jax.nn.one_hot(
            topk_indices_for_count, local_num_experts, dtype=jnp.int32) *
                             valid_mask[:, None].astype(jnp.int32))
        group_sizes_local = group_sizes_local.sum(axis=0)

        if compacted:
            # INVARIANT: group_sizes_local must sum to the number of rows
            # actually present in the routed buffer. gmm_v2 walks m offsets
            # from it with no clamp to size_m and no bounds checks
            # (megablox/gmm_v2.py:725, :1388, :1450), so an unclamped histogram
            # runs the grouped matmul past both its row buffers and its SMEM
            # metadata array. Clamp here, after every branch that can produce
            # the histogram, not inside one of them.
            #
            # The sort is stable and keyed by expert, so group `e` owns the
            # contiguous destination slots [offset[e], offset[e] + n[e]); the
            # clipped difference is exactly |that range intersect [0,
            # capacity)|, i.e. the same count `kept_mask` above leaves behind.
            offsets = jnp.cumsum(group_sizes_local) - group_sizes_local
            group_sizes_local = (
                jnp.clip(offsets + group_sizes_local, 0, routed_capacity) -
                jnp.clip(offsets, 0, routed_capacity))

    if use_ep and use_sparse_core:
        # The threshold selects the *algorithm*, so it is compared against the
        # uncompacted routed-row count: a capacity must change the size of the
        # buffer, never which kernel runs. `sort_keys.size` is
        # `num_tokens_local * topk` whether or not a capacity is in force, so
        # this is byte-identical with the capacity off. (Whether the threshold
        # should instead see the compacted count is a live question, but it is
        # only answerable once the threshold's own value is re-derived -- the
        # shipped recipe sets 32768 routed rows, which is 3-6x past the
        # measured crossover.)
        if sort_keys.size <= onehot_moe_permute_threshold:
            # Use one-hot matmul for permutation, which can be faster
            # for small batch size
            onehot = jax.nn.one_hot(token_indices_sorted,
                                    num_tokens_local,
                                    dtype=hidden_states_local.dtype)
            x = onehot @ hidden_states_local
        else:
            # Match uLLM/reference EP routing: materialize the valid local
            # expert prefix through SparseCore ragged_gather. Invalid
            # non-local rows are sorted after the valid prefix.
            valid_count = group_sizes_local.sum(dtype=jnp.int32)
            x = ragged_gather_v2(
                hidden_states_local,
                token_indices_sorted,
                jnp.array([0], dtype=jnp.int32),
                valid_count[None],
            )
    else:
        x = hidden_states_local[token_indices_sorted]
    return (x, group_sizes_local, argsort_revert_indices, topk_weights_flat,
            valid_mask)


def moe_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    argsort_revert_indices: jax.Array,
    topk_weights_flat: jax.Array,
    valid_mask_flat: jax.Array,
    *,
    activation: str,
    num_tokens: int,
    topk: int,
    use_ep: bool,
    use_sparse_core: bool,
    onehot_moe_permute_threshold: int = 0,
    rhs_quant_dtype: jnp.dtype | None = None,
    skip_padded_tokens: bool = False,
) -> jax.Array:
    """Run grouped GEMM for routed tokens and reduce back to tokens.

    The valid-mask gating below is a no-op for non-EP runs (mask is all
    True) and keeps the single code path uniform for both EP and non-EP.
    EP uses SparseCore ragged_gather + ragged_gather_reduce for token movement.
    """
    group_offset = jnp.array([0], dtype=jnp.int32)

    gmm1_res = gmm_wrapper(
        x,
        w1,
        w1_scale,
        w1_bias,
        group_sizes,
        group_offset,
        zero_initialize=False,
        fuse_act=activation,
        rhs_quant_dtype=rhs_quant_dtype,
    )
    packing_factor = get_packing_factor(w2.dtype, rhs_quant_dtype)
    gmm1_res = gmm1_res[:, :w2.shape[1] * packing_factor]

    topk_weights = topk_weights_flat.reshape((num_tokens, topk))
    valid_mask = valid_mask_flat.reshape((num_tokens, topk))

    gmm2_res = gmm_wrapper(gmm1_res,
                           w2,
                           w2_scale,
                           w2_bias,
                           group_sizes,
                           group_offset,
                           zero_initialize=False,
                           preferred_element_type=x.dtype,
                           rhs_quant_dtype=rhs_quant_dtype)
    if use_ep and use_sparse_core:
        # Selection is on the uncompacted routed-row count -- see the matching
        # comment in `prepare_routed_gmm_inputs`. `argsort_revert_indices` is
        # destination-major and is never truncated by a capacity, so its size
        # is `num_tokens * topk` in every configuration.
        if argsort_revert_indices.size <= onehot_moe_permute_threshold:
            # Use onehot + matmul for unpermutation, which can be faster
            # for small batch size.
            if skip_padded_tokens:
                # Zero out the GMM output rows of padded tokens that
                # skipped the GMM computation, otherwise those
                # uninitialized values can be NaN and the one-hot matmul
                # would spread them to every token in the batch.
                computed = jnp.arange(
                    gmm2_res.shape[0],
                    dtype=jnp.int32) < group_sizes.sum(dtype=jnp.int32)
                gmm2_res = jnp.where(computed[:, None], gmm2_res, 0)
            revert_indices = argsort_revert_indices.reshape(num_tokens, topk)
            onehot = jax.nn.one_hot(revert_indices,
                                    gmm2_res.shape[0],
                                    dtype=gmm2_res.dtype)
            # revert_indices is a permutation, so each routed row lands in
            # exactly one (token, k) slot: the products and the sum over k
            # never round, and `combine` holds the topk weights exactly. Only
            # the row reduction below accumulates; pin it to f32 to match the
            # SparseCore combine kernels and their fallback.
            combine = (onehot * topk_weights[..., None] *
                       valid_mask[..., None]).sum(axis=1)
            return jnp.matmul(combine,
                              gmm2_res,
                              preferred_element_type=jnp.float32).astype(
                                  x.dtype)
        return ragged_gather_reduce(
            gmm2_res,
            argsort_revert_indices,
            topk_weights_flat,
            valid_mask_flat,
            reduce_group_size=topk,
        ).astype(x.dtype)

    token_hidden = gmm2_res[argsort_revert_indices]
    token_topk_hidden = token_hidden.reshape(
        (num_tokens, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights,
                                                            axis=-1)
    token_topk_hidden = jnp.where(valid_mask[:, :, None], token_topk_hidden, 0)
    # FP32 top-k weights can promote BF16 outputs; keep the custom-op dtype.
    return token_topk_hidden.sum(axis=1).astype(x.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "activation",
        "use_ep",
        "use_sparse_core",
        "onehot_moe_permute_threshold",
        "rhs_quant_dtype",
        "skip_padded_tokens",
        "ep_size",
    ),
)
def fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    experts_start: jax.Array | None = None,
    topk: int = 1,
    activation: str = "silu",
    use_ep: bool = False,
    use_sparse_core: bool = True,
    onehot_moe_permute_threshold: int = 0,
    rhs_quant_dtype: jnp.dtype | None = None,
    skip_padded_tokens: bool = False,
    ep_size: int = 1,
) -> jax.Array:
    """Run MoE with precomputed expert ids and weights.

    For linear EP placement, ``experts_start`` is the first global expert id
    owned by this shard, passed as a 0-d int32 array (real traced data, not a
    Python int/compile-time constant -- every EP rank owns a different value,
    and binding it as a constant would make each rank JIT-compile a
    structurally different program under the same custom-op name, which
    desyncs the in-graph EP collectives). The kernel remaps global ids to
    local ids with an elementwise subtract and masks non-local experts.
    ``use_ep`` is a static flag enabling EP routing (identical across ranks,
    safe to bind as a compile-time constant); ``use_sparse_core`` is a static
    flag selecting the #193 SparseCore ragged gather/gather-reduce (vs the
    pre-#193 plain-JAX path).

    For packed weights, `rhs_quant_dtype` is used to specify the logical dtype of the weights.
    Currently, only INT4 logical weights packed inside INT32 or UINT32 carrier containers are
    supported. Weights must be packed along the contracting dimension (K-axis) in LSB-first order.
    """

    # Convert 3D scale [experts, blocks, N] to 4D [experts, blocks, 1, N] expected by gmm_v2.
    if w1_scale is not None and w1_scale.ndim == 3:
        w1_scale = jnp.expand_dims(w1_scale, 2)
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jnp.expand_dims(w2_scale, 2)

    num_tokens, hidden_size = hidden_states.shape

    # NVFP4 weights arrive as native float4_e2m1fn in gmm_v2's K-major layout
    # (unpacked once at load; see fused_moe.load_kmajor_fp4). gmm_v2 picks the
    # regime from the block size: block-16 -> W4A16, block >= MXU -> W4A8.
    _, padded_hidden_size, _ = w1.shape
    packing_factor = get_packing_factor(w1.dtype, rhs_quant_dtype)
    padded_hidden_size *= packing_factor

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)

    # Pin topk_weights to hidden_states.dtype so the EP mask, GMM
    # reduce, and `where` propagate in BF16 — some custom_routing_fn
    # callers return FP32 weights and would otherwise promote downstream.
    topk_weights = topk_weights.astype(hidden_states.dtype)

    if use_ep and experts_start is not None:
        local_ids = topk_ids - experts_start
        valid = (local_ids >= 0) & (local_ids < w1.shape[0])
        topk_weights = jnp.where(valid, topk_weights,
                                 jnp.zeros_like(topk_weights))
        topk_ids = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))

    # `ep_size` is identical on every rank (unlike `experts_start`), so binding
    # the capacity as a compile-time constant keeps the ranks' programs
    # structurally identical -- which it must be, or XLA cannot shrink the
    # routed buffer at all.
    capacity_alpha = envs.TPU_MOE_ROUTED_CAPACITY_ALPHA
    routed_capacity = (routed_buffer_capacity(
        num_tokens, topk, ep_size, alpha=capacity_alpha) if
                       (use_ep and ep_size > 1 and capacity_alpha > 0) else
                       None)

    (x, group_sizes, argsort_revert_indices, topk_weights_flat,
     valid_mask) = prepare_routed_gmm_inputs(
         hidden_states,
         topk_ids,
         topk_weights,
         local_num_experts=w1.shape[0],
         topk=topk,
         use_ep=use_ep,
         use_sparse_core=use_sparse_core,
         onehot_moe_permute_threshold=onehot_moe_permute_threshold,
         skip_padded_tokens=skip_padded_tokens,
         routed_capacity=routed_capacity,
     )
    x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))
    x = moe_gmm(
        x,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        activation=activation,
        num_tokens=num_tokens,
        topk=topk,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        rhs_quant_dtype=rhs_quant_dtype,
        skip_padded_tokens=skip_padded_tokens,
    )
    return x[:num_tokens, :hidden_size]
