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

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
from vllm_torchtpu.kernels.sparse_core.ragged_gather import \
    ragged_gather as ragged_gather_v1
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import \
    ragged_gather_reduce as ragged_gather_reduce_v1
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2 import \
    ragged_gather_reduce as ragged_gather_reduce_v2
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import \
    ragged_gather_v2

if envs.RAGGED_GATHER_VERSION == "v1":
    ragged_gather = ragged_gather_v1
else:
    ragged_gather = ragged_gather_v2
if envs.RAGGED_GATHER_REDUCE_VERSION == "v1":
    ragged_gather_reduce = ragged_gather_reduce_v1
else:
    ragged_gather_reduce = ragged_gather_reduce_v2


# WARNING: [TEMP WORKAROUND] GMM tile size override function to support v6e-8 testing
def wrap_get_tuned_block_sizes(
    dims,
    lhs_cfgs,
    rhs_cfgs,
    vmem_limit_bytes,
    fuse_act=None,
):
    from vllm_torchtpu.kernels.megablox.gmm_v2 import TileSizes
    from vllm_torchtpu.kernels.megablox.tuned_block_sizes import get_tuned_block_sizes

    n = dims.size_n
    if fuse_act is not None:
        n //= 2

    tm, tk, tn = get_tuned_block_sizes(
        m=dims.size_m,
        k=dims.size_k,
        n=n,
        num_total_groups=256,
        num_current_groups=dims.size_group,
        lhs_dtype=str(lhs_cfgs.dtype),
        rhs_dtype=str(rhs_cfgs.dtype),
        quant_block_size=rhs_cfgs.quant_block_size,
    )
    return TileSizes(tile_m=tm, tile_k=tk, tile_n=tn)


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
        tile_info=wrap_get_tuned_block_sizes,
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

    valid_mask = topk_indices_flat >= 0
    sort_keys = jnp.where(valid_mask, topk_indices_flat, local_num_experts)
    sorted_indices = jnp.argsort(sort_keys)
    argsort_revert_indices = jnp.argsort(sorted_indices)

    token_indices_sorted = token_indices_flat[sorted_indices]
    topk_indices_for_count = jnp.where(valid_mask, topk_indices_flat, 0)
    group_sizes_local = (jax.nn.one_hot(
        topk_indices_for_count, local_num_experts, dtype=jnp.int32) *
                         valid_mask[:, None].astype(jnp.int32))
    group_sizes_local = group_sizes_local.sum(axis=0)

    if use_ep and use_sparse_core:
        if token_indices_sorted.shape[0] <= onehot_moe_permute_threshold:
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
            x = ragged_gather(
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
        rhs_quant_dtype=w1.dtype,
    )
    gmm1_res = gmm1_res[:, :w2.shape[1]]

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
                           rhs_quant_dtype=w2.dtype)

    if use_ep and use_sparse_core:
        if argsort_revert_indices.size <= onehot_moe_permute_threshold:
            # Use onehot + matmul for unpermutation, which can be faster
            # for small batch size.
            revert_indices = argsort_revert_indices.reshape(num_tokens, topk)
            onehot = jax.nn.one_hot(revert_indices,
                                    argsort_revert_indices.size,
                                    dtype=gmm2_res.dtype)
            combine = (onehot * topk_weights[..., None] *
                       valid_mask[..., None]).sum(axis=1)
            return (combine @ gmm2_res).astype(x.dtype)
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
    experts_start: int | None = None,
    topk: int = 1,
    activation: str = "silu",
    use_ep: bool = False,
    use_sparse_core: bool = True,
    onehot_moe_permute_threshold: int = 0,
) -> jax.Array:
    """Run MoE with precomputed expert ids and weights."""
    if w1.dtype == jnp.uint8:
        w1 = w1.view(jnp.float4_e2m1fn)
    if w2.dtype == jnp.uint8:
        w2 = w2.view(jnp.float4_e2m1fn)
    num_tokens, hidden_size = hidden_states.shape
    _, padded_hidden_size, _ = w1.shape

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)

    # Pin topk_weights to hidden_states.dtype so the EP mask, GMM
    # reduce, and `where` propagate in BF16 — some custom_routing_fn
    # callers return FP32 weights and would otherwise promote downstream.
    topk_weights = topk_weights.astype(hidden_states.dtype)

    if experts_start is not None:
        local_ids = topk_ids - experts_start
        valid = (local_ids >= 0) & (local_ids < w1.shape[0])
        topk_weights = jnp.where(valid, topk_weights,
                                 jnp.zeros_like(topk_weights))
        topk_ids = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))

    import os
    is_ds_v4 = (w1.shape[0] in (32, 256) and w1.shape[1] == 4096 and w1.shape[2] == 4096)
    is_quantized_dtype = w1.dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.float4_e2m1fn, jnp.uint8, jnp.int8)
    use_native = (is_ds_v4 or os.getenv("VLLM_USE_NATIVE_MOE", "0") == "1") and os.getenv("VLLM_FORCE_GMM", "0") != "1"
    if is_quantized_dtype and os.getenv("VLLM_USE_NATIVE_MOE", "0") != "1":
        use_native = False
 
    if use_native:
        from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_act_fn
        from vllm_torchtpu.layers.vllm.linear_common import _quantized_matmul_jax
 
        output = jnp.zeros((num_tokens, hidden_size), dtype=hidden_states.dtype)
        for e in range(w1.shape[0]):
            mask = (topk_ids == e)
            expert_weights = jnp.where(mask, topk_weights, 0.0).sum(axis=1)
 
            if w1_scale is not None:
                gate_up = _quantized_matmul_jax(hidden_states, w1[e], w1_scale[e])
            else:
                gate_up = hidden_states @ w1[e]
            if w1_bias is not None:
                gate_up += w1_bias[e]
 
            act = apply_act_fn(gate_up, activation)
 
            if w2_scale is not None:
                out = _quantized_matmul_jax(act, w2[e].T, w2_scale[e])
            else:
                out = act @ w2[e]
            if w2_bias is not None:
                out += w2_bias[e]
 
            output += out * expert_weights[:, None]
        return output

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
    )
    return x[:num_tokens, :hidden_size]
