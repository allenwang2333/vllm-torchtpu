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
"""CSA sparse MLA for DeepSeek-V4.1: the V4 kernel on top-k pre-encoded as physical
cache rows, so each CSA layer skips the page-table lookup."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    DEFAULT_VMEM_LIMIT_BYTES,
    _attention_kernel,
    _largest_divisor,
    align_to,
    cdiv,
    gather_page_ids,
    prepare_outputs,
    prepare_q_inputs,
    prepare_swa_inputs,
)


def encode_csa_gather_indices(
    topk_indices: jax.Array,  # i32[num_tokens, topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    total_num_pages: int,
    page_size: int,
) -> jax.Array:
    """Maps CSA top-k token indices to physical cache rows; padding (-1) gets a
    scattered row, stored bitwise inverted so it stays negative."""
    num_tokens, topk = topk_indices.shape
    max_num_seqs = cu_q_lens.shape[0] - 1
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    if eff_max_seqs < max_num_seqs:
        pages_per_seq = page_indices.shape[0] // max_num_seqs
        page_indices = page_indices[: eff_max_seqs * pages_per_seq]
        cu_q_lens = cu_q_lens[: eff_max_seqs + 1]
        max_num_seqs = eff_max_seqs
    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs, dtype=jnp.int32),
        tokens_per_seq,
        total_repeat_length=num_tokens,
    )
    seq_page_ids = topk_indices // page_size
    token_offset = topk_indices % page_size
    page_ids = gather_page_ids(
        page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
    )
    is_padding = topk_indices == -1
    flat_element_index = jnp.arange(num_tokens * topk, dtype=jnp.int32).reshape(
        num_tokens, topk
    )
    # Large primes spread the padding rows across the cache.
    scattered_page_ids = (flat_element_index * 104729) % total_num_pages
    scattered_token_offset = (flat_element_index * 15485863) % page_size
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
    raw_gather_indices = page_ids * page_size + token_offset
    return jnp.where(is_padding, ~raw_gather_indices, raw_gather_indices)


# The compressor writes this step's compressed KV tokens to the cache before
# this runs, so the encoded indices may point at them.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk], already encoded
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_q_heads]
    *,
    sm_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int = 128,
    attention_kernel_batch_size: int = 16,
) -> jax.Array:
    """DeepSeek-V4 sparse_ragged_paged_attention on encode_csa_gather_indices output."""
    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    _, num_q_heads, head_dim = q.shape
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, num_q_heads - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )
    assert swa_accumution.dtype == q.dtype
    swa_accumution, swa_l, swa_m = prepare_swa_inputs(swa_accumution, swa_l, swa_m)

    topk = topk_indices.shape[-1]
    kv_lens = jnp.sum(topk_indices >= 0, axis=-1)
    gather_indices = jnp.where(topk_indices < 0, ~topk_indices, topk_indices).ravel()
    num_valid_tokens = cu_q_lens[distribution[2]]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 6

    for i in range(cdiv(q.shape[0], gather_and_attention_chunk_size)):
        start_pos = i * gather_and_attention_chunk_size
        end_pos = min(start_pos + gather_and_attention_chunk_size, q.shape[0])
        chunk_size = end_pos - start_pos
        indices = gather_indices[start_pos * topk : end_pos * topk]

        batch_size = _largest_divisor(chunk_size, attention_kernel_batch_size)
        valid_in_chunk = jnp.maximum(
            0, jnp.minimum(num_valid_tokens, end_pos) - start_pos
        )
        batch_end = start_pos + cdiv(valid_in_chunk, batch_size) * batch_size
        num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

        gathered_nope, gathered_rope = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices,
            num_valid_indices=num_valid_indices,
            top_k=topk,
        )
        gathered_nope = gathered_nope.reshape(chunk_size, topk, -1)
        gathered_rope = gathered_rope.reshape(chunk_size, topk // 4, -1)
        lanes = gathered_nope.shape[-1]
        end_seq_idx = jnp.maximum(start_pos, batch_end)
        grid = (cdiv(end_seq_idx - start_pos, batch_size),)
        bq_buf = pltpu.VMEM((2, batch_size, num_q_heads, head_dim), q.dtype)
        bl_buf = pltpu.VMEM((2, batch_size, align_to(num_q_heads, 128)), jnp.float32)
        scratch_shapes = [
            pltpu.VMEM((2, batch_size, topk * 4, lanes), jnp.uint8),
            pltpu.VMEM((2, batch_size, (topk // 4) * 2, lanes), jnp.bfloat16),
            bq_buf,
            bq_buf,
            bl_buf,
            bl_buf,
            bq_buf,
            pltpu.SemaphoreType.DMA((7, 2)),
        ]
        scalar_prefetches = (
            kv_lens,
            jnp.array([start_pos, end_seq_idx, num_valid_tokens], jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.full((2,), -1, jnp.int32),
        )
        scope_name = f"SparseMLA-p_{topk}-bz_{batch_size}-gcz_{chunk_size}"
        q = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _attention_kernel, sm_scale=sm_scale, batch_size=batch_size
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=hbm,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                    disable_bounds_checks=True,
                ),
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={5: 0},
                name=scope_name,
            )
        )(
            *scalar_prefetches,
            attention_sinks,
            q,
            gathered_nope,
            gathered_rope,
            swa_accumution,
            swa_l,
            swa_m,
        )
    return prepare_outputs(q, actual_num_q_heads, actual_head_dim)
