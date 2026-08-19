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
"""Numeric parity for the `USE_BATCHED_RPA_SEQ_ON_LANE` KV-cache layout."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    wrapper)

_DECODE_BLOCKS = configs.BlockSizes(bq_sz=1,
                                    bq_c_sz=1,
                                    bkv_sz=256,
                                    batch_size=8,
                                    n_buffer=3)
_PREFILL_BLOCKS = configs.BlockSizes(bq_sz=1792,
                                     bq_c_sz=28,
                                     bkv_sz=256,
                                     batch_size=2,
                                     n_buffer=3)


def _build_cache(kv_layout, target_k, target_v, kv_len, total_pages, page_size,
                 num_q_heads, num_kv_heads, head_dim, q_dtype, kv_dtype,
                 page_indices, sm_scale, decode_blocks, prefill_blocks):
    """Prime a KV cache by letting the kernel write the whole sequence into it.

    Queries stay in the activation dtype while the new keys/values and the cache
    carry `kv_dtype` -- that is the split the server runs with an fp8 KV cache,
    and the kernel requires K/V and the cache to agree.
    """
    cache_shape = wrapper.get_kv_cache_shape(total_pages,
                                             page_size,
                                             num_kv_heads,
                                             head_dim,
                                             kv_dtype,
                                             kv_layout=kv_layout)
    _, cache = wrapper.ragged_paged_attention(
        jnp.zeros((kv_len, num_q_heads, head_dim), q_dtype),
        jnp.asarray(target_k, kv_dtype),
        jnp.asarray(target_v, kv_dtype),
        jnp.zeros(cache_shape, kv_dtype),
        jnp.array([kv_len], jnp.int32),
        page_indices,
        jnp.array([0, kv_len], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        sm_scale=sm_scale,
        decode_block_sizes=decode_blocks,
        prefill_block_sizes=prefill_blocks,
        kv_layout=kv_layout,
    )
    return cache


# head_dim / heads / page size / KV dtype. The first is the small shape the
# kernel was developed against; the second is what Qwen3.5-397B-A17B-FP8 serves
# with (`--attention-backend CUSTOM --kv-cache-dtype fp8 --block-size 256`),
# which no parity case covered.
# The fixed block sizes above were chosen for the small shape; at head_dim 256
# with 32 query heads they ask for 168 MB of a 64 MB VMEM budget, so the
# production shape lets the wrapper size its own blocks the way the server does.
_SHAPES = [
    pytest.param(128, 2, 4, 128, jnp.bfloat16, True, id="hd128-bf16-p128"),
    pytest.param(256,
                 2,
                 32,
                 256,
                 jnp.float8_e4m3fn,
                 False,
                 id="hd256-fp8kv-p256"),
]


@pytest.mark.parametrize(
    "head_dim, num_kv_heads, num_q_heads, page_size, kv_dtype, fixed_blocks",
    _SHAPES)
@pytest.mark.parametrize(
    "kv_len, q_len, distribution",
    [
        (32, 16, (0, 0, 1)),
        (33, 1, (1, 1, 1)),
        (4096, 1, (1, 1, 1)),
    ],
)
def test_seq_along_lane_matches_head_along_sublane(kv_len, q_len, distribution,
                                                   head_dim, num_kv_heads,
                                                   num_q_heads, page_size,
                                                   kv_dtype, fixed_blocks):
    rng = np.random.default_rng(0)
    dtype = jnp.bfloat16
    decode_blocks = _DECODE_BLOCKS if fixed_blocks else None
    prefill_blocks = _PREFILL_BLOCKS if fixed_blocks else None
    total_pages = max(8, -(-kv_len // 16) + 1)
    pages_per_seq = total_pages
    sm_scale = head_dim**-0.5
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    target_k, target_v = r(kv_len, num_kv_heads,
                           head_dim), r(kv_len, num_kv_heads, head_dim)
    query = r(q_len, num_q_heads, head_dim)
    new_k, new_v = target_k[kv_len - q_len:kv_len], target_v[kv_len -
                                                             q_len:kv_len]

    kv_lens = jnp.array([kv_len], jnp.int32)
    cu_q_lens = jnp.array([0, q_len], jnp.int32)
    distribution = jnp.asarray(distribution, jnp.int32)

    def run(kv_layout):
        cache = _build_cache(kv_layout, target_k, target_v, kv_len,
                             total_pages, page_size, num_q_heads, num_kv_heads,
                             head_dim, dtype, kv_dtype, page_indices, sm_scale,
                             decode_blocks, prefill_blocks)
        out, _ = wrapper.ragged_paged_attention(
            jnp.asarray(query, dtype),
            jnp.asarray(new_k, kv_dtype),
            jnp.asarray(new_v, kv_dtype),
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            decode_block_sizes=decode_blocks,
            prefill_block_sizes=prefill_blocks,
            kv_layout=kv_layout,
        )
        return np.asarray(out.astype(jnp.float32))

    out_sublane = run(configs.KVLayout.HEAD_ALONG_SUBLANE)
    out_lane = run(configs.KVLayout.SEQ_ALONG_LANE)

    # An fp8 cache rounds every stored key/value, so the two layouts agree to
    # fp8's own granularity rather than bf16's.
    atol = 2e-2 if kv_dtype == jnp.bfloat16 else 1.5e-1
    np.testing.assert_allclose(out_lane, out_sublane, atol=atol, rtol=0)
    assert np.isfinite(out_lane).all()


@pytest.mark.parametrize("kv_layout", list(configs.KVLayout))
def test_non_power_of_two_page_size_is_rejected(kv_layout):
    """A page size that is not a power of two must fail loudly.

    Page addresses inside the kernel are built with `page_size_log2` shifts and
    `page_size_mask` ands, so a non-power-of-two page silently lands new KV on
    the wrong page. 4224 is the exact value this model would hit: it is the
    unified GDN pool's state-fit block, and it is a multiple of 128, so the
    layout check alone lets it through.
    """
    head_dim, num_kv_heads, num_q_heads, page_size = 256, 2, 32, 4224
    dtype = jnp.bfloat16
    cache_shape = wrapper.get_kv_cache_shape(4,
                                             page_size,
                                             num_kv_heads,
                                             head_dim,
                                             dtype,
                                             kv_layout=kv_layout)
    with pytest.raises(ValueError, match="power of two"):
        wrapper.ragged_paged_attention(
            jnp.zeros((1, num_q_heads, head_dim), dtype),
            jnp.zeros((1, num_kv_heads, head_dim), dtype),
            jnp.zeros((1, num_kv_heads, head_dim), dtype),
            jnp.zeros(cache_shape, dtype),
            jnp.array([1], jnp.int32),
            jnp.arange(4, dtype=jnp.int32),
            jnp.array([0, 1], jnp.int32),
            jnp.array([1, 1, 1], jnp.int32),
            sm_scale=head_dim**-0.5,
            decode_block_sizes=_DECODE_BLOCKS,
            prefill_block_sizes=_PREFILL_BLOCKS,
            kv_layout=kv_layout,
        )
