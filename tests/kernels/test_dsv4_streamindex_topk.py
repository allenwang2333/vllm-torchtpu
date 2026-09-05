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
"""Tests for Deepseek V4 StreamIndex Top-K kernel."""

import ast
import inspect
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from google3.experimental.users.hwanginho.deepseek_v4.streamindex_topk.streamindex_topk import \
        streamindex_topk
except ImportError:
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        streamindex_topk

# The `causal_row_offset` internals. Imported unconditionally and by their real
# names: a guarded import that skipped these tests when the symbol moved would
# hide exactly the regression they exist to catch.
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    _effective_row_lengths, _fast_path_max_valid_idx,
    _last_visible_compressed_row, _visible_compressed_rows)


# =====================================================================
# Helper functions from the compressor implementation
# =====================================================================
def _to_byte_lane(x: jax.Array) -> jax.Array:
    """Reinterpret each element of ``x``'s trailing dim as raw bytes."""
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def quantize_fp8_ue8m0(x: jax.Array, block_size: int):
    """Block fp8 quantization with UE8M0 (power-of-two) block scales."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4,
                    None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    scale = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
    return q, scale


# =====================================================================

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def streamindex_topk_ref(
    q,
    weights,
    kv,
    block_table,
    T_list,
    S_list,
    cu_q_lens,
    k,
    comp_ratio,
    H_I,
    H_KV,
):
    """Naive NumPy reference implementation for StreamIndex Top-K."""
    num_tokens = q.shape[0]
    expected_topk = np.full((num_tokens, k), -1, dtype=np.int32)
    B = len(T_list)

    for b in range(B):
        T_seq = T_list[b]
        S_total = S_list[b]
        q_start = cu_q_lens[b]

        if T_seq == 0:
            continue

        S_valid = S_total // comp_ratio
        seq_blocks = block_table[b]
        seq_kv = np.concatenate([kv[p] for p in seq_blocks], axis=0)

        naive_scores = np.full((T_seq, max(S_valid, k)),
                               -np.inf,
                               dtype=np.float32)

        for t_idx in range(T_seq):
            global_t = q_start + t_idx
            q_abs_pos = (S_total - T_seq) + t_idx

            for s_idx in range(S_valid):
                if s_idx * comp_ratio > q_abs_pos:
                    continue

                score = 0.0
                for h in range(H_I):
                    h_kv = h // (H_I // H_KV)
                    inner = np.dot(q[global_t, h], seq_kv[s_idx, h_kv])
                    score += max(0.0, inner) * weights[global_t, h]

                naive_scores[t_idx, s_idx] = score

        seq_expected = np.argsort(-naive_scores, axis=-1)[:, :k]

        for t_idx in range(T_seq):
            for i in range(k):
                idx = seq_expected[t_idx, i]
                if naive_scores[t_idx, idx] == -np.inf:
                    expected_topk[q_start + t_idx, i] = -1
                else:
                    expected_topk[q_start + t_idx, i] = idx

    return expected_topk


@pytest.mark.parametrize(
    "B, num_tokens, page_size, max_blocks, H_I, H_KV, D, k, compression_ratio,"
    " bq_sz, bkv_p, seq_lens_list, cu_q_lens_list",
    [
        # Small standard case
        (2, 6, 16, 8, 4, 1, 64, 512, 2, 32, 8, [4, 6], [0, 3, 6]),
        # # Single token batch (Decode Phase)
        (1, 1, 16, 4, 2, 1, 32, 512, 1, 16, 8, [10], [0, 1]),
        # Large batch, multiple tokens per sequence (Prefill Phase)
        (
            4,
            20,
            32,
            16,
            8,
            1,
            128,
            512,
            4,
            64,
            4,
            [10, 20, 30, 40],
            [0, 5, 10, 15, 20],
        ),
        # Odd chunk sizes
        (2, 5, 8, 6, 4, 1, 16, 512, 1, 16, 16, [8, 12], [0, 2, 5]),
        # Single head for KV but multiple heads for Queries (MQA/GQA pattern)
        (2, 10, 16, 8, 8, 1, 64, 512, 2, 32, 8, [16, 24], [0, 4, 10]),
        # # High D, High K
        (1, 8, 16, 4, 2, 1, 256, 512, 1, 32, 8, [60], [0, 8]),
        # Mixed batch: sequence 0 has 1 token (decode), sequence 1 has 5 tokens
        # (prefill)
        (2, 6, 16, 8, 4, 1, 64, 512, 2, 32, 8, [4, 6], [0, 1, 6]),
    ],
)
def test_streamindex_topk_shape(
    B,
    num_tokens,
    page_size,
    max_blocks,
    H_I,
    H_KV,
    D,
    k,
    compression_ratio,
    bq_sz,
    bkv_p,
    seq_lens_list,
    cu_q_lens_list,
):
    """Tests the shape and basic execution bounds of streamindex_topk."""
    _ = H_KV
    print(f"\n{'-'*60}")
    print(f"SHAPE TEST: B={B}, Tokens={num_tokens}, k={k}")
    print(f"{'-'*60}")

    query_projection = jnp.zeros((num_tokens, H_I, D), dtype=jnp.float32)
    indexer_weights = jnp.zeros((num_tokens, H_I), dtype=jnp.float32)

    num_pages = max_blocks * B
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128
    kv_cache = jnp.zeros((num_pages, page_size // 4, 4, width),
                         dtype=jnp.uint8)

    block_table = jnp.zeros((B, max_blocks), dtype=jnp.int32)
    page_indices = block_table.flatten()
    cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

    print("Inputs:")
    print(f"  - query_projection: {query_projection.shape}")
    print(f"  - kv_cache:         {kv_cache.shape}")
    print(f"  - seq_lens:         {seq_lens_list}")
    print(f"  - cu_q_lens:        {cu_q_lens_list}")

    # Count number of decode sequences (T == 1) at the beginning of the batch
    num_decodes = 0
    while (num_decodes < B
           and (cu_q_lens_list[num_decodes + 1] - cu_q_lens_list[num_decodes])
           == 1):
        num_decodes += 1
    distribution = (num_decodes, num_decodes, B)
    expected_shape = (num_tokens, k)

    seq_lens = jnp.array(seq_lens_list, dtype=jnp.int32)

    out_shape_idxs = jax.eval_shape(
        streamindex_topk,
        query_projection,
        indexer_weights,
        kv_cache,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    print("\nOutputs:")
    print(f"  - Expected shape: {expected_shape}, dtype: int32")
    print(f"  - Actual shape:   {out_shape_idxs.shape}, dtype:"
          f" {out_shape_idxs.dtype}")

    assert out_shape_idxs.shape == expected_shape
    assert out_shape_idxs.dtype == jnp.int32
    print("Result: PASS")


@pytest.mark.parametrize(
    "B, T_list, S_list, page_size, H_I, H_KV, D, k, comp_ratio, bq_sz, bkv_p,"
    " block_table_list",
    [
        # 0. Single sequence, single page (page_indices shape (1,))
        (1, [1], [8], 8, 4, 1, 16, 1024, 1, 1, 16, [[0]]),
        # 1. Single sequence, highly fragmented block table
        (1, [4], [16], 8, 4, 1, 16, 1024, 2, 8, 16, [[2, 0]]),
        # 2. Batched Decode (T=1, B=2)
        (2, [1, 1], [12, 16], 8, 4, 1, 16, 1024, 1, 8, 16, [[1, 3], [0, 2]]),
        # 3. High GQA (8 Query Heads, 2 KV Heads)
        (1, [6], [20], 4, 8, 1, 16, 1024, 1, 8, 32, [[4, 1, 3, 0, 2]]),
        # 4. Multi-Batch Prefill with variable query/sequence lengths
        (2, [5, 3], [16, 16], 8, 4, 1, 16, 1024, 1, 8, 16, [[3, 1], [2, 4]]),
        # 5. Dummy sequences / Padding (B=3, but sequence 1 has 0 tokens)
        (
            3,
            [2, 0, 3],
            [16, 0, 8],
            8,
            4,
            1,
            16,
            1024,
            1,
            8,
            16,
            [[1, 2], [0, 0], [4, 3]],
        ),
        # 6. Mixed batch: sequence 0 has 1 token, sequence 1 has 5 tokens
        (2, [1, 5], [12, 16], 8, 4, 1, 16, 1024, 2, 8, 16, [[1, 3], [2, 0]]),
        (
            1,
            [4],
            [384],
            16,
            4,
            1,
            16,
            1024,
            1,
            8,
            8,
            [[
                13, 2, 21, 7, 0, 18, 5, 11, 23, 1, 9, 16, 3, 20, 6, 14, 22, 4,
                10, 17, 8, 15, 19, 12
            ]],
        ),
    ],
)
def test_streamindex_topk_numerical_correctness(
    B,
    T_list,
    S_list,
    page_size,
    H_I,
    H_KV,
    D,
    k,
    comp_ratio,
    bq_sz,
    bkv_p,
    block_table_list,
):
    """Executes randomized input data against a naive NumPy ground truth."""
    print(f"\n{'='*60}")
    print(f"BATCHED NUMERICAL TEST (B={B})")
    print(f"{'='*60}")

    np.random.seed(42)

    # 1. Setup Random Tensors
    num_tokens = sum(T_list)
    q = np.random.randn(num_tokens, H_I, D).astype(np.float32)
    weights = np.random.uniform(-1.5, 1.5,
                                size=(num_tokens, H_I)).astype(np.float32)

    # Create a unified physical KV Cache pool large enough for all block indices
    max_physical_page = np.max(block_table_list)
    float32_kv = np.random.randn(max_physical_page + 1, page_size, H_KV,
                                 D).astype(np.float32)

    # Pack cache using compressor's quantize_fp8_ue8m0 and _to_byte_lane
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128

    cache_kv = np.zeros((max_physical_page + 1, page_size // 4, 4, width),
                        dtype=np.uint8)
    dequantized_kv = np.zeros_like(float32_kv)

    for p in range(max_physical_page + 1):
        for s in range(page_size):
            for h in range(H_KV):
                row_kv = float32_kv[p, s, h]
                # Quantize using D as block size (each head query key has dimension D)
                q_jax, scale_jax = quantize_fp8_ue8m0(jnp.array(row_kv), D)
                q_bytes = np.array(_to_byte_lane(q_jax))
                scale_bytes = np.array(_to_byte_lane(scale_jax))
                dq = np.array(q_jax).astype(np.float32)
                dequantized_kv[p, s, h] = dq * float(scale_jax[0])
                record = np.concatenate([q_bytes, scale_bytes], axis=-1)
                record = np.pad(record, (0, width - record.shape[-1]))

                w_idx = s // 4
                lane_idx = s % 4
                cache_kv[p, w_idx, lane_idx] = record

    block_table = np.array(block_table_list, dtype=np.int32)
    page_indices = block_table.flatten()
    seq_lens = np.array(S_list, dtype=np.int32)
    cu_q_lens = np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32)

    print("Configuration:")
    print(f"  - Tokens total: {num_tokens}, Sequences(B): {B}, K: {k}")
    print(f"  - GQA: {H_I} Query Heads -> {H_KV} KV Heads")
    print(f"  - Compression Ratio: {comp_ratio}")

    print("\nInputs Generated:")
    print(f"  - Query Tensor: {q.shape}")
    print(f"  - KV Cache:     {cache_kv.shape}")
    print(f"  - Block Table:  {block_table.tolist()}")

    # =====================================================================
    # 2. NAIVE COMPUTATION (The Ground Truth)
    # =====================================================================
    print("\n[1/3] Computing exact ground-truth using naive NumPy loops...")
    expected_topk = streamindex_topk_ref(
        q=q,
        weights=weights,
        kv=dequantized_kv,
        block_table=block_table,
        T_list=T_list,
        S_list=S_list,
        cu_q_lens=cu_q_lens,
        k=k,
        comp_ratio=comp_ratio,
        H_I=H_I,
        H_KV=H_KV,
    )

    print("\nGROUND TRUTH (Naive NumPy):")
    print(expected_topk)

    # =====================================================================
    # 3. Pallas KERNEL COMPUTATION
    # =====================================================================
    print(
        "\n[2/3] Executing optimized JAX Kernel (streamindex_pallas_topk)...")
    # Count number of decode sequences (T == 1) at the beginning of the batch
    num_decodes = 0
    while num_decodes < B and T_list[num_decodes] == 1:
        num_decodes += 1
    distribution = (num_decodes, num_decodes, B)

    actual_topk = streamindex_topk(
        q=jnp.array(q),
        indexer_weights=jnp.array(weights),
        cache_kv=jnp.array(cache_kv),
        seq_lens=jnp.array(seq_lens),
        page_indices=jnp.array(page_indices),
        cu_q_lens=jnp.array(cu_q_lens),
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    actual_topk_np = np.array(actual_topk)

    print("\nACTUAL OUTPUT (JAX/XLA Kernel):")
    print(actual_topk_np)

    # =====================================================================
    # 4. VERIFY
    # =====================================================================
    print("\n[3/3] Verifying absolute correctness...")
    np.testing.assert_array_equal(
        np.sort(actual_topk_np, axis=-1),
        np.sort(expected_topk, axis=-1),
        err_msg="JAX Pallas Kernel Top-K math did not match Naive Ground Truth",
    )
    print(
        "MATCH VERIFIED! JAX kernel handles batches, fragmentation, and dummy"
        " padding.\n")


def test_streamindex_topk_quantized():
    """Verifies correctness of streamindex_topk on FP8 packed cache."""
    np.random.seed(42)

    T_seq = 4
    S_seq = 2048
    page_size = 16
    H_I = 2
    H_KV = 1
    D = 128
    k = 512
    comp_ratio = 4
    bkv_p = 8  # page_size * bkv_p = 128 (TPU DMA contract of the scores kernel)
    bq_sz = 1

    q = np.random.randn(T_seq, H_I, D).astype(np.float32)
    weights = np.random.uniform(0.5, 1.5, size=(T_seq, H_I)).astype(np.float32)

    S_valid = S_seq // comp_ratio
    # We need enough pages to hold S_valid compressed tokens.
    num_pages = (S_valid + page_size - 1) // page_size
    float32_kv = np.random.randn(num_pages, page_size, H_KV,
                                 D).astype(np.float32)

    # Pack cache using compressor's quantize_fp8_ue8m0 and _to_byte_lane
    width = 256
    cache_kv = np.zeros((num_pages, page_size // 4, 4, width), dtype=np.uint8)

    # Dequantized KV for naive ground truth
    dequantized_kv = np.zeros_like(float32_kv)

    for p in range(num_pages):
        for s in range(page_size):
            for h_kv in range(H_KV):
                row_kv = float32_kv[p, s, h_kv]

                # Quantize using compressor helper
                # block_size = 128 since we want 1 scale factor for D=128
                q_jax, scale_jax = quantize_fp8_ue8m0(jnp.array(row_kv), 128)

                # Convert to bytes
                q_bytes = np.array(_to_byte_lane(q_jax))
                scale_bytes = np.array(_to_byte_lane(scale_jax))

                # Store dequantized version exactly as hardware sees it for accurate
                # validation
                dq = np.array(q_jax).astype(np.float32)
                dequantized_kv[p, s, h_kv] = dq * float(scale_jax[0])

                record = np.concatenate([q_bytes, scale_bytes], axis=-1)
                record = np.pad(record, (0, width - record.shape[-1]))

                w_idx = s // 4
                lane_idx = s % 4
                cache_kv[p, w_idx, lane_idx] = record

    # Compute expected top-k using exact dequantized keys
    seq_kv = np.concatenate([dequantized_kv[p] for p in range(num_pages)],
                            axis=0)

    naive_scores = np.full((T_seq, max(S_valid, k)), -np.inf, dtype=np.float32)

    for t_idx in range(T_seq):
        for s_idx in range(S_valid):
            score = 0.0
            for h in range(H_I):
                h_kv = h // (H_I // H_KV)
                inner = np.dot(q[t_idx, h], seq_kv[s_idx, h_kv])
                score += max(0.0, inner) * weights[t_idx, h]
            naive_scores[t_idx, s_idx] = score

    expected_topk = np.argsort(-naive_scores, axis=-1)[:, :k]
    for t_idx in range(T_seq):
        for i in range(k):
            idx = expected_topk[t_idx, i]
            if naive_scores[t_idx, idx] == -np.inf:
                expected_topk[t_idx, i] = -1

    # Pallas parameters
    actual_topk = streamindex_topk(
        q=jnp.array(q),
        indexer_weights=jnp.array(weights),
        cache_kv=jnp.array(cache_kv),
        seq_lens=jnp.array([S_seq], dtype=jnp.int32),
        page_indices=jnp.arange(num_pages, dtype=jnp.int32),
        cu_q_lens=jnp.array([0, T_seq], dtype=jnp.int32),
        distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    actual_topk_np = np.array(actual_topk)

    np.testing.assert_array_equal(
        np.sort(actual_topk_np, axis=-1),
        np.sort(expected_topk, axis=-1),
    )


# =====================================================================
# Early exit
# =====================================================================
# `enable_early_exit=True` must change cost, never the answer. Once a
# sequence's compressed length is <= k, top-k selects every visible position,
# so the scores decide nothing and the scoring kernel can be skipped outright.
# The guard is batch-wide, so the tests below pin both halves of that: the
# shortcut agrees with the scoring path when the whole batch is short, and is
# inert the moment any single sequence is not.

# `page_size * bkv_p` is 128, the TPU DMA contract of the scores kernel.
_EE_PAGE_SIZE = 128
_EE_PAGES_PER_SEQ = 32
_EE_BKV_P = 1
_EE_BQ_SZ = 4
_EE_H_I = 4
_EE_D = 128


def _ee_pack_cache(keys):
    """fp8-quantize ``[num_pages, page_size, D]`` keys into the uint8 cache.

    Same record layout the tests above build row by row -- ``[fp8 x D | one
    e8m0 scale byte | pad]`` rounded up to a 128-lane group, addressed as
    ``[page, slot // 4, slot % 4, width]`` -- just quantized in one shot.
    """
    num_pages, page_size, head_dim = keys.shape
    fp8, scale = quantize_fp8_ue8m0(jnp.asarray(keys), head_dim)
    record = jnp.concatenate(
        [_to_byte_lane(fp8), _to_byte_lane(scale)], axis=-1)
    width = -(-record.shape[-1] // 128) * 128
    record = jnp.pad(record, ((0, 0), (0, 0), (0, width - record.shape[-1])))
    return np.asarray(record).reshape(num_pages, page_size // 4, 4, width)


def _ee_inputs(q_lens, seq_lens, seed):
    """One kernel invocation at ``compression_ratio=1``.

    Every sequence gets a shuffled, disjoint set of physical pages so a bug in
    the page walk cannot hide behind sequential layout.
    """
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    num_tokens = int(sum(q_lens))
    num_pages = num_seqs * _EE_PAGES_PER_SEQ

    block_table = rng.permutation(num_pages).astype(np.int32).reshape(
        num_seqs, _EE_PAGES_PER_SEQ)
    keys = rng.standard_normal((num_pages, _EE_PAGE_SIZE, _EE_D),
                               dtype=np.float32)

    # `distribution` requires the decode-only sequences to lead the batch.
    num_decodes = 0
    while num_decodes < num_seqs and q_lens[num_decodes] == 1:
        num_decodes += 1

    return {
        "q":
        rng.standard_normal((num_tokens, _EE_H_I, _EE_D), dtype=np.float32),
        "indexer_weights":
        rng.uniform(0.25, 1.75, (num_tokens, _EE_H_I)).astype(np.float32),
        "cache_kv":
        _ee_pack_cache(keys),
        "seq_lens":
        np.asarray(seq_lens, np.int32),
        "page_indices":
        block_table.reshape(-1),
        "cu_q_lens":
        np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
        "distribution":
        np.array([num_decodes, num_decodes, num_seqs], np.int32),
    }


def _ee_both_paths(q_lens,
                   seq_lens,
                   k,
                   seed=11,
                   compression_ratio=1,
                   causal_row_offset=0):
    """Run the same batch with the flag off and on.

    ``compression_ratio`` / ``causal_row_offset`` default to the values the
    four tests below were written against, so those tests are unchanged; the
    parameters exist for the ratio-2 / ratio-4 cases further down, which are
    the geometries the fast path has never been covered at.
    """
    inputs = _ee_inputs(q_lens, seq_lens, seed)

    def run(enable_early_exit):
        return np.asarray(
            streamindex_topk(**{
                name: jnp.asarray(value)
                for name, value in inputs.items()
            },
                             k=k,
                             compression_ratio=compression_ratio,
                             causal_row_offset=causal_row_offset,
                             num_kv_pages_per_block=_EE_BKV_P,
                             num_queries_per_block=_EE_BQ_SZ,
                             enable_early_exit=enable_early_exit))

    return run(False), run(True)


def _ee_visible_counts(q_lens, seq_lens):
    """Causally visible compressed positions per query token.

    At ``compression_ratio=1`` the queries are the tail of their sequence, so
    query ``i`` of a sequence of length ``S`` sits at absolute position
    ``S - q_len + i`` and sees every position up to and including it.
    """
    counts = []
    for q_len, seq_len in zip(q_lens, seq_lens):
        counts.extend(seq_len - q_len + i + 1 for i in range(q_len))
    return counts


def _ee_assert_same_selection(early, baseline):
    """Both paths must select the same positions, in whatever order.

    The scoring path emits winners in ``approx_max_k`` order and the early-exit
    path in increasing KV position. Order is deliberately unspecified, so
    comparing the raw rows would assert something the kernel never promised --
    compare the selected sets.
    """
    assert early.shape == baseline.shape, (early.shape, baseline.shape)
    for token, (row_e, row_b) in enumerate(zip(early, baseline)):
        kept_e = np.sort(row_e[row_e >= 0])
        kept_b = np.sort(row_b[row_b >= 0])
        np.testing.assert_array_equal(
            kept_e, kept_b, f"token {token}: early exit selected a different "
            "set of positions than the scoring path")
        assert np.all(row_e[len(kept_e):] < 0), (
            f"token {token}: -1 padding is not a suffix: {row_e}")


def _ee_assert_every_visible_position(actual, q_lens, seq_lens, k):
    """With the whole batch under k the answer is exact, not a ranking.

    Every visible position wins, so the expected index set is closed-form and
    the check does not depend on score arithmetic at all.
    """
    for token, n_visible in enumerate(_ee_visible_counts(q_lens, seq_lens)):
        assert n_visible <= k, "test setup: sequence is not short"
        row = actual[token]
        np.testing.assert_array_equal(
            np.sort(row[row >= 0]), np.arange(n_visible),
            f"token {token}: not every visible position was reported")
        assert np.all(row[n_visible:] < 0), (
            f"token {token}: -1 padding is not a suffix: {row}")


def _ee_assert_full_rows(actual, q_lens, seq_lens, k):
    """Cheap sanity for the long-sequence cases.

    Correctness of the scoring path itself is what the tests above cover; here
    the claim under test is that the flag is inert, so this only rules out
    gross breakage.
    """
    counts = _ee_visible_counts(q_lens, seq_lens)
    limits = []
    for q_len, seq_len in zip(q_lens, seq_lens):
        limits.extend([seq_len] * q_len)
    for token, (n_visible, limit) in enumerate(zip(counts, limits)):
        assert n_visible > k, "test setup: sequence is not long"
        row = actual[token]
        assert np.all(row >= 0), f"token {token}: padded a full-length row"
        assert row.max() < limit, (
            f"token {token}: index {row.max()} past seq_len {limit}")


def test_streamindex_topk_early_exit_all_short():
    """Whole batch under k: the global fast path answers on its own."""
    q_lens, seq_lens, k = (1, 1, 1), (64, 96, 128), 512
    baseline, early = _ee_both_paths(q_lens, seq_lens, k)

    _ee_assert_every_visible_position(early, q_lens, seq_lens, k)
    _ee_assert_same_selection(early, baseline)


def test_streamindex_topk_early_exit_no_short_is_inert():
    """Every sequence over k: nothing is skipped, so nothing may change.

    Here the flag must be inert, which is a stronger claim than agreeing as
    sets -- no row took the shortcut, so the rows are bit-identical.
    """
    q_lens, seq_lens, k = (1, 1, 4), (2048, 3072, 4096), 128
    baseline, early = _ee_both_paths(q_lens, seq_lens, k)

    _ee_assert_full_rows(early, q_lens, seq_lens, k)
    np.testing.assert_array_equal(early, baseline)


def test_streamindex_topk_early_exit_is_batch_wide():
    """The guard is batch-wide, not per sequence.

    Two short decodes alongside one long prefill: `jnp.max(seq_lens)` is over
    k, so every token goes back through the scoring kernel including the short
    ones, and the flag must be bit-for-bit inert.
    """
    q_lens, seq_lens, k = (1, 1, 4), (64, 96, 4096), 512
    baseline, early = _ee_both_paths(q_lens, seq_lens, k)

    np.testing.assert_array_equal(early, baseline)


def test_streamindex_topk_early_exit_keeps_causal_mask():
    """The shortcut is still causal: a prefill token sees only its past.

    The whole sequence fits in k, so a bug that returned `[0, k)` instead of
    `[0, position]` would still look plausible -- this is what catches it.
    """
    q_lens, seq_lens, k = (4, ), (64, ), 256
    _, early = _ee_both_paths(q_lens, seq_lens, k)

    _ee_assert_every_visible_position(early, q_lens, seq_lens, k)
    # Four query tokens at the tail of a 64-long sequence sit at absolute
    # positions 60..63, so they keep 61..64 positions respectively.
    kept = [int(np.count_nonzero(row >= 0)) for row in early]
    assert kept == [61, 62, 63, 64], kept


# =====================================================================
# causal_row_offset
# =====================================================================
# `causal_row_offset` generalises one predicate: compressed row `j` is visible
# to raw query position `p` iff `j * compression_ratio + causal_row_offset <=
# p`. Offset 0 is what this kernel has always done (a row is visible as soon as
# the query enters it); offset `compression_ratio - 1` is GLM-5.3 kpool's rule
# (a pool is a candidate only once its *last* token is visible).
#
# Two production models -- DeepSeek-V4 CSA and GLM-5.2 DSA -- run this file
# today, so the load-bearing claim is not "the offset works" but "offset 0
# changed nothing". The tests below pin that three ways: against frozen copies
# of the pre-change arithmetic, against a from-the-predicate brute force, and
# by asserting neither existing call site passes the argument at all.
#
# There is deliberately no "the offset is a no-op at ratio 1 for every value"
# test. That claim is algebraically false -- see
# `test_offset_is_not_a_no_op_at_ratio_one` -- and ratio 1 is safe only because
# the structural guard admits nothing but 0 there.


def _pre_change_effective_row_lengths(
    seq_lens,
    cu_q_lens,
    distribution,
    num_tokens,
    num_positions,
    compression_ratio,
):
    """Verbatim `_effective_row_lengths` as of the commit before this change.

    Frozen on purpose. It is the only artifact that can prove `offset=0`
    changed nothing, so it must never be "kept in sync" with the kernel -- if
    a future edit makes this disagree at offset 0, the disagreement *is* the
    regression.
    """
    max_num_seqs = seq_lens.shape[0]
    num_seqs = distribution[2]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_mask = token_ids[:, None] >= cu_q_lens[None, 1:max_num_seqs + 1]
    seq_mask = jnp.where(
        jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False)
    seq_ids = jnp.sum(seq_mask, axis=1)

    q_start = cu_q_lens[seq_ids]
    q_len = cu_q_lens[seq_ids + 1] - q_start
    seq_len = seq_lens[seq_ids]
    token_pos = seq_len - q_len + (token_ids - q_start)
    kv_len = seq_len // compression_ratio
    visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
    valid = token_ids < cu_q_lens[num_seqs]
    return jnp.where(valid, jnp.minimum(visible, num_positions), 0)


def _pre_change_fast_path_max_valid_idx(seq_lens, cu_q_lens, num_tokens,
                                        compression_ratio):
    """Verbatim index arithmetic from the pre-change `_fast_path` closure.

    Same contract as above: frozen, never updated. This is the delicate
    site: it is an inclusive max *index*, and rewriting it in the count form
    would, at offset 0, hand every query one extra in-range key id with no
    assert, no NaN and no shape error.
    """
    token_idx = jnp.arange(num_tokens)
    seq_idx = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], token_idx, side="right"),
        seq_lens.shape[0] - 1,
    )
    seq_len = seq_lens[seq_idx]
    q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
    q_start = cu_q_lens[seq_idx]
    q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
    max_valid_idx = jnp.minimum(
        seq_len // compression_ratio - 1,
        q_abs_pos // compression_ratio,
    )
    return jnp.where(token_idx < cu_q_lens[-1], max_valid_idx, -1)


def _brute_force_visible_rows(token_pos, compression_ratio, causal_row_offset):
    """Enumerate the predicate. No division, no closed form."""
    return sum(1 for j in range(token_pos // compression_ratio + 2)
               if j * compression_ratio + causal_row_offset <= token_pos)


def _random_offset_batch(rng, allow_multi_token):
    """A plausible `(seq_lens, cu_q_lens, distribution)` triple.

    Decode-only sequences lead the batch, as `distribution` requires; both
    arrays carry the trailing padding the runner passes; `num_tokens` overruns
    the real token count so the padded-row path is exercised too.
    """
    max_num_seqs = int(rng.integers(1, 6))
    num_seqs = int(rng.integers(1, max_num_seqs + 1))
    num_decodes = int(rng.integers(0, num_seqs +
                                   1)) if allow_multi_token else num_seqs

    q_lens, seq_lens = [], []
    for i in range(num_seqs):
        q_len = 1 if i < num_decodes else int(rng.integers(2, 9))
        # Mix small and large so every residue mod compression_ratio and both
        # sides of the `kv_len` clamp show up across trials.
        span = int(rng.integers(0, 64)) if rng.random() < 0.5 else int(
            rng.integers(0, 4096))
        q_lens.append(q_len)
        seq_lens.append(q_len + span)

    total = int(sum(q_lens))
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cu_q_lens = np.full(max_num_seqs + 1, total, dtype=np.int32)
    cu_q_lens[:num_seqs + 1] = cu
    padded_seq_lens = np.zeros(max_num_seqs, dtype=np.int32)
    padded_seq_lens[:num_seqs] = seq_lens

    return {
        "seq_lens":
        jnp.asarray(padded_seq_lens),
        "cu_q_lens":
        jnp.asarray(cu_q_lens),
        "distribution":
        jnp.asarray([num_decodes, num_decodes, num_seqs], dtype=jnp.int32),
        "num_tokens":
        total + int(rng.integers(0, 5)),
        "num_positions":
        int(rng.choice([128, 1024, 4096, 65536])),
        "q_lens":
        q_lens,
        "seq_lens_list":
        seq_lens,
    }


@pytest.mark.parametrize("compression_ratio", [1, 2, 3, 4, 8, 128])
def test_visible_row_count_matches_the_predicate(compression_ratio):
    """The count form, against a from-the-definition enumeration."""
    offsets = range(compression_ratio) if compression_ratio <= 8 else (0, 1,
                                                                       63, 127)
    for causal_row_offset in offsets:
        for token_pos in range(0, 260):
            assert int(
                _visible_compressed_rows(
                    token_pos, compression_ratio,
                    causal_row_offset)) == _brute_force_visible_rows(
                        token_pos, compression_ratio,
                        causal_row_offset), (token_pos, compression_ratio,
                                             causal_row_offset)


@pytest.mark.parametrize("compression_ratio", [1, 2, 3, 4, 8, 128])
def test_last_visible_row_is_an_index_not_a_count(compression_ratio):
    """Count and index are one apart, and nothing downstream notices.

    The count form differs from the index form by exactly one, and every value
    either produces is a legal in-range row id -- so nothing downstream can
    tell them apart. The only thing that can is the defining predicate: the
    returned row must itself be visible, and the next row must not be.
    """
    offsets = range(compression_ratio) if compression_ratio <= 8 else (0, 1,
                                                                       63, 127)
    for causal_row_offset in offsets:
        for token_pos in range(0, 260):
            idx = int(
                _last_visible_compressed_row(token_pos, compression_ratio,
                                             causal_row_offset))
            count = _brute_force_visible_rows(token_pos, compression_ratio,
                                              causal_row_offset)
            assert idx == count - 1, (token_pos, compression_ratio,
                                      causal_row_offset, idx, count)
            if idx >= 0:
                assert idx * compression_ratio + causal_row_offset <= token_pos, (
                    f"row {idx} is not visible to position {token_pos} at "
                    f"ratio {compression_ratio} offset {causal_row_offset} -- "
                    "this is the count form masquerading as the index form")
            else:
                assert idx == -1, idx
            assert (
                (idx + 1) * compression_ratio + causal_row_offset
                > token_pos), (
                    f"row {idx + 1} is visible to position {token_pos}, so "
                    f"{idx} is not the last visible row")


@pytest.mark.parametrize("compression_ratio", [1, 2, 3, 4, 128])
def test_mask_row_index_at_offset_zero_is_bit_identical(compression_ratio):
    """Site `:372`: the Pallas causal mask. `offset=0` must be `q_pos // r`."""
    q_pos = jnp.arange(0, 4096, dtype=jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(_last_visible_compressed_row(q_pos, compression_ratio, 0)),
        np.asarray(q_pos // compression_ratio),
    )


@pytest.mark.parametrize("compression_ratio", [1, 4, 128])
def test_effective_row_lengths_at_offset_zero_is_bit_identical(
        compression_ratio):
    """Site `:504`: the top-k row length, against the frozen pre-change copy.

    This is the real DeepSeek-V4 protection -- array equality over randomized
    `(seq_lens, cu_q_lens, distribution)`, not a claim about ratio 1.
    """
    rng = np.random.default_rng(0xC5A0 + compression_ratio)
    for trial in range(200):
        case = _random_offset_batch(rng, allow_multi_token=True)
        kwargs = {
            key: case[key]
            for key in ("seq_lens", "cu_q_lens", "distribution", "num_tokens",
                        "num_positions")
        }
        expected = _pre_change_effective_row_lengths(
            compression_ratio=compression_ratio, **kwargs)
        actual = _effective_row_lengths(compression_ratio=compression_ratio,
                                        causal_row_offset=0,
                                        **kwargs)
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected),
            f"trial {trial}: offset 0 is not bit-identical to the pre-change "
            f"row lengths at ratio {compression_ratio}")


@pytest.mark.parametrize("compression_ratio", [1, 2, 4, 128])
def test_fast_path_max_valid_idx_at_offset_zero_is_bit_identical(
        compression_ratio):
    """Site `:850-853`: the early-exit path, against the frozen pre-change copy.

    Ratios 2 and 4 with `q_len > 1` rows are the specific configuration the
    count-vs-index defect shows up in and that this file has never covered:
    at the final position of a sequence the `seq_len // r - 1` clamp hides the
    extra row, so only a mid-sequence multi-token row exposes it.
    """
    rng = np.random.default_rng(0xFA57 + compression_ratio)
    for trial in range(200):
        case = _random_offset_batch(rng, allow_multi_token=True)
        kwargs = {
            key: case[key]
            for key in ("seq_lens", "cu_q_lens", "num_tokens")
        }
        expected = _pre_change_fast_path_max_valid_idx(
            compression_ratio=compression_ratio, **kwargs)
        actual = _fast_path_max_valid_idx(compression_ratio=compression_ratio,
                                          causal_row_offset=0,
                                          **kwargs)
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected),
            f"trial {trial}: offset 0 is not bit-identical to the pre-change "
            f"fast path at ratio {compression_ratio}")


@pytest.mark.parametrize(
    "compression_ratio, causal_row_offset",
    [(1, 1), (1, -1), (1, 3), (4, 4), (4, 5), (4, -1), (128, 128)],
)
def test_causal_row_offset_outside_a_row_is_rejected(compression_ratio,
                                                     causal_row_offset):
    """The structural guard, at every entry point that takes the offset.

    `(1, 1)` is the important row: it is what makes "ratio 1 admits only offset
    0" true by construction rather than by an algebraic claim that happens to
    be false. The forbidden repair is a `compression_ratio == 1 ->
    causal_row_offset = 0` clamp inside the kernel, which would be a silent
    ratio-dependent special case in a file shared by three models.
    """
    case = _random_offset_batch(np.random.default_rng(1), False)

    with pytest.raises(ValueError, match="causal_row_offset"):
        _effective_row_lengths(
            seq_lens=case["seq_lens"],
            cu_q_lens=case["cu_q_lens"],
            distribution=case["distribution"],
            num_tokens=case["num_tokens"],
            num_positions=case["num_positions"],
            compression_ratio=compression_ratio,
            causal_row_offset=causal_row_offset,
        )

    with pytest.raises(ValueError, match="causal_row_offset"):
        _fast_path_max_valid_idx(
            seq_lens=case["seq_lens"],
            cu_q_lens=case["cu_q_lens"],
            num_tokens=case["num_tokens"],
            compression_ratio=compression_ratio,
            causal_row_offset=causal_row_offset,
        )


@pytest.mark.parametrize("compression_ratio, causal_row_offset", [(1, 1),
                                                                  (4, 4),
                                                                  (4, -1)])
def test_streamindex_topk_rejects_an_out_of_range_offset(
        compression_ratio, causal_row_offset):
    """The guard fires at trace time, before any TPU lowering.

    It also pins that `causal_row_offset` is a *static* argument: were it
    traced, `0 <= causal_row_offset` would raise jax's
    `TracerBoolConversionError`, not a `ValueError`.
    """
    with pytest.raises(ValueError, match="causal_row_offset"):
        streamindex_topk(
            jnp.zeros((4, 4, 128), jnp.float32),
            jnp.zeros((4, 4), jnp.float32),
            jnp.zeros((8, 32, 4, 256), jnp.uint8),
            jnp.asarray([64], jnp.int32),
            jnp.arange(8, dtype=jnp.int32),
            jnp.asarray([0, 4], jnp.int32),
            jnp.asarray([0, 0, 1], jnp.int32),
            k=128,
            compression_ratio=compression_ratio,
            causal_row_offset=causal_row_offset,
            num_kv_pages_per_block=1,
            num_queries_per_block=4,
        )


def test_offset_is_not_a_no_op_at_ratio_one():
    """Pins the false claim that motivated the structural guard.

    An earlier draft of this work asserted "the offset is a no-op at ratio 1
    for every value" and proposed to protect GLM-5.2 DSA with it. At ratio 1
    the visible-row count is `p + 1 - off`, which moves with `off` at every
    value. The arithmetic below is what would happen if the guard were removed
    or clamped -- it is here so nobody re-derives the false claim.
    """
    positions = jnp.arange(64, dtype=jnp.int32)
    at_zero = np.asarray(_visible_compressed_rows(positions, 1, 0))
    at_one = np.asarray(_visible_compressed_rows(positions, 1, 1))
    np.testing.assert_array_equal(at_zero, np.arange(1, 65))
    np.testing.assert_array_equal(at_one, np.arange(0, 64))
    assert not np.array_equal(at_zero, at_one)


@pytest.mark.parametrize("seq_len", [12, 13, 14, 15, 16, 2047, 2048, 2049])
def test_glm_pool_rule_at_ratio_four_offset_three(seq_len):
    """Offset `r - 1` is GLM's `floor((p + 1) / index_kpool)` pool count.

    One whole-sequence prefill, so every position `p` in the sequence is a
    query row. `L % 4 != 0` is deliberately in the parameter list: at
    `L % 4 == 0` the last pool is complete and an off-by-one at the tail
    cannot fire.
    """
    compression_ratio, causal_row_offset = 4, 3
    cu_q_lens = jnp.asarray([0, seq_len], jnp.int32)
    row_lengths = np.asarray(
        _effective_row_lengths(
            seq_lens=jnp.asarray([seq_len], jnp.int32),
            cu_q_lens=cu_q_lens,
            distribution=jnp.asarray([0, 0, 1], jnp.int32),
            num_tokens=seq_len,
            num_positions=1 << 20,
            compression_ratio=compression_ratio,
            causal_row_offset=causal_row_offset,
        ))
    expected = np.asarray([
        min(seq_len // compression_ratio, (p + 1) // compression_ratio)
        for p in range(seq_len)
    ])
    np.testing.assert_array_equal(row_lengths, expected)

    # And the corresponding index form, which the Pallas mask uses. It is not
    # clamped by `kv_len` -- the kernel's separate `valid_mask` does that -- so
    # it is `floor((p + 1) / 4) - 1` outright, and `-1` for p in 0..2 where no
    # pool is complete yet.
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    np.testing.assert_array_equal(
        np.asarray(
            _last_visible_compressed_row(positions, compression_ratio,
                                         causal_row_offset)),
        np.asarray([(p + 1) // compression_ratio - 1 for p in range(seq_len)]),
    )


@pytest.mark.parametrize(
    "compression_ratio, causal_row_offset",
    [(1, 0), (2, 0), (2, 1), (4, 0), (4, 3), (128, 0), (128, 127)],
)
def test_mask_and_row_length_carry_the_same_offset(compression_ratio,
                                                   causal_row_offset):
    """The Pallas mask and the top-k row length must differ by exactly one.

    `streamindex_topk` masks with `bq_pos_compressed` (an inclusive index) and
    selects with `_effective_row_lengths` (a count). Rewriting only one of them
    changes the candidate set by exactly one row per query -- and every short
    sequence hides it, because there top-k keeps everything visible either way.
    The token positions here are recomputed from the batch spec rather than
    from the kernel's own sequence-id search, so this is not the implementation
    checking itself.
    """
    rng = np.random.default_rng(0x17 + compression_ratio + causal_row_offset)
    for trial in range(100):
        case = _random_offset_batch(rng, allow_multi_token=True)
        row_lengths = np.asarray(
            _effective_row_lengths(
                seq_lens=case["seq_lens"],
                cu_q_lens=case["cu_q_lens"],
                distribution=case["distribution"],
                num_tokens=case["num_tokens"],
                num_positions=case["num_positions"],
                compression_ratio=compression_ratio,
                causal_row_offset=causal_row_offset,
            ))

        expected = []
        for q_len, seq_len in zip(case["q_lens"], case["seq_lens_list"]):
            kv_len = seq_len // compression_ratio
            for i in range(q_len):
                token_pos = seq_len - q_len + i
                mask_idx = int(
                    _last_visible_compressed_row(token_pos, compression_ratio,
                                                 causal_row_offset))
                expected.append(
                    min(kv_len, mask_idx + 1, case["num_positions"]))
        # Padded token rows carry no candidates at all.
        expected.extend([0] * (case["num_tokens"] - len(expected)))

        np.testing.assert_array_equal(
            row_lengths, np.asarray(expected, dtype=row_lengths.dtype),
            f"trial {trial}: row length disagrees with the mask index at "
            f"ratio {compression_ratio} offset {causal_row_offset}")


def test_causal_row_offset_defaults_to_zero():
    """Nobody has to pass it, and the default is the historical convention."""
    assert inspect.signature(
        streamindex_topk).parameters["causal_row_offset"].default == 0


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py",
        "src/vllm_torchtpu/layers/vllm/custom_ops/deepseek_v4/"
        "deepseek_v4_indexer.py",
    ],
)
def test_existing_call_sites_pass_no_causal_row_offset(relative_path):
    """GLM-5.2 DSA and DeepSeek-V4 CSA must keep the default, silently.

    Parsed rather than imported: these modules need `torch_tpu` at import
    time, and the question is purely syntactic. Finding zero calls is a
    failure, so a rename cannot make this pass vacuously.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    source = (repo_root / relative_path).read_text()
    tree = ast.parse(source)

    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id ==
             "streamindex_topk") or (isinstance(node.func, ast.Attribute)
                                     and node.func.attr == "streamindex_topk"))
    ]
    assert calls, f"no streamindex_topk call found in {relative_path}"
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert "causal_row_offset" not in passed, (
            f"{relative_path}:{call.lineno} now passes causal_row_offset; "
            "DeepSeek-V4 CSA and GLM-5.2 DSA are supposed to stay on the "
            "default 0 convention")


# ---------------------------------------------------------------------------
# Early exit at compression_ratio != 1 (device only).
#
# `_ee_both_paths` above hard-coded ratio 1, so all four early-exit tests are
# ratio-1 only and "the DSv4 file stays green" does not by itself protect the
# fast path's index arithmetic. These are the missing cases: `q_len > 1` rows
# at ratio 2 and 4, at offset 0 (DeepSeek-V4 bit-identity) and at offset
# `r - 1` (GLM's pool rule). They need a TPU, and ride into the device job.
# ---------------------------------------------------------------------------


def _ee_expected_counts(q_lens, seq_lens, compression_ratio,
                        causal_row_offset):
    """Visible compressed rows per query token, straight from the predicate."""
    counts = []
    for q_len, seq_len in zip(q_lens, seq_lens):
        kv_len = seq_len // compression_ratio
        for i in range(q_len):
            token_pos = seq_len - q_len + i
            visible = (0 if token_pos < causal_row_offset else
                       (token_pos - causal_row_offset) // compression_ratio +
                       1)
            counts.append(min(kv_len, visible))
    return counts


@pytest.mark.parametrize(
    "compression_ratio, causal_row_offset",
    [(2, 0), (2, 1), (4, 0), (4, 3)],
)
def test_streamindex_topk_early_exit_at_compression_ratio(
        compression_ratio, causal_row_offset):
    """Fast path vs scoring path with a multi-token row and a real ratio.

    `max(seq_lens) // compression_ratio <= k` so the shortcut is taken, and the
    whole batch fits under k so the expected index set is closed-form -- the
    check does not depend on score arithmetic at all. Offset 0 is the
    DeepSeek-V4 convention; offset `r - 1` is GLM's, where a query at position
    `p` may only see pools whose last token is at or before `p`.
    """
    q_lens = (1, 1, 5)
    # Under `k * compression_ratio` so the shortcut triggers, and deliberately
    # not multiples of the ratio: at a ratio-aligned final position the two
    # conventions agree, which is exactly how this off-by-one stays hidden.
    k = 512
    seq_lens = (64 * compression_ratio + 1, 200 * compression_ratio + 3,
                480 * compression_ratio - 2)
    assert max(seq_lens) // compression_ratio <= k, "test setup: not short"
    assert _ee_expected_counts(
        q_lens, seq_lens, compression_ratio, 0) != _ee_expected_counts(
            q_lens, seq_lens, compression_ratio,
            compression_ratio - 1), ("test setup: the two conventions agree "
                                     "on every row, so this proves nothing")

    baseline, early = _ee_both_paths(q_lens,
                                     seq_lens,
                                     k,
                                     compression_ratio=compression_ratio,
                                     causal_row_offset=causal_row_offset)

    expected = _ee_expected_counts(q_lens, seq_lens, compression_ratio,
                                   causal_row_offset)
    for token, n_visible in enumerate(expected):
        row = early[token]
        np.testing.assert_array_equal(
            np.sort(row[row >= 0]), np.arange(n_visible),
            f"token {token}: early exit did not report exactly the visible "
            f"rows at ratio {compression_ratio} offset {causal_row_offset}")
        assert np.all(row[n_visible:] < 0), (
            f"token {token}: -1 padding is not a suffix: {row}")

    _ee_assert_same_selection(early, baseline)


def main(argv):
    del argv

    raise SystemExit(pytest.main([__file__, "-p", "no:cacheprovider"]))


if __name__ == "__main__":
    from absl import app

    app.run(main)
