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
"""Tests for the pure array-manipulation and validation logic in
`attention_interface.py` that does not require running a real Pallas
kernel: cache reshaping, the guarded-SMEM minibatching split, and the
argument checks that raise before any kernel dispatch.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.mla.kv_cache_utils import (KVCacheLayout,
                                                      KVCacheType,
                                                      SparseMLAKVCacheSpec)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.core import attention_interface

LKV_DIM = 512
ROPE_DIM = 64
KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
NOPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.NOPE,
                                        KVCacheLayout.TENSORCORE, 16, 32,
                                        LKV_DIM, KV_PACKING)
ROPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.ROPE,
                                        KVCacheLayout.TENSORCORE, 16, 32,
                                        ROPE_DIM, KV_PACKING)
ROPE_SC_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.ROPE,
                                           KVCacheLayout.SPARSECORE, 16, 32,
                                           ROPE_DIM, KV_PACKING)


class UpdateCacheTest(parameterized.TestCase):
    """`update_cache` is plain jnp reshape/scatter; no kernel involved."""

    def test_decode_writes_one_token_per_sequence(self):
        K, L, S, H = 2, 3, 4, 5
        cache = jnp.zeros((K, L, S, H))
        operand = jnp.arange(2 * K * H, dtype=jnp.float32).reshape(2, K, 1, H)
        indices = jnp.array([1, 5], dtype=jnp.int32)

        updated = attention_interface.update_cache(False, cache, indices,
                                                   operand)

        flat = updated.reshape(K, L * S, H)
        for b in range(2):
            for k in range(K):
                np.testing.assert_array_equal(flat[k, indices[b]],
                                              operand[b, k, 0])
        # Everything else stays zero.
        mask = jnp.ones((L * S, ), dtype=bool).at[indices].set(False)
        for k in range(K):
            np.testing.assert_array_equal(flat[k, mask], 0)

    def test_prefill_writes_whole_blocks(self):
        B, K, T, H = 1, 2, 4, 3
        S = 2
        L = 4
        cache = jnp.zeros((K, L, S, H))
        operand = jnp.arange(B * K * T * H,
                             dtype=jnp.float32).reshape(B, K, T, H)
        indices = jnp.array([0, 1], dtype=jnp.int32)

        updated = attention_interface.update_cache(True, cache, indices,
                                                   operand)

        expected = cache.at[:, indices, :, :].set(
            jnp.swapaxes(operand, 0, 1).reshape(K, B * T // S, S, H))
        np.testing.assert_array_equal(updated, expected)

    def test_prefill_sliding_window_keeps_only_the_tail(self):
        B, K, T, H = 1, 2, 6, 3
        sliding_window = 4
        S = 2
        L = 4
        cache = jnp.zeros((K, L, S, H))
        operand = jnp.arange(B * K * T * H,
                             dtype=jnp.float32).reshape(B, K, T, H)
        indices = jnp.array([0, 1], dtype=jnp.int32)

        updated = attention_interface.update_cache(
            True,
            cache,
            indices,
            operand,
            prefill_seq_len=jnp.array(T),
            sliding_window=sliding_window)

        # start = max(0, T - sliding_window) = 2: only the last 4 tokens land
        # in the cache, not the first two.
        tail = operand[:, :, T - sliding_window:, :]
        expected = cache.at[:, indices, :, :].set(
            jnp.swapaxes(tail, 0, 1).reshape(K, sliding_window // S, S, H))
        np.testing.assert_array_equal(updated, expected)


class PagedAttentionGuardedSmemTest(parameterized.TestCase):
    """The SMEM-guard minibatching split, with a stub kernel standing in
    for the real Pallas `paged_attention` (the split logic is independent
    of what the kernel itself does)."""

    def test_small_batch_calls_kernel_directly(self):
        calls = []

        def fake_kernel(q, k_pages, v_pages, lengths, page_indices):
            calls.append(page_indices.shape)
            return q + 1.0

        q = jnp.zeros((4, 2, 8))
        lengths = jnp.zeros((4, ), jnp.int32)
        page_indices = jnp.zeros((4, 2), jnp.int32)

        out = attention_interface.paged_attention_with_guarded_smem(
            fake_kernel, q, q, q, lengths, page_indices)

        self.assertEqual(calls, [(4, 2)])
        np.testing.assert_array_equal(out, jnp.ones_like(q))

    def test_oversized_batch_splits_into_minibatches(self):
        # The XLA compiler for this pattern is a separate (and TPU-specific)
        # concern; this test only checks the Python-level minibatch split, so
        # it runs eagerly rather than through the function's `jax.jit`.
        calls = []

        def fake_kernel(q, k_pages, v_pages, lengths, page_indices):
            calls.append(page_indices.shape)
            return q + 1.0

        max_n = attention_interface.MAX_ALLOWED_PAGE_INDICES_N
        batch_size = 4
        blocks_per_seq = max_n // batch_size + 1  # size > max_n
        self.assertGreater(batch_size * blocks_per_seq, max_n)

        q = jnp.zeros((batch_size, 1, 1))
        lengths = jnp.zeros((batch_size, ), jnp.int32)
        page_indices = jnp.zeros((batch_size, blocks_per_seq), jnp.int32)

        with jax.disable_jit():
            out = attention_interface.paged_attention_with_guarded_smem(
                fake_kernel, q, q, q, lengths, page_indices)

        self.assertGreater(len(calls), 1)
        self.assertEqual(out.shape, q.shape)
        np.testing.assert_array_equal(out, jnp.ones_like(q))


class ShardedRaggedPagedAttentionValidationTest(parameterized.TestCase):
    """`attention_sink` support is gated on head_dim==64, checked before the
    function ever touches `mesh` or dispatches a kernel."""

    def test_attention_sink_requires_hd64(self):
        head_dim = 32  # != 64, so use_hd64 is False
        q = jnp.zeros((4, 2, head_dim))
        kv_lens = jnp.zeros((1, ), jnp.int32)
        page_indices = jnp.zeros((1, ), jnp.int32)
        cu_q_lens = jnp.zeros((1, ), jnp.int32)
        distribution = jnp.zeros((3, ), jnp.int32)
        attention_sink = jnp.zeros((2, ))

        with self.assertRaisesRegex(NotImplementedError, "head_dim==64"):
            attention_interface.sharded_ragged_paged_attention(
                None,  # mesh: unused before the raise
                q,
                q,
                q,
                jnp.zeros((1, )),  # kv_cache: unused before the raise
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sink,
                sm_scale=1.0)


class SparseMlaAttentionLayoutValidationTest(parameterized.TestCase):
    """`sparse_mla_attention` rejects mismatched NoPE/RoPE cache layouts
    before dispatching to the kernel."""

    def test_mismatched_layouts_raise(self):
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x", ))

        with self.assertRaisesRegex(ValueError,
                                    "matching NoPE and RoPE layouts"):
            attention_interface.sparse_mla_attention(
                jnp.zeros((1, 2, LKV_DIM), jnp.bfloat16),
                jnp.zeros((1, 2, ROPE_DIM), jnp.bfloat16),
                jnp.zeros((1, LKV_DIM), jnp.float8_e4m3fn),
                jnp.zeros((1, ROPE_DIM), jnp.float8_e4m3fn),
                jnp.zeros(NOPE_SPEC.shape, NOPE_SPEC.jax_dtype),
                jnp.zeros(ROPE_SC_SPEC.shape, ROPE_SC_SPEC.jax_dtype),
                jnp.zeros((1, 1), jnp.int32),
                jnp.zeros((1, ), jnp.int32),
                jnp.zeros((4, ), jnp.int32),
                jnp.zeros((2, ), jnp.int32),
                jnp.zeros((3, ), jnp.int32),
                mesh,
                NOPE_SPEC,
                ROPE_SC_SPEC,
            )
