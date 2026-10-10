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
"""Tests for the parts of `attention_interface.py` that can run without a
real TPU kernel.

Most of `attention_interface.py` is glue around Pallas kernels: it reshapes
inputs, picks kernel parameters, checks arguments, and then hands off to a
kernel. The kernels themselves are tested in their own files under
`tests/kernels/`. This file covers the glue:

* writing new tokens into the KV cache (`update_cache`), which is ordinary
  array reshaping and scattering with no kernel involved;
* the SMEM-overflow guard around paged attention, which decides whether to
  call the kernel once or split the batch into smaller pieces; and
* the up-front argument checks that raise a clear error before any kernel
  is ever called.

Because none of these tests dispatch a kernel, they run the same way on any
TPU generation and on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.core import attention_interface

# Sparse-MLA head dimensions. These are the only sizes the fp8 gather kernel
# accepts (see the assert in `sparse_mla_attention`), so the layout test below
# has to use them to get past the size check and reach the layout check.
LKV_DIM = 512
ROPE_DIM = 64
KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
# Two cache specs in the TensorCore layout and one in the SparseCore layout,
# so a test can hand `sparse_mla_attention` a mismatched pair.
NOPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.NOPE, KVCacheLayout.TENSORCORE, 16, 32, LKV_DIM, KV_PACKING
)
ROPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.ROPE, KVCacheLayout.TENSORCORE, 16, 32, ROPE_DIM, KV_PACKING
)
ROPE_SC_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.ROPE, KVCacheLayout.SPARSECORE, 16, 32, ROPE_DIM, KV_PACKING
)


class UpdateCacheTest(parameterized.TestCase):
    """`update_cache` writes freshly computed keys or values into the paged
    KV cache.

    The cache is laid out as `(kv_heads, num_blocks, block_size, head_dim)`.
    The function has two very different code paths depending on whether it
    is handling a prefill (many tokens per sequence, written a whole block
    at a time) or a decode step (one token per sequence, written one slot at
    a time). Each test below builds a small cache, calls the function, and
    compares the result to the same update written out by hand.
    """

    def test_decode_writes_one_token_per_sequence(self):
        """Decode: each sequence contributes one new token, and it lands at
        the flat slot index given for that sequence. Every other slot in the
        cache must be left untouched."""
        K, L, S, H = 2, 3, 4, 5
        cache = jnp.zeros((K, L, S, H))
        operand = jnp.arange(2 * K * H, dtype=jnp.float32).reshape(2, K, 1, H)
        indices = jnp.array([1, 5], dtype=jnp.int32)

        updated = attention_interface.update_cache(False, cache, indices, operand)

        flat = updated.reshape(K, L * S, H)
        for b in range(2):
            for k in range(K):
                np.testing.assert_array_equal(flat[k, indices[b]], operand[b, k, 0])
        # Everything else stays zero.
        mask = jnp.ones((L * S,), dtype=bool).at[indices].set(False)
        for k in range(K):
            np.testing.assert_array_equal(flat[k, mask], 0)

    def test_prefill_writes_whole_blocks(self):
        """Prefill: the new tokens for a sequence are split into block-sized
        chunks and each chunk is written to the block it was assigned. With
        T=4 tokens and a block size of 2 that is two blocks."""
        B, K, T, H = 1, 2, 4, 3
        S = 2
        L = 4
        cache = jnp.zeros((K, L, S, H))
        operand = jnp.arange(B * K * T * H, dtype=jnp.float32).reshape(B, K, T, H)
        indices = jnp.array([0, 1], dtype=jnp.int32)

        updated = attention_interface.update_cache(True, cache, indices, operand)

        expected = cache.at[:, indices, :, :].set(
            jnp.swapaxes(operand, 0, 1).reshape(K, B * T // S, S, H)
        )
        np.testing.assert_array_equal(updated, expected)

    def test_prefill_sliding_window_keeps_only_the_tail(self):
        """Prefill with a sliding window: when a prompt is longer than the
        window, only the most recent `sliding_window` tokens are kept. The
        earlier tokens must not be written to the cache at all."""
        B, K, T, H = 1, 2, 6, 3
        sliding_window = 4
        S = 2
        L = 4
        cache = jnp.zeros((K, L, S, H))
        operand = jnp.arange(B * K * T * H, dtype=jnp.float32).reshape(B, K, T, H)
        indices = jnp.array([0, 1], dtype=jnp.int32)

        updated = attention_interface.update_cache(
            True,
            cache,
            indices,
            operand,
            prefill_seq_len=jnp.array(T),
            sliding_window=sliding_window,
        )

        # start = max(0, T - sliding_window) = 2: only the last 4 tokens land
        # in the cache, not the first two.
        tail = operand[:, :, T - sliding_window :, :]
        expected = cache.at[:, indices, :, :].set(
            jnp.swapaxes(tail, 0, 1).reshape(K, sliding_window // S, S, H)
        )
        np.testing.assert_array_equal(updated, expected)


class PagedAttentionGuardedSmemTest(parameterized.TestCase):
    """`paged_attention_with_guarded_smem` protects the paged attention
    kernel from running out of SMEM.

    The kernel keeps every sequence's page table in SMEM, which is small
    and fixed in size. When the total number of page indices is under
    `MAX_ALLOWED_PAGE_INDICES_N`, the function just calls the kernel once.
    When it is over, the function splits the batch into equal-sized smaller
    batches, runs the kernel on each one, and stitches the outputs back
    together.

    The real kernel is replaced with a stub that records how it was called
    and returns `q + 1`. The splitting logic does not depend on what the
    kernel computes, so the stub is enough to check it.
    """

    def test_small_batch_calls_kernel_directly(self):
        """A batch under the limit is passed straight to the kernel in a
        single call, with the page table unchanged."""
        calls = []

        def fake_kernel(q, k_pages, v_pages, lengths, page_indices):
            calls.append(page_indices.shape)
            return q + 1.0

        q = jnp.zeros((4, 2, 8))
        lengths = jnp.zeros((4,), jnp.int32)
        page_indices = jnp.zeros((4, 2), jnp.int32)

        out = attention_interface.paged_attention_with_guarded_smem(
            fake_kernel, q, q, q, lengths, page_indices
        )

        self.assertEqual(calls, [(4, 2)])
        np.testing.assert_array_equal(out, jnp.ones_like(q))

    def test_oversized_batch_splits_into_minibatches(self):
        """A batch over the limit is split up: the kernel is called more
        than once, each time on a slice of the page table, and the combined
        output has the same shape as the input.

        This runs with `jax.disable_jit()`. The split is plain Python (a
        `for` loop and some reshapes), which is all this test is checking.
        Compiling the split for a real TPU at the size needed to trigger it
        is a separate concern for the kernel owners, and on some hosts it
        currently crashes the XLA compiler.
        """
        calls = []

        def fake_kernel(q, k_pages, v_pages, lengths, page_indices):
            calls.append(page_indices.shape)
            return q + 1.0

        max_n = attention_interface.MAX_ALLOWED_PAGE_INDICES_N
        batch_size = 4
        blocks_per_seq = max_n // batch_size + 1  # size > max_n
        self.assertGreater(batch_size * blocks_per_seq, max_n)

        q = jnp.zeros((batch_size, 1, 1))
        lengths = jnp.zeros((batch_size,), jnp.int32)
        page_indices = jnp.zeros((batch_size, blocks_per_seq), jnp.int32)

        with jax.disable_jit():
            out = attention_interface.paged_attention_with_guarded_smem(
                fake_kernel, q, q, q, lengths, page_indices
            )

        self.assertGreater(len(calls), 1)
        self.assertEqual(out.shape, q.shape)
        np.testing.assert_array_equal(out, jnp.ones_like(q))


class ShardedRaggedPagedAttentionValidationTest(parameterized.TestCase):
    """Argument checks in `sharded_ragged_paged_attention` that fail fast,
    before the function builds the sharded kernel call."""

    def test_attention_sink_requires_hd64(self):
        """Attention sinks are only implemented in the head_dim=64 kernel
        variant. Passing one with any other head size must raise a clear
        `NotImplementedError` rather than a confusing kernel failure. The
        check happens before `mesh` or `kv_cache` are read, so both can be
        placeholders here."""
        head_dim = 32  # != 64, so use_hd64 is False
        q = jnp.zeros((4, 2, head_dim))
        kv_lens = jnp.zeros((1,), jnp.int32)
        page_indices = jnp.zeros((1,), jnp.int32)
        cu_q_lens = jnp.zeros((1,), jnp.int32)
        distribution = jnp.zeros((3,), jnp.int32)
        attention_sink = jnp.zeros((2,))

        with self.assertRaisesRegex(NotImplementedError, "head_dim==64"):
            attention_interface.sharded_ragged_paged_attention(
                None,  # mesh: unused before the raise
                q,
                q,
                q,
                jnp.zeros((1,)),  # kv_cache: unused before the raise
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sink,
                sm_scale=1.0,
            )


class SparseMlaAttentionLayoutValidationTest(parameterized.TestCase):
    """Argument checks in `sparse_mla_attention` that fail fast, before the
    function builds the sharded kernel call."""

    def test_mismatched_layouts_raise(self):
        """The NoPE and RoPE caches must use the same memory layout (both
        TensorCore or both SparseCore). Handing the function one of each
        must raise a `ValueError` naming both layouts. All the array
        arguments here are zeros of the right shape; only the two layout
        specs matter for this check."""
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x",))

        with self.assertRaisesRegex(ValueError, "matching NoPE and RoPE layouts"):
            attention_interface.sparse_mla_attention(
                jnp.zeros((1, 2, LKV_DIM), jnp.bfloat16),
                jnp.zeros((1, 2, ROPE_DIM), jnp.bfloat16),
                jnp.zeros((1, LKV_DIM), jnp.float8_e4m3fn),
                jnp.zeros((1, ROPE_DIM), jnp.float8_e4m3fn),
                jnp.zeros(NOPE_SPEC.shape, NOPE_SPEC.jax_dtype),
                jnp.zeros(ROPE_SC_SPEC.shape, ROPE_SC_SPEC.jax_dtype),
                jnp.zeros((1, 1), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((4,), jnp.int32),
                jnp.zeros((2,), jnp.int32),
                jnp.zeros((3,), jnp.int32),
                mesh,
                NOPE_SPEC,
                ROPE_SC_SPEC,
            )


class AttentionInterfaceWrappersTest(parameterized.TestCase):
    """CPU unit tests for the sharded wrapper glue in `attention_interface.py`."""

    def _mesh(self, axis_name="model"):
        return jax.sharding.Mesh(np.array(jax.local_devices()[:1]), (axis_name,))

    def test_sharded_flash_and_paged_and_splash_wrappers(self):
        from unittest import mock

        mesh = self._mesh("model")
        q4 = jnp.ones((1, 2, 4, 8), dtype=jnp.float32)
        seg = jnp.zeros((1, 4), dtype=jnp.int32)

        with mock.patch.object(
            attention_interface,
            "flash_attention",
            side_effect=lambda q, k, v, **kw: q + k + v,
        ):
            fn = attention_interface.sharded_flash_attention(
                mesh, causal=False, sm_scale=0.5, vmem_limit_bytes=1024
            )
            out = fn(q4, q4, q4, seg)
            self.assertEqual(out.shape, q4.shape)

        q3 = jnp.ones((2, 2, 8), dtype=jnp.float32)
        kv_pages = jnp.ones((2, 4, 32, 8), dtype=jnp.float32)
        lengths = jnp.ones((2,), dtype=jnp.int32)
        page_indices = jnp.zeros((2, 2), dtype=jnp.int32)

        with (
            mock.patch.object(
                attention_interface,
                "paged_attention",
                side_effect=lambda q, k, v, lens, p, **kw: q + 1.0,
            ),
            mock.patch.object(attention_interface, "get_megacore", return_value=True),
        ):
            paged_fn = attention_interface.sharded_paged_attention(
                mesh, attn_logits_soft_cap=30.0
            )
            out3 = paged_fn(q3, kv_pages, kv_pages, lengths, page_indices)
            self.assertEqual(out3.shape, q3.shape)

            oversized = jnp.zeros(
                (1, attention_interface.MAX_ALLOWED_PAGE_INDICES_N + 1),
                dtype=jnp.int32,
            )
            with self.assertRaisesRegex(ValueError, "smem OOM"):
                paged_fn(q3[:1], kv_pages, kv_pages, lengths[:1], oversized)

        with (
            mock.patch.object(
                attention_interface.splash.BlockSizes,
                "get_default",
                return_value="bs",
            ),
            mock.patch.object(
                attention_interface.splash,
                "make_splash_mqa_single_device",
                side_effect=lambda mask, **kw: lambda q, k, v, s: q + 2.0,
            ),
            mock.patch.object(
                attention_interface.splash,
                "make_splash_mha_single_device",
                side_effect=lambda mask, **kw: lambda q, k, v, s: q + 3.0,
            ),
        ):
            mqa_fn = attention_interface.sharded_splash_attention(
                mesh, window_size=4, attn_logits_soft_cap=10.0, is_mqa=True
            )
            mha_fn = attention_interface.sharded_splash_attention(
                mesh, window_size=4, attn_logits_soft_cap=None, is_mqa=False
            )
            np.testing.assert_allclose(mqa_fn(q4, q4, q4), q4 + 2.0)
            np.testing.assert_allclose(mha_fn(q4, q4, q4), q4 + 3.0)

    def test_sharded_rpa_and_attention_and_bundled_and_mla(self):
        from types import SimpleNamespace
        from unittest import mock

        from vllm_torchtpu.kernels.experimental.batched_rpa import (
            configs as batched_rpa_configs,
        )
        from vllm_torchtpu.kernels.ragged_paged_attention.v3 import (
            kernel as rpa_v3_kernel,
        )
        from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

        mesh = self._mesh("model")
        q = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
        k = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
        v = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((4, 16, 2, 2, 128), dtype=jnp.bfloat16)
        md = AttentionMetadata(
            input_positions=jnp.arange(4, dtype=jnp.int32),
            block_tables=jnp.arange(4, dtype=jnp.int32),
            seq_lens=jnp.array([4], dtype=jnp.int32),
            query_start_loc=jnp.array([0, 4], dtype=jnp.int32),
            request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        )

        def fake_rpa(*args, **kwargs):
            return args[0], args[3]

        # 1. shard=True with HEAD_ALONG_SUBLANE and SEQ_ALONG_LANE
        new_kv, out = attention_interface.attention(
            kv_cache,
            q,
            k,
            v,
            md,
            mesh,
            head_dim_original=None,
            sm_scale=None,
            rpa_func=fake_rpa,
            kv_layout=batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE,
        )
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(new_kv.shape, kv_cache.shape)

        kv_cache_sal = jnp.zeros((4, 2, 2, 128, 16), dtype=jnp.bfloat16)
        new_kv2, out2 = attention_interface.attention(
            kv_cache_sal,
            q,
            k,
            v,
            md,
            mesh,
            head_dim_original=128,
            sm_scale=0.125,
            rpa_func=fake_rpa,
            kv_layout=batched_rpa_configs.KVLayout.SEQ_ALONG_LANE,
        )
        self.assertEqual(out2.shape, q.shape)
        self.assertEqual(new_kv2.shape, kv_cache_sal.shape)

        # 2. shard=False with kv_block_cap (both with and without attention_chunk_size)
        with mock.patch.object(
            attention_interface.rpa_default,
            "get_default_block_sizes",
            return_value={"bq_sz": 16, "bkv_sz": 64, "bq_csz": 8, "bkv_csz": 32},
        ):
            out_local, _ = attention_interface.sharded_ragged_paged_attention(
                mesh,
                q,
                k,
                v,
                kv_cache,
                md.seq_lens,
                md.block_tables,
                md.query_start_loc,
                md.request_distribution,
                None,
                sm_scale=0.125,
                attention_chunk_size=64,
                rpa_func=fake_rpa,
                shard=False,
                kv_block_cap=32,
            )
            self.assertEqual(out_local.shape, q.shape)
            out_local2, _ = attention_interface.sharded_ragged_paged_attention(
                mesh,
                q,
                k,
                v,
                kv_cache,
                md.seq_lens,
                md.block_tables,
                md.query_start_loc,
                md.request_distribution,
                None,
                sm_scale=0.125,
                attention_chunk_size=None,
                rpa_func=fake_rpa,
                shard=False,
                kv_block_cap=32,
            )
            self.assertEqual(out_local2.shape, q.shape)

        # 3. head_dim == 64 with attention_sink
        q64 = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        sink = jnp.ones((2,), dtype=jnp.float32)
        with mock.patch.object(
            attention_interface,
            "ragged_paged_attention_hd64",
            side_effect=fake_rpa,
        ):
            out_hd64, _ = attention_interface.sharded_ragged_paged_attention(
                mesh,
                q64,
                q64,
                q64,
                kv_cache,
                md.seq_lens,
                md.block_tables,
                md.query_start_loc,
                md.request_distribution,
                sink,
                sm_scale=0.125,
                shard=True,
            )
            self.assertEqual(out_hd64.shape, q64.shape)

        # 4. attention_bundled
        bundle = jnp.zeros((4, 2, 16, 2, 2, 128), dtype=jnp.bfloat16)
        with mock.patch.object(
            rpa_v3_kernel,
            "ragged_paged_attention_bundled",
            side_effect=lambda q, k, v, b, *a, **kw: (q, b),
        ):
            b_out, q_out = attention_interface.attention_bundled(
                bundle,
                jnp.int32(0),
                q,
                k,
                v,
                md,
                mesh,
                head_dim_original=None,
                sm_scale=None,
            )
            self.assertEqual(b_out.shape, bundle.shape)
            self.assertEqual(q_out.shape, q.shape)
            b_out2, q_out2 = attention_interface.attention_bundled(
                bundle,
                jnp.int32(1),
                q,
                k,
                v,
                md,
                mesh,
                head_dim_original=128,
                sm_scale=0.125,
            )
            self.assertEqual(b_out2.shape, bundle.shape)
            self.assertEqual(q_out2.shape, q.shape)

        # 5. mla_attention (bf16 and fp8 cache)
        q_tna = jnp.ones((4, 2, 512), dtype=jnp.bfloat16)
        q_rope = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        k_sa = jnp.ones((4, 512), dtype=jnp.bfloat16)
        k_rope = jnp.ones((4, 64), dtype=jnp.bfloat16)
        mla_cache_bf16 = jnp.zeros((4, 8, 2, 640), dtype=jnp.bfloat16)
        mla_cache_fp8 = jnp.zeros((4, 8, 4, 640), dtype=jnp.float8_e4m3fn)
        fake_tuned = SimpleNamespace(
            num_kv_pages_per_block=1,
            num_queries_per_block=16,
            vmem_limit_bytes=1024,
            decode_batch_size=1,
            q_split=1,
        )
        with (
            mock.patch.object(
                attention_interface, "get_tuned_params", return_value=fake_tuned
            ),
            mock.patch.object(
                attention_interface,
                "mla_ragged_paged_attention",
                side_effect=lambda q_hm, qr, ks, kr, c, *a, **kw: (q_hm, c),
            ),
        ):
            c1, o1 = attention_interface.mla_attention(
                q_tna,
                q_rope,
                k_sa,
                k_rope,
                mla_cache_bf16,
                md,
                mesh,
                num_attention_heads=2,
                qk_nope_head_dim=512,
                sm_scale=None,
            )
            self.assertEqual(o1.shape, q_tna.shape)
            self.assertEqual(c1.shape, mla_cache_bf16.shape)
            c2, o2 = attention_interface.mla_attention(
                q_tna,
                q_rope,
                k_sa,
                k_rope,
                mla_cache_fp8,
                md,
                mesh,
                num_attention_heads=2,
                qk_nope_head_dim=512,
                sm_scale=0.5,
            )
            self.assertEqual(o2.shape, q_tna.shape)
            self.assertEqual(c2.shape, mla_cache_fp8.shape)

    def test_sparse_mla_and_dcp_cpu_execution(self):
        from types import SimpleNamespace
        from unittest import mock

        mesh = self._mesh("x")
        ql_nope = jnp.zeros((2, 2, LKV_DIM), jnp.bfloat16)
        q_pe = jnp.zeros((2, 2, ROPE_DIM), jnp.bfloat16)
        kv_c = jnp.zeros((2, LKV_DIM), jnp.float8_e4m3fn)
        k_pe = jnp.zeros((2, ROPE_DIM), jnp.float8_e4m3fn)
        nope_cache = jnp.zeros(NOPE_SPEC.shape, NOPE_SPEC.jax_dtype)
        rope_cache = jnp.zeros(ROPE_SPEC.shape, ROPE_SPEC.jax_dtype)
        topk_idx = jnp.zeros((2, 4), jnp.int32)
        seq_lens = jnp.array([2], jnp.int32)
        block_tables = jnp.zeros((4,), jnp.int32)
        query_start_loc = jnp.array([0, 2], jnp.int32)
        dist = jnp.array([0, 1, 1], jnp.int32)

        with (
            mock.patch.object(
                attention_interface,
                "update_sparse_mla_kv_cache",
                side_effect=lambda nc, rc, *a, **kw: (nc, rc),
            ),
            mock.patch.object(
                attention_interface.mla_dispatch,
                "applies",
                return_value=True,
            ),
            mock.patch.object(
                attention_interface.mla_dispatch,
                "attention_from_parts",
                side_effect=lambda qn, *a, **kw: qn,
            ),
            mock.patch.object(
                attention_interface,
                "sparse_mla_ragged_paged_attention",
                side_effect=lambda q, *a, **kw: q,
            ),
        ):
            _, _, out_md = attention_interface.sparse_mla_attention(
                ql_nope,
                q_pe,
                kv_c,
                k_pe,
                nope_cache,
                rope_cache,
                topk_idx,
                seq_lens,
                block_tables,
                query_start_loc,
                dist,
                mesh,
                NOPE_SPEC,
                ROPE_SPEC,
                allow_masked_dense=True,
            )
            self.assertEqual(out_md.shape, ql_nope.shape)

            _, _, out_sp = attention_interface.sparse_mla_attention(
                ql_nope,
                q_pe,
                kv_c,
                k_pe,
                nope_cache,
                rope_cache,
                topk_idx,
                seq_lens,
                block_tables,
                query_start_loc,
                dist,
                mesh,
                NOPE_SPEC,
                ROPE_SPEC,
                k_scale=1.5,
                allow_masked_dense=False,
            )
            self.assertEqual(out_sp.shape, ql_nope.shape)

        # DCP validation and body execution on CPU via fake 2-rank mesh + stub shard_map
        fake_dcp_mesh = SimpleNamespace(
            axis_names=("dcp",),
            shape={"dcp": 2},
        )
        with self.assertRaisesRegex(ValueError, "dcp_size > 1"):
            attention_interface.sparse_mla_attention_dcp(
                ql_nope,
                q_pe,
                kv_c,
                k_pe,
                nope_cache,
                rope_cache,
                jnp.zeros((2, 4), jnp.int32),
                seq_lens,
                block_tables,
                query_start_loc,
                dist,
                mesh=fake_dcp_mesh,
                nope_spec=NOPE_SPEC,
                rope_spec=ROPE_SPEC,
                dcp_size=1,
                interleave_size=1,
            )
        with self.assertRaisesRegex(ValueError, "requires a mesh"):
            attention_interface.sparse_mla_attention_dcp(
                ql_nope,
                q_pe,
                kv_c,
                k_pe,
                nope_cache,
                rope_cache,
                jnp.zeros((4, 4), jnp.int32),
                seq_lens,
                block_tables,
                query_start_loc,
                dist,
                mesh=SimpleNamespace(axis_names=("x",), shape={"x": 2}),
                nope_spec=NOPE_SPEC,
                rope_spec=ROPE_SPEC,
                dcp_size=2,
                interleave_size=1,
            )
        with self.assertRaisesRegex(ValueError, "does not match mesh axis"):
            attention_interface.sparse_mla_attention_dcp(
                ql_nope,
                q_pe,
                kv_c,
                k_pe,
                nope_cache,
                rope_cache,
                jnp.zeros((8, 4), jnp.int32),
                seq_lens,
                block_tables,
                query_start_loc,
                dist,
                mesh=fake_dcp_mesh,
                nope_spec=NOPE_SPEC,
                rope_spec=ROPE_SPEC,
                dcp_size=4,
                interleave_size=1,
            )
        with self.assertRaisesRegex(ValueError, "one row per"):
            attention_interface.sparse_mla_attention_dcp(
                ql_nope,
                q_pe,
                kv_c,
                k_pe,
                nope_cache,
                rope_cache,
                jnp.zeros((3, 4), jnp.int32),
                seq_lens,
                block_tables,
                query_start_loc,
                dist,
                mesh=fake_dcp_mesh,
                nope_spec=NOPE_SPEC,
                rope_spec=ROPE_SPEC,
                dcp_size=2,
                interleave_size=1,
            )

        with self.assertRaisesRegex(ValueError, "matching NoPE and RoPE layouts"):
            attention_interface.sparse_mla_attention_dcp(
                ql_nope,
                q_pe,
                kv_c,
                k_pe,
                nope_cache,
                rope_cache,
                jnp.zeros((4, 4), jnp.int32),
                seq_lens,
                block_tables,
                query_start_loc,
                dist,
                mesh=fake_dcp_mesh,
                nope_spec=NOPE_SPEC,
                rope_spec=ROPE_SC_SPEC,
                dcp_size=2,
                interleave_size=1,
            )

        with (
            mock.patch.object(
                attention_interface.shard_map,
                "shard_map",
                side_effect=lambda fn, **kw: (
                    lambda ql, qp, kvc, kpe, nc, rc, topk, *rest: fn(
                        ql, qp, kvc, kpe, nc, rc, topk[: ql.shape[0]], *rest
                    )
                ),
            ),
            mock.patch.object(
                attention_interface.streamindex_topk,
                "cp_rank_as_data",
                return_value=jnp.int32(0),
            ),
            mock.patch.object(
                attention_interface,
                "update_sparse_mla_kv_cache_dcp",
                side_effect=lambda nc, rc, *a, **kw: (nc, rc),
            ),
            mock.patch.object(
                attention_interface.sparse_mla_kernel,
                "sparse_ragged_paged_attention",
                side_effect=lambda q, *a, **kw: (
                    q,
                    jnp.zeros((q.shape[0], q.shape[1]), jnp.float32),
                ),
            ),
        ):
            nc_out, rc_out, dcp_out, lse_out = (
                attention_interface.sparse_mla_attention_dcp(
                    ql_nope,
                    q_pe,
                    kv_c,
                    k_pe,
                    nope_cache,
                    rope_cache,
                    jnp.array([[0, -1], [-1, -1], [1, -1], [-1, -1]], jnp.int32),
                    seq_lens,
                    block_tables,
                    query_start_loc,
                    dist,
                    mesh=fake_dcp_mesh,
                    nope_spec=NOPE_SPEC,
                    rope_spec=ROPE_SPEC,
                    k_scale=1.5,
                    dcp_size=2,
                    interleave_size=1,
                )
            )
            self.assertEqual(dcp_out.shape, ql_nope.shape)
            self.assertEqual(lse_out.shape, (2, 2))
            self.assertEqual(nc_out.shape, nope_cache.shape)
            self.assertEqual(rc_out.shape, rope_cache.shape)
