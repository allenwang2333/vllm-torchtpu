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
"""Shared KV numerical tests; native SparseCore tests require a TPU."""

import functools
from unittest import skipUnless

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.deepseek_v4.core_attention import (shared_kv,
                                                              sparse_mla)


def reference(q, kv, mask, acc, swa_sum, m, sinks, scale):
    """Independent per-query attention over only that query's selected keys."""
    q, kv, acc, swa_sum, m, sinks = [
        np.asarray(x, dtype=np.float32)
        for x in (q, kv, acc, swa_sum, m, sinks)
    ]
    output = []
    for t in range(len(q)):
        selected = kv[np.asarray(mask[t], dtype=bool)]
        scores = q[t] @ selected.T * scale
        rowmax = np.max(scores, axis=1) if len(selected) else np.full(
            q.shape[1], -np.inf)
        maximum = np.maximum(np.maximum(rowmax, m[t]), sinks)
        maximum = np.where(np.isneginf(maximum), 0, maximum)
        p = np.exp(scores - maximum[:, None])
        rescale = np.exp(m[t] - maximum)
        denominator = rescale * swa_sum[t] + p.sum(axis=1) + np.exp(sinks -
                                                                    maximum)
        value = rescale[:, None] * acc[t] + p @ selected
        output.append(value /
                      np.where(denominator > 0, denominator, 1)[:, None])
    return np.stack(output)


class SharedKVTest(parameterized.TestCase):

    @parameterized.product(case=((16, 13, 24), (16, 13, 96), (23, 21, 24),
                                 (512, 503, 24)),
                           group_size=(4, 8),
                           chunk_size=(64, 256))
    @skipUnless(jax.default_backend() == "tpu", "SparseCore requires TPU")
    def test_full_operator(self, case, group_size, chunk_size):
        tokens, valid_tokens, keys_per_query = case
        rng = np.random.default_rng(87)
        heads, dim, topk, capacity = 32, 512, 128, 128
        # Three different requests, with a boundary inside a 4-token group.
        ends = [0, 1, 5, valid_tokens]
        page_table = rng.permutation(24).astype(np.int32).reshape(3, 8)
        physical_rows = 24 * 16
        nope = np.zeros((physical_rows, 512), np.uint8)
        nope[:, :448] = rng.integers(0,
                                     80, (physical_rows, 448),
                                     dtype=np.uint8)
        nope[:, 448:455] = 127
        rope = jnp.asarray(rng.standard_normal((physical_rows, 64)),
                           jnp.bfloat16)
        bits = jax.lax.bitcast_convert_type(rope, jnp.uint16)
        rope_bytes = jnp.concatenate([(bits >> 8).astype(jnp.uint8),
                                      (bits & 255).astype(jnp.uint8)],
                                     axis=1)
        cache_nope = jnp.asarray(nope.reshape(24, 16, 4, 128))
        cache_rope = rope_bytes.reshape(24, 4, 4, 128)
        decoded = jnp.concatenate(
            [lax_bitcast_fp8(jnp.asarray(nope[:, :448])), rope], axis=1)
        indices = np.full((tokens, topk), -1, np.int32)
        physical_mask = np.zeros((valid_tokens, physical_rows), np.int32)
        for seq in range(3):
            for t in range(ends[seq], ends[seq + 1]):
                length = 0 if t % 7 == 0 else keys_per_query
                pool = 32 if keys_per_query < 32 else 128
                chosen = rng.permutation(pool)[:length]
                indices[t, :length] = chosen
                physical_mask[t, page_table[seq, chosen // 16] * 16 +
                              chosen % 16] = 1
        q = jnp.asarray(rng.standard_normal((tokens, heads, dim)),
                        jnp.bfloat16)
        acc = jnp.asarray(rng.standard_normal(q.shape), jnp.bfloat16)
        swa_sum = jnp.ones((tokens, heads), jnp.float32) * 4
        m = jnp.zeros((tokens, heads), jnp.float32)
        sinks = jnp.zeros((heads, ), jnp.float32)
        expected = reference(q[:valid_tokens], decoded, physical_mask,
                             acc[:valid_tokens], swa_sum[:valid_tokens],
                             m[:valid_tokens], sinks, dim**-0.5)

        args = (q, cache_nope, cache_rope, jnp.asarray(indices),
                jnp.asarray(page_table.reshape(-1)),
                jnp.asarray(ends, jnp.int32), jnp.array([1, 1, 3], jnp.int32),
                sinks, acc, swa_sum, m)
        kwargs = dict(sm_scale=dim**-0.5,
                      attention_shared_kv_capacity=capacity,
                      attention_shared_kv_group_size=group_size,
                      gather_and_attention_chunk_size=chunk_size,
                      attention_shared_kv_min_queries=0)
        actual = sparse_mla.sparse_ragged_paged_attention(*args, **kwargs)
        np.testing.assert_allclose(actual[:valid_tokens],
                                   expected,
                                   atol=0.006,
                                   rtol=0.025)
        np.testing.assert_array_equal(actual[valid_tokens:], q[valid_tokens:])

    @parameterized.product(group_size=(2, 4, 8),
                           capacity=(16, 32, 128),
                           topk=(13, 16))
    @skipUnless(jax.default_backend() == "tpu", "SparseCore requires TPU")
    def test_union_membership_and_overflow(self, group_size, capacity, topk):
        rng = np.random.default_rng(13)
        indices = np.full((3 * group_size, topk), -1, np.int32)
        for t in range(2 * group_size):
            # Physical addresses differ across requests even when their logical
            # positions match; include overlapping, disjoint, and empty rows.
            base = 0 if t < group_size else 1024 * (t + 1)
            count = min(topk, (0, 7, 13, 16)[t % 4])
            indices[t, :count] = base + rng.permutation(24)[:count]
        fn = jax.jit(
            functools.partial(shared_kv.build_shared_kv_indices,
                              group_size=group_size,
                              capacity=capacity))
        ids, masks, lengths = map(np.asarray, fn(jnp.asarray(indices)))
        for g in range(3):
            rows = indices[g * group_size:(g + 1) * group_size]
            union = sorted({int(v) for row in rows for v in row if v >= 0})
            self.assertEqual(lengths[g], len(union))
            np.testing.assert_array_equal(ids[g, :min(len(union), capacity)],
                                          union[:capacity])
            for t, row in enumerate(rows):
                expected = np.zeros(capacity, np.int32)
                for k, v in enumerate(union[:capacity]):
                    expected[k] = int(v in row)
                np.testing.assert_array_equal(masks[g, t], expected)

    @parameterized.parameters(640, 1024, 1280)
    @skipUnless(jax.default_backend() == "tpu", "SparseCore requires TPU")
    def test_rotated_bit_membership(self, capacity):
        rng = np.random.default_rng(91)
        indices = np.stack([
            rng.choice(896, 512, replace=False) for _ in range(16)
        ]).astype(np.int32)

        @jax.jit
        def run(x):
            *data, lengths = shared_kv.prepare_shared_kv_indices(x,
                                                                 group_size=4)
            out, bits = shared_kv.pack_shared_kv_indices(*data,
                                                         group_size=4,
                                                         capacity=capacity,
                                                         rotate=True,
                                                         padding_modulus=8192)
            return out, bits, lengths

        ids, bits, lengths = map(np.asarray, run(jnp.asarray(indices)))
        for g in range(4):
            rows = indices[g * 4:(g + 1) * 4]
            union = np.unique(rows)
            self.assertEqual(lengths[g], len(union))
            active = bits[g, 0] != 0
            self.assertTrue(((ids[g] >= 0) & (ids[g] < 8192)).all())
            np.testing.assert_array_equal(np.sort(ids[g, active]),
                                          union[:capacity])
            for t in range(4):
                np.testing.assert_array_equal(
                    (bits[g, 0] >> t) & 1, active & np.isin(ids[g], rows[t]))

    @parameterized.product(group_size=(2, 4, 8), heads=(32, 64, 128))
    def test_shared_softmax_and_pv(self, group_size, heads):
        rng = np.random.default_rng(42)
        q = jnp.asarray(rng.standard_normal((group_size, heads, 128)),
                        jnp.bfloat16)
        kv = jnp.asarray(rng.standard_normal((128, 128)), jnp.bfloat16)
        mask = rng.integers(0, 2, (group_size, 128), dtype=np.int32)
        mask[0] = 0
        mask[:, -7:] = 0
        kv = kv.at[-7:].set(jnp.nan)
        acc = jnp.asarray(rng.standard_normal(q.shape), jnp.bfloat16)
        swa_sum = jnp.ones(q.shape[:2], jnp.float32) * 3
        m = jnp.asarray(rng.standard_normal(q.shape[:2]), jnp.float32)
        # Include a disabled sink and a large sink (stable softmax required).
        sinks = jnp.linspace(-10, 100, heads).at[0].set(-jnp.inf)
        actual = jax.jit(
            functools.partial(shared_kv.shared_attention,
                              sm_scale=128**-0.5))(q, kv, jnp.asarray(mask),
                                                   acc, swa_sum, m, sinks)
        expected = reference(q, kv, mask, acc, swa_sum, m, sinks, 128**-0.5)
        np.testing.assert_allclose(actual, expected, atol=0.003, rtol=0.015)
        self.assertTrue(np.isfinite(actual).all())

    def test_completely_empty_attention(self):
        q = jnp.ones((4, 32, 128), jnp.bfloat16)
        result = shared_kv.shared_attention(q,
                                            jnp.full((128, 128), jnp.nan,
                                                     jnp.bfloat16),
                                            jnp.zeros((4, 128), jnp.int32),
                                            jnp.zeros_like(q),
                                            jnp.zeros((4, 32)),
                                            jnp.full((4, 32), -jnp.inf),
                                            jnp.full((32, ), -jnp.inf),
                                            sm_scale=1.0)
        np.testing.assert_array_equal(result, np.zeros(q.shape))

    @parameterized.product(group_size=(2, 4, 8),
                           heads=(32, 64),
                           capacity=(128, 1024))
    def test_pallas_shared_kernel(self, group_size, heads, capacity):
        rng = np.random.default_rng(29)
        tokens = 16
        groups = tokens // group_size
        q = jnp.asarray(rng.standard_normal((tokens, heads, 512)),
                        jnp.bfloat16)
        nope = np.zeros((groups, capacity, 512), np.uint8)
        nope[..., :448] = rng.integers(0,
                                       80, (groups, capacity, 448),
                                       dtype=np.uint8)
        nope[..., 448:455] = 127  # e8m0 1.0 for each 64-element block
        nope[:, -5:, 448:455] = 255  # invalid cached rows decode to NaN
        nope = jnp.asarray(nope.reshape(groups, capacity, 4, 128))
        rope = jnp.asarray(rng.standard_normal((groups, capacity, 64)),
                           jnp.bfloat16)
        mask = rng.integers(0,
                            2, (groups, group_size, capacity),
                            dtype=np.int32)
        mask[:, :, -5:] = 0
        mask[0, 0] = 0
        acc = jnp.asarray(rng.standard_normal(q.shape), jnp.bfloat16)
        swa_sum = jnp.ones((tokens, heads), jnp.float32) * 4
        m = jnp.zeros((tokens, heads), jnp.float32)
        sinks = jnp.zeros((heads, ), jnp.float32)
        expected = []
        for g in range(groups):
            rows = slice(g * group_size, (g + 1) * group_size)
            raw = nope[g].reshape(capacity, 512)
            decoded = lax_bitcast_fp8(raw[:, :448])
            kv = jnp.concatenate([decoded, rope[g]], axis=1)
            expected.append(
                reference(q[rows], kv, mask[g], acc[rows], swa_sum[rows],
                          m[rows], sinks, 512**-0.5))
        actual = sparse_mla._run_shared_attention(
            q,
            nope,
            rope,
            jnp.asarray(mask),
            sinks,
            acc,
            jnp.pad(swa_sum, ((0, 0), (0, 128 - heads))),
            jnp.pad(m, ((0, 0), (0, 128 - heads))),
            group_size=group_size,
            batch_size=16,
            sm_scale=512**-0.5,
            interpret=jax.default_backend() == "cpu")
        np.testing.assert_allclose(actual,
                                   np.concatenate(expected),
                                   atol=0.004,
                                   rtol=0.02)
        self.assertTrue(np.isfinite(actual).all())


def lax_bitcast_fp8(raw):
    return jax.lax.bitcast_convert_type(raw,
                                        jnp.float8_e4m3fn).astype(jnp.bfloat16)


if __name__ == "__main__":
    absltest.main()
