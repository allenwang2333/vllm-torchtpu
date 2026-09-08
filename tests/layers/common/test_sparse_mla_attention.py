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
"""Functional tests for the GLM-5.2 sparse (DSA) MLA attention path.

The kernel is validated against a pure-numpy MLA reference: `topk_indices`
are synthesized (causal or random subsets), so the reference knows exactly
which KV entries each token attends to and computes softmax(q @ k^T) @ v in
float32 over the same fp8-rounded latents the kernel reads. With causal
arange indices and kv_len <= topk this is exactly dense MLA attention, which
makes the first cases a dense-parity check of the full path
(insert -> SparseCore gather -> attention kernel).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME, _partition_winners_by_owner, cp_global_to_local)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (KVCacheLayout,
                                                      KVCacheType,
                                                      SparseMLAKVCacheSpec)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.common import attention_interface

P = jax.sharding.PartitionSpec

LKV_DIM = 512
ROPE_DIM = 64
NUM_HEADS = 16
PAGE_SIZE = 32
PAGES_PER_SEQ = 4
TOTAL_PAGES = 16
TOKEN_PAD = 16  # attention kernel batch size; token count must be a multiple

# Layouts pinned explicitly rather than read from the environment: these
# tests check the exact geometry dsa_gather is written against.
KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
NOPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.NOPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, LKV_DIM, KV_PACKING)
ROPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.ROPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, ROPE_DIM, KV_PACKING)


def _empty_pair():
    """Zeroed (nope, rope) caches in dsa_gather's native tiled layouts."""
    return (jnp.zeros(NOPE_SPEC.shape, NOPE_SPEC.jax_dtype),
            jnp.zeros(ROPE_SPEC.shape, ROPE_SPEC.jax_dtype))


def _quantize_fp8(x: np.ndarray, k_scale: float) -> jax.Array:
    return jnp.asarray(x / k_scale).astype(jnp.float8_e4m3fn)


def _dequantize(x: jax.Array, k_scale: float) -> np.ndarray:
    return np.asarray(x.astype(jnp.float32)) * k_scale


def _causal_topk(positions: list[int], topk: int) -> np.ndarray:
    """topk_indices row per token: [0..pos] then -1 padding."""
    rows = np.full((len(positions), topk), -1, np.int32)
    for i, pos in enumerate(positions):
        assert pos + 1 <= topk
        rows[i, :pos + 1] = np.arange(pos + 1)
    return rows


class SparseMlaAttentionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(1234)
        self.k_scale = 1.5
        self.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)

    def _require_tpu(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA kernel requires a TPU (SparseCore).")

    def _make_mesh(self):
        return jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x", ))

    def _random_latents(self, n):
        kv_c = self.rng.standard_normal((n, LKV_DIM)).astype(np.float32)
        k_pe = self.rng.standard_normal((n, ROPE_DIM)).astype(np.float32)
        return kv_c, k_pe

    def _random_queries(self, num_tokens):
        ql_nope = jnp.asarray(self.rng.standard_normal(
            (num_tokens, NUM_HEADS, LKV_DIM)).astype(np.float32),
                              dtype=jnp.bfloat16)
        q_pe = jnp.asarray(self.rng.standard_normal(
            (num_tokens, NUM_HEADS, ROPE_DIM)).astype(np.float32),
                           dtype=jnp.bfloat16)
        return ql_nope, q_pe

    def _block_tables(self, num_seqs):
        """Distinct physical pages per sequence, permuted."""
        perm = self.rng.permutation(TOTAL_PAGES)[:num_seqs * PAGES_PER_SEQ]
        return jnp.asarray(perm.reshape(num_seqs, PAGES_PER_SEQ).reshape(-1),
                           dtype=jnp.int32)

    def _empty_cache(self):
        return _empty_pair()

    def _call(self, kv_cache, ql_nope, q_pe, kv_c_fp8, k_pe_fp8, topk_rows,
              seq_lens, query_start_loc, distribution, block_tables, mesh):
        nope_cache, rope_cache, output = \
            attention_interface.sparse_mla_attention(
            ql_nope,
            q_pe,
            kv_c_fp8,
            k_pe_fp8,
            kv_cache[0],
            kv_cache[1],
            jnp.asarray(topk_rows, dtype=jnp.int32),
            jnp.asarray(seq_lens, dtype=jnp.int32),
            block_tables,
            jnp.asarray(query_start_loc, dtype=jnp.int32),
            jnp.asarray(distribution, dtype=jnp.int32),
            mesh,
            NOPE_SPEC,
            ROPE_SPEC,
            sm_scale=self.sm_scale,
            k_scale=self.k_scale,
        )
        return (nope_cache, rope_cache), output

    def _reference(self, ql_nope, q_pe, topk_rows, q_lens, query_start_loc,
                   kv_c_deq, k_pe_deq):
        """f32 MLA attention over exactly the selected (dequantized) KVs.

        kv_c_deq/k_pe_deq: per-sequence arrays [seq_len, dim] of the values
        the cache actually holds after fp8 rounding.
        """
        num_tokens = ql_nope.shape[0]
        q = np.concatenate([
            np.asarray(ql_nope.astype(jnp.float32)),
            np.asarray(q_pe.astype(jnp.float32))
        ], -1)  # [T, N, 576]
        out = np.zeros((num_tokens, NUM_HEADS, LKV_DIM), np.float32)
        mask = np.zeros(num_tokens, bool)
        for s in range(len(q_lens)):
            for local in range(q_lens[s]):
                t = query_start_loc[s] + local
                sel = topk_rows[t][topk_rows[t] >= 0]
                keys = np.concatenate([kv_c_deq[s][sel], k_pe_deq[s][sel]],
                                      -1)  # [n, 576]
                scores = q[t] @ keys.T * self.sm_scale  # [N, n]
                scores -= scores.max(-1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(-1, keepdims=True)
                out[t] = probs @ kv_c_deq[s][sel]
                mask[t] = True
        return out, mask

    def _check(self, output, expected, valid_mask):
        got = np.asarray(output.astype(jnp.float32))[valid_mask]
        np.testing.assert_allclose(got,
                                   expected[valid_mask],
                                   rtol=2e-2,
                                   atol=2e-2)

    @parameterized.named_parameters(
        dict(testcase_name="single_prefill", q_lens=[48], topk=64),
        dict(testcase_name="mixed_prefill", q_lens=[40, 24], topk=64),
        dict(testcase_name="padded_tokens", q_lens=[20, 16], topk=64),
    )
    def test_causal_matches_dense_reference(self, q_lens, topk):
        """Causal-arange topk == dense MLA; validates the full path."""
        self._require_tpu()
        num_seqs = len(q_lens)
        total = sum(q_lens)
        num_tokens = math.ceil(total / TOKEN_PAD) * TOKEN_PAD
        query_start_loc = np.concatenate([[0], np.cumsum(q_lens)])

        kv_c_deq, k_pe_deq = [], []
        kv_c_rows, k_pe_rows, positions = [], [], []
        for s, q_len in enumerate(q_lens):
            kv_c, k_pe = self._random_latents(q_len)
            kv_c_fp8 = _quantize_fp8(kv_c, self.k_scale)
            k_pe_fp8 = _quantize_fp8(k_pe, self.k_scale)
            kv_c_rows.append(kv_c_fp8)
            k_pe_rows.append(k_pe_fp8)
            kv_c_deq.append(_dequantize(kv_c_fp8, self.k_scale))
            k_pe_deq.append(_dequantize(k_pe_fp8, self.k_scale))
            positions.extend(range(q_len))

        pad = num_tokens - total
        kv_c_fp8 = jnp.concatenate(
            kv_c_rows + [jnp.zeros((pad, LKV_DIM), jnp.float8_e4m3fn)])
        k_pe_fp8 = jnp.concatenate(
            k_pe_rows + [jnp.zeros((pad, ROPE_DIM), jnp.float8_e4m3fn)])
        # Padding tokens still need >= 1 valid topk entry (kernel contract).
        topk_rows = np.concatenate(
            [_causal_topk(positions, topk),
             _causal_topk([0] * pad, topk)])

        ql_nope, q_pe = self._random_queries(num_tokens)
        mesh = self._make_mesh()
        _, output = self._call(self._empty_cache(), ql_nope, q_pe, kv_c_fp8,
                               k_pe_fp8, topk_rows, q_lens, query_start_loc,
                               [0, num_seqs, num_seqs],
                               self._block_tables(num_seqs), mesh)

        expected, valid = self._reference(ql_nope, q_pe, topk_rows, q_lens,
                                          query_start_loc, kv_c_deq, k_pe_deq)
        self._check(output, expected, valid)

    def test_sparse_subset_selection(self):
        """Random (non-causal) subsets: topk selection is actually honored."""
        self._require_tpu()
        q_len, topk = 32, 16
        kv_c, k_pe = self._random_latents(q_len)
        kv_c_fp8 = _quantize_fp8(kv_c, self.k_scale)
        k_pe_fp8 = _quantize_fp8(k_pe, self.k_scale)

        topk_rows = np.full((q_len, topk), -1, np.int32)
        for pos in range(q_len):
            n_sel = min(pos + 1, 8)
            sel = self.rng.choice(pos + 1, size=n_sel, replace=False)
            sel[0] = pos  # always include self
            topk_rows[pos, :n_sel] = np.sort(sel)

        ql_nope, q_pe = self._random_queries(q_len)
        mesh = self._make_mesh()
        _, output = self._call(self._empty_cache(), ql_nope, q_pe, kv_c_fp8,
                               k_pe_fp8, topk_rows, [q_len], [0, q_len],
                               [0, 1, 1], self._block_tables(1), mesh)

        expected, valid = self._reference(
            ql_nope, q_pe, topk_rows, [q_len], [0, q_len],
            [_dequantize(kv_c_fp8, self.k_scale)],
            [_dequantize(k_pe_fp8, self.k_scale)])
        self._check(output, expected, valid)

    def test_prefill_then_decode(self):
        """Two steps: decode gathers KVs inserted by an earlier call."""
        self._require_tpu()
        prior_lens, topk = [24, 40], 128
        num_seqs = len(prior_lens)
        mesh = self._make_mesh()
        block_tables = self._block_tables(num_seqs)

        kv_c_deq, k_pe_deq = [], []
        kv_cache = self._empty_cache()

        # Step 1: prefill both sequences (output ignored).
        kv_c_rows, k_pe_rows, positions = [], [], []
        for q_len in prior_lens:
            kv_c, k_pe = self._random_latents(q_len)
            kv_c_rows.append(_quantize_fp8(kv_c, self.k_scale))
            k_pe_rows.append(_quantize_fp8(k_pe, self.k_scale))
            positions.extend(range(q_len))
        total = sum(prior_lens)
        query_start_loc = np.concatenate([[0], np.cumsum(prior_lens)])
        ql_nope, q_pe = self._random_queries(total)
        kv_cache, _ = self._call(kv_cache, ql_nope, q_pe,
                                 jnp.concatenate(kv_c_rows),
                                 jnp.concatenate(k_pe_rows),
                                 _causal_topk(positions, topk), prior_lens,
                                 query_start_loc, [0, num_seqs, num_seqs],
                                 block_tables, mesh)
        for s in range(num_seqs):
            kv_c_deq.append(_dequantize(kv_c_rows[s], self.k_scale))
            k_pe_deq.append(_dequantize(k_pe_rows[s], self.k_scale))

        # Step 2: one decode token per sequence, attending causally over the
        # prefilled context plus itself.
        num_tokens = TOKEN_PAD
        kv_c_new, k_pe_new = self._random_latents(num_seqs)
        kv_c_fp8 = _quantize_fp8(kv_c_new, self.k_scale)
        k_pe_fp8 = _quantize_fp8(k_pe_new, self.k_scale)
        pad = num_tokens - num_seqs
        topk_rows = np.concatenate([
            _causal_topk(prior_lens, topk),  # decode pos == prior_len
            _causal_topk([0] * pad, topk),
        ])
        ql_nope, q_pe = self._random_queries(num_tokens)
        seq_lens = [n + 1 for n in prior_lens]
        _, output = self._call(
            kv_cache, ql_nope, q_pe,
            jnp.concatenate(
                [kv_c_fp8,
                 jnp.zeros((pad, LKV_DIM), jnp.float8_e4m3fn)]),
            jnp.concatenate(
                [k_pe_fp8,
                 jnp.zeros((pad, ROPE_DIM),
                           jnp.float8_e4m3fn)]), topk_rows, seq_lens,
            list(range(num_seqs + 1)), [num_seqs, num_seqs, num_seqs],
            block_tables, mesh)

        for s in range(num_seqs):
            kv_c_deq[s] = np.concatenate(
                [kv_c_deq[s],
                 _dequantize(kv_c_fp8[s:s + 1], self.k_scale)])
            k_pe_deq[s] = np.concatenate(
                [k_pe_deq[s],
                 _dequantize(k_pe_fp8[s:s + 1], self.k_scale)])
        expected, valid = self._reference(ql_nope, q_pe, topk_rows,
                                          [1] * num_seqs,
                                          list(range(num_seqs + 1)), kv_c_deq,
                                          k_pe_deq)
        self._check(output, expected, valid)


# =====================================================================
# DCP: log-sum-exp export and the cross-shard merge.
# =====================================================================

DCP_TOKENS = 64  # multiple of TOKEN_PAD, fits PAGES_PER_SEQ * PAGE_SIZE
DCP_TOPK = 64
# The DCP shard rule from streamindex_topk.cp_local_to_global: compressed
# position `p` is owned by rank `(p // interleave_c) % dcp_size`. Anything
# below `dsa_gather.TILE_SUBROWS` would split a rope tile across ranks.
DCP_INTERLEAVE_C = 4


def _shard_topk_by_owner(topk_rows: np.ndarray, dcp_size: int,
                         interleave_c: int):
    """Partition each token's top-k list by the position's owning DCP rank.

    Returns `(shards, counts)` with `shards[r]` shaped like `topk_rows`.
    The cache stays replicated here, so entries keep their *global* indices
    and no local-index remap is involved: this isolates the LSE merge, which
    is the only part the kernel is responsible for.

    A rank that owns none of a token's top-k still has to be handed a row with
    one valid entry (the kernel derives `kv_len` from the `-1` tail and
    requires `kv_len >= 1`). Such a row computes a real but meaningless
    softmax, so `counts` records the emptiness and the merge masks it out.
    """
    num_tokens, topk = topk_rows.shape
    shards = np.full((dcp_size, num_tokens, topk), -1, np.int32)
    counts = np.zeros((dcp_size, num_tokens), np.int32)
    for t in range(num_tokens):
        row = topk_rows[t][topk_rows[t] >= 0]
        for r in range(dcp_size):
            sel = row[(row // interleave_c) % dcp_size == r]
            counts[r, t] = sel.size
            shards[r, t, :sel.size] = sel
            if sel.size == 0:
                shards[r, t, 0] = 0  # dummy; masked by `counts` at merge time
    return shards, counts


def _merge_partials(outs: np.ndarray, lses: np.ndarray,
                    counts: np.ndarray) -> np.ndarray:
    """`o = sum_r softmax_r(lse) * o_r` over the DCP axis, in f32.

    outs: f32[D, T, N, Dh] normalized per-shard outputs.
    lses: f32[D, T, N] per-shard log-sum-exp.
    counts: i32[D, T] valid entries per shard, 0 meaning "did not participate".
    """
    lses = np.where(counts[:, :, None] > 0, lses, -np.inf)
    shift = lses.max(0, keepdims=True)
    # A token no shard owns would give 0/0; it cannot occur for a real top-k
    # list but the guard keeps the failure a zero rather than a NaN.
    all_empty = ~np.isfinite(shift)
    shift = np.where(all_empty, 0.0, shift)
    weights = np.exp(lses - shift[0])  # [D, T, N]
    denom = weights.sum(0)
    weights = np.where(denom[None] > 0, weights / np.where(denom == 0, 1,
                                                           denom)[None], 0.0)
    return (weights[..., None] * outs).sum(0)


class SparseMlaLseTest(parameterized.TestCase):
    """`return_lse` and the KV-sharded (DCP) merge it exists to enable.

    Unlike the PCP class above, DCP splits a single token's top-k across
    ranks, so each rank sees a *partial* softmax and parity can only be
    approximate. The tolerance here is set by the kernel's bf16 output dtype,
    not by the merge: the merge itself is exact in f32 given exact partials.
    """

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(20260906)
        self.k_scale = 1.5
        self.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA kernel requires a TPU (SparseCore).")

    def _prefilled(self):
        """One prefilled sequence: populated caches plus kernel-ready inputs."""
        kv_c = self.rng.standard_normal(
            (DCP_TOKENS, LKV_DIM)).astype(np.float32)
        k_pe = self.rng.standard_normal(
            (DCP_TOKENS, ROPE_DIM)).astype(np.float32)
        kv_c_fp8 = _quantize_fp8(kv_c, self.k_scale)
        k_pe_fp8 = _quantize_fp8(k_pe, self.k_scale)
        ql_nope = jnp.asarray(self.rng.standard_normal(
            (DCP_TOKENS, NUM_HEADS, LKV_DIM)).astype(np.float32),
                              dtype=jnp.bfloat16)
        q_pe = jnp.asarray(self.rng.standard_normal(
            (DCP_TOKENS, NUM_HEADS, ROPE_DIM)).astype(np.float32),
                           dtype=jnp.bfloat16)
        topk_rows = _causal_topk(list(range(DCP_TOKENS)), DCP_TOPK)
        block_tables = jnp.asarray(
            self.rng.permutation(TOTAL_PAGES)[:PAGES_PER_SEQ], dtype=jnp.int32)
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x", ))

        # Run the layer once purely to insert this step's rows into the caches;
        # every assertion below then drives the kernel directly, because
        # `return_lse` is a kernel-level flag.
        nope_cache, rope_cache, _ = attention_interface.sparse_mla_attention(
            ql_nope, q_pe, kv_c_fp8, k_pe_fp8, *_empty_pair(),
            jnp.asarray(topk_rows, jnp.int32),
            jnp.asarray([DCP_TOKENS], jnp.int32), block_tables,
            jnp.asarray([0, DCP_TOKENS], jnp.int32),
            jnp.asarray([0, 1, 1], jnp.int32), mesh, NOPE_SPEC, ROPE_SPEC,
            sm_scale=self.sm_scale, k_scale=self.k_scale)

        return dict(
            q=jnp.concatenate([ql_nope, q_pe], axis=-1),
            nope_cache=nope_cache,
            rope_cache=rope_cache,
            topk_rows=topk_rows,
            block_tables=block_tables,
            cu_q_lens=jnp.asarray([0, DCP_TOKENS], jnp.int32),
            distribution=jnp.asarray([0, 1, 1], jnp.int32),
            kv_c_deq=_dequantize(kv_c_fp8, self.k_scale),
            k_pe_deq=_dequantize(k_pe_fp8, self.k_scale),
        )

    def _attend(self, inp, topk_rows, return_lse):
        return sparse_mla_kernel.sparse_ragged_paged_attention(
            inp["q"],
            inp["nope_cache"],
            inp["rope_cache"],
            jnp.asarray(topk_rows, jnp.int32),
            inp["block_tables"],
            inp["cu_q_lens"],
            inp["distribution"],
            sm_scale=self.sm_scale,
            k_scale=self.k_scale,
            return_lse=return_lse,
        )

    def test_return_lse_does_not_perturb_the_output(self):
        """The flag is additive: the attention result must be bit-identical."""
        inp = self._prefilled()
        base = self._attend(inp, inp["topk_rows"], False)
        out, lse = self._attend(inp, inp["topk_rows"], True)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(base))
        self.assertEqual(lse.dtype, jnp.float32)
        self.assertEqual(lse.shape, (DCP_TOKENS, NUM_HEADS))

    def test_lse_matches_a_reference_logsumexp(self):
        inp = self._prefilled()
        _, lse = self._attend(inp, inp["topk_rows"], True)

        keys = np.concatenate([inp["kv_c_deq"], inp["k_pe_deq"]], -1)
        q = np.asarray(inp["q"].astype(jnp.float32))
        expected = np.zeros((DCP_TOKENS, NUM_HEADS), np.float32)
        for t in range(DCP_TOKENS):
            sel = inp["topk_rows"][t][inp["topk_rows"][t] >= 0]
            scores = q[t] @ keys[sel].T * self.sm_scale  # [N, n]
            m = scores.max(-1)
            expected[t] = m + np.log(np.exp(scores - m[:, None]).sum(-1))

        np.testing.assert_allclose(np.asarray(lse),
                                   expected,
                                   rtol=2e-2,
                                   atol=2e-2)

    @parameterized.named_parameters(
        dict(testcase_name="d2", dcp_size=2),
        dict(testcase_name="d4", dcp_size=4),
        dict(testcase_name="d8", dcp_size=8),
    )
    def test_merging_position_sharded_partials_reproduces_the_whole(
            self, dcp_size):
        """The core DCP claim, at the kernel boundary.

        Splitting a token's top-k across ranks by KV *position* and merging the
        per-rank `(output, lse)` pairs must reproduce the unsplit result. With
        `_causal_topk` the early tokens own fewer entries than there are ranks,
        so the empty-shard path is exercised too.
        """
        inp = self._prefilled()
        whole = np.asarray(self._attend(inp, inp["topk_rows"],
                                        False).astype(jnp.float32))

        shards, counts = _shard_topk_by_owner(inp["topk_rows"], dcp_size,
                                              DCP_INTERLEAVE_C)
        self.assertTrue(
            (counts.sum(0) == (inp["topk_rows"] >= 0).sum(-1)).all(),
            "the shard split must partition the top-k list exactly")

        outs, lses = [], []
        for r in range(dcp_size):
            out_r, lse_r = self._attend(inp, shards[r], True)
            outs.append(np.asarray(out_r.astype(jnp.float32)))
            lses.append(np.asarray(lse_r))

        merged = _merge_partials(np.stack(outs), np.stack(lses), counts)
        np.testing.assert_allclose(merged, whole, rtol=2e-2, atol=2e-2)

    def test_an_empty_shard_contributes_nothing(self):
        """A rank owning none of a token's top-k must not shift the merge.

        Its `lse` is meaningless (the kernel was handed a dummy entry), so the
        merge has to drop it on `counts`, not on the value of `lse`.
        """
        inp = self._prefilled()
        whole = np.asarray(self._attend(inp, inp["topk_rows"],
                                        False).astype(jnp.float32))
        out, lse = self._attend(inp, inp["topk_rows"], True)

        # Two shards: one holds the entire list, the other holds nothing.
        empty_rows = np.full_like(inp["topk_rows"], -1)
        empty_rows[:, 0] = 0
        out_e, lse_e = self._attend(inp, empty_rows, True)
        counts = np.stack([(inp["topk_rows"] >= 0).sum(-1),
                           np.zeros(DCP_TOKENS, np.int32)])

        merged = _merge_partials(
            np.stack([
                np.asarray(out.astype(jnp.float32)),
                np.asarray(out_e.astype(jnp.float32))
            ]), np.stack([np.asarray(lse), np.asarray(lse_e)]), counts)
        np.testing.assert_allclose(merged, whole, rtol=2e-2, atol=2e-2)


# =====================================================================
# DCP: attention over a position-sharded latent cache.
#
# Step 3 (`streamindex_topk_dcp`) proves the *selection* survives sharding
# the KV positions. This proves the *attention* does: each rank attends
# only over the winners it holds, and the LSE merge puts the pieces back
# together into the same answer the unsharded path gives.
# =====================================================================

DCP_SEQ = 256  # positions in the (single) sequence
DCP_ATTN_TOPK = 64


def _dcp_caches(dcp_size):
    """Zeroed caches whose *global* page axis is `dcp_size` local shards."""
    return tuple(
        jnp.zeros((dcp_size * spec.shape[0], ) + spec.shape[1:], jnp.uint8)
        for spec in (NOPE_SPEC, ROPE_SPEC))


def _random_causal_topk(rng, num_tokens, topk):
    """Token t attends to a random subset of [0, t], as a `-1`-padded prefix."""
    rows = np.full((num_tokens, topk), -1, np.int32)
    for t in range(num_tokens):
        n = min(topk, t + 1)
        rows[t, :n] = np.sort(rng.choice(t + 1, size=n, replace=False))
    return rows


class SparseMlaDcpTest(parameterized.TestCase):
    """`sparse_mla_attention_dcp` + `merge_dcp_lse_partials`, end to end.

    The comparison is against the same pure-numpy f32 MLA reference the
    unsharded tests use, not against the unsharded kernel, so a layout bug
    that happened to be symmetric between the two paths cannot hide.
    """

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(20260907)
        self.k_scale = 1.5
        self.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)

    def _require_devices(self, n):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA kernel requires a TPU (SparseCore).")
        if len(jax.devices()) < n:
            self.skipTest(f"needs {n} devices, have {len(jax.devices())}")

    def _inputs(self):
        kv_c = self.rng.standard_normal((DCP_SEQ, LKV_DIM)).astype(np.float32)
        k_pe = self.rng.standard_normal((DCP_SEQ, ROPE_DIM)).astype(np.float32)
        kv_c_fp8 = _quantize_fp8(kv_c, self.k_scale)
        k_pe_fp8 = _quantize_fp8(k_pe, self.k_scale)
        ql_nope = jnp.asarray(self.rng.standard_normal(
            (DCP_SEQ, NUM_HEADS, LKV_DIM)).astype(np.float32),
                              dtype=jnp.bfloat16)
        q_pe = jnp.asarray(self.rng.standard_normal(
            (DCP_SEQ, NUM_HEADS, ROPE_DIM)).astype(np.float32),
                           dtype=jnp.bfloat16)
        return dict(
            kv_c_fp8=kv_c_fp8,
            k_pe_fp8=k_pe_fp8,
            ql_nope=ql_nope,
            q_pe=q_pe,
            topk=_random_causal_topk(self.rng, DCP_SEQ, DCP_ATTN_TOPK),
            kv_c_deq=_dequantize(kv_c_fp8, self.k_scale),
            k_pe_deq=_dequantize(k_pe_fp8, self.k_scale),
            seq_lens=jnp.asarray([DCP_SEQ], jnp.int32),
            cu_q_lens=jnp.asarray([0, DCP_SEQ], jnp.int32),
            distribution=jnp.asarray([0, 1, 1], jnp.int32),
        )

    def _numpy_reference(self, inp):
        """f32 MLA over exactly the selected, fp8-rounded latents."""
        q = np.concatenate([
            np.asarray(inp["ql_nope"].astype(jnp.float32)),
            np.asarray(inp["q_pe"].astype(jnp.float32))
        ], -1)
        keys = np.concatenate([inp["kv_c_deq"], inp["k_pe_deq"]], -1)
        out = np.zeros((DCP_SEQ, NUM_HEADS, LKV_DIM), np.float32)
        for t in range(DCP_SEQ):
            sel = inp["topk"][t][inp["topk"][t] >= 0]
            scores = q[t] @ keys[sel].T * self.sm_scale
            scores -= scores.max(-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(-1, keepdims=True)
            out[t] = probs @ inp["kv_c_deq"][sel]
        return out

    def _run_dcp(self, inp, dcp_size, interleave_size):
        """Shard, attend, merge. Returns (merged_output, dropped)."""
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp_size]),
                                 (DCP_AXIS_NAME, ))
        sharded = jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME))

        lists, dropped = _partition_winners_by_owner(
            jnp.asarray(inp["topk"], jnp.int32), dcp_size, interleave_size,
            DCP_ATTN_TOPK)
        # [T, D, C] -> [D, T, C] -> [D*T, C], the rank-major stacking
        # `streamindex_topk_dcp` produces and the shard_map expects.
        local_topk = jnp.asarray(lists).transpose(1, 0, 2).reshape(
            dcp_size * DCP_SEQ, DCP_ATTN_TOPK)

        # One virtual page spans `PAGE_SIZE * dcp_size` global positions.
        vpages = math.ceil(DCP_SEQ / (PAGE_SIZE * dcp_size))
        block_tables = jnp.asarray(
            self.rng.permutation(NOPE_SPEC.num_pages)[:vpages], jnp.int32)

        nope, rope = (jax.device_put(c, sharded) for c in _dcp_caches(dcp_size))
        _, _, out, lse = attention_interface.sparse_mla_attention_dcp(
            inp["ql_nope"],
            inp["q_pe"],
            inp["kv_c_fp8"],
            inp["k_pe_fp8"],
            nope,
            rope,
            jax.device_put(local_topk, sharded),
            inp["seq_lens"],
            block_tables,
            inp["cu_q_lens"],
            inp["distribution"],
            mesh,
            NOPE_SPEC,
            ROPE_SPEC,
            sm_scale=self.sm_scale,
            k_scale=self.k_scale,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
        )
        merged = attention_interface.merge_dcp_lse_partials(
            out.reshape(dcp_size, DCP_SEQ, NUM_HEADS, LKV_DIM),
            lse.reshape(dcp_size, DCP_SEQ, NUM_HEADS))
        return np.asarray(merged), int(dropped)

    @parameterized.named_parameters(
        dict(testcase_name="d2_c4", dcp_size=2, interleave_size=4),
        dict(testcase_name="d4_c4", dcp_size=4, interleave_size=4),
        dict(testcase_name="d8_c4", dcp_size=8, interleave_size=4),
        dict(testcase_name="d2_c8", dcp_size=2, interleave_size=8),
    )
    def test_dcp_merge_matches_the_unsharded_reference(self, dcp_size,
                                                       interleave_size):
        self._require_devices(dcp_size)
        inp = self._inputs()
        merged, dropped = self._run_dcp(inp, dcp_size, interleave_size)
        self.assertEqual(dropped, 0)
        np.testing.assert_allclose(merged,
                                   self._numpy_reference(inp),
                                   rtol=2e-2,
                                   atol=2e-2)

    def test_dcp_write_mask_partitions_the_cache(self):
        """Each rank must hold exactly the positions it owns -- no more.

        If the ownership mask in the scatter disagreed with the interleave the
        readers assume, the merge would still produce *a* number; it would just
        be one that double-counted some positions and dropped others. Counting
        written slots catches that directly.
        """
        self._require_devices(4)
        dcp_size, interleave_size = 4, 4
        inp = self._inputs()
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp_size]),
                                 (DCP_AXIS_NAME, ))
        sharded = jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME))
        vpages = math.ceil(DCP_SEQ / (PAGE_SIZE * dcp_size))
        block_tables = jnp.arange(vpages, dtype=jnp.int32)
        nope, rope = (jax.device_put(c, sharded) for c in _dcp_caches(dcp_size))

        lists, _ = _partition_winners_by_owner(
            jnp.asarray(inp["topk"], jnp.int32), dcp_size, interleave_size,
            DCP_ATTN_TOPK)
        local_topk = jnp.asarray(lists).transpose(1, 0, 2).reshape(
            dcp_size * DCP_SEQ, DCP_ATTN_TOPK)

        new_nope, _, _, _ = attention_interface.sparse_mla_attention_dcp(
            inp["ql_nope"], inp["q_pe"], inp["kv_c_fp8"], inp["k_pe_fp8"],
            nope, rope, jax.device_put(local_topk, sharded), inp["seq_lens"],
            block_tables, inp["cu_q_lens"], inp["distribution"], mesh,
            NOPE_SPEC, ROPE_SPEC, sm_scale=self.sm_scale,
            k_scale=self.k_scale, dcp_size=dcp_size,
            interleave_size=interleave_size)

        # A written latent row is all-zero with probability ~0, so the
        # nonzero (page, slot) pairs are exactly the written ones.
        per_rank = np.asarray(new_nope).reshape(dcp_size, NOPE_SPEC.num_pages,
                                                NOPE_SPEC.page_size, -1)
        written = (per_rank != 0).any(-1)

        # `block_tables` is the identity here, so the expected slots follow
        # straight from the layout: rank r holds global position p iff
        # `(p // c) % D == r`, at local index `cp_global_to_local(p)`.
        expected = np.zeros_like(written)
        for pos in range(DCP_SEQ):
            r = (pos // interleave_size) % dcp_size
            local = int(cp_global_to_local(jnp.int32(pos), dcp_size,
                                            interleave_size))
            expected[r, local // NOPE_SPEC.page_size,
                     local % NOPE_SPEC.page_size] = True
        np.testing.assert_array_equal(written.sum((1, 2)),
                                      np.full(dcp_size, DCP_SEQ // dcp_size))
        np.testing.assert_array_equal(written, expected)

    def test_a_rank_owning_nothing_is_dropped_by_the_merge(self):
        """The empty-shard path, isolated.

        A rank with no winners for a token is still handed a dummy key, so its
        partial output is a real number that must be excluded on the strength
        of `lse = -inf` alone. Force that case by giving one token winners on a
        single rank only, and check the merge ignores the other three.
        """
        self._require_devices(4)
        dcp_size, interleave_size = 4, 4
        inp = self._inputs()
        topk = inp["topk"].copy()
        # interleave_size=4, dcp_size=4 -> positions 0..3 are all on rank 0.
        topk[DCP_SEQ - 1] = -1
        topk[DCP_SEQ - 1, :4] = [0, 1, 2, 3]
        inp["topk"] = topk

        merged, dropped = self._run_dcp(inp, dcp_size, interleave_size)
        self.assertEqual(dropped, 0)
        np.testing.assert_allclose(merged[DCP_SEQ - 1],
                                   self._numpy_reference(inp)[DCP_SEQ - 1],
                                   rtol=2e-2,
                                   atol=2e-2)
