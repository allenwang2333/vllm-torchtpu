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
"""`sparse_mla_attention_dcp` variants must match the reference bit for bit.

Each variant changes how the op is fed, never what it computes, so the bar is
`array_equal` on all four outputs -- both cache shards, the partial output and
the partial lse -- against the default call on the same inputs:

* `dcp_rank` (TPU_DCP_RANK_OPERAND): the rank as a sharded operand instead of
  the per-call `all_to_all`.

The inputs are the decode shape the GLM-5.2 MTP drafter and backbone run:
16 token rows, four sequences of q_len 4 with the last one empty (rows 12..15
are padding the grid never reaches), rows where a rank owns nothing, rows
past the top-k cap (the overflow fix-up), on the SparseCore cache layout.
"""

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.core import attention_interface

P = jax.sharding.PartitionSpec
DCP_AXIS = streamindex_topk.DCP_AXIS_NAME

LKV_DIM = 512
ROPE_DIM = 64
NUM_HEADS = 16
PAGE_SIZE = 32
PAGES_PER_SEQ = 4
TOTAL_PAGES = 16  # per rank
NUM_SEQS = 4
Q_LEN = 4
NUM_TOKENS = 16
TOPK = 1024
TOPK_CAP = 512
TABLE_ROWS = 64  # the taller shared top-k table, per rank
INTERLEAVE = 4

KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
NOPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.NOPE,
    KVCacheLayout.SPARSECORE,
    TOTAL_PAGES,
    PAGE_SIZE,
    LKV_DIM,
    KV_PACKING,
)
ROPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.ROPE,
    KVCacheLayout.SPARSECORE,
    TOTAL_PAGES,
    PAGE_SIZE,
    ROPE_DIM,
    KV_PACKING,
)

# The served decode configuration of the DCP sparse MLA path.
SERVED_ENV = {
    "TPU_DCP_MLA_TOPK_CAP": str(TOPK_CAP),
    "TPU_DCP_MLA_CAPPED_PREP": "1",
    "TPU_DCP_MLA_SPLIT_Q_LSE": "1",
    "TPU_DCP_MLA_SMALL_CHUNK_BATCH": "4",
}


def _finite_fp8_bytes(rng, shape, dtype):
    """Random cache words whose every byte is a small finite e4m3 value."""
    itemsize = jnp.dtype(dtype).itemsize
    n = int(np.prod(shape)) * itemsize
    # Exponent <= 7 (|v| < 2) with a random sign: never NaN, never huge.
    b = rng.integers(0, 0x40, size=n, dtype=np.uint8)
    b |= rng.integers(0, 2, size=n, dtype=np.uint8) << 7
    return jnp.asarray(b.view(np.dtype(jnp.dtype(dtype).name)).reshape(shape))


class SparseMlaAttentionDcpVariantsTest(parameterized.TestCase):
    def _require(self, dcp_size):
        devices = jax.devices()
        if devices[0].platform != "tpu":
            self.skipTest("sparse MLA kernel requires a TPU (SparseCore).")
        if len(devices) < dcp_size:
            self.skipTest(f"needs {dcp_size} devices, have {len(devices)}")

    def _case(self, dcp_size, seed):
        rng = np.random.default_rng(seed)
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp_size]), (DCP_AXIS,))
        sharded = jax.sharding.NamedSharding(mesh, P(DCP_AXIS))

        def cache(spec):
            shape = (dcp_size * spec.shape[0],) + tuple(spec.shape[1:])
            return jax.device_put(
                _finite_fp8_bytes(rng, shape, spec.jax_dtype), sharded
            )

        ql_nope = jnp.asarray(
            rng.standard_normal((NUM_TOKENS, NUM_HEADS, LKV_DIM)), jnp.bfloat16
        )
        q_pe = jnp.asarray(
            rng.standard_normal((NUM_TOKENS, NUM_HEADS, ROPE_DIM)), jnp.bfloat16
        )
        kv_c = jnp.asarray(
            rng.standard_normal((NUM_TOKENS, LKV_DIM)) * 0.5, jnp.float8_e4m3fn
        )
        k_pe = jnp.asarray(
            rng.standard_normal((NUM_TOKENS, ROPE_DIM)) * 0.5, jnp.float8_e4m3fn
        )

        # Three live sequences of q_len 4 and an empty fourth: rows 12..15 are
        # padding past `cu_q_lens[distribution[2]]`.
        live = NUM_SEQS - 1
        cu = np.array([0, 4, 8, 12, 12], np.int32)
        virtual_page = PAGE_SIZE * dcp_size
        seq_lens = np.zeros(NUM_SEQS, np.int32)
        seq_lens[:live] = rng.integers(Q_LEN, PAGES_PER_SEQ * virtual_page, live)
        block_tables = rng.permutation(TOTAL_PAGES).astype(np.int32)
        distribution = np.array([0, 0, live], np.int32)

        # Each rank's slice of every token's top-k: local indices, prefix
        # packed, -1 padded. Per rank, some tokens own nothing (all -1), some
        # few entries, some more than the cap (the overflow fix-up).
        local_len = PAGES_PER_SEQ * PAGE_SIZE
        topk = np.full((dcp_size, NUM_TOKENS, TOPK), -1, np.int32)
        for r in range(dcp_size):
            for t in range(cu[live]):
                kind = (r + 3 * t) % 4
                n = {0: 0, 1: int(rng.integers(1, 64)), 2: 300, 3: 700}[kind]
                topk[r, t, :n] = rng.integers(0, local_len, n)
        assert (topk[:, : cu[live], 0] == -1).any(), "no owns-nothing row"
        assert ((topk >= 0).sum(-1) > TOPK_CAP).any(), "no overflow row"

        # The shared top-k table: the same rows on top, junk below them that
        # the op must never read.
        table = rng.integers(0, local_len, (dcp_size, TABLE_ROWS, TOPK)).astype(
            np.int32
        )
        table[:, :NUM_TOKENS] = topk

        args = dict(
            ql_nope=ql_nope,
            q_pe=q_pe,
            kv_c_normed=kv_c,
            k_pe=k_pe,
            kv_cache_nope=cache(NOPE_SPEC),
            kv_cache_rope=cache(ROPE_SPEC),
            seq_lens=jnp.asarray(seq_lens),
            block_tables=jnp.asarray(block_tables),
            query_start_loc=jnp.asarray(cu),
            request_distribution=jnp.asarray(distribution),
        )
        rows = jax.device_put(jnp.asarray(topk.reshape(-1, TOPK)), sharded)
        table = jax.device_put(jnp.asarray(table.reshape(-1, TOPK)), sharded)
        rank = jax.device_put(jnp.arange(dcp_size, dtype=jnp.int32), sharded)
        return mesh, args, rows, table, rank

    def _run(self, dcp_size, mesh, args, topk, env, **kw):
        with mock.patch.dict(os.environ, env):
            out = attention_interface.sparse_mla_attention_dcp(
                args["ql_nope"],
                args["q_pe"],
                args["kv_c_normed"],
                args["k_pe"],
                args["kv_cache_nope"],
                args["kv_cache_rope"],
                topk,
                args["seq_lens"],
                args["block_tables"],
                args["query_start_loc"],
                args["request_distribution"],
                mesh,
                NOPE_SPEC,
                ROPE_SPEC,
                sm_scale=0.04,
                k_scale=1.5,
                dcp_size=dcp_size,
                # The served --cp-kv-cache-interleave-size; the rope cache
                # packs WORD_BYTES = 4 tokens a word, so 4 is the minimum.
                interleave_size=INTERLEAVE,
                **kw,
            )
        return [np.asarray(jax.block_until_ready(x)) for x in out]

    @parameterized.product(
        dcp_size=[2, 8],
        variant=["rank"],
        capped_prep=[True, False],
    )
    def test_variant_is_bit_identical(self, dcp_size, variant, capped_prep):
        self._require(dcp_size)
        mesh, args, rows, table, rank = self._case(dcp_size, seed=dcp_size)
        env = dict(SERVED_ENV, TPU_DCP_MLA_CAPPED_PREP="1" if capped_prep else "0")
        want = self._run(dcp_size, mesh, args, rows, env)

        kw = {}
        topk = rows
        if "rank" in variant:
            kw["dcp_rank"] = rank
        if "fused" in variant:
            kw["post_fused"] = True
        if "table" in variant:
            topk = table
        got = self._run(dcp_size, mesh, args, topk, env, **kw)

        for name, g, w in zip(("nope", "rope", "output", "lse"), got, want):
            self.assertEqual(g.shape, w.shape, name)
            np.testing.assert_array_equal(g, w, err_msg=name)
        lse = want[3]
        self.assertTrue(np.isneginf(lse).any(), "no owns-nothing lse")
        self.assertTrue(np.isfinite(lse).any())


if __name__ == "__main__":
    absltest.main()
