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
"""`sparse_mla_dcp_draft_prep` (TPU_DRAFT_HOIST_ATTN_META) against the
per-call prep of `sparse_mla_attention_dcp`, bit for bit.

The MTP IndexShare draft's loop steps call the DCP sparse MLA op with the same
top-k rows, block tables and query layout, and seq_lens advancing by one per
step. The hoisted path builds every step's prep once and hands one step's to
each call; the caches, partial outputs and lse must equal the per-call path's
exactly, step after step. Needs a TPU (SparseCore caches) with >= 2 devices.
"""

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import DCP_AXIS_NAME
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.layers.core import attention_interface as ai

LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
NUM_HEADS = 16
TOPK = 2048

# The served GLM-5.2 AgentX env (decode-relevant DCP sparse MLA options).
_SERVED_ENV = {
    "TPU_DCP_MLA_TOPK_CAP": "512",
    "TPU_DCP_MLA_TOPK_CAP_CHUNK": "256",
    "TPU_DCP_MLA_CAPPED_PREP": "1",
    "TPU_DCP_MLA_SPLIT_Q_LSE": "1",
    "TPU_DCP_MLA_SMALL_CHUNK_BATCH": "4",
    "TPU_DCP_MLA_PACKED_GATHER": "0",
}


def _problem(rng, *, dcp, num_tokens, num_reqs, kernel_num_reqs, pages_per_seq,
             num_pages, counts_mode):
    nope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE, KVCacheLayout.SPARSECORE, num_pages, PAGE_SIZE, LKV_DIM
    )
    rope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.ROPE, KVCacheLayout.SPARSECORE, num_pages, PAGE_SIZE, ROPE_DIM
    )

    def rand_cache(spec):
        shape = (dcp * spec.shape[0],) + tuple(spec.shape[1:])
        return rng.integers(0, 2**32, size=shape, dtype=np.uint32)

    # Loop-step layout: one query per request, padded to num_tokens.
    qsl = np.minimum(np.arange(kernel_num_reqs + 1), num_reqs).astype(np.int32)
    rd = np.array([num_reqs] * 3, np.int32)
    # Step 1's seq_lens; the virtual page of the next positions stays in range.
    max_len = (pages_per_seq * dcp - 2) * PAGE_SIZE
    seq_lens = rng.integers(PAGE_SIZE, max_len, size=kernel_num_reqs).astype(np.int32)
    # Cross a virtual-page and an interleave boundary between the steps.
    seq_lens[0] = PAGE_SIZE * dcp
    seq_lens[1] = 63
    block_tables = rng.integers(
        0, 4 * num_pages, size=kernel_num_reqs * pages_per_seq
    ).astype(np.int32)

    # Per rank, per token: prefix-packed local indices, -1 tail.
    if counts_mode == "within_cap":
        pool = np.array([0, 1, 63, 64, 128, 255, 511, 512])
    else:
        pool = np.array([0, 1, 130, 511, 512, 513, 700, 1025, 2048])
    counts = rng.choice(pool, size=(dcp, num_tokens))
    idx = rng.integers(0, pages_per_seq * PAGE_SIZE, size=(dcp, num_tokens, TOPK))
    live = np.arange(TOPK)[None, None, :] < counts[..., None]
    topk = np.where(live, idx, -1).astype(np.int32).reshape(dcp * num_tokens, TOPK)

    def q(dim):
        return rng.normal(size=(num_tokens, NUM_HEADS * dcp, dim)).astype(np.float32)

    def latent(dim):
        v = rng.normal(size=(num_tokens, dim)).astype(np.float32)
        return jnp.asarray(v).astype(jnp.float8_e4m3fn)

    return dict(
        nope_spec=nope_spec,
        rope_spec=rope_spec,
        nope=rand_cache(nope_spec),
        rope=rand_cache(rope_spec),
        qsl=qsl,
        rd=rd,
        seq_lens=seq_lens,
        block_tables=block_tables,
        topk=topk,
        steps=[
            dict(
                ql_nope=jnp.asarray(q(LKV_DIM), jnp.bfloat16),
                q_pe=jnp.asarray(q(ROPE_DIM), jnp.bfloat16),
                kv_c=latent(LKV_DIM),
                k_pe=latent(ROPE_DIM),
            )
            for _ in range(3)
        ],
    )


class SparseMlaDcpDraftPrepTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore DCP sparse MLA needs a TPU")
        if len(jax.devices()) < 2:
            self.skipTest("DCP needs >= 2 devices")

    @parameterized.product(
        counts_mode=("within_cap", "overflow"),
        num_tokens=(16, 32),
        interleave_size=(4, 16),
    )
    def test_hoisted_prep_is_bit_identical(self, counts_mode, num_tokens,
                                           interleave_size):
        with mock.patch.dict(os.environ, _SERVED_ENV):
            self._check(counts_mode, num_tokens, interleave_size)

    def test_hoisted_prep_default_env(self):
        """Without the capped prep env: the physical-index prep is hoisted."""
        env = dict(_SERVED_ENV, TPU_DCP_MLA_CAPPED_PREP="0",
                   TPU_DCP_MLA_SMALL_CHUNK_BATCH="1", TPU_DCP_MLA_SPLIT_Q_LSE="0")
        with mock.patch.dict(os.environ, env):
            self._check("overflow", 16, 16)

    def _check(self, counts_mode, num_tokens, interleave_size):
        devices = np.array(jax.devices())
        dcp = len(devices)
        mesh = Mesh(devices, (DCP_AXIS_NAME,))
        rng = np.random.default_rng(20261002 + num_tokens + interleave_size)
        num_steps = 3
        pr = _problem(
            rng,
            dcp=dcp,
            num_tokens=num_tokens,
            num_reqs=4,
            kernel_num_reqs=8,
            pages_per_seq=32,
            num_pages=64,
            counts_mode=counts_mode,
        )
        shard = NamedSharding(mesh, P(DCP_AXIS_NAME))
        rep = NamedSharding(mesh, P())
        topk = jax.device_put(pr["topk"], NamedSharding(mesh, P(DCP_AXIS_NAME, None)))
        seq_lens = jax.device_put(pr["seq_lens"], rep)
        bt = jax.device_put(pr["block_tables"], rep)
        qsl = jax.device_put(pr["qsl"], rep)
        rd = jax.device_put(pr["rd"], rep)
        kw = dict(dcp_size=dcp, interleave_size=interleave_size)

        def attend(caches, step, sl, draft_prep=None):
            s = pr["steps"][step]
            return ai.sparse_mla_attention_dcp(
                s["ql_nope"], s["q_pe"], s["kv_c"], s["k_pe"],
                caches[0], caches[1], topk, sl, bt, qsl, rd,
                mesh, pr["nope_spec"], pr["rope_spec"],
                sm_scale=0.1, k_scale=0.75, draft_prep=draft_prep, **kw,
            )

        def fresh_caches():
            return (jax.device_put(pr["nope"], shard), jax.device_put(pr["rope"], shard))

        prep = ai.sparse_mla_dcp_draft_prep(
            topk, seq_lens, bt, qsl, rd, mesh, pr["nope_spec"],
            num_steps=num_steps, **kw,
        )
        self.assertLen(prep, ai.DCP_DRAFT_PREP_SHARED + 3 * num_steps)

        ref_caches, hoist_caches = fresh_caches(), fresh_caches()
        sl = seq_lens
        for step in range(num_steps):
            ref = jax.block_until_ready(attend(ref_caches, step, sl))
            got = jax.block_until_ready(
                attend(hoist_caches, step, sl, ai.DcpDraftPrep.for_step(prep, step))
            )
            for name, r, g in zip(("nope", "rope", "out", "lse"), ref, got):
                r, g = np.asarray(r), np.asarray(g)
                if r.dtype == jnp.bfloat16:
                    r, g = r.view(np.uint16), g.view(np.uint16)
                np.testing.assert_array_equal(
                    g, r, err_msg=f"step {step}: {name} differs from the per-call path"
                )
            # Positive control: the output is attention, not the aliased q.
            self.assertGreater(
                np.abs(np.asarray(ref[2], np.float32)).max(), 0.0
            )
            ref_caches, hoist_caches = ref[:2], got[:2]
            sl = sl + 1

        # Positive control: a stale write plan (step 0's at step 1) moves the
        # write, so the comparison above can see a wrong plan.
        base = fresh_caches()
        r0 = attend(base, 0, seq_lens)
        right = attend(r0[:2], 1, seq_lens + 1, ai.DcpDraftPrep.for_step(prep, 1))
        base = fresh_caches()
        r0 = attend(base, 0, seq_lens)
        wrong = attend(r0[:2], 1, seq_lens + 1, ai.DcpDraftPrep.for_step(prep, 0))
        self.assertFalse(
            np.array_equal(np.asarray(right[0]), np.asarray(wrong[0])),
            "a stale write plan left the nope cache unchanged",
        )

    def test_hoisted_hlo_drops_index_prep(self):
        """The hoisted call carries none of the per-call prep: no rank id
        all-to-all, no capped-window-indices kernel, no searchsorted loop."""
        with mock.patch.dict(os.environ, _SERVED_ENV):
            devices = np.array(jax.devices())
            dcp = len(devices)
            mesh = Mesh(devices, (DCP_AXIS_NAME,))
            rng = np.random.default_rng(7)
            pr = _problem(rng, dcp=dcp, num_tokens=16, num_reqs=4, kernel_num_reqs=8,
                          pages_per_seq=32, num_pages=64, counts_mode="overflow")
            kw = dict(dcp_size=dcp, interleave_size=16)
            s = pr["steps"][0]
            prep = ai.sparse_mla_dcp_draft_prep(
                pr["topk"], pr["seq_lens"], pr["block_tables"], pr["qsl"], pr["rd"],
                mesh, pr["nope_spec"], num_steps=2, **kw,
            )

            def hlo(draft_prep):
                def f(*a):
                    return ai.sparse_mla_attention_dcp(
                        *a[:11], mesh, pr["nope_spec"], pr["rope_spec"],
                        sm_scale=0.1, k_scale=0.75, draft_prep=draft_prep, **kw,
                    )
                args = (s["ql_nope"], s["q_pe"], s["kv_c"], s["k_pe"], pr["nope"],
                        pr["rope"], pr["topk"], pr["seq_lens"], pr["block_tables"],
                        pr["qsl"], pr["rd"])
                return jax.jit(f).lower(*args).as_text()

            base = hlo(None)
            hoisted = hlo(ai.DcpDraftPrep.for_step(prep, 0))
            # The rank id (`cp_rank_as_data`) is an all-to-all.
            self.assertIn("all_to_all", base)
            self.assertNotIn("all_to_all", hoisted, "rank id left in the hoisted path")
            self.assertNotIn("capped_window_indices", hoisted)
            self.assertLess(len(hoisted), len(base))


if __name__ == "__main__":
    absltest.main()
