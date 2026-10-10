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
"""The fused CSA + SWA kernel against the SWA kernel followed by SparseMLA."""

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from tests.kernels.deepseek_v4.sparse_mla_test import create_cache
from vllm_torchtpu.kernels.deepseek_v4.core_attention import (
    mla_swa,
    sparse_mla,
    sparse_mla_swa,
)

pytestmark = pytest.mark.tpu_test

NUM_HEADS = 128
HEAD_DIM = 512
WINDOW = 128
SWA_PAGE = 32  # SWA tokens per logical page
CSA_PAGE = 64  # compressed tokens (int32 rows) per page; 32 SWA tokens
COMPRESS = 4
TOPK = 256
MAX_LEN = 1024


def _swa_reference(q, new_kv, cache, meta, sinks, sm_scale):
    return mla_swa.mla_sliding_window_ragged_paged_attention(
        q,
        new_kv,
        cache,
        *meta,
        sinks,
        sm_scale=sm_scale,
        sliding_window=WINDOW,
        logical_page_size=SWA_PAGE,
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 32, 32),
        q_compute_block_size=4,
        unnormalized_output=True,
    )


class FusedSwaTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(7)

    def _rand(self, shape, dtype=jnp.bfloat16):
        return jnp.array(self.rng.standard_normal(shape, dtype=np.float32)).astype(
            dtype
        )

    def _run(self, old_lens, q_lens, num_decode, num_tokens, nope_is_sw_cache):
        num_seqs = len(q_lens)
        max_num_seqs = num_seqs + 2
        # Peakier than HEAD_DIM**-0.5, so a dropped or extra key shows.
        sm_scale = 0.25
        old_lens = np.array(old_lens + [0, 0], np.int32)
        q_lens = np.array(q_lens + [0, 0], np.int32)
        kv_lens = old_lens + q_lens

        # One page pool: CSA NoPE pages first, then shuffled SWA pages.
        csa_pages_per_seq = MAX_LEN // COMPRESS // CSA_PAGE
        swa_pages_per_seq = MAX_LEN // SWA_PAGE
        num_csa_pages = max_num_seqs * csa_pages_per_seq
        num_swa_pages = max_num_seqs * swa_pages_per_seq
        _, csa_nope, csa_rope = create_cache(
            self.rng, num_csa_pages, CSA_PAGE, HEAD_DIM, jnp.bfloat16
        )
        csa_nope = csa_nope.reshape(num_csa_pages, CSA_PAGE, 128)
        pool = jnp.concatenate(
            [csa_nope, jnp.zeros((num_swa_pages, CSA_PAGE, 128), jnp.int32)]
        )
        main_pi = jnp.arange(num_csa_pages, dtype=jnp.int32)
        swa_pi = jnp.array(
            num_csa_pages + self.rng.permutation(num_swa_pages), jnp.int32
        )
        sinks = jnp.array(self.rng.random(NUM_HEADS, dtype=np.float32))

        def cu(lens):
            return jnp.array(np.concatenate([[0], np.cumsum(lens)]), jnp.int32)

        # Step 1: write each sequence's old tokens to the SWA cache.
        n1 = max(int(old_lens.sum()), 1)
        dist1 = jnp.array([0, 0, num_seqs], jnp.int32)
        meta1 = (jnp.array(old_lens), swa_pi, cu(old_lens), dist1)
        _, pool, _, _ = _swa_reference(
            self._rand((n1, NUM_HEADS, HEAD_DIM)),
            self._rand((n1, HEAD_DIM)),
            pool,
            meta1,
            sinks,
            sm_scale,
        )

        # Step 2, the one compared.
        q = self._rand((num_tokens, NUM_HEADS, HEAD_DIM))
        new_kv = self._rand((num_tokens, HEAD_DIM))
        cu2 = cu(q_lens)
        dist2 = jnp.array([num_decode, num_decode, num_seqs], jnp.int32)
        topk = np.full((num_tokens, TOPK), -1, np.int32)
        t = 0
        for s in range(num_seqs):
            for j in range(q_lens[s]):
                num_c = (old_lens[s] + j + 1) // COMPRESS
                pick = self.rng.permutation(num_c)[:TOPK]
                topk[t, : len(pick)] = pick
                t += 1
        topk = jnp.array(topk)
        meta2 = (jnp.array(kv_lens), swa_pi, cu2, dist2)

        ref_swa, ref_pool, ref_l, ref_m = _swa_reference(
            q, new_kv, jnp.copy(pool), meta2, sinks, sm_scale
        )
        ref_out = sparse_mla.sparse_ragged_paged_attention(
            q,
            ref_pool if nope_is_sw_cache else pool,
            csa_rope,
            topk,
            main_pi,
            cu2,
            dist2,
            sinks,
            ref_swa,
            ref_l,
            ref_m,
            sm_scale=sm_scale,
        )
        out, out_pool = sparse_mla_swa.sparse_swa_ragged_paged_attention(
            q,
            new_kv,
            jnp.copy(pool),
            *meta2,
            pool,
            csa_rope,
            topk,
            main_pi,
            cu2,
            dist2,
            sinks,
            sm_scale=sm_scale,
            sliding_window=WINDOW,
            logical_page_size=SWA_PAGE,
            nope_is_sw_cache=nope_is_sw_cache,
        )

        np.testing.assert_array_equal(np.asarray(out_pool), np.asarray(ref_pool))
        valid = int(cu2[num_seqs])
        ref = np.asarray(ref_out[:valid], np.float32)
        got = np.asarray(out[:valid], np.float32)
        self.assertTrue(np.isfinite(ref).all())
        self.assertTrue(np.isfinite(got).all())
        # Relative L2 per token and head; the reference rounds its SWA
        # partials to bf16, the fused kernel does not.
        err = np.linalg.norm(got - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
        worst = np.unravel_index(err.argmax(), err.shape)
        print(f"worst rel L2 {err.max():.3e} at {worst} of {valid} tokens")
        self.assertLess(err.max(), 1e-2)

    @parameterized.named_parameters(
        ("aliased", True),
        ("separate", False),
    )
    def test_mixed_decode_and_prefill(self, nope_is_sw_cache):
        # 5 decode tokens, then chunks with and without a cached prefix,
        # shorter and longer than the window, crossing batch boundaries.
        self._run(
            old_lens=[3, 130, 300, 127, 40, 200, 0, 60, 500],
            q_lens=[1, 1, 1, 1, 1, 150, 40, 7, 37],
            num_decode=5,
            num_tokens=256,
            nope_is_sw_cache=nope_is_sw_cache,
        )

    def test_decode_only(self):
        self._run(
            old_lens=[0, 5, 127, 128, 129, 400, 31, 32, 33, 64],
            q_lens=[1] * 10,
            num_decode=10,
            num_tokens=16,
            nope_is_sw_cache=True,
        )

    def test_one_long_prefill(self):
        self._run(
            old_lens=[256],
            q_lens=[512],
            num_decode=0,
            num_tokens=512,
            nope_is_sw_cache=True,
        )


if __name__ == "__main__":
    absltest.main()
