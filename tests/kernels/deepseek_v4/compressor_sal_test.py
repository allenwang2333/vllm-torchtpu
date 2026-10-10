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
"""`compressor_forward(record_layout=SEQ_ALONG_LANE)` on the indexer's array.

The SEQ_ALONG_LANE records must be the reader's own HAS -> SAL conversion of
the HEAD_ALONG_SUBLANE records, the state pages must not change, and the
spare sublane groups of each record page must stay untouched.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store.compressor_v1 import (
    compressor_forward,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    convert_cache_to_seq_along_lane,
)

pytestmark = pytest.mark.tpu_test

_HEAD_DIM = 128
_RECORD_GROUPS = _HEAD_DIM // 4 + 1
_COMPRESS_RATIO = 4
_PAGE_TOKENS = 256  # Compressed tokens a page: vLLM block_size 1024.
_STATE_BLOCK = 16  # Token states a page, floored to CSA's.
_NUM_PAGES = 16


def _run(layout, cache, starts, q_lens, num_decodes, hidden_size=7168, seed=0):
    rng = np.random.default_rng(seed)
    positions = np.concatenate(
        [np.arange(s, s + q) for s, q in zip(starts, q_lens)]
    ).astype(np.int32)
    num_reqs = len(q_lens)
    max_pos = int(positions.max()) + 1
    # Disjoint pages: two record pages per request, the rest host state.
    perm = rng.permutation(_NUM_PAGES)
    kv_pages = perm[: 2 * num_reqs].reshape(num_reqs, 2)
    state_pages = perm[2 * num_reqs :]
    state_cols = -(-max_pos // _STATE_BLOCK)
    block_table = np.stack(
        [
            np.roll(state_pages, i)[np.arange(state_cols) % len(state_pages)]
            for i in range(num_reqs)
        ]
    ).astype(np.int32)
    state_width = 2 * 2 * _HEAD_DIM
    out, _, _ = compressor_forward(
        hidden_states=jnp.asarray(
            rng.standard_normal((positions.size, hidden_size)), jnp.float32
        ),
        wkv_wgate=jnp.asarray(
            rng.standard_normal((hidden_size, state_width)) * 0.02, jnp.float32
        ),
        ape=jnp.asarray(
            rng.standard_normal((_COMPRESS_RATIO, state_width // 2)) * 0.1,
            jnp.float32,
        ),
        norm_weight=jnp.asarray(1 + 0.1 * rng.standard_normal(_HEAD_DIM), jnp.float32),
        cos_sin_cache=jnp.asarray(rng.standard_normal((max_pos, 64)), jnp.float32),
        positions=jnp.asarray(positions),
        block_table=jnp.asarray(block_table.reshape(-1)),
        query_start_loc=jnp.asarray(
            np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32
        ),
        kv_block_table=jnp.asarray(kv_pages.reshape(-1), jnp.int32),
        cache=jnp.asarray(cache),
        rope_cache=None,
        distribution=jnp.asarray([num_decodes, num_decodes, num_reqs], jnp.int32),
        state_block_size=_STATE_BLOCK,
        head_dim=_HEAD_DIM,
        compress_ratio=_COMPRESS_RATIO,
        overlap=True,
        rms_eps=1e-6,
        quant_block=_HEAD_DIM,
        record_layout=layout,
    )
    # The (page, slot) of every record: one per `compress_ratio` positions.
    req = np.repeat(np.arange(num_reqs), q_lens)
    boundary = (positions + 1) % _COMPRESS_RATIO == 0
    kv_idx = positions[boundary] // _COMPRESS_RATIO
    records = (kv_pages[req[boundary], kv_idx // _PAGE_TOKENS], kv_idx % _PAGE_TOKENS)
    return np.asarray(out), sorted(kv_pages.reshape(-1).tolist()), records


class CompressorSeqAlongLaneTest(parameterized.TestCase):
    @parameterized.named_parameters(
        # Two decodes, then chunks; the 60-token chunk at 1003 crosses into its
        # request's second record page (compressed token 256).
        ("decode_and_prefill", [3, 1, 0, 1003], [1, 1, 37, 60], 2),
        ("prefill_only", [0, 512, 400], [128, 22, 96], 0),
    )
    def test_sal_records_match_has_conversion(self, starts, q_lens, num_decodes):
        if jax.default_backend() != "tpu":
            self.skipTest("compressor kernels require a TPU")
        rng = np.random.default_rng(1)
        # Junk in every byte: each layout must overwrite exactly its own.
        cache = rng.integers(0, 256, (_NUM_PAGES, _PAGE_TOKENS // 4, 4, 256), np.uint8)
        # Stored state must be finite f32: it is read back as such.
        cache_f32 = cache.view(np.float32)
        cache_f32[...] = rng.standard_normal(cache_f32.shape).astype(np.float32)
        has, kv_pages, (page, slot) = _run(
            KVLayout.HEAD_ALONG_SUBLANE, cache, starts, q_lens, num_decodes
        )
        sal, _, _ = _run(KVLayout.SEQ_ALONG_LANE, cache, starts, q_lens, num_decodes)
        other = [p for p in range(_NUM_PAGES) if p not in kv_pages]
        np.testing.assert_array_equal(sal[other], has[other], err_msg="state pages")

        converted = np.asarray(
            convert_cache_to_seq_along_lane(jnp.asarray(has), _HEAD_DIM)
        )
        expected = cache[:, :_RECORD_GROUPS].copy()
        expected[page, :, :, slot] = converted[page, :, :, slot]
        assert page.size > 0
        np.testing.assert_array_equal(
            sal[kv_pages][:, :_RECORD_GROUPS], expected[kv_pages], err_msg="records"
        )
        np.testing.assert_array_equal(
            sal[kv_pages][:, _RECORD_GROUPS:],
            cache[kv_pages][:, _RECORD_GROUPS:],
            err_msg="spare groups",
        )

    def test_page_geometry_mismatch_raises(self):
        """Refused before any kernel runs, so this needs no TPU."""
        cache = np.zeros((_NUM_PAGES, 128, 4, 256), np.uint8)  # block_size 2048
        with self.assertRaisesRegex(ValueError, "cannot reuse"):
            _run(KVLayout.SEQ_ALONG_LANE, cache, [0], [8], 0, hidden_size=256)


if __name__ == "__main__":
    absltest.main()
