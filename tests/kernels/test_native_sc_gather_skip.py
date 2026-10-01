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
"""Batch-level skip in the native-SC gather.

Under DCP most of a token's top-k row is `-1` padding. A gather batch whose
occurrences are *all* padding is skipped outright. These tests cover the mask
that decides that, and that skipping never disturbs a batch it kept.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla.sparse import native_sc_gather as nsg

PAGES = 4096
CHUNK = 32
TOPK = 256
OUT_SIZE = CHUNK * TOPK
ATOMS_PER_BATCH = 1
OCC_PER_BATCH = ATOMS_PER_BATCH * nsg.OCCURRENCES_PER_ATOM
NUM_BATCHES = OUT_SIZE // OCC_PER_BATCH


def _live_rows(kv_len):
    """Left-packed liveness, the shape `topk_indices != -1` produces."""
    return np.broadcast_to(
        (np.arange(TOPK)[None, :] < kv_len).astype(np.int32), (CHUNK, TOPK)
    )


class BatchLiveMaskTest(parameterized.TestCase):
    def test_absent_live_marks_every_batch_live(self):
        mask = nsg.batch_live_mask(None, NUM_BATCHES, OCC_PER_BATCH)
        self.assertEqual(mask.shape, (NUM_BATCHES * nsg.NUM_LANES,))
        np.testing.assert_array_equal(np.asarray(mask), 1)

    def test_a_batch_is_dead_only_when_every_occurrence_is_padding(self):
        live = np.zeros((NUM_BATCHES, OCC_PER_BATCH), np.int32)
        live[3, OCC_PER_BATCH - 1] = 1  # one survivor keeps the batch alive
        mask = np.asarray(
            nsg.batch_live_mask(
                jnp.asarray(live.reshape(-1)), NUM_BATCHES, OCC_PER_BATCH
            )
        ).reshape(NUM_BATCHES, nsg.NUM_LANES)
        self.assertTrue((mask[3] == 1).all())
        self.assertEqual(mask.sum(), nsg.NUM_LANES)

    def test_the_flag_is_identical_across_all_sixteen_lanes(self):
        # The SC lowering only loads int32 vectors of shape (16,), so the
        # predicate reads lane 0 and the other fifteen must agree with it.
        rng = np.random.default_rng(0)
        live = rng.integers(0, 2, (NUM_BATCHES, OCC_PER_BATCH)).astype(np.int32)
        mask = np.asarray(
            nsg.batch_live_mask(
                jnp.asarray(live.reshape(-1)), NUM_BATCHES, OCC_PER_BATCH
            )
        ).reshape(NUM_BATCHES, nsg.NUM_LANES)
        np.testing.assert_array_equal(mask, np.broadcast_to(mask[:, :1], mask.shape))
        np.testing.assert_array_equal(mask[:, 0], live.max(axis=1))

    @parameterized.parameters((TOPK, 0.0), (TOPK // 2, 0.5), (TOPK // 4, 0.75))
    def test_left_packed_rows_kill_the_expected_fraction(self, kv_len, want_dead):
        live = _live_rows(kv_len)
        mask = np.asarray(
            nsg.batch_live_mask(
                jnp.asarray(live.reshape(-1)), NUM_BATCHES, OCC_PER_BATCH
            )
        ).reshape(NUM_BATCHES, nsg.NUM_LANES)[:, 0]
        self.assertAlmostEqual(1.0 - mask.mean(), want_dead)

    def test_rejects_a_live_array_of_the_wrong_size(self):
        rng = np.random.default_rng(0)
        nope, rope, indices = _build_cache(rng)
        with self.assertRaisesRegex(ValueError, "live has"):
            nsg.dsa_gather_native_sc(
                nope,
                rope,
                indices,
                out_size=OUT_SIZE,
                atoms_per_batch=ATOMS_PER_BATCH,
                live=jnp.ones((OUT_SIZE - 1,), jnp.int32),
            )


def _build_cache(rng):
    nope = jnp.asarray(
        rng.integers(0, 2**31, (PAGES, 128), dtype=np.int64).astype(np.uint32)
    )
    rope = jnp.asarray(
        rng.integers(0, 2**31, (PAGES // 4, 128), dtype=np.int64).astype(np.uint32)
    )
    indices = jnp.asarray(rng.integers(0, PAGES, (CHUNK, TOPK)).astype(np.int32))
    return nope, rope, indices


class SkipPreservesKeptBatchesTest(parameterized.TestCase):
    def _gather(self, nope, rope, indices, live):
        return nsg.dsa_gather_native_sc(
            nope,
            rope,
            indices,
            out_size=OUT_SIZE,
            atoms_per_batch=ATOMS_PER_BATCH,
            live=live,
        )

    def test_an_all_live_mask_is_bit_identical_to_no_mask(self):
        rng = np.random.default_rng(1)
        nope, rope, indices = _build_cache(rng)
        base_nope, base_rope = self._gather(nope, rope, indices, None)
        got_nope, got_rope = self._gather(
            nope, rope, indices, jnp.ones((OUT_SIZE,), jnp.int32)
        )
        np.testing.assert_array_equal(np.asarray(got_nope), np.asarray(base_nope))
        np.testing.assert_array_equal(np.asarray(got_rope), np.asarray(base_rope))

    @parameterized.parameters(TOPK // 2, TOPK // 4, TOPK // 8)
    def test_kept_batches_match_the_unskipped_gather(self, kv_len):
        rng = np.random.default_rng(2)
        nope, rope, indices = _build_cache(rng)
        live = _live_rows(kv_len)
        base_nope, _ = self._gather(nope, rope, indices, None)
        got_nope, _ = self._gather(nope, rope, indices, jnp.asarray(live.reshape(-1)))

        alive = np.asarray(
            nsg.batch_live_mask(
                jnp.asarray(live.reshape(-1)), NUM_BATCHES, OCC_PER_BATCH
            )
        ).reshape(NUM_BATCHES, nsg.NUM_LANES)[:, 0]
        self.assertGreater((alive == 0).sum(), 0, "nothing was skipped")

        rows = np.repeat(alive, OCC_PER_BATCH).astype(bool)
        np.testing.assert_array_equal(
            np.asarray(got_nope)[rows], np.asarray(base_nope)[rows]
        )


if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "float32")
    absltest.main()
