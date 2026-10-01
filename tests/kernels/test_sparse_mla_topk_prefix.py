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
"""`topk_prefix_width` must change the cost of sparse MLA, never the answer."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM

TOPK = 256
PREFIX = 64
QUERY_TOKENS = 128


def _build_problem(rng, kv_lens, *, k_scale=1.0, topk=TOPK):
    """A sparse-MLA problem whose row `t` holds `kv_lens[t]` real winners.

    Everything past `kv_lens[t]` is the "-1" suffix padding the top-k producer
    emits, which is what `topk_prefix_width` slices away.
    """
    page_size = 256
    cache_tokens = 8192
    pages = cache_tokens // page_size
    num_heads = 8
    query_tokens = len(kv_lens)

    def quantized_bytes(dim):
        values = rng.random((pages, page_size, dim), dtype=np.float32) * 2 - 1
        fp8 = jnp.asarray(values / k_scale).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    nope_bytes = quantized_bytes(NOPE_DIM)
    rope_bytes = quantized_bytes(ROPE_DIM)
    tensorcore_nope = nope_bytes.reshape(pages, page_size, 4, 128)
    tensorcore_rope = np.pad(rope_bytes, ((0, 0), (0, 0), (0, ROPE_DIM))).reshape(
        pages, page_size // 4, 4, 128
    )

    page_table = rng.permutation(pages).astype(np.int32)
    inverse_page_table = np.argsort(page_table)
    physical = np.stack(
        [rng.choice(cache_tokens, size=topk, replace=False) for _ in range(query_tokens)]
    ).astype(np.int32)
    logical = (
        inverse_page_table[physical // page_size] * page_size + physical % page_size
    ).astype(np.int32)

    # Left-packed real winners, "-1" suffix -- the contract the slice relies on.
    keep = np.arange(topk)[None, :] < np.asarray(kv_lens)[:, None]
    logical = np.where(keep, logical, -1).astype(np.int32)

    q = jnp.asarray(
        rng.normal(size=(query_tokens, num_heads, HEAD_DIM)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    return {
        "q": q,
        "tensorcore_nope": jnp.asarray(tensorcore_nope),
        "tensorcore_rope": jnp.asarray(tensorcore_rope),
        "sparsecore_nope": jnp.asarray(
            native_sc_cache_layout.pack_nope(tensorcore_nope)
        ),
        "sparsecore_rope": jnp.asarray(
            native_sc_cache_layout.pack_rope_banded(tensorcore_rope)
        ),
        "topk": jnp.asarray(logical),
        "page_table": jnp.asarray(page_table),
        "cu_q_lens": jnp.array([0, query_tokens], jnp.int32),
        "distribution": jnp.array([0, 1, 1], jnp.int32),
    }


def _attend(problem, layout, **extra):
    nope, rope = (
        (problem["sparsecore_nope"], problem["sparsecore_rope"])
        if layout == sparse_mla.SPARSECORE_CACHE_LAYOUT
        else (problem["tensorcore_nope"], problem["tensorcore_rope"])
    )
    out = sparse_mla.sparse_ragged_paged_attention(
        problem["q"],
        nope,
        rope,
        problem["topk"],
        problem["page_table"],
        problem["cu_q_lens"],
        problem["distribution"],
        sm_scale=0.125,
        gather_and_attention_chunk_size=64,
        attention_kernel_batch_size=16,
        cache_layout=layout,
        **extra,
    )
    return np.asarray(jax.block_until_ready(out), dtype=np.float32)


LAYOUTS = (sparse_mla.TENSORCORE_CACHE_LAYOUT, sparse_mla.SPARSECORE_CACHE_LAYOUT)


class TopkPrefixWidthTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA attention requires a TPU")

    @parameterized.parameters(*LAYOUTS)
    def test_narrow_matches_wide_when_every_row_fits(self, layout):
        rng = np.random.default_rng(20260930)
        kv_lens = rng.integers(1, PREFIX + 1, size=QUERY_TOKENS)
        problem = _build_problem(rng, kv_lens)

        wide = _attend(problem, layout)
        narrow = _attend(problem, layout, topk_prefix_width=PREFIX)

        np.testing.assert_allclose(narrow, wide, rtol=1e-3, atol=1e-3)

    @parameterized.parameters(*LAYOUTS)
    def test_an_empty_row_is_handled(self, layout):
        rng = np.random.default_rng(20260930)
        kv_lens = rng.integers(1, PREFIX + 1, size=QUERY_TOKENS)
        kv_lens[0] = 0
        kv_lens[-1] = PREFIX
        problem = _build_problem(rng, kv_lens)

        wide = _attend(problem, layout)
        narrow = _attend(problem, layout, topk_prefix_width=PREFIX)

        self.assertTrue(np.isfinite(narrow).all())
        np.testing.assert_allclose(narrow, wide, rtol=1e-3, atol=1e-3)

    @parameterized.parameters(*LAYOUTS)
    def test_a_row_over_the_width_falls_back_instead_of_dropping_winners(self, layout):
        """The guard is correctness, not speed: truncation is silent otherwise.

        A row with `kv_len > prefix_width` still passes every mask inside the
        kernel -- `k_span < kv_len` is true for all `prefix_width` gathered
        rows -- so a dropped winner shows up only as a wrong number here.
        """
        rng = np.random.default_rng(20260930)
        kv_lens = rng.integers(1, PREFIX + 1, size=QUERY_TOKENS)
        kv_lens[7] = PREFIX + 1  # one row overflows, by the smallest margin
        problem = _build_problem(rng, kv_lens)

        wide = _attend(problem, layout)
        narrow = _attend(problem, layout, topk_prefix_width=PREFIX)

        np.testing.assert_allclose(narrow, wide, rtol=1e-3, atol=1e-3)

    def test_rejects_an_unusable_width(self):
        rng = np.random.default_rng(20260930)
        problem = _build_problem(rng, np.full(QUERY_TOKENS, PREFIX))
        for bad in (0, -8, TOPK + 4):
            with self.assertRaisesRegex(ValueError, "topk_prefix_width"):
                _attend(problem, LAYOUTS[0], topk_prefix_width=bad)
        # SparseCore ROPE packing needs a multiple of ROPE_PACKING.
        with self.assertRaisesRegex(ValueError, "topk_prefix_width"):
            _attend(
                problem,
                sparse_mla.SPARSECORE_CACHE_LAYOUT,
                topk_prefix_width=PREFIX + 1,
            )


if __name__ == "__main__":
    absltest.main()
