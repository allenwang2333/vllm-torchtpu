# SPDX-License-Identifier: Apache-2.0
"""100% statement and branch coverage tests for DeepSeek-V4.1 indexer custom ops
and kernels (TD7.37)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41 import indexer as indexer_kernel
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
    deepseek_v41_indexer as indexer_ops,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_cache import (
    INDEX_HEAD_DIM,
    ROW_WORDS,
)


@pytest.fixture(autouse=True)
def _mock_tpu_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr("jax._src.pallas.mosaic.tpu_info.get_tpu_info", lambda: info)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _make_native_pool_and_scales(
    num_pages: int = 4, page_size: int = 128, seed: int = 0
):
    rng = np.random.default_rng(seed)
    total_keys = num_pages * page_size
    word_rows = total_keys // 4
    pool = jnp.asarray(
        rng.integers(0, 0x3FFF3FFF, size=(word_rows, INDEX_HEAD_DIM), dtype=np.int32)
    )
    scales = jnp.asarray(
        rng.integers(
            120,
            130,
            size=(num_pages, page_size // ROW_WORDS, ROW_WORDS),
            dtype=np.int32,
        )
    )
    return pool, scales


class TestKernelsDeepseekV41Indexer:
    def test_paged_scores_validation_errors(self):
        pool, scales = _make_native_pool_and_scales(2, 128)
        block_table = jnp.zeros((2, 2), dtype=jnp.int32)
        requests = jnp.zeros((4,), dtype=jnp.int32)
        lengths = jnp.ones((4,), dtype=jnp.int32) * 16
        weights = jnp.ones((4, 2), dtype=jnp.float32)

        # Invalid rows or dim
        q_bad_rows = jnp.zeros((0, 2, 128), dtype=jnp.float8_e4m3fn)
        with pytest.raises(ValueError, match="1..32 rows and 128 channels"):
            indexer_kernel.paged_scores(
                q_bad_rows,
                weights,
                pool,
                scales,
                block_table,
                requests,
                lengths,
                128,
                page_size=128,
                page_stride=128,
            )

        q_bad_dim = jnp.zeros((4, 2, 64), dtype=jnp.float8_e4m3fn)
        with pytest.raises(ValueError, match="1..32 rows and 128 channels"):
            indexer_kernel.paged_scores(
                q_bad_dim,
                weights,
                pool,
                scales,
                block_table,
                requests,
                lengths,
                128,
                page_size=128,
                page_stride=128,
            )

        # Invalid dtype
        q_bad_dtype = jnp.zeros((4, 2, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="FP8 E4M3 queries"):
            indexer_kernel.paged_scores(
                q_bad_dtype,
                weights,
                pool,
                scales,
                block_table,
                requests,
                lengths,
                128,
                page_size=128,
                page_stride=128,
            )

        # Invalid capacity or page_size
        q_ok = jnp.zeros((4, 2, 128), dtype=jnp.float8_e4m3fn)
        with pytest.raises(
            ValueError, match="Positive capacity and complete native pages"
        ):
            indexer_kernel.paged_scores(
                q_ok,
                weights,
                pool,
                scales,
                block_table,
                requests,
                lengths,
                0,
                page_size=128,
                page_stride=128,
            )

        # Misaligned page_size / page_stride / row_offset
        with pytest.raises(ValueError, match="128-key aligned"):
            indexer_kernel.paged_scores(
                q_ok,
                weights,
                pool,
                scales,
                block_table,
                requests,
                lengths,
                128,
                page_size=256,
                page_stride=128,
                row_offset=64,
            )

    def test_paged_scores_interpret_single_and_multi_head(self):
        pool, scales = _make_native_pool_and_scales(2, 128, seed=1)
        block_table = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        requests = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        lengths = jnp.array([64, 0, 128, 96], dtype=jnp.int32)

        # Single head (heads == 1 -> loop over range(1, heads) has 0 iterations)
        q_1h = jnp.ones((4, 1, 128), dtype=jnp.float8_e4m3fn)
        w_1h = jnp.ones((4, 1), dtype=jnp.float32)
        scores_1h = indexer_kernel.paged_scores(
            q_1h,
            w_1h,
            pool,
            scales,
            block_table,
            requests,
            lengths,
            128,
            page_size=128,
            page_stride=128,
            interpret=True,
        )
        assert scores_1h.shape == (4, 128)

        # Multi-head (heads == 2 -> loop over range(1, heads) executes)
        q_2h = jnp.ones((4, 2, 128), dtype=jnp.float8_e4m3fn)
        w_2h = jnp.ones((4, 2), dtype=jnp.float32)
        scores_2h = indexer_kernel.paged_scores(
            q_2h,
            w_2h,
            pool,
            scales,
            block_table,
            requests,
            lengths,
            128,
            page_size=128,
            page_stride=128,
            interpret=True,
        )
        assert scores_2h.shape == (4, 128)

    def test_native_paged_scores_validation_and_execution(self):
        pool, scales = _make_native_pool_and_scales(2, 128, seed=2)
        block_table = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        requests = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        lengths = jnp.array([64, 96, 128, 32], dtype=jnp.int32)
        weights = jnp.ones((4, 2), dtype=jnp.float32)

        # Invalid query dtype/shape
        with pytest.raises(ValueError, match="1..32 BF16 queries of width 128"):
            indexer_kernel.native_paged_scores(
                jnp.zeros((4, 2, 128), dtype=jnp.float32),
                weights,
                pool,
                scales,
                block_table,
                requests,
                lengths,
                128,
                page_size=128,
                page_stride=128,
            )

        # Invalid page alignment/size
        q_ok = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
        with pytest.raises(ValueError, match="128-aligned pages no larger than 2048"):
            indexer_kernel.native_paged_scores(
                q_ok,
                weights,
                pool,
                scales,
                block_table,
                requests,
                lengths,
                128,
                page_size=4096,
                page_stride=4096,
            )

        # Single head (heads == 1) and multi-block (steps > 1 so following() runs)
        q_1h = jnp.ones((2, 1, 128), dtype=jnp.bfloat16)
        w_1h = jnp.ones((2, 1), dtype=jnp.float32)
        out_1h = indexer_kernel.native_paged_scores(
            q_1h,
            w_1h,
            pool,
            scales,
            block_table,
            jnp.array([0, 1], dtype=jnp.int32),
            jnp.array([256, 256], dtype=jnp.int32),
            256,
            page_size=128,
            page_stride=128,
            interpret=True,
        )
        assert out_1h.shape == (2, 256)

        # Multi-head (heads == 2)
        out_2h = indexer_kernel.native_paged_scores(
            q_ok,
            weights,
            pool,
            scales,
            block_table,
            requests,
            lengths,
            256,
            page_size=128,
            page_stride=128,
            interpret=True,
        )
        assert out_2h.shape == (4, 256)

    def test_pipeline_score_topk_even_and_odd_tiles(self):
        def prepare(tile):
            return tile + 1

        def score(prepared):
            return prepared * 10

        def select(tile, scores):
            return scores + tile

        def write(tile, chosen, result):
            return result.at[tile].set(chosen)

        out2 = indexer_kernel.pipeline_score_topk(
            2, prepare, score, select, write, jnp.zeros((2,), dtype=jnp.int32)
        )
        np.testing.assert_array_equal(
            np.asarray(out2), np.array([10, 21], dtype=np.int32)
        )

        out3 = indexer_kernel.pipeline_score_topk(
            3, prepare, score, select, write, jnp.zeros((3,), dtype=jnp.int32)
        )
        np.testing.assert_array_equal(
            np.asarray(out3), np.array([10, 21, 32], dtype=np.int32)
        )

    def test_td8327_indexer_jnp_arange_explicit_int32(self, monkeypatch):
        orig_arange = jnp.arange
        seen_dtypes: list[object] = []

        def _spy_arange(*args, **kwargs):
            seen_dtypes.append(kwargs.get("dtype"))
            return orig_arange(*args, **kwargs)

        monkeypatch.setattr(indexer_kernel.jnp, "arange", _spy_arange)

        pool, scales = _make_native_pool_and_scales(2, 128, seed=3)
        block_table = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        requests = jnp.array([0, 1], dtype=jnp.int32)
        lengths = jnp.array([64, 128], dtype=jnp.int32)
        weights = jnp.ones((2, 1), dtype=jnp.float32)

        indexer_kernel.paged_scores(
            jnp.ones((2, 1, 128), dtype=jnp.float8_e4m3fn),
            weights,
            pool,
            scales,
            block_table,
            requests,
            lengths,
            128,
            page_size=128,
            page_stride=128,
            interpret=True,
        )
        indexer_kernel.native_paged_scores(
            jnp.ones((2, 1, 128), dtype=jnp.bfloat16),
            weights,
            pool,
            scales,
            block_table,
            requests,
            lengths,
            128,
            page_size=128,
            page_stride=128,
            interpret=True,
        )

        assert len(seen_dtypes) >= 3
        assert all(dt == jnp.int32 for dt in seen_dtypes)


class TestDeepseekV41IndexerOps:
    def test_indexer_cache_init_and_methods(self, monkeypatch):
        bad_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(index_head_dim=64))
        )
        monkeypatch.setattr(indexer_ops, "get_current_vllm_config", lambda: bad_cfg)
        with pytest.raises(ValueError, match="128-dimensional keys"):
            indexer_ops.IndexerCache()

        good_cfg = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(index_head_dim=128))
        )
        monkeypatch.setattr(indexer_ops, "get_current_vllm_config", lambda: good_cfg)

        def fake_super_init(self, *args, **kwargs):
            torch.nn.Module.__init__(self)
            self.kwargs = kwargs
            self._bound_cache = None

        def fake_super_bind(self, cache):
            self._bound_cache = cache

        monkeypatch.setattr(
            indexer_ops.upstream.DeepseekV4IndexerCache, "__init__", fake_super_init
        )
        monkeypatch.setattr(
            indexer_ops.upstream.DeepseekV4IndexerCache,
            "bind_kv_cache",
            fake_super_bind,
        )

        cache_mod = indexer_ops.IndexerCache()
        assert cache_mod.kwargs["head_dim"] == 128
        assert cache_mod.kwargs["dtype"] == torch.uint8
        assert cache_mod.get_attn_backend() is indexer_ops.AttentionBackend

        # get_kv_cache_spec
        monkeypatch.setattr(
            indexer_ops.upstream.DeepseekV4IndexerCache,
            "get_kv_cache_spec",
            lambda self, cfg: "base_spec",
        )
        with mock.patch(
            "vllm_torchtpu.runner.kv_cache_dsv4.deepseek_v41_cache_spec",
            return_value="wrapped_spec",
        ) as mock_spec:
            assert cache_mod.get_kv_cache_spec(good_cfg) == "wrapped_spec"
            mock_spec.assert_called_once_with("base_spec", good_cfg, indexer=True)

        # bind_kv_cache: raw_cache is None and isinstance(cache, torch.Tensor)
        t_cache = torch.zeros((2, 128, 128), dtype=torch.uint8)
        cache_mod.bind_kv_cache(t_cache)
        assert cache_mod.bound_cache is t_cache

        # bind_kv_cache: invalid cache tuple
        with pytest.raises(ValueError, match="requires a \\(keys, scales\\) pair"):
            cache_mod.bind_kv_cache("invalid")
        with pytest.raises(ValueError, match="requires a \\(keys, scales\\) pair"):
            cache_mod.bind_kv_cache((t_cache,))

        # bind_kv_cache: 2-tuple with raw_cache=None
        scales = torch.zeros((2, 1, 128), dtype=torch.int32)
        cache_mod.bind_kv_cache((t_cache, scales), raw_cache=None)
        assert cache_mod.bound_cache is t_cache

        # bind_kv_cache: 2-tuple with raw_cache provided
        raw_cache = torch.zeros((64, 128), dtype=torch.int32)
        layout = ((2, 128, 128), torch.uint8, 128 * 128, 0)
        cache_mod.bind_kv_cache((t_cache, scales), raw_cache=raw_cache, layout=layout)
        assert cache_mod.native().pool is raw_cache
        assert cache_mod.native().scales is scales

    def test_index_record_layout_validation_and_success(self):
        raw = torch.zeros((64, 128), dtype=torch.int32)
        scales = torch.zeros((2, 1, 128), dtype=torch.int32)
        valid_layout = ((2, 128, 128), torch.uint8, 128 * 128, 0)

        # 1. Invalid dtype or trailing shape
        with pytest.raises(ValueError, match="128-byte uint8 keys"):
            indexer_ops.index_record_layout(
                raw, scales, ((2, 128, 64), torch.uint8, 128 * 128, 0)
            )

        # 2. Invalid raw tensor
        with pytest.raises(ValueError, match="native int32 word rows"):
            indexer_ops.index_record_layout(
                torch.zeros((64, 64), dtype=torch.int32), scales, valid_layout
            )

        # 3. Invalid stride_bytes or offset_bytes
        with pytest.raises(ValueError, match="complete word rows at pool offset zero"):
            indexer_ops.index_record_layout(
                raw, scales, ((2, 128, 128), torch.uint8, 128 * 128, 4)
            )

        # 4. Invalid scales tensor
        with pytest.raises(ValueError, match="companion int32 array"):
            indexer_ops.index_record_layout(raw, None, valid_layout)

        # 5. Shape[1] != page_size or raw.nbytes mismatch
        with pytest.raises(ValueError, match="fill the native NoPE array"):
            indexer_ops.index_record_layout(
                torch.zeros((32, 128), dtype=torch.int32), scales, valid_layout
            )

        # 6. Overflow int32 record addresses
        huge_pages = (torch.iinfo(torch.int32).max // 128) + 2
        fake_raw = SimpleNamespace(
            dtype=torch.int32,
            ndim=2,
            shape=(huge_pages * 32, 128),
            nbytes=huge_pages * 128 * 128,
        )
        fake_scales = SimpleNamespace(
            dtype=torch.int32,
            ndim=3,
            shape=(huge_pages, 1, 128),
            element_size=lambda: 4,
        )
        huge_layout = ((huge_pages, 128, 128), torch.uint8, 128 * 128, 0)
        with pytest.raises(ValueError, match="fit int32"):
            indexer_ops.index_record_layout(fake_raw, fake_scales, huge_layout)

        # 7. Valid record layout
        rec_layout = indexer_ops.index_record_layout(raw, scales, valid_layout)
        assert rec_layout.page_size == 128
        assert rec_layout.page_stride == 128

    def test_quantize_pack_and_gather_index_keys(self):
        with pytest.raises(ValueError, match="128 channels"):
            indexer_ops.quantize_index_rows(jnp.zeros((2, 64), dtype=jnp.bfloat16))

        keys_bf16 = jnp.ones((4, 128), dtype=jnp.bfloat16) * 2.0
        packed, scale_words = indexer_ops.pack_index_keys(keys_bf16)
        assert packed.shape == (4, 128)
        assert packed.dtype == jnp.uint8
        assert scale_words.shape == (4,)

        pool, scales = _make_native_pool_and_scales(2, 128, seed=3)
        # Non-multiple of 4 width -> arbitrary_rows branch
        rec_unaligned = jnp.array([[0, 1, 2], [4, 5, 6]], dtype=jnp.int32)
        k_unaligned, s_unaligned = indexer_ops.gather_index_keys(
            pool, scales, rec_unaligned, 128
        )
        assert k_unaligned.shape == (2, 3, 128)
        assert s_unaligned.shape == (2, 3)

        # Multiple of 4 width with grouped slabs -> four_key_slabs branch
        rec_aligned = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)
        k_aligned, s_aligned = indexer_ops.gather_index_keys(
            pool, scales, rec_aligned, 128
        )
        assert k_aligned.shape == (2, 4, 128)
        assert s_aligned.shape == (2, 4)
        np.testing.assert_allclose(
            np.asarray(k_aligned[:, :3].astype(jnp.float32)),
            np.asarray(k_unaligned[:, :3].astype(jnp.float32)),
        )

    def test_select_topk_and_select_indices_jax_branches(self, monkeypatch):
        scores = jnp.array(
            [
                [1.0, 5.0, 3.0, 2.0, -jnp.inf, 4.0, 0.5, 1.5] * 2,
                [2.0, 1.0, 6.0, 0.0, 3.0, -jnp.inf, 2.5, 0.5] * 2,
            ],
            dtype=jnp.float32,
        )

        # _select_topk with sparsecore=True and k=16 (mocking sparsecore_topk)
        def fake_sparsecore_topk(padded, k, row_lengths=None, return_scores=False):
            vals, idx = jax.lax.top_k(padded, k)
            # Mark last column as invalid (-1) to exercise underfull filler logic
            idx = idx.at[:, -1].set(-1)
            vals = vals.at[:, -1].set(-jnp.inf)
            words = jax.lax.bitcast_convert_type(vals, jnp.uint32)
            return idx, words

        with mock.patch(
            "vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk.sparsecore_topk",
            side_effect=fake_sparsecore_topk,
        ):
            vals_sc, idx_sc = indexer_ops._select_topk(scores, 16, sparsecore=True)
            assert vals_sc.shape == (2, 16)
            assert idx_sc.shape == (2, 16)

        # select_indices_jax validation errors
        cands = jnp.zeros((2, 2), dtype=jnp.int32)
        lens = jnp.array([8, 12], dtype=jnp.int32)
        with pytest.raises(ValueError, match="Positive top-k"):
            indexer_ops.select_indices_jax(scores, lens, cands, 0, 4, False, False)
        with pytest.raises(ValueError, match="must not filter their own scores"):
            indexer_ops.select_indices_jax(scores, lens, cands, 4, 4, True, True)
        with pytest.raises(ValueError, match="Candidate rows must match"):
            indexer_ops.select_indices_jax(
                scores, lens, jnp.zeros((3, 2), dtype=jnp.int32), 4, 4, False, False
            )

        # Branch: scores_masked=True, candidate_write=False, candidate_filter=False
        sel_masked, pub_masked = indexer_ops.select_indices_jax(
            scores, lens, cands, 4, 4, False, False, scores_masked=True
        )
        assert sel_masked.shape == (2, 4)
        assert pub_masked is cands

        # Branch: candidate_write=True and candidates.shape[1] * block_size < width
        sel_w, pub_w = indexer_ops.select_indices_jax(
            scores, lens, cands, 4, 4, True, False
        )
        assert sel_w.shape == (2, 4)
        assert pub_w.shape == (2, 2)

        # Branch: candidate_filter=True and candidates.shape[1] * block_size < width
        sel_f, pub_f = indexer_ops.select_indices_jax(
            scores,
            lens,
            jnp.array([[0, 1], [1, 2]], dtype=jnp.int32),
            4,
            4,
            False,
            True,
        )
        assert sel_f.shape == (2, 4)
        assert pub_f.shape == (2, 2)

    def test_paged_indexer_jax_full_and_candidate_context(self):
        pool, scales = _make_native_pool_and_scales(2, 128, seed=4)
        q = jnp.ones((2, 2, 128), dtype=jnp.bfloat16)
        weights = jnp.ones((2, 2), dtype=jnp.float32)
        records = jnp.arange(256, dtype=jnp.int32).reshape(2, 128)
        lengths = jnp.array([64, 96], dtype=jnp.int32)
        candidates = jnp.array([[0, 1, 1, -1], [1, 2, 0, 3]], dtype=jnp.int32)

        # 1. candidate_filter=False -> full_context
        sel_full, pub_full = indexer_ops.paged_indexer_jax(
            q,
            pool,
            scales,
            records,
            weights,
            lengths,
            candidates,
            topk=8,
            block_size=16,
            candidate_write=True,
            candidate_filter=False,
            page_stride=128,
        )
        assert sel_full.shape == (2, 8)
        assert pub_full.shape == (2, 4)

        # 2. candidate_filter=True and small candidate buffer -> candidate_context
        sel_cand, pub_cand = indexer_ops.paged_indexer_jax(
            q,
            pool,
            scales,
            records,
            weights,
            lengths,
            candidates,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=True,
            page_stride=128,
        )
        assert sel_cand.shape == (2, 8)
        assert pub_cand is candidates

    def test_bounded_indexer_jax_zero_padding_and_padded_paths(self):
        pool, scales = _make_native_pool_and_scales(2, 128, seed=5)
        block_table = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)

        # Case A: padding == 0 (count = 4 <= QUERY_TILE)
        q4 = jnp.ones((4, 2, 128), dtype=jnp.bfloat16)
        w4 = jnp.ones((4, 2), dtype=jnp.float32)
        lens4 = jnp.array([64, 96, 128, 80], dtype=jnp.int32)
        reqs4 = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        cands4 = jnp.array([[0, 1], [1, 2], [0, 2], [1, 0]], dtype=jnp.int32)

        # A1: neither candidate_write nor candidate_filter (padding=0)
        sel_a1, pub_a1 = indexer_ops.bounded_indexer_jax(
            q4,
            pool,
            scales,
            block_table,
            w4,
            lens4,
            cands4,
            reqs4,
            capacity=128,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            page_size=128,
            page_stride=128,
            row_offset=0,
        )
        assert sel_a1.shape == (4, 8)
        assert pub_a1 is cands4

        # A2: candidate_write=True (fori_loop full_context path, padding=0)
        sel_a2, pub_a2 = indexer_ops.bounded_indexer_jax(
            q4,
            pool,
            scales,
            block_table,
            w4,
            lens4,
            cands4,
            reqs4,
            capacity=128,
            topk=8,
            block_size=16,
            candidate_write=True,
            candidate_filter=False,
            page_size=128,
            page_stride=128,
            row_offset=0,
        )
        assert sel_a2.shape == (4, 8)
        assert pub_a2.shape == (4, 2)

        # A3: candidate_filter=True (fori_loop candidate_context path, padding=0)
        sel_a3, pub_a3 = indexer_ops.bounded_indexer_jax(
            q4,
            pool,
            scales,
            block_table,
            w4,
            lens4,
            cands4,
            reqs4,
            capacity=128,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=True,
            page_size=128,
            page_stride=128,
            row_offset=0,
        )
        assert sel_a3.shape == (4, 8)
        assert pub_a3.shape == (4, 2)

        # Case B: padding > 0 (count = 36 > QUERY_TILE=32 -> padded_count=64)
        q36 = jnp.ones((36, 1, 128), dtype=jnp.bfloat16)
        w36 = jnp.ones((36, 1), dtype=jnp.float32)
        lens36 = jnp.full((36,), 64, dtype=jnp.int32)
        reqs36 = jnp.zeros((36,), dtype=jnp.int32)
        cands36 = jnp.zeros((36, 2), dtype=jnp.int32)

        # B1: neither candidate_write nor candidate_filter with padding > 0
        sel_b1, pub_b1 = indexer_ops.bounded_indexer_jax(
            q36,
            pool,
            scales,
            block_table,
            w36,
            lens36,
            cands36,
            reqs36,
            capacity=128,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            page_size=128,
            page_stride=128,
            row_offset=0,
        )
        assert sel_b1.shape == (36, 8)
        assert pub_b1.shape == (36, 2)

        # B2: candidate_filter=True with padding > 0
        sel_b2, pub_b2 = indexer_ops.bounded_indexer_jax(
            q36,
            pool,
            scales,
            block_table,
            w36,
            lens36,
            cands36,
            reqs36,
            capacity=128,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=True,
            page_size=128,
            page_stride=128,
            row_offset=0,
        )
        assert sel_b2.shape == (36, 8)
        assert pub_b2.shape == (36, 2)

    def test_write_index_keys_and_update_and_select(self, monkeypatch):
        pool, scales = _make_native_pool_and_scales(2, 128, seed=6)

        # Validation errors
        with pytest.raises(ValueError, match="Index keys must be BF16"):
            indexer_ops.write_index_keys(
                pool,
                scales,
                jnp.array([0], dtype=jnp.int32),
                jnp.zeros((1, 128), dtype=jnp.float32),
                128,
            )
        with pytest.raises(ValueError, match="One physical record index per key"):
            indexer_ops.write_index_keys(
                pool,
                scales,
                jnp.array([0, 1], dtype=jnp.int32),
                jnp.zeros((1, 128), dtype=jnp.bfloat16),
                128,
            )

        # Empty keys early return
        p_empty, s_empty = indexer_ops.write_index_keys(
            pool,
            scales,
            jnp.zeros((0,), dtype=jnp.int32),
            jnp.zeros((0, 128), dtype=jnp.bfloat16),
            128,
        )
        assert p_empty is pool
        assert s_empty is scales

        # TPU backend branch
        monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
        with mock.patch.object(
            indexer_ops, "write_index_cache", return_value=("tpu_pool", "tpu_scales")
        ) as mock_wc:
            res_tpu = indexer_ops.write_index_keys(
                pool,
                scales,
                jnp.array([0], dtype=jnp.int32),
                jnp.ones((1, 128), dtype=jnp.bfloat16),
                128,
            )
            assert res_tpu == ("tpu_pool", "tpu_scales")
            mock_wc.assert_called_once()

        monkeypatch.setattr(jax, "default_backend", lambda: "cpu")

        # write_paged_index_keys and update_and_select_jax on CPU
        slots = jnp.array([0, 1, -1, 3], dtype=jnp.int32)
        keys = jnp.ones((4, 128), dtype=jnp.bfloat16) * 1.5
        p_updated, s_updated = indexer_ops.write_paged_index_keys(
            pool, scales, slots, keys, 128, 128, 0
        )
        assert p_updated.shape == pool.shape
        assert s_updated.shape == scales.shape

        block_table = jnp.array([[0, 1]], dtype=jnp.int32)
        q = jnp.ones((4, 1, 128), dtype=jnp.bfloat16)
        weights = jnp.ones((4, 1), dtype=jnp.float32)
        lengths = jnp.array([4, 4, 4, 4], dtype=jnp.int32)
        candidates = jnp.zeros((4, 2), dtype=jnp.int32)
        requests = jnp.zeros((4,), dtype=jnp.int32)

        u_pool, u_scales, sel, pub = indexer_ops.update_and_select_jax(
            q,
            pool,
            scales,
            block_table,
            weights,
            lengths,
            candidates,
            slots,
            keys,
            requests,
            capacity=128,
            topk=4,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            page_size=128,
            page_stride=128,
            row_offset=0,
        )
        assert u_pool.shape == pool.shape
        assert u_scales.shape == scales.shape
        assert sel.shape == (4, 4)
        assert pub.shape == (4, 2)

    def test_initialize_indexer_and_custom_op_wrappers(self, monkeypatch):
        monkeypatch.setattr(indexer_ops, "_indexer_op", None)
        monkeypatch.setattr(indexer_ops, "_write_index_op", None)

        registered_fakes = {}

        class _FakeOp:
            def __init__(self, name, fn):
                self.name = name
                self.fn = fn
                self.calls = []

            def register_fake(self, fake_fn):
                registered_fakes[self.name] = fake_fn

            def __call__(self, *args):
                self.calls.append(args)
                return registered_fakes[self.name](*args)

        def fake_jax_op(name, fn, **kwargs):
            return _FakeOp(name, fn)

        with (
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=fake_jax_op),
            mock.patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
                return_value=SimpleNamespace(mesh="dummy_mesh"),
            ),
        ):
            indexer_ops.initialize_indexer()
            first_op = indexer_ops._indexer_op
            assert first_op is not None
            # Second call returns early
            indexer_ops.initialize_indexer()
            assert indexer_ops._indexer_op is first_op

        layout = SimpleNamespace(geometry=(128, 128, 0))
        cache = torch.zeros((64, 128), dtype=torch.int32)
        scales = torch.zeros((2, 1, 128), dtype=torch.int32)
        keys = torch.zeros((4, 128), dtype=torch.bfloat16)

        # publish_index_keys with int64 slots and int32 slots
        indexer_ops.publish_index_keys(
            cache,
            scales,
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            keys,
            layout=layout,
        )
        assert indexer_ops._write_index_op.calls[-1][2].dtype == torch.int32

        slots_i32 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        indexer_ops.publish_index_keys(cache, scales, slots_i32, keys, layout=layout)
        assert indexer_ops._write_index_op.calls[-1][2] is slots_i32

        # paged_indexer with non-int32 / 2D lengths (converts/flattens)
        q = torch.zeros((4, 2, 128), dtype=torch.bfloat16)
        weights = torch.zeros((4, 2), dtype=torch.float32)
        out_converted = indexer_ops.paged_indexer(
            q,
            cache,
            scales,
            torch.zeros((2, 2), dtype=torch.int64),
            torch.zeros((4,), dtype=torch.int64),
            128,
            weights,
            torch.ones((4, 1), dtype=torch.int64),
            torch.zeros((4, 2), dtype=torch.int64),
            torch.zeros((4,), dtype=torch.int64),
            keys,
            layout=layout,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
        )
        assert len(out_converted) == 4
        assert out_converted[2].shape == (4, 8)

        # paged_indexer with already-int32 / 1D tensors (skips all conversions)
        bt_i32 = torch.zeros((2, 2), dtype=torch.int32)
        req_i32 = torch.zeros((4,), dtype=torch.int32)
        len_i32 = torch.ones((4,), dtype=torch.int32)
        cand_i32 = torch.zeros((4, 2), dtype=torch.int32)
        out_fast = indexer_ops.paged_indexer(
            q,
            cache,
            scales,
            bt_i32,
            req_i32,
            128,
            weights,
            len_i32,
            cand_i32,
            slots_i32,
            keys,
            layout=layout,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
        )
        assert len(out_fast) == 4
        last_args = indexer_ops._indexer_op.calls[-1]
        assert last_args[3] is bt_i32
        assert last_args[5] is len_i32
        assert last_args[6] is cand_i32
        assert last_args[7] is slots_i32
        assert last_args[9] is req_i32
