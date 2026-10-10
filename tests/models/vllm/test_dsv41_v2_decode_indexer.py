"""100% diff line and branch coverage for / decode indexer.

Target modules and diff lines:
1. vllm_torchtpu.models.vllm.deepseek_v41.v2.cache
   - gather_cache_page_values (lines 523-550):
     * ValueError on ROW_WORDS % record_words != 0
     * ValueError on (page_records * record_words) % ROW_WORDS != 0
     * Valid and negative (-1 sentinel) first_records
     * Bitwise equivalence against gather_cache_values(split=True)
2. vllm_torchtpu.models.vllm.deepseek_v41.v2.attention
   - _index_page_records (lines 389-410):
     * Not PagedCacheRows instance -> 0
     * dtype != torch.bfloat16 (e.g. fp8, fp32, None) -> 0
     * row_bytes % record_bytes != 0 -> 0
     * (per_page * record_bytes) % row_bytes != 0 -> 0
     * row_offset % per_row != 0 -> 0
     * page_stride % per_row != 0 -> 0
     * Valid qualifying layout -> per_page
   - DeepseekV41Attention._select_paged_indices (lines 1351-1490):
     * split=True (ranks > 1, part = (count + ranks - 1) // ranks, padding = ranks)
     * split=False (ranks = 1, padding = 0, no slice / no gather)
     * cached_query_split hit vs miss (stage_attention_cache_scope)
     * candidate_write=True vs candidate_write=False
     * candidate_filter=True vs False
     * page_records passed to paged_indexer
3. vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels
   - Constants _INDEX_KEYS_PER_ITER = 65536, _INDEX_MAX_TILE = 8192
   - paged_index_scores_jax (lines 1318-1360, 1410-1425):
     * tokens <= 8 -> _INDEX_MAX_TILE (8192) tile budget
     * tokens = 16 -> 4096 tile budget
     * tokens >= 128 -> 512 tile floor
     * width clamping (-(-width // 128) * 128)
     * requests is not None and tokens <= records.shape[0] (decode take)
     * requests is not None and tokens > records.shape[0] (prefill row_step)
     * requests is None
     * pages=True (gather_cache_page_values branch)
     * pages=False: page_records is None
     * pages=False: tile % page_records != 0
     * Numerical equivalence of pages=True vs pages=False
   - _visible_top_k (lines 1535-1555):
     * SparseCore path with rows < 16 (unpadded)
     * SparseCore path with rows >= 16
     * Fallback path: unordered=False, lengths=None, width < 2048,
       width % 16 != 0, k % 16 != 0
   - paged_indexer_jax, update_and_select_jax, initialize_indexer, paged_indexer
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.models.vllm.deepseek_v41.v2.attention as v2_attn
import vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels as v2_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import (
    _index_page_records,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _INDEX_KEYS_PER_ITER,
    _INDEX_MAX_TILE,
    _visible_top_k,
    initialize_indexer,
    paged_index_scores_jax,
    paged_indexer,
    update_and_select_jax,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    ROW_WORDS,
    PagedCacheRows,
    gather_cache_page_values,
    gather_cache_values,
)


class TestGatherCachePageValues:
    """Coverage for v2/cache.py gather_cache_page_values."""

    def test_value_error_row_words_not_divisible_by_record_words(self):
        """ROW_WORDS % record_words != 0 raises ValueError."""
        cache = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
        # elements = 10 -> record_words = 5; 128 % 5 != 0
        with pytest.raises(ValueError, match="Pages must cover whole pool word rows"):
            gather_cache_page_values(
                cache,
                jnp.array([0], dtype=jnp.int32),
                page_records=4,
                elements=10,
            )

    def test_value_error_page_records_not_covering_whole_rows(self):
        """(page_records * record_words) % ROW_WORDS != 0 raises ValueError."""
        cache = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
        # elements = 128 -> record_words = 64; 128 % 64 == 0
        # page_records = 3 -> 3 * 64 = 192 % 128 != 0
        with pytest.raises(ValueError, match="Pages must cover whole pool word rows"):
            gather_cache_page_values(
                cache,
                jnp.array([0], dtype=jnp.int32),
                page_records=3,
                elements=128,
            )

    def test_positive_and_negative_first_records(self):
        """Positive records gather contiguous page rows; negative take spread."""
        rng = np.random.default_rng(42)
        cache_data = rng.integers(-(2**31), 2**31, (64, ROW_WORDS)).astype(np.int32)
        cache = jnp.asarray(cache_data & 0x3FFF3FFF)  # finite BF16 words
        elements = 128  # record_words = 64 -> per_row = 2
        page_records = 4  # rows = 4 * 64 // 128 = 2 rows per page

        # first_records with valid offsets and negative sentinels (-1, -5)
        first_records = jnp.asarray([0, 2, -1, 4, -5], dtype=jnp.int32)
        out = gather_cache_page_values(cache, first_records, page_records, elements)
        assert out.shape == (5 * page_records, elements)
        assert out.dtype == jnp.bfloat16

        # Compare valid records against gather_cache_values(split=True)
        for idx in [0, 1, 3]:
            first = int(first_records[idx])
            page_slice = out[idx * page_records : (idx + 1) * page_records]
            ref_records = jnp.arange(first, first + page_records, dtype=jnp.int32)
            ref_values = gather_cache_values(cache, ref_records, elements, split=True)
            np.testing.assert_array_equal(
                np.asarray(page_slice), np.asarray(ref_values)
            )

    def test_single_row_page_and_small_cache(self):
        """Small cache and single-row pages cover spread %= max(1, ...)."""
        cache = jnp.zeros((1, ROW_WORDS), dtype=jnp.int32)
        elements = 256  # record_words = 128 -> per_row = 1
        page_records = 1  # rows = 1
        first_records = jnp.asarray([-1, 0, -3], dtype=jnp.int32)
        out = gather_cache_page_values(cache, first_records, page_records, elements)
        assert out.shape == (3, elements)
        assert out.dtype == jnp.bfloat16


class TestIndexPageRecords:
    """Coverage for v2/attention.py _index_page_records."""

    def test_not_paged_cache_rows_returns_zero(self):
        """Non-PagedCacheRows object returns 0."""
        assert _index_page_records(None) == 0
        assert _index_page_records(torch.zeros(10, 10)) == 0

    def test_non_bf16_dtype_returns_zero(self):
        """Dtype other than torch.bfloat16 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        layout_fp32 = ((4, 16, 64), torch.float32, 32 * 64 * 4, 0)
        cache_fp32 = PagedCacheRows(raw, layout_fp32)
        assert _index_page_records(cache_fp32) == 0

        # Mock PagedCacheRows with FP8 dtype
        mock_fp8 = MagicMock(spec=PagedCacheRows)
        mock_fp8.dtype = torch.float8_e4m3fn
        assert _index_page_records(mock_fp8) == 0

    def test_row_bytes_not_divisible_by_record_bytes_returns_zero(self):
        """row_bytes % record_bytes != 0 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 512, BF16 -> record_bytes = 1024; ROW_BYTES = 512; 512 % 1024 != 0
        layout = ((4, 16, 512), torch.bfloat16, 32 * 1024, 0)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 0

    def test_page_record_bytes_not_divisible_by_row_bytes_returns_zero(self):
        """(per_page * record_bytes) % row_bytes != 0 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 128 -> record_bytes = 256; ROW_BYTES = 512;
        # per_page = 3 -> 768 % 512 != 0
        layout = ((4, 3, 128), torch.bfloat16, 32 * 256, 0)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 0

    def test_row_offset_not_aligned_returns_zero(self):
        """index_cache.row_offset % per_row != 0 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 128 -> record_bytes = 256 -> per_row = 2
        # offset = 1 record -> offset % 2 == 1 != 0
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 1 * 256)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 0

    def test_page_stride_not_aligned_returns_zero(self):
        """index_cache.page_stride % per_row != 0 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 128 -> record_bytes = 256 -> per_row = 2
        # stride = 33 records -> stride % 2 == 1 != 0
        layout = ((4, 16, 128), torch.bfloat16, 33 * 256, 0)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 0

    def test_valid_qualifying_layout_returns_per_page(self):
        """Properly aligned BF16 layout returns per_page."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 128 -> record_bytes = 256 -> per_row = 2
        # per_page = 16, stride = 32, offset = 16 -> all % 2 == 0
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 16 * 256)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 16

    def test_2bank_lo_rows_alignment(self):
        """2-bank cache returns per_page if lo_rows*per_row % per_page == 0, else 0."""
        raw_lo_aligned = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        raw_hi = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 0)
        cache_aligned = PagedCacheRows((raw_lo_aligned, raw_hi), layout)
        assert _index_page_records(cache_aligned) == 16

        raw_lo_unaligned = torch.zeros((5, ROW_WORDS), dtype=torch.int32)
        cache_unaligned = PagedCacheRows((raw_lo_unaligned, raw_hi), layout)
        assert _index_page_records(cache_unaligned) == 0


class TestAttentionKernelsDecodeIndexer:
    """Coverage for v2/attention_kernels.py / changes."""

    def test_constants_and_tile_budget_spectrum(self):
        """Test _INDEX_KEYS_PER_ITER = 262144 and _INDEX_MAX_TILE = 16384 budget."""
        assert _INDEX_KEYS_PER_ITER == 262144
        assert _INDEX_MAX_TILE == 16384

        # 1. tokens <= 16: budget capped at _INDEX_MAX_TILE (16384)
        pool = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
        q = jnp.zeros((1, 2, 128), dtype=jnp.bfloat16)
        records = jnp.full((1, 16384), -1, dtype=jnp.int32)
        weights = jnp.zeros((1, 2), dtype=jnp.float32)
        lengths = jnp.array([10], dtype=jnp.int32)

        # tokens = 1 -> budget = min(max(512, min(16384, 262144 // 1)), 16384) = 16384
        scores = paged_index_scores_jax(q, pool, records, weights, lengths)
        assert scores.shape == (1, 16384)

        # 2. tokens = 16 -> min(max(512, min(16384, 262144 // 16)), 16384) = 16384
        q16 = jnp.zeros((16, 2, 128), dtype=jnp.bfloat16)
        weights16 = jnp.zeros((16, 2), dtype=jnp.float32)
        lengths16 = jnp.full((16,), 10, dtype=jnp.int32)
        scores16 = paged_index_scores_jax(q16, pool, records, weights16, lengths16)
        assert scores16.shape == (16, 16384)

        # 3. tokens = 512 -> budget hits 512 floor
        q512 = jnp.zeros((512, 2, 128), dtype=jnp.bfloat16)
        weights512 = jnp.zeros((512, 2), dtype=jnp.float32)
        lengths512 = jnp.full((512,), 10, dtype=jnp.int32)
        scores512 = paged_index_scores_jax(q512, pool, records, weights512, lengths512)
        assert scores512.shape == (512, 16384)

        # 4. Small width clamps budget via -(-width // 128) * 128
        records_short = jnp.full((1, 256), -1, dtype=jnp.int32)
        scores_short = paged_index_scores_jax(q, pool, records_short, weights, lengths)
        assert scores_short.shape == (1, 256)

    def test_paged_index_scores_pages_vs_unpaged_equivalence(self):
        """paged_index_scores_jax with pages=True vs pages=False match."""
        rng = np.random.default_rng(200)
        heads, dim = 2, 128
        tokens, width = 4, 1024
        page_records = 32

        pool = jnp.asarray(
            rng.integers(-(2**31), 2**31, (256, ROW_WORDS)), dtype=jnp.int32
        ) & jnp.int32(0x3FFF3FFF)
        records = rng.integers(0, 256 * 2, (tokens, width)).astype(np.int32)
        # Align records to contiguous page blocks of 32
        for r in range(tokens):
            for p in range(0, width, page_records):
                base = int(records[r, p] // 2 * 2)
                records[r, p : p + page_records] = base + np.arange(page_records)
        records[:, -10:] = -1  # invalid tail
        records = jnp.asarray(records)

        q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
        weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
        lengths = jnp.asarray(rng.integers(50, width, tokens), dtype=jnp.int32)

        # 1. Unpaged (page_records = None)
        unpaged_scores = paged_index_scores_jax(
            q, pool, records, weights, lengths, page_records=None
        )
        # 2. Paged (page_records = 32)
        paged_scores = paged_index_scores_jax(
            q, pool, records, weights, lengths, page_records=page_records
        )
        # 3. Non-dividing page_records (e.g. 300, tile % 300 != 0 -> pages = False)
        nondiv_scores = paged_index_scores_jax(
            q, pool, records, weights, lengths, page_records=300
        )

        valid = np.asarray(records >= 0)
        np.testing.assert_allclose(
            np.asarray(paged_scores)[valid],
            np.asarray(unpaged_scores)[valid],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(nondiv_scores)[valid],
            np.asarray(unpaged_scores)[valid],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_paged_index_scores_request_branches(self):
        """Covers requests is not None (decode tokens <= records vs prefill)."""
        heads, dim = 2, 128
        pool = jnp.zeros((64, ROW_WORDS), dtype=jnp.int32)

        # Branch 1a: requests is not None, tokens <= records.shape[0],
        # tokens >= 6 and tokens % 6 == 0 (grouped branch with tokens=6 and tokens=18)
        q_dec = jnp.zeros((18, heads, dim), dtype=jnp.bfloat16)
        records_dec = jnp.full((24, 256), -1, dtype=jnp.int32)
        weights_dec = jnp.zeros((18, heads), dtype=jnp.float32)
        lengths_dec = jnp.full((18,), 10, dtype=jnp.int32)
        requests_dec = jnp.zeros((18,), dtype=jnp.int32)

        res_dec_18 = paged_index_scores_jax(
            q_dec,
            pool,
            records_dec,
            weights_dec,
            lengths_dec,
            requests=requests_dec,
            page_records=16,
        )
        assert res_dec_18.shape == (18, 256)

        res_dec = paged_index_scores_jax(
            q_dec[:6],
            pool,
            records_dec,
            weights_dec[:6],
            lengths_dec[:6],
            requests=requests_dec[:6],
            page_records=16,
        )
        assert res_dec.shape == (6, 256)

        # Branch 1b: requests is not None, tokens <= records.shape[0],
        # tokens not divisible by 6 (tokens = 2 and tokens = 7)
        res_dec_2 = paged_index_scores_jax(
            q_dec[:2],
            pool,
            records_dec,
            weights_dec[:2],
            lengths_dec[:2],
            requests=requests_dec[:2],
            page_records=16,
        )
        assert res_dec_2.shape == (2, 256)

        res_dec_7 = paged_index_scores_jax(
            q_dec[:7],
            pool,
            records_dec,
            weights_dec[:7],
            lengths_dec[:7],
            requests=requests_dec[:7],
            page_records=16,
        )
        assert res_dec_7.shape == (7, 256)

        # Branch 1c: TP=4 DSpark K=5 decode where tokens = 24 > records.shape[0] = 16
        # but eff_tokens = tokens // 6 = 4 <= records.shape[0] = 16 (uses fast
        # _score_word_rows path instead of prefill row_step loop).
        q_tp4 = jnp.zeros((24, heads, dim), dtype=jnp.bfloat16)
        records_tp4 = jnp.full((16, 256), -1, dtype=jnp.int32)
        weights_tp4 = jnp.zeros((24, heads), dtype=jnp.float32)
        lengths_tp4 = jnp.full((24,), 10, dtype=jnp.int32)
        requests_tp4 = jnp.repeat(jnp.arange(4, dtype=jnp.int32), 6)

        res_tp4 = paged_index_scores_jax(
            q_tp4,
            pool,
            records_tp4,
            weights_tp4,
            lengths_tp4,
            requests=requests_tp4,
            page_records=16,
        )
        assert res_tp4.shape == (24, 256)

        # Branch 2: requests is not None and eff_tokens > records.shape[0]
        # (prefill row_step loop)
        q_pref = jnp.zeros((4, heads, dim), dtype=jnp.bfloat16)
        records_pref = jnp.full((2, 256), -1, dtype=jnp.int32)
        weights_pref = jnp.zeros((4, heads), dtype=jnp.float32)
        lengths_pref = jnp.array([10, 10, 20, 20], dtype=jnp.int32)
        requests_pref = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        res_pref = paged_index_scores_jax(
            q_pref,
            pool,
            records_pref,
            weights_pref,
            lengths_pref,
            requests=requests_pref,
            page_records=16,
        )
        assert res_pref.shape == (4, 256)

    def test_visible_top_k_sparsecore_and_fallback_branches(self, monkeypatch):
        """Covers _visible_top_k unpadded rows < 16, rows >= 16, and fallback."""
        sparsecore_calls = []

        def mock_sc_top_k(scores, k, lengths):
            sparsecore_calls.append((scores.shape, k))
            return jnp.zeros((scores.shape[0], k), dtype=jnp.int32)

        monkeypatch.setattr(v2_kernels, "_sparsecore_top_k", mock_sc_top_k)

        bound = jnp.int32(2048)

        # 1. SparseCore branch with rows < 16 (e.g. rows = 3, unpadded)
        scores_3 = jnp.zeros((3, 2048), dtype=jnp.float32)
        lengths_3 = jnp.array([100, 200, 300], dtype=jnp.int32)
        out_3 = _visible_top_k(scores_3, 16, bound, lengths_3, unordered=True)
        assert out_3.shape == (3, 16)
        assert sparsecore_calls[-1] == ((3, 2048), 16)

        # 2. SparseCore branch with rows >= 16 (e.g. rows = 16)
        scores_16 = jnp.zeros((16, 2048), dtype=jnp.float32)
        lengths_16 = jnp.full((16,), 100, dtype=jnp.int32)
        out_16 = _visible_top_k(scores_16, 32, bound, lengths_16, unordered=True)
        assert out_16.shape == (16, 32)
        assert sparsecore_calls[-1] == ((16, 2048), 32)

        # 3. Fallback branch: unordered=False (calls select via switch)
        scores_fb = jnp.zeros((2, 2048), dtype=jnp.float32)
        lengths_fb = jnp.array([10, 20], dtype=jnp.int32)
        out_fb = _visible_top_k(scores_fb, 16, bound, lengths_fb, unordered=False)
        assert out_fb.shape == (2, 16)

        # 4. Fallback branch: width < 2048
        scores_narrow = jnp.zeros((2, 1024), dtype=jnp.float32)
        out_narrow = _visible_top_k(
            scores_narrow, 16, jnp.int32(1024), lengths_fb, unordered=True
        )
        assert out_narrow.shape == (2, 16)

        # 5. Fallback branch: width % 16 != 0
        scores_unaligned_w = jnp.zeros((2, 2050), dtype=jnp.float32)
        out_unaligned_w = _visible_top_k(
            scores_unaligned_w, 16, jnp.int32(2050), lengths_fb, unordered=True
        )
        assert out_unaligned_w.shape == (2, 16)

        # 6. Fallback branch: k % 16 != 0
        out_unaligned_k = _visible_top_k(
            scores_fb, 10, bound, lengths_fb, unordered=True
        )
        assert out_unaligned_k.shape == (2, 10)

        # 7. Fallback branch: lengths is None
        out_no_len = _visible_top_k(scores_fb, 16, bound, None, unordered=True)
        assert out_no_len.shape == (2, 16)

        # 8. Fallback branch: scores.shape[0] >= 16 and columns >= 2048
        # (triggers _threshold_top_k)
        scores_thresh = jnp.zeros((16, 2048), dtype=jnp.float32)
        out_thresh = _visible_top_k(
            scores_thresh, 16, bound, lengths_16, unordered=False
        )
        assert out_thresh.shape == (16, 16)

    def test_paged_indexer_and_wrappers_dispatch(self, monkeypatch):
        """Covers paged_indexer, update_and_select_jax, and initialize_indexer."""
        # 1. paged_indexer wrapper with batch_max=None and page_records
        called_args = []

        def mock_op(*args):
            called_args.append(args)
            return (
                args[1],
                torch.zeros(args[0].shape[0], args[9], dtype=torch.int32),
                args[5],
            )

        monkeypatch.setattr(v2_kernels, "_indexer_op", mock_op)

        q = torch.zeros(4, 2, 128)
        pool = torch.zeros(16, ROW_WORDS, dtype=torch.int32)
        cache_obj = SimpleNamespace(raw=pool)
        records = torch.zeros(4, 256, dtype=torch.int32)
        weights = torch.zeros(4, 256)
        lengths = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        candidates = torch.zeros(4, 8, dtype=torch.int32)
        write_records = torch.zeros(0, dtype=torch.int32)
        write_keys = torch.zeros(0, 128)
        requests = torch.arange(4, dtype=torch.int32)

        # Without batch_max (defaults to lengths.max()) and without
        # page_records (defaults to 0)
        paged_indexer(
            q,
            cache_obj,
            records,
            weights,
            lengths,
            candidates,
            write_records,
            write_keys,
            requests,
            topk=16,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
        )
        assert len(called_args) == 1
        # page_records arg is index 15
        assert called_args[0][15] == 0
        # batch_max arg is index 9
        assert int(called_args[0][9].item()) == 40

        # With explicit batch_max and page_records
        paged_indexer(
            q,
            cache_obj,
            records,
            weights,
            lengths,
            candidates,
            write_records,
            write_keys,
            requests,
            topk=16,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            batch_max=torch.tensor(50, dtype=torch.int32),
            page_records=32,
        )
        assert len(called_args) == 2
        assert called_args[1][15] == 32
        assert int(called_args[1][9].item()) == 50

        # 2. update_and_select_jax with page_records=0 vs page_records=64
        q_j = jnp.zeros((2, 2, 128), dtype=jnp.bfloat16)
        pool_j = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
        records_j = jnp.full((2, 256), -1, dtype=jnp.int32)
        weights_j = jnp.zeros((2, 2), dtype=jnp.float32)
        lengths_j = jnp.array([10, 20], dtype=jnp.int32)
        cand_j = jnp.full((2, 4), -1, dtype=jnp.int32)
        wrec_j = jnp.zeros((0,), dtype=jnp.int32)
        wkeys_j = jnp.zeros((0, 128), dtype=jnp.bfloat16)
        req_j = jnp.arange(2, dtype=jnp.int32)

        up_pool0, sel0, pub0 = update_and_select_jax(
            q_j,
            pool_j,
            records_j,
            weights_j,
            lengths_j,
            cand_j,
            wrec_j,
            wkeys_j,
            req_j,
            topk=16,
            block_size=16,
            batch_max=jnp.array(20, dtype=jnp.int32),
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
            page_records=0,
        )
        assert sel0.shape == (2, 16)

        up_pool64, sel64, pub64 = update_and_select_jax(
            q_j,
            pool_j,
            records_j,
            weights_j,
            lengths_j,
            cand_j,
            wrec_j,
            wkeys_j,
            req_j,
            topk=16,
            block_size=16,
            batch_max=jnp.array(20, dtype=jnp.int32),
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
            page_records=64,
        )
        assert sel64.shape == (2, 16)

    def test_initialize_indexer_and_fake_handler(self):
        """Covers initialize_indexer custom op and fake tensor registration."""
        from jax.sharding import Mesh

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with set_vllm_model_wrapper_context(mesh=mesh):
            v2_kernels._indexer_op = None
            initialize_indexer()
            assert v2_kernels._indexer_op is not None
            # Test early return branch
            saved_op = v2_kernels._indexer_op
            initialize_indexer()
            assert v2_kernels._indexer_op is saved_op

        # Verify fake registration by calling op with meta device tensors
        q = torch.empty((4, 2, 128), dtype=torch.bfloat16, device="meta")
        pool = torch.empty((16, ROW_WORDS), dtype=torch.int32, device="meta")
        records = torch.empty((4, 256), dtype=torch.int32, device="meta")
        weights = torch.empty((4, 256), dtype=torch.float32, device="meta")
        lengths = torch.empty((4,), dtype=torch.int32, device="meta")
        candidates = torch.empty((4, 8), dtype=torch.int32, device="meta")
        write_recs = torch.empty((0,), dtype=torch.int32, device="meta")
        keys = torch.empty((0, 128), dtype=torch.bfloat16, device="meta")
        requests = torch.empty((4,), dtype=torch.int32, device="meta")
        batch_max = torch.empty((1,), dtype=torch.int32, device="meta")

        f_pool, f_sel, f_pub = v2_kernels._indexer_op(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            write_recs,
            keys,
            requests,
            batch_max,
            16,
            16,
            False,
            False,
            False,
            32,
        )
        assert f_pool.shape == pool.shape
        assert f_sel.shape == (4, 16)
        assert f_pub.shape == candidates.shape


class TestAttentionSelectPagedIndicesDecode:
    """Coverage for v2/attention.py _select_paged_indices / changes."""

    def _setup_mock_attn(self, monkeypatch, ranks=8, tp_rank=0, candidate_write=False):
        """Constructs an attention module mock with target attributes."""
        attn: v2_attn.DeepseekV41Attention = object.__new__(
            v2_attn.DeepseekV41Attention
        )
        torch.nn.Module.__init__(attn)
        attn.compress_ratio = 2
        attn._tp_rank = tp_rank
        attn._fp8_kv = True
        attn._config = {"index_topk": 4}
        attn.candidate_block_size = 16
        attn.candidate_block_buffer = torch.full((128, 8), -1, dtype=torch.int32)
        attn.indexer = SimpleNamespace(
            head_dim=128,
            n_head=2,
            indexer_op=SimpleNamespace(
                candidate_blocks=torch.zeros(1),
                candidate_write=candidate_write,
            ),
        )
        attn._project_index_q = lambda q, p: q

        monkeypatch.setattr(
            v2_attn, "get_tensor_model_parallel_world_size", lambda: ranks
        )
        monkeypatch.setattr(
            v2_attn,
            "_gather_query_rows",
            lambda x: torch.cat([x] * ranks, dim=0),
        )
        return attn

    @pytest.mark.parametrize("count", [1, 16, 17, 32])
    def test_select_paged_indices_split_part_and_padding_calculation(
        self, count, monkeypatch
    ):
        """Verifies part = (count + ranks - 1) // ranks and padding = ranks."""
        ranks = 8
        attn = self._setup_mock_attn(monkeypatch, ranks=ranks)

        captured_args = {}

        def mock_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            captured_args["q_part_shape"] = q_part.shape
            captured_args["lengths_part_shape"] = lengths_part.shape
            captured_args["page_records"] = kwargs.get("page_records")
            expected_part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
            return (
                index_cache.raw,
                torch.zeros(expected_part, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", mock_paged_indexer)

        # Mock PagedCacheRows with qualifying layout (per_page = 16)
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 16 * 256)
        index_cache_mock = PagedCacheRows(raw, layout)

        qr = torch.zeros(count, 128)
        positions = torch.arange(count, dtype=torch.int32)
        records = torch.zeros(count, 256, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)
        weights = torch.zeros(count, 2)
        lengths = torch.full((count, 1), 50, dtype=torch.int32)
        swa_slots = torch.zeros(count, 1, dtype=torch.int32)

        with v2_attn.stage_attention_cache_scope():
            selected = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )

        assert selected.shape == (count, 4)
        expected_part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
        assert captured_args["q_part_shape"][0] == expected_part
        assert captured_args["lengths_part_shape"][0] == expected_part
        assert captured_args["page_records"] == 16

    def test_select_paged_indices_candidate_write_and_published_buffer(
        self, monkeypatch
    ):
        """candidate_write=True gathers published rows and updates buffer."""
        ranks = 2
        count = 6
        attn = self._setup_mock_attn(monkeypatch, ranks=ranks, candidate_write=True)

        def mock_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            part = q_part.shape[0]
            # published part
            pub_part = torch.full((part, 8), 42, dtype=torch.int32)
            return (
                index_cache.raw,
                torch.zeros(part, 4, dtype=torch.int32),
                pub_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", mock_paged_indexer)

        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        index_cache_mock = SimpleNamespace(raw=raw)

        qr = torch.zeros(count, 128)
        positions = torch.arange(count, dtype=torch.int32)
        records = torch.zeros(count, 256, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)
        weights = torch.zeros(count, 2)
        lengths = torch.full((count, 1), 50, dtype=torch.int32)
        swa_slots = torch.zeros(count, 1, dtype=torch.int32)

        # Also supply pending_index_write to test pending_index_write is not None
        pending_write = (
            torch.tensor([1, 2], dtype=torch.int32),
            torch.zeros(2, 128),
        )

        selected = attn._select_paged_indices(
            qr,
            positions,
            index_cache_mock,
            records,
            requests,
            weights,
            lengths,
            swa_slots,
            pending_write,
        )

        assert selected.shape == (count, 4)
        # candidate_block_buffer [:count] must have received 42
        assert (attn.candidate_block_buffer[:count] == 42).all()
        # and tail must be padded with -1
        assert (attn.candidate_block_buffer[count:] == -1).all()

    def test_select_paged_indices_split_false_and_no_candidates(self, monkeypatch):
        """split=False (ranks=1) bypasses query splitting and row gathers."""
        ranks = 1
        count = 5
        attn = self._setup_mock_attn(monkeypatch, ranks=ranks)
        attn.indexer.indexer_op.candidate_blocks = None  # has_candidates = False

        gather_called = []
        monkeypatch.setattr(
            v2_attn,
            "_gather_query_rows",
            lambda x: gather_called.append(True) or x,
        )

        def mock_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            assert q_part.shape[0] == count
            assert candidates_part.shape == (count, 0)
            return (
                index_cache.raw,
                torch.zeros(count, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", mock_paged_indexer)

        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        index_cache_mock = SimpleNamespace(raw=raw)

        qr = torch.zeros(count, 128)
        positions = torch.arange(count, dtype=torch.int32)
        records = torch.zeros(count, 256, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)
        weights = torch.zeros(count, 2)
        lengths = torch.full((count, 1), 50, dtype=torch.int32)
        swa_slots = torch.zeros(count, 1, dtype=torch.int32)

        selected = attn._select_paged_indices(
            qr,
            positions,
            index_cache_mock,
            records,
            requests,
            weights,
            lengths,
            swa_slots,
            None,
        )

        assert selected.shape == (count, 4)
        assert len(gather_called) == 0  # No gather called when split=False

    def test_stage_cache_hit_and_miss_branches(self, monkeypatch):
        """stage_cache miss computes split; subsequent call hits cached_query_split."""
        ranks = 4
        count = 12
        attn = self._setup_mock_attn(monkeypatch, ranks=ranks)

        call_count = 0

        def mock_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            nonlocal call_count
            call_count += 1
            part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
            assert q_part.shape[0] == part
            return (
                index_cache.raw,
                torch.zeros(part, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", mock_paged_indexer)

        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        index_cache_mock = SimpleNamespace(raw=raw)

        qr = torch.zeros(count, 128)
        positions = torch.arange(count, dtype=torch.int32)
        records = torch.zeros(count, 256, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)
        weights = torch.zeros(count, 2)
        lengths = torch.full((count, 1), 50, dtype=torch.int32)
        swa_slots = torch.zeros(count, 1, dtype=torch.int32)

        with v2_attn.stage_attention_cache_scope():
            # Call 1: cache miss
            sel1 = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )
            assert sel1.shape == (count, 4)
            assert call_count == 1

            # Call 2: cache hit
            sel2 = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )
            assert sel2.shape == (count, 4)
            assert call_count == 2

            # Call 3: 2-bank PagedCacheRows returning (lo, hi) tuple
            raw_lo = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
            raw_hi = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
            layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 0)
            index_cache_2b = PagedCacheRows((raw_lo, raw_hi), layout)
            monkeypatch.setattr(
                v2_attn,
                "paged_indexer",
                lambda q_part, idx_c, *a, **kw: (
                    (torch.ones_like(raw_lo), torch.full_like(raw_hi, 2)),
                    torch.zeros(q_part.shape[0], 4, dtype=torch.int32),
                    a[3],
                ),
            )
            sel3 = attn._select_paged_indices(
                qr,
                positions,
                index_cache_2b,
                records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )
            assert sel3.shape == (count, 4)
            assert torch.all(index_cache_2b.raw == 1)
            assert torch.all(index_cache_2b.raw_hi == 2)

    def test_prestrided_pages_and_verify_tile_192(self):
        """Verifies prestrided_pages=True and 16K tile preservation up to tokens=192."""
        rng = np.random.default_rng(77)
        heads, dim = 2, 128
        tokens, width, page_records = 6, 1024, 32
        pool = jnp.asarray(
            rng.integers(-(2**31), 2**31, (256, ROW_WORDS)), dtype=jnp.int32
        ) & jnp.int32(0x3FFF3FFF)
        records = np.zeros((tokens, width), dtype=np.int32)
        for r in range(tokens):
            for p in range(0, width, page_records):
                base = int(rng.integers(0, 256) * 2)
                records[r, p : p + page_records] = base + np.arange(page_records)
        records[:, -page_records:] = -1
        records_j = jnp.asarray(records)
        first_records_j = records_j[:, ::page_records]

        q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
        weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
        lengths = jnp.full((tokens,), width, dtype=jnp.int32)

        scores_full = paged_index_scores_jax(
            q, pool, records_j, weights, lengths, page_records=page_records
        )
        scores_prestrided = paged_index_scores_jax(
            q,
            pool,
            first_records_j,
            weights,
            lengths,
            page_records=page_records,
            prestrided_pages=True,
        )
        np.testing.assert_array_equal(
            np.asarray(scores_prestrided), np.asarray(scores_full)
        )

        # Verify tokens=144, 168, 192 with requests (DP=2 x TP=4 verify batches)
        # maintain 16K tile budget without collapsing to 8K.
        for tok in (144, 168, 192):
            n_req = tok // 6
            q_v = jnp.zeros((tok, heads, dim), dtype=jnp.bfloat16)
            rec_v = jnp.full((n_req, 16384), -1, dtype=jnp.int32)
            w_v = jnp.zeros((tok, heads), dtype=jnp.float32)
            len_v = jnp.full((tok,), 32, dtype=jnp.int32)
            req_v = jnp.repeat(jnp.arange(n_req, dtype=jnp.int32), 6)
            out_v = paged_index_scores_jax(
                q_v,
                pool,
                rec_v,
                w_v,
                len_v,
                requests=req_v,
                page_records=page_records,
            )
            assert out_v.shape == (tok, 16384)

        # Cover pages=True with non-bfloat16 q (gather_cache_page_values step loop)
        scores_fp32 = paged_index_scores_jax(
            q.astype(jnp.float32),
            pool,
            records_j,
            weights,
            lengths,
            page_records=page_records,
        )
        assert scores_fp32.shape == (tokens, width)

    def test_paged_indexer_prestrided_candidate_and_full_context(self):
        """Verifies pre-strided candidate_context and full_context gathers."""
        rng = np.random.default_rng(88)
        tokens, width, topk = 6, 1024, 64
        heads, dim = 2, 128
        pool = jnp.asarray(
            rng.integers(-(2**31), 2**31, (512, ROW_WORDS)), dtype=jnp.int32
        ) & jnp.int32(0x3FFF3FFF)

        for block_size, page_records in ((8, 32), (16, 8)):
            records = np.zeros((tokens, width), dtype=np.int32)
            for r in range(tokens):
                for p in range(0, width, page_records):
                    base = int(rng.integers(0, 512) * 2)
                    records[r, p : p + page_records] = base + np.arange(page_records)
            records[0, -page_records:] = -1
            records_j = jnp.asarray(records)

            q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
            weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
            lengths = jnp.asarray(rng.integers(128, width, tokens), dtype=jnp.int32)
            lengths = lengths.at[-1].set(0)
            candidates = jnp.asarray(
                rng.integers(-1, width // block_size + 2, (tokens, 16)),
                dtype=jnp.int32,
            )
            requests = jnp.arange(tokens, dtype=jnp.int32)

            for cand_filter in (False, True):
                sel_ref, pub_ref = v2_kernels.paged_indexer_jax(
                    q,
                    pool,
                    records_j,
                    weights,
                    lengths,
                    candidates,
                    requests,
                    topk=topk,
                    block_size=block_size,
                    candidate_write=not cand_filter,
                    candidate_filter=cand_filter,
                    unordered=False,
                    page_records=None,
                )
                sel_paged, pub_paged = v2_kernels.paged_indexer_jax(
                    q,
                    pool,
                    records_j,
                    weights,
                    lengths,
                    candidates,
                    requests,
                    topk=topk,
                    block_size=block_size,
                    candidate_write=not cand_filter,
                    candidate_filter=cand_filter,
                    unordered=False,
                    page_records=page_records,
                )
                np.testing.assert_array_equal(
                    np.asarray(sel_paged), np.asarray(sel_ref)
                )
                np.testing.assert_array_equal(
                    np.asarray(pub_paged), np.asarray(pub_ref)
                )

    @pytest.mark.parametrize("use_owner_perm", [False, True])
    def test_td8221_select_paged_indices_clamped_mine_parity_and_no_cat(
        self, monkeypatch, use_owner_perm
    ):
        """TD8.221: clamped_mine + masked_fill matches reference without torch.cat."""
        orig_cat = torch.cat
        ranks = 4
        count = 10
        tp_rank = 2
        attn = self._setup_mock_attn(
            monkeypatch, ranks=ranks, tp_rank=tp_rank, candidate_write=False
        )
        attn._tp_rank_1d = torch.tensor([tp_rank], dtype=torch.int32)
        attn.candidate_block_buffer = torch.arange(128 * 8, dtype=torch.int32).reshape(
            128, 8
        )
        captured_parts: list[tuple[torch.Tensor, ...]] = []

        def capture_project_q(q_part, pos_part):
            captured_parts.append((q_part, pos_part))
            return q_part

        attn._project_index_q = capture_project_q
        captured_indexer: list[tuple[torch.Tensor, ...]] = []

        def fake_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            captured_indexer.append(
                (weights_part, lengths_part, candidates_part, requests_part)
            )
            return (
                index_cache.raw,
                torch.zeros(q_part.shape[0], 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", fake_paged_indexer)
        monkeypatch.setattr(
            v2_attn, "_gather_query_rows", lambda s: s.repeat(ranks, 1)[:count]
        )

        cat_calls: list[int] = []

        def spy_cat(tensors, *args, **kwargs):
            cat_calls.append(len(tensors))
            return orig_cat(tensors, *args, **kwargs)

        monkeypatch.setattr(torch, "cat", spy_cat)

        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        index_cache_mock = SimpleNamespace(raw=raw)
        qr = torch.arange(count * 16, dtype=torch.float32).reshape(count, 16) + 1.0
        positions = torch.arange(100, 100 + count, dtype=torch.int32)
        records = torch.zeros(count, 32, dtype=torch.int32)
        requests = torch.arange(20, 20 + count, dtype=torch.int32)
        weights = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2) + 0.5
        lengths = torch.arange(1, count + 1, dtype=torch.int32)[:, None]
        swa_slots = torch.arange(count, dtype=torch.int32)
        swa_slots[1] = -1

        if use_owner_perm:
            owner_perm = torch.tensor(
                [0, 1, 2, -1, 3, 4, -1, 5, 9, -1, 0, 1, 6, 7, 8, -1],
                dtype=torch.int32,
            )
            part = 4
            padding = 1
            ref_mine = torch.tensor([9, count, 0, 1], dtype=torch.int32)
        else:
            owner_perm = None
            part, padding = v2_attn._query_split(count, ranks, kv_shard=False)
            ref_mine = tp_rank * part + torch.arange(part, dtype=torch.int32)

        active_lengths = lengths.masked_fill(swa_slots[:, None] < 0, 0)
        ref_qr = orig_cat((qr, qr.new_full((padding, 16), 0))).index_select(0, ref_mine)
        ref_pos = orig_cat((positions, positions.new_full((padding,), 0))).index_select(
            0, ref_mine
        )
        ref_w = orig_cat((weights, weights.new_full((padding, 2), 0))).index_select(
            0, ref_mine
        )
        ref_l = orig_cat(
            (active_lengths, active_lengths.new_full((padding, 1), 0))
        ).index_select(0, ref_mine)
        ref_req = orig_cat((requests, requests.new_full((padding,), 0))).index_select(
            0, ref_mine
        )
        candidates = attn.candidate_block_buffer[:count]
        ref_cand = (
            attn.candidate_block_buffer[:part]
            if use_owner_perm
            else orig_cat(
                (candidates, candidates.new_full((padding, 8), -1))
            ).index_select(0, ref_mine)
        )
        cat_calls.clear()

        with v2_attn.stage_attention_cache_scope():
            for _ in range(2):
                attn._select_paged_indices(
                    qr,
                    positions,
                    index_cache_mock,
                    records,
                    requests,
                    weights,
                    lengths,
                    swa_slots,
                    None,
                    owner_perm=owner_perm,
                )

        assert cat_calls == []
        assert len(captured_parts) == 2
        for (q_p, pos_p), (w_p, l_p, c_p, req_p) in zip(
            captured_parts, captured_indexer, strict=True
        ):
            assert torch.equal(q_p, ref_qr)
            assert torch.equal(pos_p, ref_pos)
            assert torch.equal(w_p, ref_w)
            assert torch.equal(l_p, ref_l)
            assert torch.equal(c_p, ref_cand)
            assert torch.equal(req_p, ref_req)
