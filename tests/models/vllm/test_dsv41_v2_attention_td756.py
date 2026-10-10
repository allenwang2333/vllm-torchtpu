# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Wave 7 Track TD7.56:
- `vllm_torchtpu.models.vllm.deepseek_v41.v2.attention`:
  1. Precomputed `self._indexer_weight_scale` in `DeepseekV41Attention.__init__`
     and single-multiply `weights = index_weights.float() * _indexer_weight_scale`
     in `_prepare_tpu`.
  2. Stage-scoped `swa_row_slots_masked` caching in `swa_slots_key` and zero-op
     reuse of `records = swa_row_slots_masked` when `swa_cache.row_offset == 0`.
  3. Stage-scoped `index_records` caching (`idx_rec_key`) and zero-offset fast
     path (`index_cache.row_offset == 0`) in `_prepare_tpu`.
  4. Stage-scoped `(lengths, lengths_comp, swa_valid_comp, requests_comp)`
     caching (`comp_inputs_key`) across all indexer source layers in both
     non-segmented and `segmented_main` modes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


def _make_paged_cache_mock(
    pages_tensor: torch.Tensor | None,
    *,
    row_offset: int | None = 0,
    rows_per_page: int = 32,
    page_stride: int = 32,
    sharded: bool = False,
) -> Any:
    c = MagicMock(spec=PagedCacheRows)
    c.numel.return_value = 1024
    c.shape = (16, rows_per_page, 512)
    c.page_stride = page_stride
    c.row_offset = row_offset
    c.sharded = sharded
    c.row_indices = lambda slots, ro=(row_offset or 0): ro + (
        slots // rows_per_page * page_stride + slots % rows_per_page
    )
    c.pages.return_value = pages_tensor
    return c


def _make_attn(
    *,
    swa_cache: Any,
    main_cache: Any | None = None,
    idx_cache: Any | None = None,
    has_indexer: bool = True,
    compress_ratio: int = 2,
) -> Any:
    attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "swa_pfx"
    attn.compressed_cache_prefix = "src_pfx"
    attn.index_source_layer_id = 0
    attn.window_size = 64
    attn.compress_ratio = compress_ratio
    attn._max_model_len = 16
    attn._config = {"index_topk": 2}
    attn._fp8_kv = True
    attn._ced_decoder = False
    attn._ced_global_from_encoder = False
    attn._local_q_attn = False
    attn.n_local_heads = 2
    attn.head_dim = 8
    attn.scale = 0.25
    attn.attn_sink = torch.zeros(2)
    attn._swa_max_requests = 16
    attn._precompact_records = False
    attn._token_parallel_attn = False
    attn._full_attn_sink = False
    attn._tp_rank = torch.tensor(0, dtype=torch.int32)
    attn._tp_rank_1d = torch.tensor([0], dtype=torch.int32)
    attn.topk_indices_buffer = torch.full((16, 4), -1, dtype=torch.int32)
    attn._rotate = lambda x, p: x
    attn.wq_b = lambda qr: qr
    attn.compressor = None
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_pfx", kv_cache=swa_cache)
    attn.kv_cache = swa_cache
    resolved_main = main_cache if main_cache is not None else swa_cache
    attn._static_forward_context = {
        "src_pfx": SimpleNamespace(kv_cache=resolved_main),
        "swa_pfx": SimpleNamespace(kv_cache=swa_cache),
    }
    if has_indexer:
        resolved_idx = idx_cache if idx_cache is not None else swa_cache
        attn.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            wk=lambda lat: lat,
            k_norm=lambda k: k,
            k_cache=SimpleNamespace(
                prefix="idx_pfx",
                compress_ratio=compress_ratio,
                kv_cache=resolved_idx,
            ),
        )
        attn._indexer_weight_scale = (8 * 2) ** -0.5
    else:
        attn.indexer = None
        attn._indexer_weight_scale = 1.0
    attn._select_paged_indices = lambda *a, **kw: torch.tensor(
        [[0, 1], [0, 2], [1, 2], [0, 1]][: a[1].shape[0]], dtype=torch.int32
    )
    return attn


@pytest.mark.cpu_test
class TestV2AttentionTD756:
    def test_init_precomputes_indexer_weight_scale(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def fake_super_init(self, *args, **kwargs):
            torch.nn.Module.__init__(self)
            self.kv_cache_torch_dtype = torch.bfloat16
            self.rotary_emb = SimpleNamespace(
                cos_sin_cache=torch.zeros((8, 4), dtype=torch.float32)
            )
            self.attn_sink = torch.zeros(2, dtype=torch.float32)
            self.indexer = getattr(self, "_preset_indexer", None)

        monkeypatch.setattr(
            v2_attn.upstream.DeepseekV4Attention, "__init__", fake_super_init
        )
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)

        vllm_cfg = SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype="auto"),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: {"index_topk": 2}),
                max_model_len=64,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
        )

        # 1. With indexer is None
        attn_no_idx = v2_attn.DeepseekV41Attention(vllm_cfg)
        assert attn_no_idx._indexer_weight_scale == 1.0

        # 2. With indexer present (head_dim=8, n_head=2 -> (16)**-0.5 == 0.25)
        def fake_super_init_with_indexer(self, *args, **kwargs):
            fake_super_init(self, *args, **kwargs)
            self.indexer = SimpleNamespace(head_dim=8, n_head=2)

        monkeypatch.setattr(
            v2_attn.upstream.DeepseekV4Attention,
            "__init__",
            fake_super_init_with_indexer,
        )
        attn_with_idx = v2_attn.DeepseekV41Attention(vllm_cfg)
        assert attn_with_idx._indexer_weight_scale == pytest.approx(0.25)

    def test_swa_row_slots_masked_reuse_and_offset_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        count = 4
        positions = torch.tensor([0, 16, 32, 48], dtype=torch.int32)
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        meta = SimpleNamespace(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            block_tables=table,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        metadata = {"swa_pfx": meta, "src_pfx": meta, "idx_pfx": meta}
        forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)

        captured_swa_records: list[torch.Tensor] = []
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention",
            lambda q, swa_c, main_c, recs, *a, **kw: (
                captured_swa_records.append(recs) or torch.zeros_like(q)
            ),
        )

        qr = torch.ones((count, 16), dtype=torch.bfloat16)
        kv = torch.ones((count, 8), dtype=torch.bfloat16)
        out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)

        # 1. row_offset == 0 inside stage_attention_cache_scope -> reuses exact same
        #    swa_row_slots_masked tensor object across layers (0 tensor ops per layer)
        swa_zero = _make_paged_cache_mock(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), row_offset=0
        )
        attn_zero_1 = _make_attn(
            swa_cache=swa_zero, has_indexer=False, compress_ratio=0
        )
        attn_zero_2 = _make_attn(
            swa_cache=swa_zero, has_indexer=False, compress_ratio=0
        )
        monkeypatch.setattr(
            v2_attn,
            "paged_attention",
            lambda q, swa_c, main_c, recs, *a, **kw: (
                captured_swa_records.append(recs) or torch.zeros_like(q)
            ),
        )
        with v2_attn.stage_attention_cache_scope():
            attn_zero_1._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            attn_zero_2._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            assert len(captured_swa_records) == 2
            assert captured_swa_records[0] is captured_swa_records[1]

            # Also test untiled (window_size=32) inside stage_attention_cache_scope
            monkeypatch.setattr(
                v2_attn,
                "_read_rows",
                lambda c, t, r, l_arg: torch.zeros((count, l_arg.shape[1], 8)),
            )
            monkeypatch.setattr(
                v2_attn,
                "sparse_attention",
                lambda q, *a, **kw: torch.zeros_like(q),
            )
            attn_zero_1.window_size = 32
            attn_zero_1._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            attn_zero_1.window_size = 64
            stage_cache = v2_attn._active_stage_attn_cache
            assert stage_cache is not None
            assert any(
                k[0] == "swa_token_slots" and v[-1] is None
                for k, v in stage_cache.items()
                if isinstance(k, tuple)
            )

        # 2. row_offset == 0 outside stage_attention_cache_scope
        captured_swa_records.clear()
        attn_zero_1._prepare_tpu(None, qr, kv, None, None, None, positions, out)
        assert len(captured_swa_records) == 1

        # 3. row_offset != 0 (e.g. 128) and has_fast_swa_offset=False
        swa_nonzero = _make_paged_cache_mock(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), row_offset=128
        )
        attn_nonzero = _make_attn(
            swa_cache=swa_nonzero, has_indexer=False, compress_ratio=0
        )
        swa_slow = _make_paged_cache_mock(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            row_offset=None,
            sharded=True,
        )
        attn_slow = _make_attn(swa_cache=swa_slow, has_indexer=False, compress_ratio=0)
        with v2_attn.stage_attention_cache_scope():
            attn_nonzero._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            attn_slow._prepare_tpu(None, qr, kv, None, None, None, positions, out)
        assert len(captured_swa_records) == 3
        expected_nz = torch.where(
            captured_swa_records[0] >= 0, captured_swa_records[0] + 128, -1
        )
        assert torch.equal(captured_swa_records[1], expected_nz)

    def test_index_records_and_compressed_inputs_stage_caching(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        count = 4
        positions = torch.tensor([0, 16, 32, 48], dtype=torch.int32)
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        meta = SimpleNamespace(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            block_tables=table,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        metadata = {"swa_pfx": meta, "src_pfx": meta, "idx_pfx": meta}
        forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention",
            lambda q, *a, **kw: torch.zeros_like(q),
        )

        qr = torch.ones((count, 16), dtype=torch.bfloat16)
        kv = torch.ones((count, 8), dtype=torch.bfloat16)
        idx_w = torch.ones((count, 2), dtype=torch.bfloat16)
        out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)

        swa_cache = _make_paged_cache_mock(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), row_offset=0
        )
        idx_zero = _make_paged_cache_mock(
            torch.tensor([[5, 6], [7, 8]], dtype=torch.int32), row_offset=0
        )
        idx_nonzero = _make_paged_cache_mock(
            torch.tensor([[5, 6], [7, 8]], dtype=torch.int32), row_offset=256
        )
        idx_slow = _make_paged_cache_mock(
            torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
            row_offset=None,
            sharded=True,
        )

        seen_idx_records: list[torch.Tensor] = []
        seen_weights: list[torch.Tensor] = []

        def record_select(_qr, _pos, _ic, idx_recs, _req, w, *a, **kw):
            seen_idx_records.append(idx_recs)
            seen_weights.append(w)
            return torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.int32)

        # 1. Two indexer source layers in same stage_attention_cache_scope:
        #    - Layer 1 populates comp_inputs_key and idx_rec_key
        #    - Layer 2 hits comp_inputs_key and idx_rec_key in stage_cache!
        attn_idx_zero_1 = _make_attn(swa_cache=swa_cache, idx_cache=idx_zero)
        attn_idx_zero_2 = _make_attn(swa_cache=swa_cache, idx_cache=idx_zero)
        attn_idx_nz_1 = _make_attn(swa_cache=swa_cache, idx_cache=idx_nonzero)
        attn_idx_nz_2 = _make_attn(swa_cache=swa_cache, idx_cache=idx_nonzero)
        attn_idx_slow = _make_attn(swa_cache=swa_cache, idx_cache=idx_slow)
        for a in (
            attn_idx_zero_1,
            attn_idx_zero_2,
            attn_idx_nz_1,
            attn_idx_nz_2,
            attn_idx_slow,
        ):
            a._select_paged_indices = record_select

        with v2_attn.stage_attention_cache_scope():
            attn_idx_zero_1._prepare_tpu(
                None, qr, kv, None, None, idx_w, positions, out
            )
            attn_idx_zero_2._prepare_tpu(
                None, qr, kv, None, None, idx_w, positions, out
            )
            attn_idx_nz_1._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)
            attn_idx_nz_2._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)
            attn_idx_slow._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)

        assert seen_idx_records[0] is seen_idx_records[1]
        assert seen_idx_records[2] is seen_idx_records[3]
        assert torch.equal(seen_idx_records[2], seen_idx_records[0] + 256)
        assert torch.allclose(seen_weights[0], torch.full((count, 2), 0.25))

        # 2. Uncached (stage_cache is None) with row_offset == 0 and row_offset != 0
        seen_idx_records.clear()
        attn_idx_zero_1._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)
        attn_idx_nz_1._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)
        assert len(seen_idx_records) == 2

        # 3. Segmented main mode (owner_perm is not None and tp_ranks > 1):
        #    Verify comp_inputs_key miss then hit across two indexer layers
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 2)
        meta.owner_perm = torch.tensor([0, 2, 1, -1], dtype=torch.int32)
        meta.owner_inv = torch.tensor([0, 2, 1, 3], dtype=torch.int32)
        meta.owner_q_loc = torch.tensor([0, 2], dtype=torch.int32)

        def record_select_seg(_qr, _pos, _ic, idx_recs, *a, **kw):
            return torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)

        attn_idx_zero_1._select_paged_indices = record_select_seg
        attn_idx_zero_2._select_paged_indices = record_select_seg
        with v2_attn.stage_attention_cache_scope():
            attn_idx_zero_1._prepare_tpu(
                None, qr, kv, None, None, idx_w, positions, out
            )
            attn_idx_zero_2._prepare_tpu(
                None, qr, kv, None, None, idx_w, positions, out
            )
            seg_stage_cache = v2_attn._active_stage_attn_cache
            assert seg_stage_cache is not None
            assert any(
                isinstance(k, tuple) and k[0] == "compressed_inputs" and k[2] is True
                for k in seg_stage_cache
            )
        # Also test segmented_main outside stage_attention_cache_scope
        attn_idx_zero_1._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)

    def test_high_conc_c64_c96_c128_metadata_and_swa_cache_hit_rates(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """TD8.171: Profile and verify host-side metadata and SWA record
        preparation and stage_cache reuse across an 18-layer decoder stage at
        high concurrency C=64 / 96 / 128 (DSpark K=5, B=384 / 576 / 768)."""
        orig_req_row_slots = v2_attn._request_row_slots
        req_slot_calls = 0

        def _counted_req_row_slots(*args, **kwargs):
            nonlocal req_slot_calls
            req_slot_calls += 1
            return orig_req_row_slots(*args, **kwargs)

        monkeypatch.setattr(v2_attn, "_request_row_slots", _counted_req_row_slots)
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention",
            lambda q, *a, **kw: torch.zeros_like(q),
        )

        for c_reqs in (64, 96, 128):
            tokens = c_reqs * 6
            positions = torch.arange(tokens, dtype=torch.int32)
            table = torch.arange(c_reqs * 4, dtype=torch.int32).reshape(c_reqs, 4)
            qsl = torch.arange(0, (c_reqs + 1) * 6, 6, dtype=torch.int32)
            seq_lens = torch.full((c_reqs,), 64, dtype=torch.int32)
            meta = SimpleNamespace(
                seq_lens=seq_lens,
                block_tables=table,
                query_start_loc=qsl,
            )
            metadata = {
                "swa_pfx": meta,
                "src_pfx": meta,
                "idx_pfx": meta,
            }
            forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
            monkeypatch.setattr(
                v2_attn,
                "get_forward_context",
                lambda ctx=forward_ctx: ctx,
            )

            qr = torch.ones((tokens, 16), dtype=torch.bfloat16)
            kv = torch.ones((tokens, 8), dtype=torch.bfloat16)
            idx_w = torch.ones((tokens, 2), dtype=torch.bfloat16)
            out = torch.zeros((tokens, 2, 8), dtype=torch.bfloat16)

            swa_l0 = _make_paged_cache_mock(
                table, row_offset=0, rows_per_page=16, page_stride=32
            )
            swa_l1 = _make_paged_cache_mock(
                table, row_offset=64, rows_per_page=16, page_stride=32
            )
            idx_l0 = _make_paged_cache_mock(
                table, row_offset=0, rows_per_page=16, page_stride=48
            )
            idx_l1 = _make_paged_cache_mock(
                table, row_offset=128, rows_per_page=16, page_stride=48
            )

            attn_src0 = _make_attn(swa_cache=swa_l0, idx_cache=idx_l0)
            attn_src0.non_causal_multi_token_decode = True
            attn_src1 = _make_attn(swa_cache=swa_l1, idx_cache=idx_l1)
            attn_src1.non_causal_multi_token_decode = True
            attn_reuse = _make_attn(
                swa_cache=swa_l1, idx_cache=idx_l1, has_indexer=False
            )
            attn_reuse.non_causal_multi_token_decode = True

            seen_records: list[torch.Tensor] = []

            def _select_stub(
                _qr,
                _pos,
                _ic,
                idx_recs,
                *a,
                _seen=seen_records,
                _tokens=tokens,
                **kw,
            ):
                _seen.append(idx_recs)
                return torch.zeros((_tokens, 2), dtype=torch.int32)

            attn_src0._select_paged_indices = _select_stub
            attn_src1._select_paged_indices = _select_stub

            req_slot_calls = 0
            with v2_attn.stage_attention_cache_scope():
                attn_src0._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)
                attn_src1._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)
                attn_reuse._prepare_tpu(None, qr, kv, None, None, None, positions, out)
                assert req_slot_calls == 1
                stage_cache = v2_attn._active_stage_attn_cache
                assert stage_cache is not None
                key_kinds = {k[0] for k in stage_cache if isinstance(k, tuple)}
                assert {
                    "swa_token_addr",
                    "swa_token_slots",
                    "compressed_inputs",
                    "index_req_slots",
                    "index_records",
                    "compressed_pages",
                    "compressed_attn",
                }.issubset(key_kinds)
            assert torch.equal(seen_records[1], seen_records[0] + 128)

    def test_td8183_40_layer_stage_cache_key_memo_and_swa_records_reuse(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """TD8.183: Verify per-layer _stage_cache_key memoization and stage-scoped
        swa_records, owner_kwargs, and fused_swa reuse across 40 layers at
        B=384 / 576 / 672 (C=64 / 96 / 112, DSpark K=5)."""
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)
        seen_swa_records: list[torch.Tensor] = []

        def _capture_sparse_attn(q, _sc, _mc, recs, *a, **kw):
            seen_swa_records.append(recs)
            return torch.zeros_like(q)

        monkeypatch.setattr(v2_attn, "sparse_fp8_attention", _capture_sparse_attn)

        for c_reqs in (64, 96, 112):
            tokens = c_reqs * 6
            positions = torch.arange(tokens, dtype=torch.int32)
            table = torch.arange(c_reqs * 4, dtype=torch.int32).reshape(c_reqs, 4)
            qsl = torch.arange(0, (c_reqs + 1) * 6, 6, dtype=torch.int32)
            seq_lens = torch.full((c_reqs,), 64, dtype=torch.int32)
            meta = SimpleNamespace(
                seq_lens=seq_lens,
                block_tables=table,
                query_start_loc=qsl,
            )
            metadata = {"swa_pfx": meta, "src_pfx": meta, "idx_pfx": meta}
            forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
            monkeypatch.setattr(
                v2_attn, "get_forward_context", lambda ctx=forward_ctx: ctx
            )

            qr = torch.ones((tokens, 16), dtype=torch.bfloat16)
            kv = torch.ones((tokens, 8), dtype=torch.bfloat16)
            idx_w = torch.ones((tokens, 2), dtype=torch.bfloat16)
            out = torch.zeros((tokens, 2, 8), dtype=torch.bfloat16)

            swa_nonzero = _make_paged_cache_mock(
                table, row_offset=64, rows_per_page=16, page_stride=32
            )
            idx_nonzero = _make_paged_cache_mock(
                table, row_offset=128, rows_per_page=16, page_stride=48
            )
            layers = []
            for i in range(40):
                has_idx = i == 0
                lyr = _make_attn(
                    swa_cache=swa_nonzero,
                    idx_cache=idx_nonzero,
                    has_indexer=has_idx,
                )
                lyr.non_causal_multi_token_decode = True
                if has_idx:
                    lyr._select_paged_indices = (
                        lambda _qr, _pos, *a, _t=tokens, **kw: torch.zeros(
                            (_t, 2), dtype=torch.int32
                        )
                    )
                layers.append((lyr, has_idx))

            seen_swa_records.clear()
            with v2_attn.stage_attention_cache_scope():
                for lyr, has_idx in layers:
                    lyr._prepare_tpu(
                        None,
                        qr,
                        kv,
                        None,
                        None,
                        idx_w if has_idx else None,
                        positions,
                        out,
                    )
                stage_cache = v2_attn._active_stage_attn_cache
                assert stage_cache is not None
                assert "fused_swa" in stage_cache
                key_kinds = {k[0] for k in stage_cache if isinstance(k, tuple)}
                assert "swa_records" in key_kinds
                # All 40 layers with identical non-zero row_offset=64 share the
                # exact same SWA records tensor instance inside the stage scope.
                assert len(seen_swa_records) == 40
                first_rec = seen_swa_records[0]
                assert all(r is first_rec for r in seen_swa_records)
                # Key memoization returns identical tuple object on second call.
                k1 = layers[0][0]._stage_cache_key("swa_records", 64, tokens)
                k2 = layers[0][0]._stage_cache_key("swa_records", 64, tokens)
                assert k1 is k2
