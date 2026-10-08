# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EXT-1 TP=8 Owner-Rank Sharded KV Cache & Token-Parallel Decode.

Covers 100% of modified lines and branches in:
- src/vllm_torchtpu/envs.py (TPU_DEEPSEEK_V41_KV_SHARD)
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/cache.py
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention.py
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import ROW_WORDS
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as v2_cache
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


# ==============================================================================
# 1. envs.py: TPU_DEEPSEEK_V41_KV_SHARD
# ==============================================================================
class TestEnvsKvShard:
    """Verify TPU_DEEPSEEK_V41_KV_SHARD default and boolean parsing."""

    def test_default_is_false(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TPU_DEEPSEEK_V41_KV_SHARD", raising=False)
        resolver = envs.environment_variables["TPU_DEEPSEEK_V41_KV_SHARD"]
        assert resolver() is False

    @pytest.mark.parametrize(
        ("val_str", "expected"),
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("0", False),
            ("false", False),
            ("False", False),
        ],
    )
    def test_parsing(
        self, monkeypatch: pytest.MonkeyPatch, val_str: str, expected: bool
    ):
        monkeypatch.setenv("TPU_DEEPSEEK_V41_KV_SHARD", val_str)
        resolver = envs.environment_variables["TPU_DEEPSEEK_V41_KV_SHARD"]
        assert resolver() is expected


# ==============================================================================
# 2. v2/cache.py: Sharded Materialization, PagedCacheRows & NativeCacheView
# ==============================================================================
class TestCacheKvShard:
    """Verify 1/TP HBM allocation and block-offset translation in v2/cache.py."""

    def test_materialize_upstream_kv_cache_sharded_layouts(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from vllm_torchtpu import kv_cache_materializer

        # Two layers in layer_compact layout with non-zero storage_offset on layer1
        global_blocks = 16
        backing = torch.zeros(global_blocks * 2 * 512, dtype=torch.uint8)
        view0 = backing[: global_blocks * 512].view(global_blocks, 1, 1, 512)
        view1 = backing[global_blocks * 512 :].view(global_blocks, 1, 1, 512)
        views = {"layer0": view0, "layer1": view1}
        monkeypatch.setattr(v2_cache, "allocate_upstream_kv_cache", lambda *a: views)
        monkeypatch.setattr(
            kv_cache_materializer,
            "layer_to_pool_index",
            lambda c: {"layer0": 0, "layer1": 0},
        )

        cfg = SimpleNamespace(
            kv_cache_tensors=[SimpleNamespace(size=global_blocks * 2 * 512)],
            num_blocks=global_blocks,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=1024))
            ],
        )
        layout_lc = SimpleNamespace(is_layer_compact=True)
        layout_bc = SimpleNamespace(is_layer_compact=False)

        # 1. Default env fallback (kv_shard=None, tp_size=None, tp_rank=None)
        monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", True)
        monkeypatch.setattr(
            "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 8
        )
        monkeypatch.setattr(
            "vllm.distributed.get_tensor_model_parallel_rank", lambda: 3
        )
        res_lc = v2_cache.materialize_upstream_kv_cache(
            cfg, torch.device("cpu"), layout_lc, word_rows=True
        )
        # Global blocks = 16, TP = 8 -> local_blocks = 2, block_offset = 6
        assert res_lc.native_cache_layouts["layer0"] == (
            (2, 1, 512),
            torch.uint8,
            512,
            0,
            6,
            True,
        )
        # layer1 offset_bytes scaled from 16 * 512 to 2 * 512 = 1024
        assert res_lc.native_cache_layouts["layer1"] == (
            (2, 1, 512),
            torch.uint8,
            512,
            2 * 512,
            6,
            True,
        )
        # Raw pool has 2 blocks * 2 layers = 4 rows (8x smaller than 32 rows)
        assert res_lc.native_raw_cache["layer0"].shape == (4, ROW_WORDS)
        # Upstream placeholder view is expanded from 1 page (stride 0 on dim 0)
        assert res_lc.kv_caches["layer0"].shape == (16, 1, 1, 512)
        assert res_lc.kv_caches["layer0"].stride(0) == 0

        # 2. Block-compact sharded layout with split_banks=True
        monkeypatch.setattr(v2_cache, "SPARSECORE_MAX_POOL_BYTES", 2 * 512)
        res_split = v2_cache.materialize_upstream_kv_cache(
            cfg,
            torch.device("cpu"),
            layout_bc,
            word_rows=True,
            kv_shard=True,
            tp_size=8,
            tp_rank=5,
        )
        assert isinstance(res_split.native_raw_cache["layer0"], tuple)
        assert res_split.native_cache_layouts["layer0"][4] == 10
        assert res_split.native_cache_layouts["layer0"][5] is True

        # 3. Hybrid Id contract v2 layout (replicated_blocks=4, global=20, TP=8 -> P=2)
        cfg_v2 = SimpleNamespace(
            kv_cache_tensors=[SimpleNamespace(size=20 * 2 * 512)],
            num_blocks=20,
            replicated_blocks=4,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=1024))
            ],
        )
        backing_v2 = torch.zeros(20 * 2 * 512, dtype=torch.uint8)
        views_v2 = {
            "layer0": backing_v2[: 20 * 512].view(20, 1, 1, 512),
            "layer1": backing_v2[20 * 512 :].view(20, 1, 1, 512),
        }
        monkeypatch.setattr(v2_cache, "allocate_upstream_kv_cache", lambda *a: views_v2)
        monkeypatch.setattr(v2_cache, "SPARSECORE_MAX_POOL_BYTES", 64 * 512)
        res_v2 = v2_cache.materialize_upstream_kv_cache(
            cfg_v2,
            torch.device("cpu"),
            layout_lc,
            word_rows=True,
            kv_shard=True,
            tp_size=8,
            tp_rank=3,
        )
        # Rank 3: local_blocks = 4 + 2 = 6, block_offset = 4 + 3*2 = 10, rep = 4
        assert res_v2.native_cache_layouts["layer0"] == (
            (6, 1, 512),
            torch.uint8,
            512,
            0,
            10,
            True,
            4,
        )
        for bad_n, bad_r in [(15, 0), (20, -1), (4, 4), (19, 4)]:
            with pytest.raises(ValueError, match="KV sharding requires num_blocks"):
                v2_cache.resolve_kv_block_shard(
                    bad_n, tp_size=8, tp_rank=0, kv_shard=True, replicated_blocks=bad_r
                )

        # 4. Unsharded fallback (kv_shard=False)
        monkeypatch.setattr(v2_cache, "allocate_upstream_kv_cache", lambda *a: views)
        monkeypatch.setattr(v2_cache, "SPARSECORE_MAX_POOL_BYTES", 32 * 512)
        monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", False)
        res_unsharded = v2_cache.materialize_upstream_kv_cache(
            cfg, torch.device("cpu"), layout_bc, word_rows=True
        )
        assert len(res_unsharded.native_cache_layouts["layer0"]) == 4

    def test_paged_cache_rows_sharded_indexing_and_cpu_io(self):
        # 16 global blocks, TP=4 -> 4 local blocks; rank 1 owns [4, 5, 6, 7]
        raw = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
        layout_6 = ((4, 4, 256), torch.bfloat16, 2048, 0, 4, True)
        paged = PagedCacheRows(raw, layout_6)
        assert paged.sharded is True
        assert paged.block_offset == 4
        assert paged.replicated_blocks == 0

        # Global slots: block 3 (unowned -> -1), block 4 row 2 (owned -> 2),
        # block 7 row 3 (owned -> 15), block 8 row 0 (unowned -> -1), -1 -> -1
        slots = torch.tensor([3 * 4 + 1, 4 * 4 + 2, 7 * 4 + 3, 8 * 4 + 0, -1])
        records = paged.row_indices(slots)
        assert records.tolist() == [-1, 2, 15, -1, -1]

        # Pages translation: global blocks [2, 4, 5, 7, 9, -1] -> [0, 0, 1, 3, 0, 0]
        bt = torch.tensor([[2, 4, 5], [7, 9, -1]], dtype=torch.int32)
        pages = paged.pages(bt)
        assert pages.tolist() == [[0, 0, 1], [3, 0, 0]]

        # Hybrid Id contract v2 (layout_7): R=2 replicated [0, 1], P=2 sharded/rank
        # Rank 1 owns global sharded blocks [4, 5] at local blocks [2, 3]
        layout_7 = ((4, 4, 256), torch.bfloat16, 2048, 0, 4, True, 2)
        paged_v2 = PagedCacheRows(
            torch.zeros((16, ROW_WORDS), dtype=torch.int32), layout_7
        )
        assert paged_v2.replicated_blocks == 2
        assert paged_v2.block_offset == 4
        # Global blocks: 0 (rep -> local 0), 1 (rep -> local 1), 2 (rank 0 -> -1),
        # 4 (rank 1 -> local 2), 5 (rank 1 -> local 3), 6 (rank 2 -> -1), -1 -> -1
        v2_slots = torch.tensor(
            [0 * 4 + 1, 1 * 4 + 3, 2 * 4 + 0, 4 * 4 + 2, 5 * 4 + 1, 6 * 4 + 0, -1]
        )
        assert paged_v2.row_indices(v2_slots).tolist() == [1, 7, -1, 10, 13, -1, -1]
        v2_bt = torch.tensor([[0, 1, 2], [4, 5, 6]], dtype=torch.int32)
        assert paged_v2.pages(v2_bt).tolist() == [[0, 1, 0], [2, 3, 0]]

        # Write values on CPU: unowned slots are ignored, owned slots are written
        write_slots = torch.tensor([2 * 4 + 0, 5 * 4 + 1, 9 * 4 + 2], dtype=torch.long)
        vals = torch.tensor(
            [[10.0] * 256, [42.0] * 256, [99.0] * 256], dtype=torch.bfloat16
        )
        paged.write(write_slots, vals)

        # Read back via __getitem__: owned block 5 returns 42.0, unowned returns 0.0
        got = paged[torch.tensor([2, 5, 9]), torch.tensor([0, 1, 2])]
        assert torch.all(got[0] == 0)
        assert torch.all(got[1] == 42.0)
        assert torch.all(got[2] == 0)

        # 2-bank sharded and unsharded CPU write and __getitem__
        raw_lo = torch.zeros((8, ROW_WORDS), dtype=torch.int32)
        raw_hi = torch.zeros((8, ROW_WORDS), dtype=torch.int32)
        paged_2b = PagedCacheRows((raw_lo, raw_hi), layout_6)
        paged_2b.write(torch.tensor([4 * 4 + 1, 6 * 4 + 2]), vals[:2])
        got_2b = paged_2b[torch.tensor([4, 6, 9]), torch.tensor([1, 2, 0])]
        assert torch.all(got_2b[0] == 10.0)
        assert torch.all(got_2b[1] == 42.0)
        assert torch.all(got_2b[2] == 0.0)

        paged_unsharded = PagedCacheRows(
            torch.zeros((16, ROW_WORDS), dtype=torch.int32),
            ((4, 4, 256), torch.bfloat16, 2048, 0),
        )
        paged_unsharded.write(torch.tensor([1]), vals[:1])
        assert torch.all(
            paged_unsharded[torch.tensor([0]), torch.tensor([1])][0] == 10.0
        )

        # FP8 sharded write on CPU: only owned block is written
        raw_fp8 = torch.zeros((4, ROW_WORDS), dtype=torch.int32)
        rope_fp8 = torch.zeros((1, ROW_WORDS), dtype=torch.int32)
        paged_fp8 = PagedCacheRows(
            raw_fp8,
            ((4, 1, 512), torch.uint8, 512, 0, 4, True),
            rope=rope_fp8,
        )
        latents = torch.randn(2, 512, dtype=torch.bfloat16)
        # Slot 1 is unowned (block 1 < 4), slot 5 is owned (block 5 -> local 1)
        paged_fp8.write(torch.tensor([1, 5]), latents)
        assert torch.all(raw_fp8[0] == 0)
        assert not torch.all(raw_fp8[1] == 0)

    def test_native_cache_view_sharded_property(self, monkeypatch: pytest.MonkeyPatch):
        class Base:
            def bind_kv_cache(self, cache):
                self._bound_cache = cache

        class View(v2_cache.NativeCacheView, Base):
            pass

        view = View()
        placeholder = torch.zeros(1, 1, 4, 256, dtype=torch.bfloat16)
        raw = torch.zeros(16, ROW_WORDS, dtype=torch.int32)
        layout_6 = ((4, 4, 256), torch.bfloat16, 2048, 0, 8, True)
        view.bind_kv_cache(placeholder, raw_cache=raw, layout=layout_6)
        assert view._tpu_kv_shard is True

        # Even when not compiling, sharded NativeCacheView returns PagedCacheRows
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        kv = view.kv_cache
        assert isinstance(kv, PagedCacheRows)
        assert kv.sharded is True
        assert kv.block_offset == 8

        # Unsharded 4-tuple layout returns _bound_cache when not compiling
        layout_4 = ((4, 4, 256), torch.bfloat16, 2048, 0)
        view.bind_kv_cache(placeholder, raw_cache=raw, layout=layout_4)
        assert view._tpu_kv_shard is False
        assert view.kv_cache is placeholder


# ==============================================================================
# 3. v2/attention.py: _query_split, DeepseekV41Attention & _prepare_tpu
# ==============================================================================
class TestAttentionKvShard:
    """Verify _query_split, _swa_max_requests, and sharded offset guards."""

    def test_query_split_kv_shard_modes(self):
        # Under kv_shard=True with count >= ranks and count % ranks == 0:
        assert v2_attn._query_split(8, 8, kv_shard=True) == (1, 8)
        assert v2_attn._query_split(16, 8, kv_shard=True) == (2, 8)
        assert v2_attn._query_split(48, 8, kv_shard=True) == (6, 8)
        assert v2_attn._query_split(288, 8, kv_shard=True) == (36, 8)

        # Fallback when kv_shard=False or count % ranks != 0 or count < ranks:
        assert v2_attn._query_split(8, 8, kv_shard=False) == (2, 16)
        assert v2_attn._query_split(10, 8, kv_shard=True) == (2, 16)
        assert v2_attn._query_split(4, 8, kv_shard=True) == (2, 16)

    def test_attention_init_swa_max_requests_under_kv_shard(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def fake_upstream_init(self, vllm_config, *args, **kwargs):
            torch.nn.Module.__init__(self)
            self.kv_cache_torch_dtype = torch.float8_e4m3fn
            self.rotary_emb = SimpleNamespace(
                cos_sin_cache=torch.zeros(64, 64, dtype=torch.float32)
            )
            self.attn_sink = torch.ones(4, dtype=torch.float32)

        monkeypatch.setattr(
            v2_attn.upstream.DeepseekV4Attention, "__init__", fake_upstream_init
        )
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 8)

        cfg = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: {}),
                max_model_len=1024,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
        )

        monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", True)
        attn_sharded = v2_attn.DeepseekV41Attention(cfg)
        assert attn_sharded._swa_max_requests is None

        monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", False)
        attn_replicated = v2_attn.DeepseekV41Attention(cfg)
        assert attn_replicated._swa_max_requests == 4

    def test_select_paged_indices_with_kv_shard(self, monkeypatch: pytest.MonkeyPatch):
        count = 8
        ranks = 8
        positions = torch.arange(count, dtype=torch.int32)
        qr = torch.ones((count, 16), dtype=torch.bfloat16)
        weights = torch.ones((count, 2), dtype=torch.float32)
        lengths = torch.full((count, 1), 4, dtype=torch.int32)
        swa_slots = torch.arange(count, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)

        attn = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        attn.compress_ratio = 2
        attn._tp_rank = 3
        attn._fp8_kv = True
        attn._config = {"index_topk": 2}
        attn.candidate_block_size = 1
        attn.candidate_block_buffer = None
        attn.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            indexer_op=SimpleNamespace(candidate_blocks=None, candidate_write=False),
        )
        attn._project_index_q = lambda q, p: q

        seen_q_shapes: list[tuple[int, ...]] = []

        def fake_paged_indexer(q, c, r, req, w, lengths_arg, *a, **kw):
            seen_q_shapes.append(tuple(q.shape))
            return c, torch.full((q.shape[0], 2), 3, dtype=torch.int32), None

        monkeypatch.setattr(v2_attn, "paged_indexer", fake_paged_indexer)
        monkeypatch.setattr(
            v2_attn, "get_tensor_model_parallel_world_size", lambda: ranks
        )
        monkeypatch.setattr(
            v2_attn,
            "_gather_query_rows",
            lambda x: x.repeat(ranks, 1),
        )

        index_cache_mock = MagicMock(spec=PagedCacheRows)
        index_cache_mock.raw = MagicMock()
        index_cache_mock.sharded = True
        index_records = torch.zeros((count, 4), dtype=torch.int32)

        with v2_attn.stage_attention_cache_scope():
            sel = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                index_records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )
            assert sel.shape == (count, 2)
            # With count=8, ranks=8, kv_shard=True -> part = 1 row per rank!
            assert seen_q_shapes == [(1, 16)]

    def test_prepare_tpu_sharded_swa_and_idx_offsets(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        count = 4
        capacity = 8
        positions = torch.tensor([0, 16, 32, 48], dtype=torch.int32)
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        meta = SimpleNamespace(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            block_tables=table,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        forward_ctx = SimpleNamespace(
            attn_metadata={"src_pfx": meta, "idx_pfx": meta},
            additional_kwargs={},
        )
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention",
            lambda q, *a, **kw: torch.zeros_like(q),
        )

        for is_sharded in (True, False):
            swa_calls: list[torch.Tensor] = []
            idx_calls: list[torch.Tensor] = []

            cache_mock = MagicMock(spec=PagedCacheRows)
            cache_mock.numel.return_value = 1024
            cache_mock.shape = (64, 32, 512)
            cache_mock.page_stride = 32
            cache_mock.row_offset = 100
            cache_mock.sharded = is_sharded
            cache_mock.row_indices = lambda s, sc=swa_calls: sc.append(s) or s
            cache_mock.pages.return_value = torch.tensor([[0, 1], [2, 3]])

            idx_cache_mock = MagicMock(spec=PagedCacheRows)
            idx_cache_mock.numel.return_value = 1024
            idx_cache_mock.shape = (64, 32, 128)
            idx_cache_mock.page_stride = 32
            idx_cache_mock.row_offset = 200
            idx_cache_mock.sharded = is_sharded
            idx_cache_mock.row_indices = lambda s, ic=idx_calls: ic.append(s) or s
            idx_cache_mock.pages.return_value = torch.tensor([[0, 1], [2, 3]])

            attn = object.__new__(v2_attn.DeepseekV41Attention)
            torch.nn.Module.__init__(attn)
            attn.prefix = "src_pfx"
            attn.compressed_cache_prefix = "src_pfx"
            attn.index_source_layer_id = 0
            attn.window_size = 64
            attn.compress_ratio = 2
            attn._max_model_len = capacity * 2
            attn._config = {"index_topk": 2}
            attn._fp8_kv = True
            attn._ced_decoder = False
            attn._ced_global_from_encoder = False
            attn._local_q_attn = False
            attn.n_local_heads = 2
            attn.head_dim = 8
            attn.scale = 0.25
            attn.attn_sink = torch.zeros(2)
            attn._swa_max_requests = None
            attn._precompact_records = True
            attn._token_parallel_attn = True
            attn._full_attn_sink = torch.zeros(2)
            attn._tp_rank_1d = torch.tensor([0], dtype=torch.int32)
            attn.topk_indices_buffer = torch.full((16, 4), -1, dtype=torch.int32)
            attn._rotate = lambda x, p: x
            attn.wq_b = lambda qr: qr
            attn.compressor = None
            attn.swa_cache_layer = SimpleNamespace(
                prefix="src_pfx", kv_cache=cache_mock
            )
            attn.kv_cache = cache_mock
            attn._static_forward_context = {
                "src_pfx": SimpleNamespace(kv_cache=cache_mock)
            }
            attn.indexer = SimpleNamespace(
                head_dim=8,
                n_head=2,
                wk=lambda lat: lat,
                k_norm=lambda k: k,
                k_cache=SimpleNamespace(
                    prefix="idx_pfx",
                    compress_ratio=2,
                    kv_cache=idx_cache_mock,
                ),
            )
            attn._select_paged_indices = lambda *a, **kw: torch.zeros(
                (count, 2), dtype=torch.int32
            )

            with v2_attn.stage_attention_cache_scope():
                out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
                qr = torch.ones((count, 16), dtype=torch.bfloat16)
                kv = torch.ones((count, 8), dtype=torch.bfloat16)
                idx_w = torch.ones((count, 2), dtype=torch.bfloat16)
                attn._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)

            # When sharded=True, row_indices must be called for both SWA and Indexer!
            if is_sharded:
                assert len(swa_calls) >= 1
                assert len(idx_calls) >= 1
            else:
                assert len(swa_calls) == 0
                assert len(idx_calls) == 0


# ==============================================================================
# 4. v2/attention_kernels.py: Token-Parallel Thresholds & Padding under KV Shard
# ==============================================================================
class TestAttentionKernelsKvShard:
    """Verify fp8_attention and fp8_attention_qo dispatch at tokens >= tp_size."""

    def test_fp8_attention_kv_shard_small_batch(self, monkeypatch: pytest.MonkeyPatch):
        import vllm.distributed as vllm_dist

        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        mock_tp = mock.MagicMock()
        mock_tp.device_group.size.return_value = 8
        captured_fns: dict[str, object] = {}

        def capture_jax_op(name, fn, *args, **kwargs):
            captured_fns[name] = fn
            return MagicMock()

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch.object(vllm_dist, "get_tp_group", return_value=mock_tp),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=capture_jax_op),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None
            attention_kernels.initialize_paged_attention()
            fp8_attn = captured_fns["pallas::dsv41v2_sparse_fp8_attention"]

            a2a_calls: list[int] = []

            def fake_a2a(x, *a, **kw):
                a2a_calls.append(x.shape[0])
                return x

            with (
                mock.patch(
                    "jax.shard_map",
                    side_effect=lambda fn, **kw: lambda *args: fn(*args),
                ),
                mock.patch("jax.lax.all_to_all", side_effect=fake_a2a),
                mock.patch.object(
                    attention_kernels,
                    "sparse_fp8_attention_jax",
                    side_effect=lambda q, *a, **kw: q,
                ),
            ):
                # 1. tokens = 8 (< 48): KV_SHARD=True uses token_parallel
                monkeypatch.setitem(
                    vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", True
                )
                out = fp8_attn(
                    jnp.zeros((8, 64, 512)),
                    jnp.zeros((8, 512)),
                    jnp.zeros((8, 128)),
                    jnp.zeros((2, 32)),
                    jnp.zeros((8, 64), dtype=jnp.int32),
                    jnp.zeros((8, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((64,)),
                    jnp.zeros((64,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    True,
                    False,
                    4096,
                )
                assert out.shape == (8, 64, 512)
                assert len(a2a_calls) == 2

                # 2. tokens = 64 (pad_rows = 0) with KV_SHARD=False -> token_parallel
                monkeypatch.setitem(
                    vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", False
                )
                out_64 = fp8_attn(
                    jnp.zeros((64, 64, 512)),
                    jnp.zeros((64, 512)),
                    jnp.zeros((64, 128)),
                    jnp.zeros((16, 32)),
                    jnp.zeros((64, 64), dtype=jnp.int32),
                    jnp.zeros((64, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((64,)),
                    jnp.zeros((64,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    True,
                    False,
                    4096,
                )
                assert out_64.shape == (64, 64, 512)
                assert len(a2a_calls) == 4

                # 3. tokens = 8 (< 48) with KV_SHARD=False -> head_parallel (no a2a)
                out_hp = fp8_attn(
                    jnp.zeros((8, 64, 512)),
                    jnp.zeros((8, 512)),
                    jnp.zeros((8, 128)),
                    jnp.zeros((2, 32)),
                    jnp.zeros((8, 64), dtype=jnp.int32),
                    jnp.zeros((8, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((64,)),
                    jnp.zeros((64,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    True,
                    False,
                    4096,
                )
                assert out_hp.shape == (8, 64, 512)
                assert len(a2a_calls) == 4

    def test_fp8_attention_qo_kv_shard_padding_and_unpadded(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        mock_tp = mock.MagicMock()
        mock_tp.device_group.size.return_value = 8
        captured_fns: dict[str, object] = {}

        def capture_jax_op(name, fn, *args, **kwargs):
            captured_fns[name] = fn
            return MagicMock()

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch("vllm.distributed.get_tp_group", return_value=mock_tp),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=capture_jax_op),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None
            attention_kernels.initialize_paged_attention()
            fn = captured_fns["pallas::dsv41v2_sparse_fp8_attention_qo"]

            monkeypatch.setitem(
                vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", True
            )
            seen_q_shapes: list[tuple[int, ...]] = []

            def fake_sparse_attn(q, *a, **kw):
                seen_q_shapes.append(tuple(q.shape))
                return q

            with (
                mock.patch(
                    "jax.shard_map",
                    side_effect=lambda body, **kw: lambda *args: body(*args),
                ),
                mock.patch("jax.lax.all_to_all", side_effect=lambda x, **kw: x),
                mock.patch(
                    "jax.lax.all_gather",
                    side_effect=lambda x, **kw: jnp.repeat(x, 8, axis=0),
                ),
                mock.patch(
                    "vllm_torchtpu.layers.adapter.linear_common._quantized_matmul_jax",
                    side_effect=lambda q_in, w, ws: jnp.zeros(
                        (q_in.shape[0], 64 * 512), dtype=jnp.bfloat16
                    ),
                ),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v4.attention._rope_jax",
                    side_effect=lambda q, pos, cs: q,
                ),
                mock.patch.object(
                    attention_kernels,
                    "sparse_fp8_attention_jax",
                    side_effect=fake_sparse_attn,
                ),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v4.attention._o_proj_jax",
                    side_effect=lambda out, pos, cs, w, ws, head_dim: jnp.zeros(
                        (out.shape[0], 1024), dtype=jnp.bfloat16
                    ),
                ),
            ):
                # 1. tokens = 16, tp_size = 8 -> tokens_local = 2 -> pad_rows = 6 (> 0)
                # 2. tokens = 64, tp_size = 8 -> tokens_local = 8 -> pad_rows = 0
                for tokens, tokens_local in [(16, 2), (64, 8)]:
                    args = (
                        jnp.zeros((tokens, 1280), dtype=jnp.bfloat16),
                        jnp.zeros((tokens_local, 1280), dtype=jnp.bfloat16),
                        jnp.zeros((1280, 256)),
                        jnp.zeros((256,)),
                        jnp.zeros((1280, 512)),
                        jnp.zeros((512,)),
                        jnp.zeros((tokens,)),
                        jnp.zeros((tokens, 64)),
                        jnp.zeros((16, 64, 128)),
                        jnp.zeros((16, 64, 128)),
                        jnp.zeros((16, 64, 64)),
                        jnp.zeros((tokens, 64), dtype=jnp.int32),
                        jnp.zeros((tokens, 128), dtype=jnp.int32),
                        jnp.zeros((1, 1), dtype=jnp.int32),
                        jnp.zeros((2,), dtype=jnp.int32),
                        jnp.zeros((64,)),
                        jnp.zeros((64,)),
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.zeros((512, 1024)),
                        jnp.zeros((1024,)),
                        jnp.zeros((512, 1024)),
                        jnp.zeros((1024,)),
                        jnp.zeros((tokens,), dtype=jnp.int32),
                        0.125,
                        64,
                        None,
                        False,
                        128,
                    )
                    res = fn(*args)
                    assert res.shape == (tokens, 128)
                # Both padded (2 + 6 = 8) and unpadded (8) calls pass 8 rows to kernel
                assert seen_q_shapes == [(8, 256, 128), (8, 256, 128)]

                # 3. With KV_SHARD=False, tokens=64 (< 128*8=1024) takes _sharded_body
                monkeypatch.setitem(
                    vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", False
                )
                res_sharded = fn(*args)
                assert res_sharded.shape == (64, 1024)


# ==============================================================================
# 5. End-to-End 8-Rank Owner-Sharded vs Replicated Bitwise Parity on CPU
# ==============================================================================
class TestEndToEndOwnerShardedParity:
    """Prove 8-rank owner-sharded write + indexer + attention is bitwise identical."""

    def test_eight_rank_sharded_write_and_attention_bitwise_parity(self):
        tp_size = 8
        num_seqs = 16  # 2 sequences per TP rank
        pages_per_seq = 2
        page_size = 16
        global_blocks = num_seqs * pages_per_seq  # 32 global blocks
        local_blocks = global_blocks // tp_size  # 4 local blocks per rank
        window = 16
        topk = 16
        heads = 8
        dim = 512
        idx_dim = 128
        idx_heads = 4
        scale = dim**-0.5

        gen = torch.Generator().manual_seed(2026)
        # Assign disjoint global blocks so rank r owns requests [2r, 2r+1]
        block_table = torch.arange(global_blocks, dtype=torch.int32).reshape(
            num_seqs, pages_per_seq
        )

        # Build 1 replicated cache set + 8 sharded cache sets (Main, SWA, Indexer K)
        rep_main_raw = torch.zeros(
            (global_blocks * page_size * 2, ROW_WORDS), dtype=torch.int32
        )
        rep_main = PagedCacheRows(
            rep_main_raw,
            ((global_blocks, page_size, dim), torch.bfloat16, page_size * dim * 2, 0),
        )
        rep_swa_raw = torch.zeros(
            (global_blocks * page_size * 2, ROW_WORDS), dtype=torch.int32
        )
        rep_swa = PagedCacheRows(
            rep_swa_raw,
            ((global_blocks, page_size, dim), torch.bfloat16, page_size * dim * 2, 0),
        )
        rep_idx_raw = torch.zeros(
            (global_blocks * page_size // 2, ROW_WORDS), dtype=torch.int32
        )
        rep_idx = PagedCacheRows(
            rep_idx_raw,
            (
                (global_blocks, page_size, idx_dim),
                torch.bfloat16,
                page_size * idx_dim * 2,
                0,
            ),
        )

        sharded_mains = []
        sharded_swas = []
        sharded_idxs = []
        for r in range(tp_size):
            m_raw = torch.zeros(
                (local_blocks * page_size * 2, ROW_WORDS), dtype=torch.int32
            )
            sharded_mains.append(
                PagedCacheRows(
                    m_raw,
                    (
                        (local_blocks, page_size, dim),
                        torch.bfloat16,
                        page_size * dim * 2,
                        0,
                        r * local_blocks,
                        True,
                    ),
                )
            )
            s_raw = torch.zeros(
                (local_blocks * page_size * 2, ROW_WORDS), dtype=torch.int32
            )
            sharded_swas.append(
                PagedCacheRows(
                    s_raw,
                    (
                        (local_blocks, page_size, dim),
                        torch.bfloat16,
                        page_size * dim * 2,
                        0,
                        r * local_blocks,
                        True,
                    ),
                )
            )
            i_raw = torch.zeros(
                (local_blocks * page_size // 2, ROW_WORDS), dtype=torch.int32
            )
            sharded_idxs.append(
                PagedCacheRows(
                    i_raw,
                    (
                        (local_blocks, page_size, idx_dim),
                        torch.bfloat16,
                        page_size * idx_dim * 2,
                        0,
                        r * local_blocks,
                        True,
                    ),
                )
            )

        # Write 32 pages * 16 rows = 512 tokens into replicated & sharded caches
        all_slots = torch.arange(global_blocks * page_size, dtype=torch.long)
        kv_values = torch.randn(
            (global_blocks * page_size, dim), generator=gen, dtype=torch.bfloat16
        )
        swa_values = torch.randn(
            (global_blocks * page_size, dim), generator=gen, dtype=torch.bfloat16
        )
        idx_values = torch.randn(
            (global_blocks * page_size, idx_dim), generator=gen, dtype=torch.bfloat16
        )
        rep_main.write(all_slots, kv_values)
        rep_swa.write(all_slots, swa_values)
        rep_idx.write(all_slots, idx_values)
        for r in range(tp_size):
            sharded_mains[r].write(all_slots, kv_values)
            sharded_swas[r].write(all_slots, swa_values)
            sharded_idxs[r].write(all_slots, idx_values)

        # Verify 8x HBM savings per rank
        assert sharded_mains[0].raw.nbytes * tp_size == rep_main.raw.nbytes
        assert sharded_swas[0].raw.nbytes * tp_size == rep_swa.raw.nbytes
        assert sharded_idxs[0].raw.nbytes * tp_size == rep_idx.raw.nbytes

        # Step 1: Verify 8-rank owner-sharded Indexer scoring matches replicated bitwise
        idx_q = jnp.asarray(
            torch.randn(
                (num_seqs, idx_heads, idx_dim), generator=gen, dtype=torch.float32
            )
            .to(torch.bfloat16)
            .float()
            .numpy(),
            dtype=jnp.bfloat16,
        )
        idx_weights = jnp.asarray(
            torch.rand(
                (num_seqs, idx_heads), generator=gen, dtype=torch.float32
            ).numpy()
        )
        kv_width = pages_per_seq * page_size
        lengths = jnp.full((num_seqs,), kv_width, dtype=jnp.int32)
        logical_idx = torch.arange(kv_width, dtype=torch.long)[None, :].expand(
            num_seqs, -1
        )
        reqs = torch.arange(num_seqs, dtype=torch.long)
        global_idx_slots = v2_attn._row_slots(rep_idx, block_table, reqs, logical_idx)
        rep_idx_records = jnp.asarray(
            rep_idx.row_indices(global_idx_slots).to(torch.int32).numpy()
        )

        ref_scores = attention_kernels.paged_index_scores_jax(
            idx_q,
            jnp.asarray(rep_idx.raw.numpy()),
            rep_idx_records,
            idx_weights,
            lengths,
        )

        part, _ = v2_attn._query_split(num_seqs, tp_size, kv_shard=True)
        assert part == 2
        rank_scores = []
        for r in range(tp_size):
            start, end = r * part, (r + 1) * part
            idx_rec_r = jnp.asarray(
                sharded_idxs[r]
                .row_indices(global_idx_slots[start:end])
                .to(torch.int32)
                .numpy()
            )
            scores_r = attention_kernels.paged_index_scores_jax(
                idx_q[start:end],
                jnp.asarray(sharded_idxs[r].raw.numpy()),
                idx_rec_r,
                idx_weights[start:end],
                lengths[start:end],
            )
            rank_scores.append(scores_r)
        sharded_scores = jnp.concatenate(rank_scores, axis=0)
        np.testing.assert_array_equal(
            np.asarray(sharded_scores), np.asarray(ref_scores)
        )

        # Step 2: Verify 8-rank owner-sharded SWA + Main attention bitwise parity
        q = jnp.asarray(
            torch.randn((num_seqs, heads, dim), generator=gen, dtype=torch.float32)
            .to(torch.bfloat16)
            .float()
            .numpy(),
            dtype=jnp.bfloat16,
        )
        sink = jnp.zeros((heads,), dtype=jnp.float32)
        q_loc = jnp.arange(num_seqs + 1, dtype=jnp.int32)

        logical_swa = torch.arange(window, dtype=torch.long)[None, :].expand(
            num_seqs, -1
        )
        global_swa_slots = v2_attn._row_slots(rep_swa, block_table, reqs, logical_swa)
        rep_swa_records = jnp.asarray(
            rep_swa.row_indices(global_swa_slots).to(torch.int32).numpy()
        )
        logical_main = torch.arange(topk, dtype=torch.long)[None, :].expand(
            num_seqs, -1
        )
        global_main_slots = v2_attn._row_slots(
            rep_main, block_table, reqs, logical_main
        )
        rep_main_records = jnp.asarray(
            rep_main.row_indices(global_main_slots).to(torch.int32).numpy()
        )

        ref_out = attention_kernels.paged_attention_jax(
            q,
            jnp.asarray(rep_swa.raw.numpy()),
            jnp.asarray(rep_main.raw.numpy()),
            rep_swa_records,
            rep_main_records,
            sink,
            scale,
        )

        rank_outputs = []
        for r in range(tp_size):
            swa_rec_r = jnp.asarray(
                sharded_swas[r].row_indices(global_swa_slots).to(torch.int32).numpy()
            )
            main_rec_r = jnp.asarray(
                sharded_mains[r].row_indices(global_main_slots).to(torch.int32).numpy()
            )
            swa_l, main_l, _ = attention_kernels.slice_token_parallel_attention_inputs(
                swa_rec_r,
                main_rec_r,
                q_loc,
                jnp.asarray(r, dtype=jnp.int32),
                tp_size,
                num_seqs,
            )
            start = r * (num_seqs // tp_size)
            end = start + (num_seqs // tp_size)
            out_r = attention_kernels.paged_attention_jax(
                q[start:end],
                jnp.asarray(sharded_swas[r].raw.numpy()),
                jnp.asarray(sharded_mains[r].raw.numpy()),
                swa_l,
                main_l,
                sink,
                scale,
            )
            rank_outputs.append(out_r)

        sharded_out = jnp.concatenate(rank_outputs, axis=0)
        np.testing.assert_array_equal(
            np.asarray(sharded_out, dtype=np.float32),
            np.asarray(ref_out, dtype=np.float32),
        )

    def test_s66_bf16_attention_and_paged_attention_token_parallel(self, monkeypatch):
        """Verify S6.6 owner-parallel bf16_attention and paged_attention."""
        from jax.sharding import Mesh

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        mock_tp = mock.MagicMock()
        mock_tp.device_group.size.return_value = 8
        captured_fns: dict[str, Any] = {}

        def capture_jax_op(name, fn, **kwargs):
            captured_fns[name] = fn
            m = mock.MagicMock()
            m._fn = fn
            return m

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch(
                "vllm.distributed.get_tp_group",
                return_value=mock_tp,
            ),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=capture_jax_op),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None
            attention_kernels.initialize_paged_attention()
            bf16_attn = captured_fns["pallas::dsv41v2_paged_bf16_attention"]
            bf16_attn_2b = captured_fns["pallas::dsv41v2_paged_bf16_attention_2bank"]

            a2a_calls: list[int] = []
            seen_swa_rows: list[int] = []

            def fake_a2a(x, *a, **kw):
                a2a_calls.append(x.shape[0])
                return x

            def fake_paged_jax(q, s_p, m_p, s_r, m_r, *a, **kw):
                seen_swa_rows.append(s_r.shape[0])
                return q

            with (
                mock.patch(
                    "jax.shard_map",
                    side_effect=lambda fn, **kw: lambda *args: fn(*args),
                ),
                mock.patch("jax.lax.all_to_all", side_effect=fake_a2a),
                mock.patch.object(
                    attention_kernels,
                    "paged_attention_jax",
                    side_effect=fake_paged_jax,
                ),
            ):
                # 1. 1-bank owner-parallel bf16_attention (KV_SHARD=True)
                monkeypatch.setitem(
                    vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", True
                )
                out_1b = bf16_attn(
                    jnp.zeros((16, 8, 512), dtype=jnp.bfloat16),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((16, 64), dtype=jnp.int32),
                    jnp.zeros((16, 0), dtype=jnp.int32),
                    jnp.zeros((8,), dtype=jnp.float32),
                    jnp.zeros((64,), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    True,
                    1,
                )
                assert out_1b.shape == (16, 8, 512)
                assert seen_swa_rows[-1] == 2
                assert len(a2a_calls) == 2

                # 2. 2-bank owner-parallel paged_attention_2bank (DSpark draft block=6)
                out_2b = bf16_attn_2b(
                    jnp.zeros((48, 8, 512), dtype=jnp.bfloat16),
                    jnp.zeros((16, 128), dtype=jnp.int32),
                    jnp.zeros((16, 128), dtype=jnp.int32),
                    jnp.zeros((16, 128), dtype=jnp.int32),
                    jnp.zeros((16, 128), dtype=jnp.int32),
                    jnp.zeros((48, 64), dtype=jnp.int32),
                    jnp.zeros((48, 0), dtype=jnp.int32),
                    jnp.zeros((8,), dtype=jnp.float32),
                    jnp.zeros((64,), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    True,
                    6,
                )
                assert out_2b.shape == (48, 8, 512)
                assert seen_swa_rows[-1] == 6
                assert len(a2a_calls) == 4

                # 3. Non-token-parallel fallback (KV_SHARD=False, token_parallel=False)
                monkeypatch.setitem(
                    vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", False
                )
                out_hp = bf16_attn(
                    jnp.zeros((16, 8, 512), dtype=jnp.bfloat16),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((16, 64), dtype=jnp.int32),
                    jnp.zeros((16, 0), dtype=jnp.int32),
                    jnp.zeros((8,), dtype=jnp.float32),
                    jnp.zeros((64,), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    False,
                    1,
                )
                assert out_hp.shape == (16, 8, 512)
                assert seen_swa_rows[-1] == 16

                # 4. Legacy 7-arg and 8-arg calls to bf16_attention
                out_leg7 = bf16_attn(
                    jnp.zeros((4, 8, 512), dtype=jnp.bfloat16),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((4, 64), dtype=jnp.int32),
                    jnp.zeros((4, 0), dtype=jnp.int32),
                    jnp.zeros((8,), dtype=jnp.float32),
                    0.125,
                )
                assert out_leg7.shape == (4, 8, 512)
                out_leg8 = bf16_attn(
                    jnp.zeros((6, 8, 512), dtype=jnp.bfloat16),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((6, 64), dtype=jnp.int32),
                    jnp.zeros((6, 0), dtype=jnp.int32),
                    jnp.zeros((8,), dtype=jnp.float32),
                    0.125,
                    6,
                )
                assert out_leg8.shape == (6, 8, 512)

        # 5. Test paged_attention wrapper with explicit full_sink & tp_rank
        mock_op_1b = mock.MagicMock(return_value=torch.zeros((8, 2, 128)))
        mock_op_2b = mock.MagicMock(return_value=torch.zeros((8, 2, 128)))
        monkeypatch.setattr(attention_kernels, "_attention_op", mock_op_1b)
        monkeypatch.setattr(attention_kernels, "_attention_2bank_op", mock_op_2b)

        raw_1b = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
        cache_1b = PagedCacheRows(
            raw_1b, ((2, 4, 128), torch.bfloat16, 4 * 128 * 2, 0, 0, True)
        )
        res_1b = attention_kernels.paged_attention(
            torch.zeros((8, 2, 128)),
            cache_1b,
            cache_1b,
            torch.zeros((8, 4), dtype=torch.int32),
            torch.zeros((8, 0), dtype=torch.int32),
            torch.zeros((2,)),
            0.125,
            1,
            token_parallel=True,
            full_sink=torch.zeros((16,)),
            tp_rank=torch.tensor([3], dtype=torch.int32),
        )
        assert res_1b.shape == (8, 2, 128)
        mock_op_1b.assert_called_once()
        assert mock_op_1b.call_args.args[-1] == 1

        cache_2b = PagedCacheRows(
            (raw_1b, raw_1b.clone()),
            ((4, 4, 128), torch.bfloat16, 4 * 128 * 2, 0, 0, True),
        )
        res_2b = attention_kernels.paged_attention(
            torch.zeros((8, 2, 128)),
            cache_2b,
            cache_2b,
            torch.zeros((8, 4), dtype=torch.int32),
            torch.zeros((8, 0), dtype=torch.int32),
            torch.zeros((2,)),
            0.125,
            6,
            token_parallel=True,
        )
        assert res_2b.shape == (8, 2, 128)
        mock_op_2b.assert_called_once()
        assert mock_op_2b.call_args.args[-1] == 6

    def test_s66_8rank_compressor_state_cache_and_hybrid_v2_parity(self):
        """Verify 8-rank CompressorStateCache ring reads/writes (R=0 & R>0)."""
        tp_size = 8
        num_seqs = 16
        seqs_per_rank = num_seqs // tp_size  # 2 requests/rank
        capacity = 8  # CircularBufferSpec ring capacity
        dim = 512
        state_dim = 2 * dim  # 1024 FP32 (8 word rows per slot)
        row_bytes = state_dim * 4  # 4096 bytes
        stride_bytes = capacity * row_bytes

        gen = torch.Generator().manual_seed(20261009)
        norm_weight = torch.ones((dim,), dtype=torch.float32)
        eps = 1e-6
        query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32)
        token_to_req = torch.arange(num_seqs, dtype=torch.int32)

        # Reference unsharded CompressorStateCache (16 blocks)
        rep_raw = torch.zeros(
            (num_seqs * capacity * (row_bytes // (ROW_WORDS * 4)), ROW_WORDS),
            dtype=torch.int32,
        )
        rep_state = PagedCacheRows(
            rep_raw,
            ((num_seqs, capacity, state_dim), torch.float32, stride_bytes, 0),
        )

        # Case 1: R = 0 pure 8-way sharded CompressorStateCache (2 blocks/rank)
        sharded_states_r0 = [
            PagedCacheRows(
                torch.zeros(
                    (
                        seqs_per_rank * capacity * (row_bytes // (ROW_WORDS * 4)),
                        ROW_WORDS,
                    ),
                    dtype=torch.int32,
                ),
                (
                    (seqs_per_rank, capacity, state_dim),
                    torch.float32,
                    stride_bytes,
                    0,
                    r * seqs_per_rank,
                    True,
                ),
            )
            for r in range(tp_size)
        ]

        # Step 1 (even position pos=4): publish ring state, latent is masked to 0
        pos_even = torch.full((num_seqs,), 4, dtype=torch.long)
        slots_even = torch.arange(num_seqs, dtype=torch.long) * capacity + (
            pos_even % capacity
        )
        kv_score_even = torch.randn(
            (num_seqs, state_dim), generator=gen, dtype=torch.float32
        )
        ref_latent_even = torch.empty((num_seqs, dim), dtype=torch.bfloat16)
        v2_attn.save_compress_norm(
            kv_score_even,
            pos_even,
            rep_state,
            slots_even,
            query_start_loc,
            token_to_req,
            norm_weight,
            eps,
            2,
            ref_latent_even,
        )
        assert torch.all(ref_latent_even == 0)

        for r in range(tp_size):
            lat_r = torch.empty((num_seqs, dim), dtype=torch.bfloat16)
            v2_attn.save_compress_norm(
                kv_score_even,
                pos_even,
                sharded_states_r0[r],
                slots_even,
                query_start_loc,
                token_to_req,
                norm_weight,
                eps,
                2,
                lat_r,
            )

        # Step 2 (odd position pos=5): reads pos=4 from ring and emits compressed latent
        pos_odd = torch.full((num_seqs,), 5, dtype=torch.long)
        slots_odd = torch.arange(num_seqs, dtype=torch.long) * capacity + (
            pos_odd % capacity
        )
        kv_score_odd = torch.randn(
            (num_seqs, state_dim), generator=gen, dtype=torch.float32
        )
        ref_latent_odd = torch.empty((num_seqs, dim), dtype=torch.bfloat16)
        v2_attn.save_compress_norm(
            kv_score_odd,
            pos_odd,
            rep_state,
            slots_odd,
            query_start_loc,
            token_to_req,
            norm_weight,
            eps,
            2,
            ref_latent_odd,
        )

        sharded_latents = []
        for r in range(tp_size):
            lat_r = torch.empty((num_seqs, dim), dtype=torch.bfloat16)
            v2_attn.save_compress_norm(
                kv_score_odd,
                pos_odd,
                sharded_states_r0[r],
                slots_odd,
                query_start_loc,
                token_to_req,
                norm_weight,
                eps,
                2,
                lat_r,
            )
            # Owner rank r's slice [r*2 : (r+1)*2] is bitwise identical to reference
            sharded_latents.append(lat_r[r * seqs_per_rank : (r + 1) * seqs_per_rank])
        combined_latent = torch.cat(sharded_latents, dim=0)
        assert torch.equal(combined_latent, ref_latent_odd)

        # Case 2: R > 0 hybrid Id contract v2 (R = 16 replicated ring blocks + P = 2)
        for r in range(tp_size):
            hybrid_state = PagedCacheRows(
                torch.zeros(
                    (
                        (num_seqs + seqs_per_rank)
                        * capacity
                        * (row_bytes // (ROW_WORDS * 4)),
                        ROW_WORDS,
                    ),
                    dtype=torch.int32,
                ),
                (
                    (num_seqs + seqs_per_rank, capacity, state_dim),
                    torch.float32,
                    stride_bytes,
                    0,
                    num_seqs + r * seqs_per_rank,
                    True,
                    num_seqs,
                ),
            )
            lat_even = torch.empty((num_seqs, dim), dtype=torch.bfloat16)
            lat_odd = torch.empty((num_seqs, dim), dtype=torch.bfloat16)
            v2_attn.save_compress_norm(
                kv_score_even,
                pos_even,
                hybrid_state,
                slots_even,
                query_start_loc,
                token_to_req,
                norm_weight,
                eps,
                2,
                lat_even,
            )
            v2_attn.save_compress_norm(
                kv_score_odd,
                pos_odd,
                hybrid_state,
                slots_odd,
                query_start_loc,
                token_to_req,
                norm_weight,
                eps,
                2,
                lat_odd,
            )
            # Under hybrid Id contract v2, EVERY rank has all B latents identical!
            assert torch.equal(lat_odd, ref_latent_odd)

    def test_s66_8rank_swa_only_and_dspark_draft_bitwise_parity(self):
        """Verify 8-rank bitwise parity for SWA-only (block=1) & DSpark (block=6)."""
        tp_size = 8
        num_seqs = 16
        draft_block = 6
        draft_tokens = num_seqs * draft_block  # 96 tokens
        pages_per_seq = 2
        global_blocks = num_seqs * pages_per_seq  # 32 pages
        local_blocks = global_blocks // tp_size  # 4 pages/rank
        page_size = 16
        window = 32
        heads = 64
        heads_local = heads // tp_size  # 8 heads/rank
        dim = 512
        scale = dim**-0.5

        gen = torch.Generator().manual_seed(20261008)
        block_table = torch.arange(global_blocks, dtype=torch.int32).reshape(
            num_seqs, pages_per_seq
        )
        rep_swa_raw = torch.zeros(
            (global_blocks * page_size * 2, ROW_WORDS), dtype=torch.int32
        )
        rep_swa = PagedCacheRows(
            rep_swa_raw,
            ((global_blocks, page_size, dim), torch.bfloat16, page_size * dim * 2, 0),
        )
        sharded_swas = [
            PagedCacheRows(
                torch.zeros(
                    (local_blocks * page_size * 2, ROW_WORDS), dtype=torch.int32
                ),
                (
                    (local_blocks, page_size, dim),
                    torch.bfloat16,
                    page_size * dim * 2,
                    0,
                    r * local_blocks,
                    True,
                ),
            )
            for r in range(tp_size)
        ]

        all_slots = torch.arange(global_blocks * page_size, dtype=torch.long)
        swa_values = torch.randn(
            (global_blocks * page_size, dim), generator=gen, dtype=torch.bfloat16
        )
        rep_swa.write(all_slots, swa_values)
        for r in range(tp_size):
            sharded_swas[r].write(all_slots, swa_values)

        full_sink = jnp.asarray(
            torch.randn((heads,), generator=gen, dtype=torch.float32).numpy()
        )
        logical_swa = torch.arange(window, dtype=torch.long)[None, :].expand(
            num_seqs, -1
        )
        reqs_decode = torch.arange(num_seqs, dtype=torch.long)
        decode_swa_slots = v2_attn._row_slots(
            rep_swa, block_table, reqs_decode, logical_swa
        )
        rep_decode_records = jnp.asarray(
            rep_swa.row_indices(decode_swa_slots).to(torch.int32).numpy()
        )
        empty_main_decode = jnp.zeros((num_seqs, 0), dtype=jnp.int32)

        # Case A: SWA-only decode layer (block = 1, compress_ratio == 0)
        q_decode = jnp.asarray(
            torch.randn((num_seqs, heads, dim), generator=gen, dtype=torch.float32)
            .to(torch.bfloat16)
            .float()
            .numpy(),
            dtype=jnp.bfloat16,
        )
        ref_swa_only = attention_kernels.paged_attention_jax(
            q_decode,
            jnp.asarray(rep_swa.raw.numpy()),
            jnp.asarray(rep_swa.raw.numpy()),
            rep_decode_records,
            empty_main_decode,
            full_sink,
            scale,
            1,
        )
        # Simulate 8-rank head-sharded Q -> all_to_all -> owner-rank paged_attention
        q_hp_ranks = [
            q_decode[:, r * heads_local : (r + 1) * heads_local, :]
            for r in range(tp_size)
        ]
        seqs_local = num_seqs // tp_size
        out_tok_ranks = []
        for r in range(tp_size):
            # all_to_all(split_axis=0, concat_axis=1) gathers all heads for owner r
            q_tok_r = jnp.concatenate(
                [
                    q_hp_ranks[src][r * seqs_local : (r + 1) * seqs_local]
                    for src in range(tp_size)
                ],
                axis=1,
            )
            swa_rec_r = jnp.asarray(
                sharded_swas[r].row_indices(decode_swa_slots).to(torch.int32).numpy()
            )[r * seqs_local : (r + 1) * seqs_local]
            out_tok_r = attention_kernels.paged_attention_jax(
                q_tok_r,
                jnp.asarray(sharded_swas[r].raw.numpy()),
                jnp.asarray(sharded_swas[r].raw.numpy()),
                swa_rec_r,
                jnp.zeros((seqs_local, 0), dtype=jnp.int32),
                full_sink,
                scale,
                1,
            )
            out_tok_ranks.append(out_tok_r)
        # Inverse all_to_all(split_axis=1, concat_axis=0) + concat across head shards
        out_hp_ranks = [
            jnp.concatenate(
                [
                    out_tok_ranks[src][:, r * heads_local : (r + 1) * heads_local, :]
                    for src in range(tp_size)
                ],
                axis=0,
            )
            for r in range(tp_size)
        ]
        sharded_swa_only = jnp.concatenate(out_hp_ranks, axis=1)
        np.testing.assert_array_equal(
            np.asarray(sharded_swa_only, dtype=np.float32),
            np.asarray(ref_swa_only, dtype=np.float32),
        )

        # Case B: DSpark draft layer (block = 6, non-causal shared window, T = 96)
        reqs_draft = torch.arange(num_seqs, dtype=torch.long).repeat_interleave(
            draft_block
        )
        logical_draft = torch.arange(window, dtype=torch.long)[None, :].expand(
            draft_tokens, -1
        )
        draft_swa_slots = v2_attn._row_slots(
            rep_swa, block_table, reqs_draft, logical_draft
        )
        rep_draft_records = jnp.asarray(
            rep_swa.row_indices(draft_swa_slots).to(torch.int32).numpy()
        )
        empty_main_draft = jnp.zeros((draft_tokens, 0), dtype=jnp.int32)
        q_draft = jnp.asarray(
            torch.randn((draft_tokens, heads, dim), generator=gen, dtype=torch.float32)
            .to(torch.bfloat16)
            .float()
            .numpy(),
            dtype=jnp.bfloat16,
        )
        ref_dspark = attention_kernels.paged_attention_jax(
            q_draft,
            jnp.asarray(rep_swa.raw.numpy()),
            jnp.asarray(rep_swa.raw.numpy()),
            rep_draft_records,
            empty_main_draft,
            full_sink,
            scale,
            draft_block,
        )
        draft_local = draft_tokens // tp_size  # 12 tokens/rank (2 blocks of 6)
        q_draft_hp = [
            q_draft[:, r * heads_local : (r + 1) * heads_local, :]
            for r in range(tp_size)
        ]
        draft_tok_ranks = []
        for r in range(tp_size):
            q_tok_r = jnp.concatenate(
                [
                    q_draft_hp[src][r * draft_local : (r + 1) * draft_local]
                    for src in range(tp_size)
                ],
                axis=1,
            )
            swa_rec_r = jnp.asarray(
                sharded_swas[r].row_indices(draft_swa_slots).to(torch.int32).numpy()
            )[r * draft_local : (r + 1) * draft_local]
            out_tok_r = attention_kernels.paged_attention_jax(
                q_tok_r,
                jnp.asarray(sharded_swas[r].raw.numpy()),
                jnp.asarray(sharded_swas[r].raw.numpy()),
                swa_rec_r,
                jnp.zeros((draft_local, 0), dtype=jnp.int32),
                full_sink,
                scale,
                draft_block,
            )
            draft_tok_ranks.append(out_tok_r)
        draft_hp_ranks = [
            jnp.concatenate(
                [
                    draft_tok_ranks[src][:, r * heads_local : (r + 1) * heads_local, :]
                    for src in range(tp_size)
                ],
                axis=0,
            )
            for r in range(tp_size)
        ]
        sharded_dspark = jnp.concatenate(draft_hp_ranks, axis=1)
        np.testing.assert_array_equal(
            np.asarray(sharded_dspark, dtype=np.float32),
            np.asarray(ref_dspark, dtype=np.float32),
        )


# ==============================================================================
# 6. S6.2 Worker OwnerLayout + S6.5/S6.6 Owner-Major Permutation Composition
# ==============================================================================
class TestS62AndS65OwnerComposition:
    """Verify S6.2 worker OwnerLayout materialization and S6.5 owner_perm/owner_inv."""

    def test_s62_resolve_worker_owner_layout_and_materialize_composition(
        self, monkeypatch
    ):
        """Verify R=3, P=5, tp=8 (M=8, N=43) row_indices & pages match OwnerLayout."""
        from vllm_torchtpu.core import owner_affine_block_pool as oabp
        from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
            materialize_upstream_kv_cache,
            resolve_worker_owner_layout,
        )

        # 1. Test resolve_worker_owner_layout (both fallback and explicit helper)
        vllm_cfg = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=8),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
            speculative_config=None,
            max_in_flight_tokens=16,
            cache_config=SimpleNamespace(enable_prefix_caching=False),
        )
        kv_cfg = SimpleNamespace(
            num_blocks=8,
            kv_cache_tensors=[SimpleNamespace(size=8 * 64 * 512)],
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=64 * 512))
            ],
        )
        monkeypatch.setattr(oabp, "replicated_blocks_needed", lambda vc, kc: 3)
        monkeypatch.delattr(oabp, "worker_owner_layout", raising=False)
        layout = resolve_worker_owner_layout(vllm_cfg, kv_cfg)
        assert layout == oabp.OwnerLayout(3, 8, 5)
        assert layout.num_blocks == 43

        bad_kv_cfg = SimpleNamespace(
            num_blocks=3,
            kv_cache_tensors=[SimpleNamespace(size=3 * 64 * 512)],
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(page_size_bytes=64 * 512))
            ],
        )
        with pytest.raises(ValueError, match="needs more than 3 KV blocks per rank"):
            resolve_worker_owner_layout(vllm_cfg, bad_kv_cfg)

        # Also test delegation when worker_owner_layout is present on oabp
        monkeypatch.setattr(
            oabp,
            "worker_owner_layout",
            lambda vc, kc: oabp.OwnerLayout(3, 8, 5),
            raising=False,
        )
        assert resolve_worker_owner_layout(vllm_cfg, kv_cfg) == oabp.OwnerLayout(
            3, 8, 5
        )

        # 2. Verify materialize_upstream_kv_cache with scheduler_blocks=43, M=8
        def fake_alloc(cfg, dev, lay):
            raw = torch.empty(
                (cfg.num_blocks, 1, 64, 256), dtype=torch.bfloat16, device="meta"
            )
            return {"model.layers.0.self_attn.attn": raw}

        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.cache.allocate_upstream_kv_cache",
            fake_alloc,
        )
        lay = SimpleNamespace(is_layer_compact=False)

        with pytest.raises(
            ValueError, match="Worker local_blocks .* must match per-rank"
        ):
            materialize_upstream_kv_cache(
                kv_cfg,
                torch.device("cpu"),
                lay,
                word_rows=True,
                kv_shard=True,
                tp_size=8,
                tp_rank=0,
                replicated_blocks=3,
                scheduler_blocks=51,
            )

        for r in range(8):
            mat = materialize_upstream_kv_cache(
                kv_cfg,
                torch.device("cpu"),
                lay,
                word_rows=True,
                kv_shard=True,
                tp_size=8,
                tp_rank=r,
                replicated_blocks=layout.num_replicated,
                scheduler_blocks=layout.num_blocks,
            )
            raw_pool = mat.native_raw_cache["model.layers.0.self_attn.attn"]
            cache_layout = mat.native_cache_layouts["model.layers.0.self_attn.attn"]
            rows_cache = PagedCacheRows(raw_pool, cache_layout)
            assert rows_cache.shape == (8, 64, 256)
            assert rows_cache.replicated_blocks == 3
            assert rows_cache.block_offset == 3 + r * 5

            all_blocks = torch.arange(layout.num_blocks, dtype=torch.long)
            slot_starts = all_blocks * 64
            actual_row_indices = rows_cache.row_indices(slot_starts)
            actual_pages = rows_cache.pages(all_blocks.to(torch.int32)[:, None])[:, 0]
            for b in range(layout.num_blocks):
                expected_local = layout.local_block_id(b, r)
                if expected_local == -1:
                    assert int(actual_row_indices[b].item()) == -1
                    assert int(actual_pages[b].item()) == 0
                else:
                    assert (
                        int(actual_row_indices[b].item())
                        == expected_local * rows_cache.page_stride
                    )
                    assert int(actual_pages[b].item()) == expected_local

    def test_s62_kv_cache_manager_initialize_kv_cache_passes_owner_layout(
        self, monkeypatch
    ):
        """Verify KVCacheManager.initialize_kv_cache forwards OwnerLayout."""
        from vllm.v1.kv_cache_interface import AttentionSpec

        from vllm_torchtpu.core import owner_affine_block_pool as oabp
        from vllm_torchtpu.runner import kv_cache_manager as kcm_mod

        monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_IMPL", "v2")
        monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", True)
        monkeypatch.setitem(vars(envs), "VLLM_TPU_BLOCK_MAJOR_KV", False)
        seen_kwargs: dict[str, Any] = {}

        def fake_materialize(cfg, dev, lay, **kwargs):
            seen_kwargs.update(kwargs)
            return SimpleNamespace(
                kv_caches={"layer0": torch.empty(0)},
                raw_tensors=[],
                native_raw_cache={},
                native_cache_layouts={},
                rope_pools={},
            )

        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.cache.resolve_worker_owner_layout",
            lambda vc, kc: oabp.OwnerLayout(3, 8, 5),
        )
        monkeypatch.setattr(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.cache.materialize_upstream_kv_cache",
            fake_materialize,
        )
        monkeypatch.setattr(
            kcm_mod.TpuPlatform, "_find_non_ssm_backend", lambda vc: None
        )
        monkeypatch.setattr(kcm_mod, "AttentionMetadataBuilder", lambda *a, **kw: None)
        monkeypatch.setattr(kcm_mod, "synchronize_tensors", lambda *a, **kw: None)
        monkeypatch.setattr(kcm_mod, "is_cache_for_ds_v4", lambda m: False)
        monkeypatch.setattr(
            kcm_mod, "check_kv_caches_cover_block_ids", lambda *a, **kw: None
        )
        monkeypatch.setattr(kcm_mod.utils, "tpu_bind_kv_cache", lambda *a, **kw: None)
        monkeypatch.setattr(kcm_mod, "has_kv_transfer_group", lambda: False)

        mgr = object.__new__(kcm_mod.KVCacheManager)
        mgr._maybe_add_kv_sharing_layers_to_kv_cache_groups = lambda kc: None
        mgr._add_shared_kv_cache_aliases = lambda kc: None
        mgr.runner = SimpleNamespace(
            vllm_config=SimpleNamespace(
                compilation_config=SimpleNamespace(
                    static_forward_context={
                        "layer0": SimpleNamespace(tpu_native_kv=True)
                    }
                ),
                parallel_config=SimpleNamespace(tensor_parallel_size=8),
            ),
            cache_config=SimpleNamespace(
                get_resolved_kv_cache_layout=lambda: SimpleNamespace(
                    name="BLOCK_COMPACT"
                )
            ),
            device=torch.device("cpu"),
            _unified_kv_layout=False,
            may_reinitialize_input_batch=lambda kc, bs: None,
            block_table_cpu=torch.zeros((1, 1), dtype=torch.int32),
            input_batch=SimpleNamespace(
                block_table=[
                    SimpleNamespace(
                        get_cpu_tensor=lambda: torch.zeros((1, 1), dtype=torch.int32)
                    )
                ]
            ),
            shared_kv_cache_layers={},
            kv_caches=[],
            _kv_cache_bundle=None,
            _precompile_dense_kv_cache_block_copies=lambda: None,
            use_spmd=False,
            _uniform_mamba_layout=True,
            speculative_config=None,
            _initialize_attention_kernels=lambda force=True: None,
            enforce_eager=True,
        )
        spec = AttentionSpec(
            block_size=64, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
        )
        kv_cfg = SimpleNamespace(
            num_blocks=8,
            has_mamba_layers=False,
            kv_cache_tensors=[SimpleNamespace(size=8 * 64 * 512, layers=["layer0"])],
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=spec, layer_names=["layer0"])
            ],
        )
        mgr.initialize_kv_cache(kv_cfg)
        assert seen_kwargs == {
            "word_rows": True,
            "replicated_blocks": 3,
            "scheduler_blocks": 43,
        }
        seen_kwargs.clear()
        monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", False)
        mgr.initialize_kv_cache(kv_cfg)
        assert seen_kwargs == {"word_rows": True}

    def test_s65_permute_unpermute_and_slice_token_parallel_with_owner_perm(
        self, monkeypatch
    ):
        """Verify JAX/Torch permute/unpermute, 2D q_loc, and KV_SHARD error guards."""
        # 1. JAX & Torch permute/unpermute helpers with -1 padding
        x_t = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
        perm_t = torch.tensor([2, 0, -1, 1], dtype=torch.int32)
        inv_t = torch.tensor([1, 3, 0], dtype=torch.int32)
        p_t = attention_kernels._permute_owner_tokens_torch(x_t, perm_t, fill_value=-1)
        assert torch.equal(
            p_t,
            torch.tensor([[30, 31], [10, 11], [-1, -1], [20, 21]], dtype=torch.int32),
        )
        u_t = attention_kernels._unpermute_owner_tokens_torch(p_t, inv_t)
        assert torch.equal(u_t, x_t)

        x_j = jnp.asarray(x_t.numpy())
        perm_j = jnp.asarray(perm_t.numpy())
        inv_j = jnp.asarray(inv_t.numpy())
        p_j = attention_kernels._permute_owner_tokens_jax(x_j, perm_j, fill_value=-1)
        np.testing.assert_array_equal(np.asarray(p_j), p_t.numpy())
        u_j = attention_kernels._unpermute_owner_tokens_jax(p_j, inv_j)
        np.testing.assert_array_equal(np.asarray(u_j), x_t.numpy())

        # 2. slice_token_parallel_attention_inputs with owner_perm + 2D q_loc
        swa_full = jnp.arange(12, dtype=jnp.int32).reshape(3, 4)
        main_full = (jnp.arange(12, dtype=jnp.int32) + 100).reshape(3, 4)
        q_loc_2d = jnp.array([[0, 2, 2], [2, 2, 3]], dtype=jnp.int32)
        s_l0, m_l0, ql_0 = attention_kernels.slice_token_parallel_attention_inputs(
            swa_full,
            main_full,
            q_loc_2d,
            jnp.asarray(0, dtype=jnp.int32),
            2,
            3,
            owner_perm=perm_j,
        )
        np.testing.assert_array_equal(
            np.asarray(s_l0), np.asarray(swa_full[jnp.array([2, 0])])
        )
        np.testing.assert_array_equal(
            np.asarray(m_l0), np.asarray(main_full[jnp.array([2, 0])])
        )
        np.testing.assert_array_equal(np.asarray(ql_0), np.array([0, 2, 2]))

        # Rank 1 has perm [-1, 1] -> row 0 masked to -1, row 1 is token 1
        swa_seg1 = swa_full[jnp.array([0, 1])]
        main_seg1 = main_full[jnp.array([0, 1])]
        s_l1, m_l1, ql_1 = attention_kernels.slice_token_parallel_attention_inputs(
            swa_seg1,
            main_seg1,
            q_loc_2d,
            jnp.asarray(1, dtype=jnp.int32),
            2,
            3,
            owner_perm=perm_j,
        )
        assert np.all(np.asarray(s_l1[0]) == -1)
        np.testing.assert_array_equal(np.asarray(s_l1[1]), np.asarray(swa_full[1]))
        assert np.all(np.asarray(m_l1[0]) == -1)
        np.testing.assert_array_equal(np.asarray(m_l1[1]), np.asarray(main_full[1]))
        np.testing.assert_array_equal(np.asarray(ql_1), np.array([0, 0, 1]))

        # Also test non-owner_perm with pre-segmented swa_records/main_records
        s_seg_plain, m_seg_plain, _ = (
            attention_kernels.slice_token_parallel_attention_inputs(
                swa_seg1,
                main_seg1,
                jnp.array([0, 2, 4], dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                2,
                4,
            )
        )
        np.testing.assert_array_equal(np.asarray(s_seg_plain), np.asarray(swa_seg1))
        np.testing.assert_array_equal(np.asarray(m_seg_plain), np.asarray(main_seg1))

        # 3. Verify bf16/fp8/fp8_qo attention with owner_perm & error guards
        from jax.sharding import Mesh

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        mock_tp = mock.MagicMock()
        mock_tp.device_group.size.return_value = 8
        captured_fns: dict[str, Any] = {}

        def capture_jax_op(name, fn, **kwargs):
            captured_fns[name] = fn
            m = mock.MagicMock()
            m._fn = fn
            return m

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch("vllm.distributed.get_tp_group", return_value=mock_tp),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=capture_jax_op),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None
            attention_kernels.initialize_paged_attention()
            bf16_fn = captured_fns["pallas::dsv41v2_paged_bf16_attention"]
            fp8_fn = captured_fns["pallas::dsv41v2_sparse_fp8_attention"]
            qo_fn = captured_fns["pallas::dsv41v2_sparse_fp8_attention_qo"]

            monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", True)

            # Error guards when T=7 is not divisible by tp_size=8 under KV_SHARD=True
            with pytest.raises(
                ValueError, match="TPU_DEEPSEEK_V41_KV_SHARD requires token count"
            ):
                bf16_fn(
                    jnp.zeros((7, 8, 512), dtype=jnp.bfloat16),
                    jnp.zeros((16, 128), dtype=jnp.int32),
                    jnp.zeros((16, 128), dtype=jnp.int32),
                    jnp.zeros((7, 64), dtype=jnp.int32),
                    jnp.zeros((7, 0), dtype=jnp.int32),
                    jnp.zeros((8,), dtype=jnp.float32),
                    jnp.zeros((64,), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    True,
                    1,
                )

            with pytest.raises(
                ValueError, match="TPU_DEEPSEEK_V41_KV_SHARD requires token count"
            ):
                fp8_fn(
                    jnp.zeros((7, 8, 512), dtype=jnp.bfloat16),
                    jnp.zeros((16, 128), dtype=jnp.int32),
                    jnp.zeros((16, 128), dtype=jnp.int32),
                    jnp.zeros((4, 128), dtype=jnp.int32),
                    jnp.zeros((7, 64), dtype=jnp.int32),
                    jnp.zeros((7, 16), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((8,), dtype=jnp.float32),
                    jnp.zeros((64,), dtype=jnp.float32),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    64,
                    None,
                    False,
                    True,
                    False,
                    4096,
                    False,
                )

            with pytest.raises(
                ValueError,
                match="TPU_DEEPSEEK_V41_KV_SHARD requires token-parallel",
            ):
                qo_fn(
                    jnp.zeros((7, 1280), dtype=jnp.bfloat16),
                    jnp.zeros((7, 1280), dtype=jnp.bfloat16),
                    jnp.zeros((1280, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros((1280, 512)),
                    jnp.zeros((512,)),
                    jnp.zeros((7,)),
                    jnp.zeros((7, 64)),
                    jnp.zeros((16, 64, 128)),
                    jnp.zeros((16, 64, 128)),
                    jnp.zeros((16, 64, 64)),
                    jnp.zeros((7, 64), dtype=jnp.int32),
                    jnp.zeros((7, 16), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((64,)),
                    jnp.zeros((64,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.zeros((512, 1024)),
                    jnp.zeros((1024,)),
                    jnp.zeros((512, 1024)),
                    jnp.zeros((1024,)),
                    jnp.zeros((7,), dtype=jnp.int32),
                    0.125,
                    64,
                    None,
                    False,
                    128,
                    True,
                )

            # Valid owner_perm (T=12 padded to T_pad=16 across 8 ranks, 2 tokens/rank)
            owner_perm_16 = jnp.array(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1],
                dtype=jnp.int32,
            )
            owner_inv_12 = jnp.arange(12, dtype=jnp.int32)
            with (
                mock.patch(
                    "jax.shard_map",
                    side_effect=lambda fn, **kw: lambda *args: fn(*args),
                ),
                mock.patch("jax.lax.all_to_all", side_effect=lambda x, **kw: x),
                mock.patch.object(
                    attention_kernels,
                    "paged_attention_jax",
                    side_effect=lambda q, *a, **kw: q,
                ),
                mock.patch.object(
                    attention_kernels,
                    "sparse_fp8_attention_jax",
                    side_effect=lambda q, *a, **kw: q,
                ),
                mock.patch(
                    "vllm_torchtpu.layers.adapter.linear_common._quantized_matmul_jax",
                    side_effect=lambda q_in, w, ws: jnp.zeros(
                        (q_in.shape[0], 64 * 512), dtype=jnp.bfloat16
                    ),
                ),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v4.attention._rope_jax",
                    side_effect=lambda q, pos, cs: q,
                ),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v4.attention._o_proj_jax",
                    side_effect=lambda out, pos, cs, w, ws, head_dim: jnp.zeros(
                        (out.shape[0], 1024), dtype=jnp.bfloat16
                    ),
                ),
            ):
                # Test bf16_fn with both full records (12) and pre-segmented records (2)
                for rec_rows in (12, 2):
                    out_bf16 = bf16_fn(
                        jnp.zeros((12, 8, 512), dtype=jnp.bfloat16),
                        jnp.zeros((16, 128), dtype=jnp.int32),
                        jnp.zeros((16, 128), dtype=jnp.int32),
                        jnp.zeros((rec_rows, 64), dtype=jnp.int32),
                        jnp.zeros((rec_rows, 0), dtype=jnp.int32),
                        jnp.zeros((8,), dtype=jnp.float32),
                        jnp.zeros((64,), dtype=jnp.float32),
                        jnp.zeros((1,), dtype=jnp.int32),
                        0.125,
                        True,
                        1,
                        owner_perm_16,
                        owner_inv_12,
                    )
                    assert out_bf16.shape == (12, 8, 512)

                # Test fp8_fn with both full records (12) and pre-segmented records (2)
                for rec_rows in (12, 2):
                    out_fp8 = fp8_fn(
                        jnp.zeros((12, 8, 512), dtype=jnp.bfloat16),
                        jnp.zeros((16, 128), dtype=jnp.int32),
                        jnp.zeros((16, 128), dtype=jnp.int32),
                        jnp.zeros((4, 128), dtype=jnp.int32),
                        jnp.zeros((rec_rows, 64), dtype=jnp.int32),
                        jnp.zeros((rec_rows, 16), dtype=jnp.int32),
                        jnp.zeros((1, 1), dtype=jnp.int32),
                        jnp.zeros((2,), dtype=jnp.int32),
                        jnp.zeros((8,), dtype=jnp.float32),
                        jnp.zeros((64,), dtype=jnp.float32),
                        jnp.zeros((1,), dtype=jnp.int32),
                        0.125,
                        64,
                        None,
                        False,
                        True,
                        False,
                        4096,
                        False,
                        owner_perm_16,
                        owner_inv_12,
                    )
                    assert out_fp8.shape == (12, 8, 512)

                # Test qo_fn with replicated_qr=True/False and full/segmented records
                for rep_qr, rec_rows in ((True, 12), (False, 2)):
                    qr_loc_in = (
                        jnp.zeros((12, 1280), dtype=jnp.bfloat16)
                        if rep_qr
                        else jnp.zeros((2, 1280), dtype=jnp.bfloat16)
                    )
                    out_qo = qo_fn(
                        jnp.zeros((12, 1280), dtype=jnp.bfloat16),
                        qr_loc_in,
                        jnp.zeros((1280, 256)),
                        jnp.zeros((256,)),
                        jnp.zeros((1280, 512)),
                        jnp.zeros((512,)),
                        jnp.zeros((12,)),
                        jnp.zeros((12, 64)),
                        jnp.zeros((16, 64, 128)),
                        jnp.zeros((16, 64, 128)),
                        jnp.zeros((16, 64, 64)),
                        jnp.zeros((rec_rows, 64), dtype=jnp.int32),
                        jnp.zeros((rec_rows, 16), dtype=jnp.int32),
                        jnp.zeros((1, 1), dtype=jnp.int32),
                        jnp.zeros((2,), dtype=jnp.int32),
                        jnp.zeros((64,)),
                        jnp.zeros((64,)),
                        jnp.zeros((1,), dtype=jnp.int32),
                        jnp.zeros((512, 1024)),
                        jnp.zeros((1024,)),
                        jnp.zeros((512, 1024)),
                        jnp.zeros((1024,)),
                        jnp.zeros((12,), dtype=jnp.int32),
                        0.125,
                        64,
                        None,
                        False,
                        128,
                        rep_qr,
                        False,
                        owner_perm_16,
                        owner_inv_12,
                    )
                    assert out_qo.shape == (12, 128)

        # 4. Torch wrappers with owner_perm and owner_inv
        monkeypatch.setattr(
            attention_kernels,
            "_attention_op",
            lambda q, *a: torch.zeros_like(q),
        )
        monkeypatch.setattr(
            attention_kernels,
            "_fp8_attention_op",
            lambda q, *a: torch.zeros_like(q),
        )
        monkeypatch.setattr(
            attention_kernels,
            "_fp8_attention_qo_op",
            lambda qr, *a: torch.zeros((qr.shape[0], 128), dtype=qr.dtype),
        )
        raw_pool = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
        rope_pool = torch.zeros((4, ROW_WORDS), dtype=torch.int32)
        bf16_cache = PagedCacheRows(
            raw_pool, ((2, 4, 128), torch.bfloat16, 4 * 128 * 2, 0)
        )
        fp8_cache = PagedCacheRows(
            raw_pool,
            ((2, 4, 512), torch.uint8, 4 * 512, 0),
            rope=rope_pool,
        )
        perm_torch = torch.tensor([2, 0, -1, 1], dtype=torch.int32)
        inv_torch = torch.tensor([1, 3, 0], dtype=torch.int32)
        for seg in (False, True):
            main_r = torch.zeros((2 if seg else 3, 4), dtype=torch.int32)
            out_pa = attention_kernels.paged_attention(
                torch.zeros((3, 2, 128)),
                bf16_cache,
                bf16_cache,
                torch.zeros((3, 4), dtype=torch.int32),
                main_r,
                torch.zeros((2,)),
                0.125,
                owner_perm=perm_torch,
                owner_inv=inv_torch,
                main_records_segmented=seg,
            )
            assert out_pa.shape == (3, 2, 128)

            out_sfa = attention_kernels.sparse_fp8_attention(
                torch.zeros((3, 2, 128)),
                bf16_cache,
                fp8_cache,
                torch.zeros((3, 4), dtype=torch.int32),
                main_r,
                torch.zeros((2,)),
                0.125,
                owner_perm=perm_torch,
                owner_inv=inv_torch,
                main_records_segmented=seg,
            )
            assert out_sfa.shape == (3, 2, 128)

            out_sqo = attention_kernels.sparse_fp8_attention_qo(
                torch.zeros((3, 64)),
                torch.zeros((3 if seg else 2, 64)),
                torch.zeros((64, 64)),
                torch.zeros((64,)),
                torch.zeros((64, 128)),
                torch.zeros((128,)),
                torch.zeros((3,), dtype=torch.long),
                torch.zeros((8, 64)),
                bf16_cache,
                fp8_cache,
                torch.zeros((3, 4), dtype=torch.int32),
                main_r,
                torch.zeros((2,)),
                torch.zeros((4,)),
                torch.tensor([0], dtype=torch.int32),
                torch.zeros((64, 128)),
                torch.zeros((128,)),
                torch.zeros((64, 128)),
                torch.zeros((128,)),
                torch.ones((3,), dtype=torch.bool),
                0.125,
                None,
                None,
                None,
                False,
                64,
                replicated_qr=seg,
                owner_perm=perm_torch,
                owner_inv=inv_torch,
                main_records_segmented=seg,
            )
            assert out_sqo.shape == (3, 128)

    def test_s65_attention_layer_owner_perm_and_not_implemented_guard(
        self, monkeypatch
    ):
        """Verify _prepare_tpu & _select_paged_indices with owner_perm."""
        from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import (
            DeepseekV41Attention,
        )

        # Build a mock attention instance for _prepare_tpu & _select_paged_indices
        attn = object.__new__(DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        monkeypatch.setattr(
            DeepseekV41Attention,
            "_all_indices_fit",
            property(lambda self: False),
        )
        attn._tp_rank = 1
        attn._tp_rank_1d = torch.tensor([1], dtype=torch.int32)
        attn.window_size = 64
        attn.n_local_heads = 2
        attn.head_dim = 128
        attn.scale = 0.125
        attn.attn_sink = torch.zeros((2,), dtype=torch.float32)
        attn._full_attn_sink = torch.zeros((4,), dtype=torch.float32)
        attn._ced_decoder = False
        attn._ced_global_from_encoder = True
        attn.compressor = None
        attn._local_q_attn = False
        attn._fp8_kv = True
        attn._precompact_records = True
        attn._token_parallel_attn = True
        attn._swa_max_requests = None
        attn._max_model_len = 256
        attn.compress_ratio = 2
        attn.compressed_cache_prefix = "main"
        attn.candidate_block_size = 1
        attn._config = {"index_topk": 4}
        attn.topk_indices_buffer = torch.full((8, 4), -1, dtype=torch.int32)
        attn.candidate_block_buffer = torch.full((8, 2), -1, dtype=torch.int32)
        attn.wq_b = lambda qr: qr.new_zeros((qr.shape[0], 2 * 128))
        attn._rotate = lambda x, pos: x
        attn._project_index_q = lambda qr, pos: qr.new_zeros((qr.shape[0], 2, 64))

        raw_swa = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        swa_cache = PagedCacheRows(
            raw_swa, ((1, 64, 128), torch.bfloat16, 64 * 128 * 2, 0, 0, True)
        )
        raw_main = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        rope_main = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
        main_cache = PagedCacheRows(
            raw_main,
            ((1, 64, 512), torch.uint8, 64 * 512, 0, 0, True),
            rope=rope_main,
        )
        raw_idx = torch.zeros((16, ROW_WORDS), dtype=torch.int32)
        idx_cache = PagedCacheRows(
            raw_idx, ((1, 64, 64), torch.bfloat16, 64 * 64 * 2, 0, 0, True)
        )

        attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=swa_cache)
        attn._static_forward_context = {"main": SimpleNamespace(kv_cache=main_cache)}

        # Metadata with owner_token_layout on source_meta ("main")
        # T=3 batch tokens across tp=2 -> owner_perm of length 4 (2 per rank)
        owner_perm = torch.tensor([2, 0, -1, 1], dtype=torch.int32)
        owner_inv = torch.tensor([1, 3, 0], dtype=torch.int32)
        owner_q_loc = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.int32)
        owner_token_layout = SimpleNamespace(
            perm=owner_perm,
            inv=owner_inv,
            query_start_loc=owner_q_loc,
        )
        swa_meta = SimpleNamespace(
            block_tables=torch.zeros((2, 1), dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([16, 16], dtype=torch.int32),
        )
        main_meta = SimpleNamespace(
            block_tables=torch.zeros((2, 1), dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
            seq_lens=torch.tensor([8, 8], dtype=torch.int32),
            owner_token_layout=owner_token_layout,
        )
        idx_meta = SimpleNamespace(
            block_tables=torch.zeros((2, 1), dtype=torch.int32),
            seq_lens=torch.tensor([8, 8], dtype=torch.int32),
        )
        metadata = {"swa": swa_meta, "main": main_meta, "idx": idx_meta}

        monkeypatch.setattr(
            v2_attn,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=metadata, additional_kwargs={}),
        )
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 2)

        # Fail if _gather_query_rows is ever called when owner_perm is present!
        monkeypatch.setattr(
            v2_attn,
            "_gather_query_rows",
            lambda x: (_ for _ in ()).throw(
                AssertionError("_gather_query_rows must not be called with owner_perm")
            ),
        )

        def fake_paged_indexer(q_part, ic, rec, w_p, l_p, c_p, wr, wk, rq_p, **kw):
            assert q_part.shape[0] == 2
            sel = torch.tensor([[0, 1, 2, 3], [0, 1, -1, -1]], dtype=torch.int32)
            pub = torch.tensor([[0, 1], [0, -1]], dtype=torch.int32)
            return ic.raw, sel, pub

        monkeypatch.setattr(v2_attn, "paged_indexer", fake_paged_indexer)

        captured_call: dict[str, Any] = {}

        def fake_sparse_fp8(q, sc, mc, sr, mr, snk, scl, pgs, sts, *a, **kw):
            captured_call["mr"] = mr
            captured_call["sts"] = sts
            captured_call["kw"] = kw
            return torch.ones_like(q)

        monkeypatch.setattr(v2_attn, "sparse_fp8_attention", fake_sparse_fp8)

        # Case 1: Indexer layer with candidate_write=True
        attn.indexer = SimpleNamespace(
            head_dim=64,
            n_head=2,
            k_cache=SimpleNamespace(prefix="idx", kv_cache=idx_cache),
            indexer_op=SimpleNamespace(candidate_blocks=2, candidate_write=True),
        )
        positions = torch.tensor([5, 6, 7], dtype=torch.long)
        qr = torch.zeros((3, 64), dtype=torch.bfloat16)
        kv = torch.zeros((3, 128), dtype=torch.bfloat16)
        weights = torch.ones((3, 2), dtype=torch.bfloat16)
        out = torch.empty((3, 2, 128), dtype=torch.bfloat16)
        attn._prepare_tpu(None, qr, kv, None, None, weights, positions, out)

        assert captured_call["kw"]["owner_perm"] is owner_perm
        assert captured_call["kw"]["owner_inv"] is owner_inv
        assert captured_call["kw"]["main_records_segmented"] is True
        assert captured_call["sts"] is owner_q_loc
        # Rank 1 segment is [-1, 1]: row 0 is invalid (-1), row 1 is token 1
        assert torch.all(captured_call["mr"][0] == -1)
        assert torch.equal(
            captured_call["mr"][1], torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        )
        assert torch.equal(
            attn.candidate_block_buffer[:2],
            torch.tensor([[0, 1], [0, -1]], dtype=torch.int32),
        )

        # Case 1b: Indexer layer with candidate_blocks=None + fused_qo=True
        attn.indexer.indexer_op.candidate_blocks = None
        attn._local_q_attn = True
        attn._replicated_local_q = True
        object.__setattr__(
            attn,
            "wq_b",
            SimpleNamespace(
                weight=torch.zeros((64, 64)), weight_scale=torch.zeros((64,))
            ),
        )
        attn._wq_b_full = torch.zeros((64, 128))
        attn._wq_b_full_scale = torch.zeros((128,))
        object.__setattr__(
            attn,
            "wo_a",
            SimpleNamespace(
                weight=torch.zeros((64, 128)), weight_scale=torch.zeros((128,))
            ),
        )
        attn._wo_a_full = torch.zeros((64, 128))
        attn._wo_a_full_scale = torch.zeros((128,))
        attn._tpu_rope_cache = torch.zeros((16, 64))
        captured_qo: dict[str, Any] = {}

        def fake_sparse_qo(*a, **kw):
            captured_qo["kw"] = kw
            return torch.ones((3, 128), dtype=torch.bfloat16)

        monkeypatch.setattr(v2_attn, "sparse_fp8_attention_qo", fake_sparse_qo)
        proj = attn._prepare_tpu(
            None, qr, kv, None, None, weights, positions, out, qr_local=qr
        )
        assert proj is not None and proj.shape == (3, 128)
        assert captured_qo["kw"]["owner_perm"] is owner_perm
        attn._local_q_attn = False
        object.__setattr__(
            attn, "wq_b", lambda q_in: q_in.new_zeros((q_in.shape[0], 2 * 128))
        )

        # Case 2: Shared-indexer layer + BF16 tiled path (self._fp8_kv = False)
        attn.indexer = None
        attn._fp8_kv = False
        attn._static_forward_context = {"main": SimpleNamespace(kv_cache=swa_cache)}
        captured_bf16: dict[str, Any] = {}

        def fake_paged_attn(q, sc, mc, sr, mr, snk, scl, blk, **kw):
            captured_bf16["mr"] = mr
            captured_bf16["kw"] = kw
            return torch.ones_like(q)

        monkeypatch.setattr(v2_attn, "paged_attention", fake_paged_attn)
        attn._prepare_tpu(None, qr, kv, None, None, None, positions, out)
        assert captured_bf16["kw"]["token_parallel"] is True
        assert captured_bf16["kw"]["main_records_segmented"] is True
        assert captured_bf16["mr"].shape == (2, 4)

        # Case 3: Non-tiled with compress_ratio > 0 & KV_SHARD=True raises error
        attn.window_size = 32
        monkeypatch.setitem(vars(envs), "TPU_DEEPSEEK_V41_KV_SHARD", True)
        with pytest.raises(
            NotImplementedError,
            match="FP8 V4.1 KV cache requires the tiled path",
        ):
            attn._prepare_tpu(None, qr, kv, None, None, None, positions, out)
