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

"""100% statement and branch coverage tests for TD7.64 modules:
- vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_attention_op
- vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer
- vllm_torchtpu.models.vllm.deepseek_v41.v1.engram
- vllm_torchtpu.models.vllm.deepseek_v41.v2.async_fence
- vllm_torchtpu.models.vllm.deepseek_v41.v2.engram
- vllm_torchtpu.models.vllm.vllm_model_wrapper_context
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import jax.numpy as jnp
import pytest
import torch
import torch.nn as nn
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_attention_op as attn_op_mod,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_indexer as idx_mod,
)
from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as vmwc_mod
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import engram as v1_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import async_fence as v2_fence
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import engram as v2_engram


class TestVllmModelWrapperContext:
    def test_context_get_and_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(vmwc_mod, "_vllm_model_wrapper_context", None)
        with pytest.raises(AssertionError, match="VllmModelWrapperContext is not set"):
            vmwc_mod.get_vllm_model_wrapper_context()

        fake_mesh = MagicMock()
        bundle = torch.zeros(2)
        with vmwc_mod.set_vllm_model_wrapper_context(
            mesh=fake_mesh,
            kv_caches=[],
            layer_name_to_kvcache_index={"l0": 0},
            vllm_config="cfg",
            kv_cache_bundle=bundle,
        ):
            ctx = vmwc_mod.get_vllm_model_wrapper_context()
            assert ctx.mesh is fake_mesh
            assert ctx.kv_caches == []
            assert ctx.layer_name_to_kvcache_index == {"l0": 0}
            assert ctx.vllm_config == "cfg"
            assert ctx.kv_cache_bundle is bundle

        with pytest.raises(AssertionError, match="VllmModelWrapperContext is not set"):
            vmwc_mod.get_vllm_model_wrapper_context()


class TestDeepseekV4AttentionOp:
    def test_run_swa_empty_and_active(self, monkeypatch: pytest.MonkeyPatch):
        q = jnp.ones((4, 2, 8), dtype=jnp.bfloat16)
        new_kv = jnp.ones((4, 8), dtype=jnp.bfloat16)
        empty_sw = jnp.zeros((0, 4, 8), dtype=jnp.bfloat16)
        kv_lens = jnp.array([2], dtype=jnp.int32)
        pages = jnp.array([0], dtype=jnp.int32)
        cu_q = jnp.array([0, 4], dtype=jnp.int32)
        dist = jnp.array([0, 0, 1], dtype=jnp.int32)
        sinks = jnp.zeros((2,), dtype=jnp.float32)

        # 1. sw_cache.shape[0] == 0
        out0, sw0, l0, m0 = attn_op_mod._run_swa(
            q,
            new_kv,
            empty_sw,
            kv_lens,
            pages,
            cu_q,
            dist,
            sinks,
            sm_scale=0.5,
            sliding_window=64,
            logical_page_size=16,
            swa_only=True,
        )
        assert out0.shape == q.shape
        assert sw0.shape == (0, 4, 8)
        assert l0.shape == (4, 2)
        assert m0.shape == (4, 2)

        # 2. sw_cache.shape[0] > 0
        called = {}

        def fake_swa_kernel(**kwargs):
            called.update(kwargs)
            return (
                q * 2,
                kwargs["cache_kv"] + 1,
                jnp.ones((4, 2), dtype=jnp.float32),
                jnp.zeros((4, 2), dtype=jnp.float32),
            )

        monkeypatch.setattr(
            attn_op_mod,
            "mla_sliding_window_ragged_paged_attention",
            fake_swa_kernel,
        )
        sw = jnp.ones((2, 4, 8), dtype=jnp.bfloat16)
        out1, sw1, _l1, _m1 = attn_op_mod._run_swa(
            q,
            new_kv,
            sw,
            kv_lens,
            pages,
            cu_q,
            dist,
            sinks,
            sm_scale=0.5,
            sliding_window=64,
            logical_page_size=16,
            swa_only=False,
            non_causal_block=True,
        )
        assert called["unnormalized_output"] is True
        assert called["non_causal_block"] is True
        assert out1.shape == q.shape
        assert sw1.shape == sw.shape

    def test_attention_hca_branches(self, monkeypatch: pytest.MonkeyPatch):
        q = jnp.ones((4, 2, 8), dtype=jnp.bfloat16)
        new_kv = jnp.ones((4, 8), dtype=jnp.bfloat16)
        sw = jnp.ones((2, 4, 8), dtype=jnp.bfloat16)
        main_kv = jnp.ones((2, 4, 8), dtype=jnp.bfloat16)
        kv_lens = jnp.array([2], dtype=jnp.int32)
        pages = jnp.array([[0, 1]], dtype=jnp.int32)
        cu_q = jnp.array([0, 4], dtype=jnp.int32)
        dist = jnp.array([0, 0, 1], dtype=jnp.int32)
        sinks = jnp.zeros((2,), dtype=jnp.float32)

        # 1. Empty main_cache_kv or (swa_only and empty sw_cache)
        out_empty, sw_empty = attn_op_mod._attention_hca(
            q,
            new_kv,
            jnp.zeros((0, 4, 8), dtype=jnp.bfloat16),
            kv_lens,
            pages,
            cu_q,
            dist,
            main_kv,
            kv_lens,
            kv_lens,
            pages,
            cu_q,
            dist,
            sinks,
            sm_scale=0.5,
            sliding_window=64,
            logical_page_size=16,
            swa_only=True,
            two_caches_same_buffer=False,
        )
        assert out_empty.shape == q.shape
        assert sw_empty.shape == (0, 4, 8)

        monkeypatch.setattr(
            attn_op_mod,
            "_run_swa",
            lambda **kw: (
                q * 3,
                sw + 10,
                jnp.ones((4, 2), dtype=jnp.float32),
                jnp.zeros((4, 2), dtype=jnp.float32),
            ),
        )

        # 2. swa_only=True
        out_swa_only, sw_upd = attn_op_mod._attention_hca(
            q,
            new_kv,
            sw,
            kv_lens,
            pages,
            cu_q,
            dist,
            main_kv,
            kv_lens,
            kv_lens,
            pages,
            cu_q,
            dist,
            sinks,
            sm_scale=0.5,
            sliding_window=64,
            logical_page_size=16,
            swa_only=True,
            two_caches_same_buffer=False,
        )
        assert jnp.array_equal(out_swa_only, q * 3)
        assert jnp.array_equal(sw_upd, sw + 10)

        # 3. swa_only=False with two_caches_same_buffer=True and False
        seen_cache_kv = []

        def fake_mla(**kw):
            seen_cache_kv.append(kw["cache_kv"])
            assert kw["page_indices"].ndim == 1
            return q * 5

        monkeypatch.setattr(attn_op_mod, "mla_ragged_paged_attention", fake_mla)

        for same_buf in (True, False):
            out_hca, _ = attn_op_mod._attention_hca(
                q,
                new_kv,
                sw,
                kv_lens,
                pages,
                cu_q,
                dist,
                main_kv,
                kv_lens,
                kv_lens,
                pages,
                cu_q,
                dist,
                sinks,
                sm_scale=0.5,
                sliding_window=64,
                logical_page_size=16,
                swa_only=False,
                two_caches_same_buffer=same_buf,
            )
            assert jnp.array_equal(out_hca, q * 5)

        assert jnp.array_equal(seen_cache_kv[0], sw + 10)
        assert jnp.array_equal(seen_cache_kv[1], main_kv)

    def test_attention_csa_and_csa_from_swa(self, monkeypatch: pytest.MonkeyPatch):
        q = jnp.ones((4, 2, 8), dtype=jnp.bfloat16)
        new_kv = jnp.ones((4, 8), dtype=jnp.bfloat16)
        sw = jnp.ones((2, 8, 8), dtype=jnp.bfloat16)
        main_kv = jnp.ones((2, 8, 8), dtype=jnp.bfloat16)
        main_rope = jnp.ones((2, 8, 4), dtype=jnp.bfloat16)
        kv_lens = jnp.array([2], dtype=jnp.int32)
        pages = jnp.array([[0, 1]], dtype=jnp.int32)
        cu_q = jnp.array([0, 4], dtype=jnp.int32)
        dist = jnp.array([0, 0, 1], dtype=jnp.int32)
        sinks = jnp.zeros((2,), dtype=jnp.float32)
        topk = jnp.array(
            [[5, -1, 2, 9], [1, 0, -1, 3], [0, 1, 2, 3], [-1, -1, 4, 1]],
            dtype=jnp.int32,
        )

        # 1. Empty main_cache_kv or main_cache_rope
        out_empty, sw_empty = attn_op_mod._attention_csa(
            q,
            new_kv,
            sw,
            kv_lens,
            pages,
            cu_q,
            dist,
            main_kv,
            kv_lens,
            topk,
            pages,
            cu_q,
            dist,
            sinks,
            jnp.zeros((0, 8, 4), dtype=jnp.bfloat16),
            sm_scale=0.5,
            sliding_window=64,
            logical_page_size=16,
            swa_only=False,
            two_caches_same_buffer=False,
        )
        assert out_empty.shape == q.shape
        assert sw_empty is sw

        monkeypatch.setattr(
            attn_op_mod,
            "_run_swa",
            lambda **kw: (
                q * 2,
                sw + 7,
                jnp.ones((4, 2), dtype=jnp.float32),
                jnp.zeros((4, 2), dtype=jnp.float32),
            ),
        )

        seen_sparse_calls = []

        def fake_sparse_mla(**kw):
            seen_sparse_calls.append(kw)
            return q * 9

        monkeypatch.setattr(
            attn_op_mod, "sparse_ragged_paged_attention", fake_sparse_mla
        )

        # 2. two_caches_same_buffer=True with logical_main_page_size=None
        out_csa1, sw_csa1 = attn_op_mod._attention_csa(
            q,
            new_kv,
            sw,
            kv_lens,
            pages,
            cu_q,
            dist,
            main_kv,
            kv_lens,
            topk,
            pages,
            cu_q,
            dist,
            sinks,
            main_rope,
            sm_scale=0.5,
            sliding_window=64,
            logical_page_size=16,
            swa_only=False,
            two_caches_same_buffer=True,
            logical_main_page_size=None,
        )
        assert jnp.array_equal(out_csa1, q * 9)
        assert jnp.array_equal(sw_csa1, sw + 7)
        assert jnp.array_equal(seen_sparse_calls[0]["cache_kv_nope"], sw + 7)
        assert seen_sparse_calls[0]["page_indices"].ndim == 1
        assert seen_sparse_calls[0]["topk_indices"].shape == (4, 4)

        # 3. two_caches_same_buffer=False with logical_main_page_size=4
        out_csa2, _ = attn_op_mod._attention_csa(
            q,
            new_kv,
            sw,
            kv_lens,
            pages,
            cu_q,
            dist,
            main_kv,
            kv_lens,
            topk,
            pages,
            cu_q,
            dist,
            sinks,
            main_rope,
            sm_scale=0.5,
            sliding_window=64,
            logical_page_size=16,
            swa_only=False,
            two_caches_same_buffer=False,
            logical_main_page_size=4,
        )
        assert jnp.array_equal(out_csa2, q * 9)
        assert jnp.array_equal(seen_sparse_calls[1]["cache_kv_nope"], main_kv)
        remapped_topk = seen_sparse_calls[1]["topk_indices"]
        assert remapped_topk.shape == (4, 128)
        # Valid entries compacted to front; row 0 had [5, -1, 2, 9] -> valid [5, 2, 9]
        # logical_main_page_size=4, main_kv.shape[1]=8 -> [9, 2, 17, -1]
        assert list(remapped_topk[0, :4]) == [9, 2, 17, -1]

    def test_head_size_backend_and_swa_cache(self, monkeypatch: pytest.MonkeyPatch):
        cfg = SimpleNamespace(kv_lora_rank=448, qk_rope_head_dim=64)
        assert attn_op_mod.get_packed_mla_head_size(cfg) == 640
        assert (
            attn_op_mod.VllmDeepseekSparseSWABackend.get_min_page_size(
                SimpleNamespace()
            )
            == 1
        )

        # _swa_block_size with window_size=None and window_size=16
        assert attn_op_mod.VllmDeepseekV4SWACache._swa_block_size(256, None) == 32
        assert attn_op_mod.VllmDeepseekV4SWACache._swa_block_size(256, 16) == 16

        # VllmDeepseekV4SWACache.__init__ and get_kv_cache_spec
        monkeypatch.setattr(
            attn_op_mod.DeepseekV4SWACache,
            "__init__",
            lambda self, hd, ws, dt, pfx, cc, backend_cls=None: setattr(
                self, "window_size", ws
            )
            or setattr(self, "cache_config", cc),
        )
        cache_cfg = SimpleNamespace(block_size=256, cache_dtype="auto")
        swa_cache = attn_op_mod.VllmDeepseekV4SWACache(
            head_dim=512,
            window_size=64,
            dtype=torch.bfloat16,
            prefix="model.layers.0.swa",
            cache_config=cache_cfg,
        )
        assert swa_cache.block_size == 32
        vllm_cfg = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=512, cache_dtype="auto")
        )
        spec = swa_cache.get_kv_cache_spec(vllm_cfg)
        assert isinstance(spec, SlidingWindowMLASpec)
        assert swa_cache.block_size == 64
        assert spec.block_size == 64


class TestDeepseekV4Indexer:
    def test_indexer_jax_empty_and_active(self, monkeypatch: pytest.MonkeyPatch):
        q = jnp.ones((4, 2, 8), dtype=jnp.bfloat16)
        pos = jnp.arange(4, dtype=jnp.int32)
        cos_sin = jnp.ones((16, 8), dtype=jnp.float32)
        idx_w = jnp.ones((4, 2), dtype=jnp.float32)
        seq_lens = jnp.array([4], dtype=jnp.int32)
        pages = jnp.array([[0, 1]], dtype=jnp.int32)
        cu_q = jnp.array([0, 4], dtype=jnp.int32)
        dist = jnp.array([0, 0, 1], dtype=jnp.int32)

        # 1. Empty cache_kv
        out_empty = idx_mod._indexer_jax(
            q,
            pos,
            cos_sin,
            idx_w,
            jnp.zeros((0, 4, 128), dtype=jnp.uint8),
            seq_lens,
            pages,
            cu_q,
            dist,
            k=8,
            compression_ratio=4,
            softmax_scale=0.5,
            n_head=4,
        )
        assert out_empty.shape == (4, 8)
        assert out_empty.dtype == jnp.int32

        # 2. Active cache_kv
        monkeypatch.setattr(
            idx_mod,
            "rope_quant",
            lambda q_in, p_in, cs_in, quant_dtype: (
                q_in,
                jnp.full((4, 2), 2.0, dtype=jnp.float32),
            ),
        )
        seen_topk = {}

        def fake_streamindex_topk(**kw):
            seen_topk.update(kw)
            return jnp.ones((4, kw["k"]), dtype=jnp.int32)

        monkeypatch.setattr(idx_mod, "streamindex_topk", fake_streamindex_topk)

        out_act = idx_mod._indexer_jax(
            q,
            pos,
            cos_sin,
            idx_w,
            jnp.ones((2, 4, 128), dtype=jnp.uint8),
            seq_lens,
            pages,
            cu_q,
            dist,
            k=8,
            compression_ratio=4,
            softmax_scale=0.5,
            n_head=4,
        )
        assert out_act.shape == (4, 8)
        # combined_scale = 0.5 * (4 ** -0.5) = 0.25; * q_scales (2.0) = 0.5
        assert jnp.allclose(seen_topk["indexer_weights"], jnp.full((4, 2), 0.5))
        assert seen_topk["page_indices"].ndim == 1

    def test_indexer_cache_and_vllm_indexer_init_and_forward(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # 1. VllmDeepseekV4IndexerCache.get_kv_cache_spec
        cache = idx_mod.VllmDeepseekV4IndexerCache.__new__(
            idx_mod.VllmDeepseekV4IndexerCache
        )
        cache.cache_config = SimpleNamespace(block_size=64)
        cache.compress_ratio = 4
        spec = cache.get_kv_cache_spec(SimpleNamespace())
        assert isinstance(spec, MLAAttentionSpec)
        assert spec.block_size == 64

        # 2. VllmDeepseekV4Indexer.__init__ and _build_indexer_op
        def fake_super_init(self, *args, **kwargs):
            nn.Module.__init__(self)
            self.topk_tokens = 8
            self.compress_ratio = 4
            self.softmax_scale = 0.25
            self.n_head = 2
            self.head_dim = 4
            self.wq_b = lambda q_in: (torch.ones((q_in.shape[0], 8)), None)
            self.compressor = MagicMock()
            self.k_cache = SimpleNamespace(
                prefix="model.layers.0.indexer",
                kv_cache=torch.ones((2, 4), dtype=torch.uint8),
            )

        monkeypatch.setattr(idx_mod.DeepseekV4Indexer, "__init__", fake_super_init)

        fake_registered = []

        class FakeJaxOp:
            def __init__(self, name, fn, **kw):
                self.name = name
                self.fn = fn

            def register_fake(self, fake_fn):
                fake_registered.append(fake_fn)

            def __call__(self, q_in, *args):
                return torch.full((q_in.shape[0], 8), 7, dtype=torch.int32)

        monkeypatch.setattr(
            idx_mod.pallas, "jax_op", lambda name, fn, **kw: FakeJaxOp(name, fn, **kw)
        )
        idx_mod._indexer_op_cache.clear()

        with vmwc_mod.set_vllm_model_wrapper_context(mesh=MagicMock()):
            indexer1 = idx_mod.VllmDeepseekV4Indexer()
            # Second construction hits _indexer_op_cache
            indexer2 = idx_mod.VllmDeepseekV4Indexer()

        assert indexer1.indexer_op is indexer2.indexer_op
        assert indexer1._prefix_key == "model.layers.0.indexer"
        assert len(fake_registered) == 1
        fake_out = fake_registered[0](torch.zeros((3, 2, 4)))
        assert fake_out.shape == (3, 8)

        # 3. forward: empty kv_cache
        hidden = torch.zeros((3, 16))
        query = torch.zeros((3, 8))
        idx_w = torch.ones((3, 2))
        pos = torch.arange(3, dtype=torch.int32)
        rotary = SimpleNamespace(cos_sin_cache=torch.ones((16, 4)))

        indexer1.k_cache.kv_cache = torch.empty(0, dtype=torch.uint8)
        out_empty = indexer1.forward(hidden, query, idx_w, pos, rotary)
        assert out_empty.shape == (3, 8)
        assert torch.all(out_empty == 0)

        # Restore non-empty kv_cache
        indexer1.k_cache.kv_cache = torch.ones((2, 4), dtype=torch.uint8)
        meta = SimpleNamespace(
            seq_lens=torch.tensor([3], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
        )

        # 4. forward: dict attn_ctx missing prefix_key -> KeyError
        monkeypatch.setattr(
            idx_mod,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"other": meta}),
        )
        with pytest.raises(KeyError, match="has no attention metadata"):
            indexer1.forward(hidden, query, idx_w, pos, rotary)

        # 5. forward: dict attn_ctx with matching prefix_key
        monkeypatch.setattr(
            idx_mod,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"model.layers.0.indexer": meta}),
        )
        out_dict = indexer1.forward(hidden, query, idx_w, pos, rotary)
        assert torch.all(out_dict == 7)

        # 6. forward: non-dict attn_ctx (direct AttentionMetadata)
        monkeypatch.setattr(
            idx_mod,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=meta),
        )
        out_direct = indexer1.forward(hidden, query, idx_w, pos, rotary)
        assert torch.all(out_direct == 7)


class TestV1V2EngramAndAsyncFence:
    def test_v1_and_v2_engram_write_and_ensure_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # 1. v1_engram.NgramHashState.write_cache & v2_engram.NgramHashState.write_cache
        v1_calls = []
        v2_calls = []
        monkeypatch.setattr(v1_engram, "write_cache", lambda *a: v1_calls.append(a))
        monkeypatch.setattr(v2_engram, "write_cache", lambda *a: v2_calls.append(a))

        s1 = object.__new__(v1_engram.NgramHashState)
        s2 = object.__new__(v2_engram.NgramHashState)
        t = torch.zeros(2, dtype=torch.int32)
        s1.write_cache(t, t, t, t)
        s2.write_cache(t, t, t, t)
        assert len(v1_calls) == 1
        assert len(v2_calls) == 1

        # 2. v1_engram.NgramHashState.ensure_cache branches
        nn.Module.__init__(s1)
        s1.use_slot_cache = True
        s1.block_size = 8
        s1._cache = None
        s1._kv_cache_ref = None
        bound = torch.ones((4, 8), dtype=torch.bfloat16)
        s1.swa_cache_module = SimpleNamespace(
            bound_cache=bound,
            kv_cache=bound,
            block_size=8,
        )

        # 2a. is_compiling() == True
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        assert s1.ensure_cache() is False

        # 2b. is_compiling() == False, empty bound_cache
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        s1.swa_cache_module.bound_cache = torch.empty(0)
        assert s1.ensure_cache() is False
        assert s1._cache is None
        assert s1._kv_cache_ref is None

        # 2c. use_slot_cache == False
        s1.swa_cache_module.bound_cache = bound
        s1.use_slot_cache = False
        assert s1.ensure_cache() is True

        # 2d. use_slot_cache == True, block_size mismatch -> allocates _cache
        s1.use_slot_cache = True
        s1.swa_cache_module.block_size = 16
        assert s1.ensure_cache() is True
        assert s1.block_size == 16
        assert s1._cache is not None and s1._cache.shape == (64,)

        # 2e. Repeated call hits weakref fast path
        cached_buf = s1._cache
        assert s1.ensure_cache() is True
        assert s1._cache is cached_buf

    def test_async_fence_install_and_run(self, monkeypatch: pytest.MonkeyPatch):
        from vllm_torchtpu.runner import tpu_runner

        calls = []

        def orig_prep(self, num_reqs: int):
            calls.append(num_reqs)
            return torch.tensor([num_reqs])

        monkeypatch.setattr(
            tpu_runner.TPUModelRunner,
            "_prepare_lookback_token_ids",
            orig_prep,
            raising=False,
        )
        v2_fence.install_lookback_fence()
        wrapped = tpu_runner.TPUModelRunner._prepare_lookback_token_ids
        v2_fence.install_lookback_fence()
        assert tpu_runner.TPUModelRunner._prepare_lookback_token_ids is wrapped

        ev1 = MagicMock()
        ev2 = MagicMock()
        events = iter([ev1, ev2])
        monkeypatch.setattr(
            torch, "tpu", SimpleNamespace(Event=lambda: next(events)), raising=False
        )
        runner = SimpleNamespace()
        assert torch.equal(wrapped(runner, 3), torch.tensor([3]))
        ev1.synchronize.assert_not_called()
        ev1.record.assert_called_once()

        assert torch.equal(wrapped(runner, 5), torch.tensor([5]))
        ev1.synchronize.assert_called_once()
        ev2.record.assert_called_once()
