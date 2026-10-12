# SPDX-License-Identifier: Apache-2.0
"""Complete line and branch coverage test suite for DeepSeek-V4.1 v2 attention."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

import torch
import torch._dynamo

if importlib.util.find_spec("vllm") is None:
    if "tpu" not in torch._dynamo.list_backends():
        torch._dynamo.register_backend(
            name="tpu", compiler_fn=lambda gm, *a, **kw: gm.forward
        )

    class _StubModule(types.ModuleType):
        def __init__(self, name: str):
            super().__init__(name)
            self.__path__ = []  # type: ignore[attr-defined]
            self.__file__ = f"<stub:{name}>"
            self.__qualname__ = name
            self.__globals__ = {}
            self.__code__ = (lambda *a, **kw: None).__code__
            self.__defaults__ = None
            self.__kwdefaults__ = None
            self.__closure__ = None

        def __mro_entries__(self, bases):
            if not any(
                isinstance(b, type) and issubclass(b, torch.nn.Module) for b in bases
            ):
                return (type(self.__name__.rsplit(".", 1)[-1], (torch.nn.Module,), {}),)
            return (type(self.__name__.rsplit(".", 1)[-1], (), {}),)

        def __or__(self, other):
            return object

        def __ror__(self, other):
            return object

        def __getattr__(self, attr: str):
            if attr.startswith("__"):
                raise AttributeError(attr)
            full = f"{self.__name__}.{attr}"
            if full in sys.modules:
                return sys.modules[full]
            sub = _StubModule(full)
            sys.modules[full] = sub
            setattr(self, attr, sub)
            return sub

        def __call__(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return type(
                "_CtxStub",
                (SimpleNamespace,),
                {
                    "__enter__": lambda s: s,
                    "__exit__": lambda s, *a: False,
                    "__call__": lambda s, *a, **kw: (
                        a[0] if len(a) == 1 and callable(a[0]) else s
                    ),
                },
            )()

    class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        PREFIXES = (
            "portpicker",
            "absl",
            "requests",
            "vllm",
            "torch_tpu",
            "tpu_info",
            "vllm_torchtpu.layers.adapter",
        )

        def find_spec(self, fullname, path, target=None):
            if any(
                fullname == p or fullname.startswith(p + ".") for p in self.PREFIXES
            ):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            if spec.name in sys.modules:
                return sys.modules[spec.name]
            mod = _StubModule(spec.name)
            sys.modules[spec.name] = mod
            if "." in spec.name:
                parent_name, attr_name = spec.name.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, attr_name, mod)
            return mod

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _StubFinder())
    import vllm.config
    import vllm.distributed
    import vllm.forward_context
    import vllm.logger
    import vllm.models.common.ops.sequence_parallel

    def _init_logger(name: str):
        log = logging.getLogger(name)
        log.warning_once = log.warning  # type: ignore[attr-defined]
        log.info_once = log.info  # type: ignore[attr-defined]
        return log

    vllm.logger.init_logger = _init_logger
    vllm.config.get_current_vllm_config = lambda: SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(index_head_dim=64))
    )
    vllm.config.get_current_vllm_config_or_none = lambda: None
    vllm.distributed.get_tensor_model_parallel_rank = lambda: 0
    vllm.distributed.get_tensor_model_parallel_world_size = lambda: 1
    vllm.distributed.tensor_model_parallel_all_gather = lambda x, dim=0: x
    vllm.forward_context.get_forward_context = lambda: SimpleNamespace(
        attn_metadata=None, additional_kwargs={}
    )
    vllm.models.common.ops.sequence_parallel.sp_all_gather = lambda x: x
    import vllm_torchtpu.layers.adapter.linear_common as lin_common

    lin_common.WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"
    lin_common.KEEP_VLLM_LAYOUT_ATTR = "_keep_vllm_layout"

import pytest

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


def _make_paged_mock(
    *,
    rows_per_page: int = 8,
    page_stride: int = 8,
    row_offset: int | None = 0,
    has_row_offset: bool = True,
    shape_is_tuple: bool = True,
    pages_tensor: torch.Tensor | None = None,
) -> MagicMock:
    m = MagicMock(spec=PagedCacheRows)
    m.numel.return_value = 1024
    m.shape = (16, rows_per_page, 8) if shape_is_tuple else None
    m.page_stride = page_stride
    m.dtype = torch.bfloat16 if has_row_offset else torch.float32
    m.row_bytes = 256
    m.raw = torch.zeros((32, 128), dtype=torch.int32)
    m.raw_hi = None
    m.row_offset = row_offset if has_row_offset else None
    m.row_indices = lambda slots, ro=(row_offset or 0): (
        ro + (slots // rows_per_page) * page_stride + (slots % rows_per_page)
    )
    m.pages = lambda bt: pages_tensor
    return m


def _base_attn(
    *, fp8_kv: bool = True, compress_ratio: int = 2
) -> v2_attn.DeepseekV41Attention:
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "model.layers.2.self_attn.attn"
    attn.compressed_cache_prefix = "model.layers.2.self_attn.attn"
    attn.window_size = 64
    attn._ced_decoder = False
    attn._ced_global_from_encoder = False
    attn._sp_sharded_input = False
    attn._local_q_attn = False
    attn._local_wo_b = False
    attn._replicated_local_q = False
    attn._fp8_kv = fp8_kv
    attn.compress_ratio = compress_ratio
    attn._config = {"index_topk": 4, "candidate_topk_blocks": 2}
    attn._max_model_len = 32
    attn._swa_max_requests = 4
    attn._precompact_records = fp8_kv
    attn._token_parallel_attn = fp8_kv
    attn._indexer_weight_scale = 0.25
    attn.n_local_heads = 2
    attn.padded_heads = 4
    attn.head_dim = 8
    attn.q_lora_rank = 4
    attn.n_local_groups = 1
    attn.candidate_block_size = 2
    attn.scale = 0.25
    attn.attn_sink = torch.zeros(2, dtype=torch.float32)
    attn._full_attn_sink = torch.zeros(2, dtype=torch.float32)
    attn._tp_rank = torch.tensor(0, dtype=torch.int32)
    attn._tp_rank_1d = torch.tensor([0], dtype=torch.int32)
    attn._tpu_rope_cache = torch.zeros((64, 4), dtype=torch.float32)
    attn.rotary_emb = SimpleNamespace(cos_sin_cache=attn._tpu_rope_cache, base=10000.0)
    attn.topk_indices_buffer = torch.full((16, 8), -1, dtype=torch.int32)
    attn.candidate_block_buffer = torch.full((16, 4), -1, dtype=torch.int32)
    attn.compressor = None
    attn.indexer = None
    attn._rotate = lambda x, pos, inverse=False: x

    class _Lin(torch.nn.Module):
        def __init__(self, out_dim: int, tuple_out: bool = False):
            super().__init__()
            self.out_dim = out_dim
            self.tuple_out = tuple_out
            self.weight = torch.zeros((out_dim, 8), dtype=torch.float8_e4m3fn)
            self.weight_scale = torch.ones(out_dim, dtype=torch.float32)

        def forward(self, x):
            res = x.new_zeros((x.shape[0], self.out_dim))
            return (res, None) if self.tuple_out else res

    attn.wq_b = _Lin(16, tuple_out=False)
    attn.q_norm = lambda x: x
    attn.kv_norm = lambda x: x
    attn._fused_wqa_wkv_gemm = lambda h: torch.zeros(
        (h.shape[0], 12), dtype=h.dtype, device=h.device
    )
    attn._o_proj = lambda o, pos: o.reshape(o.shape[0], -1)
    attn.wo_b = _Lin(4, tuple_out=True)
    attn.wo_a = SimpleNamespace(
        weight=torch.zeros((16, 8), dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones(8, dtype=torch.float32),
    )
    attn._prepare_and_attn_fn = attn._prepare_tpu
    return attn


class TestStandaloneHelpers:
    def test_candidate_blocks_and_compressed_indices(self):
        with pytest.raises(ValueError, match="invalid candidate block"):
            v2_attn.select_candidate_blocks(torch.zeros((2, 4)), [2, 2], 1, 0)
        with pytest.raises(ValueError, match="invalid candidate block"):
            v2_attn.select_candidate_blocks(torch.zeros((2, 4)), [2, 2], -1, 2)
        empty = v2_attn.select_candidate_blocks(torch.zeros((2, 0)), [[0], [0]], 1, 2)
        assert empty.shape == (2, 0) and empty.dtype == torch.bool

        logits = torch.tensor([[1.0, 2.0, 5.0, 0.5, 3.0]], dtype=torch.float32)
        kept = v2_attn.select_candidate_blocks(logits, [[5]], 2, 2)
        assert kept.shape == (1, 5)
        assert kept[0, 4].item() is True

        with pytest.raises(ValueError, match="nonnegative"):
            v2_attn.select_compressed_indices(torch.zeros((2, 4)), [[2], [2]], -1)

        scores = torch.tensor(
            [[3.0, -torch.inf, 9.0, 1.0], [2.0, 4.0, 8.0, 6.0]], dtype=torch.float32
        )
        lengths = torch.tensor([[3], [4]], dtype=torch.int32)
        cand = torch.tensor(
            [[True, True, True, False], [True, False, True, True]], dtype=torch.bool
        )
        sel = v2_attn.select_compressed_indices(scores, lengths, 3, candidates=cand)
        assert sel.tolist() == [[0, 2, -1], [0, 2, 3]]

    def test_sparse_attention_all_dtypes_and_blocks(self):
        q_empty = torch.randn((1, 2, 2, 4), dtype=torch.bfloat16)
        kv_empty = torch.randn((1, 0, 4), dtype=torch.bfloat16)
        sink = torch.zeros(2, dtype=torch.float32)
        idx_empty = torch.zeros((1, 2, 0), dtype=torch.int32)
        assert torch.equal(
            v2_attn.sparse_attention(q_empty, kv_empty, sink, idx_empty, 0.5),
            torch.zeros_like(q_empty),
        )

        # BF16 online 64-block loop (>64 indices to cover multi-iteration branch)
        q_bf16 = torch.randn((1, 2, 2, 4), dtype=torch.bfloat16)
        kv_bf16 = torch.randn((1, 70, 4), dtype=torch.bfloat16)
        idx_70 = torch.arange(70, dtype=torch.int32).expand(1, 2, 70).clone()
        idx_70[..., -1] = -1
        out_bf16 = v2_attn.sparse_attention(q_bf16, kv_bf16, sink, idx_70, 0.5)
        assert out_bf16.shape == q_bf16.shape and out_bf16.dtype == torch.bfloat16

        # FP32 standard softmax branch
        out_fp32 = v2_attn.sparse_attention(
            q_bf16.float(), kv_bf16.float(), sink, idx_70, 0.5
        )
        assert out_fp32.shape == q_bf16.shape and out_fp32.dtype == torch.float32

    def test_addressing_and_cache_slot_helpers(self, monkeypatch):
        meta_empty = SimpleNamespace(
            seq_lens=torch.zeros((0,), dtype=torch.int32),
            block_tables=torch.zeros((0, 2), dtype=torch.int32),
            query_start_loc=torch.tensor([0], dtype=torch.int32),
        )
        pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        assert torch.all(
            v2_attn._cache_slots(torch.empty((2, 4)), meta_empty, pos) == -1
        )

        meta = SimpleNamespace(
            seq_lens=torch.tensor([5, 6], dtype=torch.int32),
            block_tables=torch.tensor([[2, 3], [4, 5]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        )
        slots_ring = v2_attn._cache_slots(
            torch.empty((8, 4)), meta, pos, compress_ratio=1, ring=True
        )
        slots_linear = v2_attn._cache_slots(
            torch.empty((8, 4)), meta, pos, compress_ratio=2, ring=False
        )
        assert slots_ring[-1].item() == -1
        assert slots_linear[-1].item() == -1

        starts = torch.tensor([1, 0], dtype=torch.int32)
        rep = v2_attn._replayed(meta, torch.tensor([1, 2, 0, 0]), starts, 2)
        assert rep.tolist() == [True, True, False, False]

        monkeypatch.setattr(
            v2_attn,
            "get_forward_context",
            lambda: SimpleNamespace(
                additional_kwargs={"deepseek_v41_window_starts": starts}
            ),
        )
        assert v2_attn._window_starts() is starts
        with v2_attn.stage_attention_cache_scope():
            assert v2_attn._window_starts() is starts
            assert v2_attn._window_starts() is starts

        # _row_slots exact parity vs 2D table gather for both int32 and int64 inputs
        table = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.int64)
        reqs = torch.tensor([0, 1], dtype=torch.int64)
        logical = torch.tensor([[0, 4, 9], [2, 5, 11]], dtype=torch.int64)
        rs = v2_attn._row_slots(torch.empty((1, 4)), table, reqs, logical)
        assert rs.tolist() == [[40, 80, 121], [162, 201, 243]]

        # _request_row_slots both page_records > 1 and per-record expansion with
        # width > cols
        fast_cache = _make_paged_mock(rows_per_page=4, page_stride=8, row_offset=0)
        page_slots = v2_attn._request_row_slots(
            fast_cache, table, width=20, page_records=4
        )
        assert page_slots.shape == (2, 5)
        page_slots_short = v2_attn._request_row_slots(
            fast_cache, table, width=8, page_records=4
        )
        assert page_slots_short.shape == (2, 2)
        plain_cache = torch.empty((1, 4))
        full_slots = v2_attn._request_row_slots(plain_cache, table, width=20)
        assert full_slots.shape == (2, 20)
        short_slots = v2_attn._request_row_slots(plain_cache, table, width=6)
        assert short_slots.shape == (2, 6)

        monkeypatch.setattr(v2_attn, "write_cache", lambda *a, **kw: None)
        v2_attn._insert(
            plain_cache, pos, torch.zeros((4, 8)), torch.ones(4, dtype=torch.bool)
        )

    def test_index_page_records_and_gather_query_rows(self, monkeypatch):
        assert v2_attn._index_page_records(torch.empty((2, 4))) == 0
        c = _make_paged_mock(rows_per_page=8, page_stride=8, row_offset=0)
        c.dtype = torch.float32
        assert v2_attn._index_page_records(c) == 0
        c.dtype = torch.bfloat16
        c.row_bytes = 300
        assert v2_attn._index_page_records(c) == 0
        c.row_bytes = 256
        c.row_offset = 1
        assert v2_attn._index_page_records(c) == 0
        c.row_offset = 0
        c.raw_hi = torch.zeros((4, 128), dtype=torch.int32)
        c.raw = torch.zeros((3, 128), dtype=torch.int32)
        assert v2_attn._index_page_records(c) == 0
        c.raw = torch.zeros((8, 128), dtype=torch.int32)
        assert v2_attn._index_page_records(c) == 8

        monkeypatch.setattr(
            v2_attn, "tensor_model_parallel_all_gather", lambda x, dim=0: x
        )
        unpadded = torch.ones((2, 128), dtype=torch.int32)
        assert v2_attn._gather_query_rows(unpadded).shape == (2, 128)
        padded = torch.ones((2, 10), dtype=torch.int32)
        assert v2_attn._gather_query_rows(padded).shape == (2, 10)
        assert v2_attn._query_split(5, 8) == (2, 16)


class TestCompressorAndBackend:
    def test_save_compress_norm_and_compressor_forward(self, monkeypatch):
        kv_score = torch.randn((4, 9), dtype=torch.float32)
        pos = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
        state = torch.zeros((4, 2, 9), dtype=torch.float32)
        slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        qsl = torch.tensor([0, 2, 4], dtype=torch.int32)
        reqs = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        norm_w = torch.ones(8, dtype=torch.float32)
        latent = torch.empty((4, 8), dtype=torch.bfloat16)

        with pytest.raises(ValueError, match="ratio must be 1 or 2"):
            v2_attn.save_compress_norm(
                kv_score, pos, state, slots, qsl, reqs, norm_w, 1e-6, 3, latent
            )
        v2_attn.save_compress_norm(
            kv_score, pos, state, None, qsl, None, norm_w, 1e-6, 1, latent
        )
        v2_attn.save_compress_norm(
            kv_score, pos, state, slots, qsl, reqs, norm_w, 1e-6, 2, latent
        )

        # DeepseekCompressor init, project, forward (dict vs None, state_cache None
        # vs stage cached vs fallback)
        def fake_comp_init(self, *a, **kw):
            torch.nn.Module.__init__(self)

        monkeypatch.setattr(
            v2_attn.DeepseekCompressor.__bases__[0], "__init__", fake_comp_init
        )
        comp = v2_attn.DeepseekCompressor()
        comp.head_dim = 8
        comp.rms_norm_eps = 1e-6
        comp.compress_ratio = 2
        comp.norm = SimpleNamespace(weight=norm_w)
        comp.fused_wkv_wgate = SimpleNamespace(weight=torch.randn((9, 8)))
        assert comp.project(torch.randn((4, 8))).shape == (4, 9)
        comp.fused_wkv_wgate._tpu_weight_flipped = True
        comp.fused_wkv_wgate.weight = torch.randn((8, 9))
        assert comp.project(torch.randn((4, 8))).shape == (4, 9)

        monkeypatch.setattr(
            v2_attn,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=None),
        )
        assert comp(kv_score, pos) is None

        meta = {
            "state": SimpleNamespace(
                seq_lens=torch.tensor([2, 2], dtype=torch.int32),
                block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
                query_start_loc=qsl,
            ),
            "k": SimpleNamespace(
                seq_lens=torch.tensor([2, 2], dtype=torch.int32),
                block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
                query_start_loc=qsl,
            ),
        }
        monkeypatch.setattr(
            v2_attn,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=meta, additional_kwargs={}),
        )
        comp.state_cache = None
        comp.compress_ratio = 1
        comp.k_cache_prefix = "k"
        comp._static_forward_context = {
            "k": SimpleNamespace(kv_cache=torch.empty((4, 2, 8)))
        }
        assert comp(kv_score, pos).shape == (4, 8)

        comp.compress_ratio = 2
        comp.state_cache = SimpleNamespace(
            prefix="state", kv_cache=torch.zeros((4, 2, 9), dtype=torch.float32)
        )
        assert comp(kv_score, pos).shape == (4, 8)
        with v2_attn.stage_attention_cache_scope():
            assert comp(kv_score, pos).shape == (4, 8)
            assert comp(kv_score, pos).shape == (4, 8)
            comp.state_cache = None
            comp.compress_ratio = 1
            assert comp(kv_score, pos).shape == (4, 8)

        # Backend, CompressorStateCache, SWACache, IndexerCache, indexer_uses_fp4
        csc = object.__new__(v2_attn.CompressorStateCache)
        assert csc.get_attn_backend() is v2_attn.AttentionBackend
        assert v2_attn.AttentionBackend.get_name() == "TORCH_TPU_DEEPSEEK_V41"
        assert v2_attn.AttentionBackend.get_kv_cache_shape(2, 4, 1, 8) == (2, 1, 4, 8)
        assert (
            v2_attn.indexer_uses_fp4(
                SimpleNamespace(
                    attention_config=SimpleNamespace(
                        resolve_indexer_kv_dtype=lambda _: "fp8"
                    )
                )
            )
            is False
        )
        with pytest.raises(ValueError, match="requires fp8"):
            v2_attn.indexer_uses_fp4(
                SimpleNamespace(
                    attention_config=SimpleNamespace(
                        resolve_indexer_kv_dtype=lambda _: "fp4"
                    )
                )
            )

        monkeypatch.setattr(v2_attn, "prefix_replay_enabled", lambda _: True)
        monkeypatch.setattr(
            v2_attn.SWACache.__bases__[1],
            "__init__",
            lambda self, *a, **kw: setattr(self, "_kw", kw),
        )
        swa_c = v2_attn.SWACache(cache_config=SimpleNamespace(block_size=16))
        assert swa_c._kw["bounded_replay"] is True

        monkeypatch.setattr(v2_attn, "prefix_replay_enabled", lambda _: False)
        swa_c2 = v2_attn.SWACache(cache_config=SimpleNamespace(block_size=16))
        assert "bounded_replay" not in swa_c2._kw

        monkeypatch.setattr(
            v2_attn.IndexerCache.__bases__[1],
            "__init__",
            lambda self, *a, **kw: setattr(self, "_kw", kw),
        )
        monkeypatch.setattr(
            v2_attn,
            "get_current_vllm_config",
            lambda: SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(index_head_dim=64)
                )
            ),
        )
        idx_c = v2_attn.IndexerCache()
        assert idx_c.get_attn_backend() is v2_attn.AttentionBackend


class TestDeepseekV41AttentionFull:
    def test_init_spec_and_weight_loading(self, monkeypatch):
        def fake_super_init(self, vllm_config=None, *a, **kw):
            torch.nn.Module.__init__(self)
            self.kv_cache_torch_dtype = vllm_config.dtype
            self.rotary_emb = SimpleNamespace(
                cos_sin_cache=torch.zeros((16, 4)), base=10000.0
            )
            self.attn_sink = torch.zeros(2)
            self.indexer = SimpleNamespace(head_dim=8, n_head=2)
            self.wq_b = SimpleNamespace(
                weight=torch.zeros((4, 8), dtype=torch.float8_e4m3fn),
                weight_scale=torch.ones(8),
            )
            self.wo_a = SimpleNamespace(
                weight=torch.zeros((8, 4), dtype=torch.float8_e4m3fn),
                weight_scale=torch.ones(4),
            )
            self.wo_b = SimpleNamespace(
                weight=torch.zeros((8, 4), dtype=torch.float8_e4m3fn),
                weight_scale=torch.ones(8),
            )
            self.compress_ratio = 2
            self.n_local_groups = 1

        base_cls = v2_attn.DeepseekV41Attention.__bases__[1]
        monkeypatch.setattr(base_cls, "__init__", fake_super_init)
        cfg_fp8 = SimpleNamespace(
            dtype=torch.float8_e4m3fn,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: {"index_topk": 4}),
                max_model_len=128,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=8),
        )
        attn_fp8 = v2_attn.DeepseekV41Attention(cfg_fp8)
        assert attn_fp8.get_padded_num_q_heads(16) == 16
        assert attn_fp8._swa_max_requests is None

        cfg_bf16 = SimpleNamespace(
            dtype=torch.bfloat16,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: {"index_topk": 4}),
                max_model_len=128,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
        )
        attn_bf16 = v2_attn.DeepseekV41Attention(cfg_bf16)
        assert attn_bf16._swa_max_requests == 2

        cfg_bad = SimpleNamespace(
            dtype=torch.float16,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(to_dict=lambda: {}), max_model_len=128
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
        )
        with pytest.raises(NotImplementedError):
            v2_attn.DeepseekV41Attention(cfg_bad)

        # get_kv_cache_spec branches
        monkeypatch.setattr(
            base_cls,
            "get_kv_cache_spec",
            lambda self, c: None,
            raising=False,
        )
        assert attn_fp8.get_kv_cache_spec(cfg_fp8) is None
        import dataclasses

        @dataclasses.dataclass
        class _Spec:
            dtype: torch.dtype
            page_size_padded: int | None

        monkeypatch.setattr(
            base_cls,
            "get_kv_cache_spec",
            lambda self, c: _Spec(torch.bfloat16, 64),
            raising=False,
        )
        assert attn_bf16.get_kv_cache_spec(cfg_bf16).dtype == torch.bfloat16
        assert attn_fp8.get_kv_cache_spec(cfg_fp8).dtype == torch.uint8

        # _init_local_q_weights all branches + process_weights_after_loading
        monkeypatch.setattr(v2_attn.envs, "TPU_DEEPSEEK_V41_DECODE_LOCAL_Q", True)
        monkeypatch.setattr(v2_attn.envs, "TPU_DEEPSEEK_V41_LOCAL_Q", True)
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)
        attn_fp8._init_local_q_weights()
        assert attn_fp8._local_q_attn is False

        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(
            v2_attn, "tensor_model_parallel_all_gather", lambda x, dim=0: x
        )
        attn_fp8._sp_sharded_input = True
        attn_fp8._init_local_q_weights()
        assert attn_fp8._local_q_attn is True and attn_fp8._local_wo_b is True

        attn_fp8.wo_b = None
        attn_fp8._init_local_q_weights()
        assert attn_fp8._local_wo_b is False

        attn_fp8.wq_b = SimpleNamespace(weight=torch.zeros((4, 8)), weight_scale=None)
        attn_fp8._init_local_q_weights()

        import vllm_torchtpu.models.vllm.deepseek_v4.attention as v4_attn_mod

        fake_o_op = MagicMock(return_value=torch.zeros((2, 4)))
        fake_rope_op = MagicMock(side_effect=lambda x, pos, cache: x)
        monkeypatch.setattr(
            v4_attn_mod.VllmDeepseekV4MLAAttention,
            "process_weights_after_loading",
            lambda self, dt: None,
        )
        monkeypatch.setattr(
            v4_attn_mod.VllmDeepseekV4MLAAttention,
            "_assert_o_proj_supported",
            lambda self: None,
        )
        monkeypatch.setattr(
            v4_attn_mod.VllmDeepseekV4MLAAttention,
            "o_proj_op",
            property(lambda self: fake_o_op),
        )
        monkeypatch.setattr(
            v4_attn_mod.VllmDeepseekV4MLAAttention,
            "kv_rope_op",
            property(lambda self: fake_rope_op),
        )
        attn_fp8.wo_a = SimpleNamespace(
            weight=MagicMock(device=SimpleNamespace(type="tpu")),
            weight_scale=torch.ones(4),
        )
        attn_fp8.process_weights_after_loading(torch.bfloat16)
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)
        attn_fp8.wo_a = SimpleNamespace(
            weight=torch.zeros((8, 4)), weight_scale=torch.ones(4)
        )
        attn_fp8.process_weights_after_loading(torch.bfloat16)

        # kv_rope_op, _rotate (cpu vs non-cpu), _o_proj, forward_mqa
        assert attn_fp8.kv_rope_op is fake_rope_op
        attn_fp8.__dict__["_kv_rope_op_instance"] = fake_rope_op
        assert attn_fp8.kv_rope_op is fake_rope_op
        t_cpu = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
        monkeypatch.setattr(
            v2_attn, "rotary", lambda v, pos, cfg, compressed, inverse: v
        )
        assert attn_fp8._rotate(t_cpu, torch.tensor([0, 1])).shape == (2, 2, 8)
        fake_dev_val = MagicMock()
        fake_dev_val.device.type = "tpu"
        attn_fp8._rotate(fake_dev_val, torch.tensor([0, 1]), inverse=False)
        attn_fp8.n_local_heads = 2
        attn_fp8.head_dim = 4
        attn_fp8.wo_b = lambda x: x
        assert attn_fp8._o_proj(
            torch.zeros((2, 2, 4), dtype=torch.bfloat16), torch.tensor([0, 1])
        ).shape == (2, 4)
        monkeypatch.setattr(v2_attn, "round_fp8_activation", lambda v, d: v)
        assert attn_fp8._quantize_indexer(torch.zeros((2, 8))).shape == (2, 8)
        sc_rope = {}
        assert attn_fp8._get_stage_rope_cache(sc_rope) is attn_fp8._tpu_rope_cache
        assert attn_fp8._get_stage_rope_cache(sc_rope) is attn_fp8._tpu_rope_cache
        with pytest.raises(RuntimeError):
            attn_fp8.forward_mqa(None, None, None, None)

    def test_write_global_kv_and_encoder_paths(self, monkeypatch):
        attn = _base_attn(fp8_kv=True, compress_ratio=2)
        attn.compressor = lambda kv_score, pos: torch.zeros(
            (pos.shape[0], 8), dtype=torch.bfloat16
        )
        idx_cache = _make_paged_mock(rows_per_page=8, page_stride=8, row_offset=0)
        kv_cache = _make_paged_mock(rows_per_page=8, page_stride=8, row_offset=16)
        attn.kv_cache = kv_cache
        attn.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            k_norm=lambda x: x,
            wk=lambda x: x,
            k_cache=SimpleNamespace(prefix="idx", kv_cache=idx_cache, compress_ratio=2),
        )
        attn._quantize_indexer = lambda x: x

        pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        kv_score = torch.randn((4, 9), dtype=torch.float32)
        bt = torch.tensor([[1, 2]], dtype=torch.int32)
        meta = {
            attn.prefix: SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=bt,
                query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            ),
            "idx": SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=bt,
                query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            ),
        }
        starts = torch.tensor([1], dtype=torch.int32)
        monkeypatch.setattr(
            v2_attn,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata=meta,
                additional_kwargs={"deepseek_v41_window_starts": starts},
            ),
        )
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)

        # Stage cache shared block table + distinct block table + fallback
        with v2_attn.stage_attention_cache_scope():
            res1 = attn._write_global_kv(kv_score, pos, defer_index_write=True)
            assert res1 is not None
            res2 = attn._write_global_kv(kv_score, pos, defer_index_write=True)
            assert res2 is not None

            # Distinct index ratio/table & non-fast offsets
            attn.indexer.k_cache.compress_ratio = 1
            meta["idx"] = SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=torch.tensor([[2, 3]], dtype=torch.int64),
                query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            )
            attn.kv_cache = _make_paged_mock(has_row_offset=False)
            attn.indexer.k_cache.kv_cache = _make_paged_mock(has_row_offset=False)
            res3 = attn._write_global_kv(kv_score, pos, defer_index_write=True)
            assert res3 is not None

        # Non-stage-cache fallback and non-deferred insert
        monkeypatch.setattr(
            v2_attn,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=meta, additional_kwargs={}),
        )
        assert attn._write_global_kv(kv_score, pos, defer_index_write=False) is None

        # write_encoder_global_kv
        attn.compressor = SimpleNamespace(
            project=lambda h: torch.zeros((h.shape[0], 9)),
            __call__=lambda kv_s, p: torch.zeros((p.shape[0], 8)),
        )
        attn.compressor = MagicMock()
        attn.compressor.project.return_value = torch.zeros((4, 9))
        attn.compressor.return_value = torch.zeros((4, 8))
        attn.write_encoder_global_kv(torch.zeros((4, 8)), pos, sp_sharded=False)
        attn.write_encoder_global_kv(torch.zeros((4, 8)), pos, sp_sharded=True)
        monkeypatch.setattr(
            v2_attn, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
        )
        attn.write_encoder_global_kv(torch.zeros((4, 8)), pos)

    def test_forward_sp_gather_and_prepare_tpu_all_paths(self, monkeypatch):
        # _sp_gather_inputs single-part vs multi-part
        qr = torch.randn((2, 4), dtype=torch.bfloat16)
        kv = torch.randn((2, 8), dtype=torch.bfloat16)
        idx_w = torch.randn((2, 2), dtype=torch.bfloat16)
        kv_sc = torch.randn((2, 4), dtype=torch.float32)
        out_sp = v2_attn._sp_gather_inputs(
            qr, kv, kv_sc, idx_w, local_q=False, num_tokens=2
        )
        assert out_sp[2].dtype == torch.float32
        out_sp_min = v2_attn._sp_gather_inputs(
            qr, kv, None, None, local_q=True, num_tokens=2
        )
        assert out_sp_min[1].shape == (2, 8)

        attn = _base_attn(fp8_kv=True, compress_ratio=2)
        swa_cache = _make_paged_mock(
            rows_per_page=16,
            page_stride=16,
            row_offset=0,
            pages_tensor=torch.tensor([[1, 2]], dtype=torch.int64),
        )
        attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=swa_cache)
        attn._static_forward_context = {
            attn.compressed_cache_prefix: SimpleNamespace(kv_cache=swa_cache)
        }
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention",
            lambda *a, **kw: torch.ones((2, 2, 8), dtype=torch.bfloat16),
        )
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention_qo",
            lambda *a, **kw: torch.ones((2, 4), dtype=torch.bfloat16),
        )
        monkeypatch.setattr(
            v2_attn,
            "paged_attention",
            lambda *a, **kw: torch.ones((2, 2, 8), dtype=torch.bfloat16),
        )

        # Empty metadata -> direct zero output when output=None, zero_() when
        # output=Tensor
        monkeypatch.setattr(
            v2_attn, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
        )
        pos = torch.tensor([0, 1], dtype=torch.int32)
        hs = torch.zeros((2, 8), dtype=torch.bfloat16)
        assert attn(pos, hs).shape == (2, 16)
        buf = torch.ones((2, 2, 8), dtype=torch.bfloat16)
        assert (
            attn._prepare_tpu(hs, hs, hs, None, None, None, pos, buf) is None
            and buf.sum().item() == 0
        )

        meta = {
            "swa": SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 2], dtype=torch.int64),
            ),
            attn.compressed_cache_prefix: SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 2], dtype=torch.int64),
            ),
            "idx": SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=torch.tensor([[1, 2]], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            ),
        }
        monkeypatch.setattr(
            v2_attn,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata=meta,
                additional_kwargs={
                    "deepseek_v41_replay_starts": torch.tensor([0], dtype=torch.int32),
                    "deepseek_v41_window_starts": torch.tensor([0], dtype=torch.int32),
                },
            ),
        )

        # Stage-cached reuse layer (3page SWA causal + non_causal, row_offset=0 &
        # !=0, non-3page SWA)
        with v2_attn.stage_attention_cache_scope():
            assert attn(pos, hs).shape == (2, 16)
            assert attn(pos, hs).shape == (2, 16)
            swa_cache.row_offset = 8
            assert attn(pos, hs).shape == (2, 16)
            assert attn(pos, hs).shape == (2, 16)

            attn.non_causal_multi_token_decode = True
            # swa_width = 128 <= 6 * 32 -> 3-page SWA non-causal + non-fast offset
            attn.window_size = 64
            attn.swa_cache_layer = SimpleNamespace(
                prefix="swa",
                kv_cache=_make_paged_mock(
                    rows_per_page=32, page_stride=32, has_row_offset=False
                ),
            )
            assert attn(pos, hs).shape == (2, 16)

            attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=swa_cache)
            attn._ced_decoder = True
            attn.window_size = 512  # > 6 * rows_per_page -> non-3page SWA
            assert attn(pos, hs).shape == (2, 16)

        # Non-stage-cache path + compressor in _prepare_tpu + tiled output buffer copy
        attn.window_size = 64
        attn.compressor = SimpleNamespace(
            project=lambda h: h.new_zeros((h.shape[0], 9))
        )
        monkeypatch.setattr(attn, "_write_global_kv", lambda kv_s, p: None)
        assert attn(pos, hs).shape == (2, 16)
        buf_tiled = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
        assert attn._prepare_tpu(hs, hs, hs, None, None, None, pos, buf_tiled) is None
        attn.compressor = None
        attn._ced_decoder = False
        swa_cache.row_offset = 0
        assert attn(pos, hs).shape == (2, 16)
        attn.swa_cache_layer = SimpleNamespace(
            prefix="swa", kv_cache=_make_paged_mock(has_row_offset=False)
        )
        attn._static_forward_context[attn.compressed_cache_prefix] = SimpleNamespace(
            kv_cache=_make_paged_mock(has_row_offset=False, pages_tensor=None)
        )
        assert attn(pos, hs).shape == (2, 16)

        # Fused QO paths (_local_wo_b True vs False) & non-fused qr_local 3D
        # fallback & mocked _prepare_and_attn_fn
        attn = _base_attn(fp8_kv=True, compress_ratio=2)
        attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=swa_cache)
        attn._static_forward_context = {
            attn.compressed_cache_prefix: SimpleNamespace(kv_cache=swa_cache)
        }
        attn._sp_sharded_input = True
        attn._local_q_attn = True
        attn._wq_b_full = torch.zeros((4, 8))
        attn._wq_b_full_scale = torch.ones(8)
        attn._wo_a_full = torch.zeros((8, 4))
        attn._wo_a_full_scale = torch.ones(4)
        attn._wo_b_full = torch.zeros((4, 8))
        attn._wo_b_full_scale = torch.ones(8)
        attn._local_wo_b = False
        assert attn(pos, hs).shape == (2, 4)
        attn._local_wo_b = True
        assert attn(pos, hs).shape == (2, 4)

        # qr_local on non-compressed layer -> _prepare_tpu returns 3D tensor
        attn.compress_ratio = 0
        assert attn(pos, hs).shape == (2, 16)

        # Mocked _prepare_and_attn_fn -> allocates o_padded fallback
        attn._prepare_and_attn_fn = lambda *a, **kw: None
        assert attn(pos, hs).shape == (2, 16)
        attn._local_q_attn = False
        assert attn(pos, hs).shape == (2, 16)

    def test_indexer_tiled_nontiled_and_paged_selection(self, monkeypatch):
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention",
            lambda *a, **kw: torch.ones((2, 2, 8), dtype=torch.bfloat16),
        )
        monkeypatch.setattr(
            v2_attn,
            "paged_attention",
            lambda *a, **kw: torch.ones((2, 2, 8), dtype=torch.bfloat16),
        )
        monkeypatch.setattr(
            v2_attn,
            "sparse_attention",
            lambda q, rows, sink, idx, sc: torch.ones_like(q),
        )
        monkeypatch.setattr(
            v2_attn, "tensor_model_parallel_all_gather", lambda x, dim=0: x
        )
        sel_width = [4]
        monkeypatch.setattr(
            v2_attn,
            "paged_indexer",
            lambda q, cache, *a, **kw: (
                (cache.raw, cache.raw)
                if getattr(cache, "raw_hi", None) is not None
                else cache.raw,
                torch.zeros((q.shape[0], sel_width[0]), dtype=torch.int32),
                torch.zeros((q.shape[0], 4), dtype=torch.int32),
            ),
        )

        pos = torch.tensor([0, 1], dtype=torch.int32)
        hs = torch.zeros((2, 8), dtype=torch.bfloat16)
        bt = torch.tensor([[0, 1]], dtype=torch.int32)
        meta = {
            "swa": SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=bt,
                query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            ),
            "model.layers.2.self_attn.attn": SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=bt,
                query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            ),
            "idx": SimpleNamespace(
                seq_lens=torch.tensor([4], dtype=torch.int32),
                block_tables=bt,
                query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            ),
        }
        monkeypatch.setattr(
            v2_attn,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=meta, additional_kwargs={}),
        )

        # 1. _all_indices_fit=True
        attn = _base_attn(fp8_kv=False, compress_ratio=2)
        attn._max_model_len = 4
        swa_cache = _make_paged_mock(
            rows_per_page=8,
            page_stride=8,
            row_offset=0,
            pages_tensor=torch.tensor([[1, 2]], dtype=torch.int32),
        )
        attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=swa_cache)
        attn._static_forward_context = {
            attn.compressed_cache_prefix: SimpleNamespace(kv_cache=swa_cache)
        }
        attn.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            wq_b=lambda qr: qr.new_zeros((qr.shape[0], 16)),
            weights_proj=lambda h: h.new_ones((h.shape[0], 2)),
            indexer_op=SimpleNamespace(candidate_blocks=None, candidate_write=False),
            k_cache=SimpleNamespace(prefix="idx", kv_cache=swa_cache, compress_ratio=2),
        )
        attn._quantize_indexer = lambda x: x
        assert attn(pos, hs).shape == (2, 16)

        # 2. Tiled paged_indexer across split=True & split=False, candidates
        # None/write/filter, width=8 & 128, fp8 pages cached
        attn._max_model_len = 256
        attn._fp8_kv = True
        attn._precompact_records = True
        with v2_attn.stage_attention_cache_scope():
            monkeypatch.setattr(
                v2_attn, "get_tensor_model_parallel_world_size", lambda: 2
            )
            # Cold candidate_write=False first to populate stage_cache[cand_key]
            # from candidate_block_buffer
            attn.indexer.indexer_op = SimpleNamespace(
                candidate_blocks=True, candidate_write=False
            )
            assert attn(pos, hs).shape == (2, 16)
            # Second call hits cached pages & cached candidates_part with width >=
            # target_width and width=128
            attn.topk_indices_buffer = torch.full((16, 128), -1, dtype=torch.int32)
            sel_width[0] = 128
            attn.indexer.indexer_op = SimpleNamespace(
                candidate_blocks=None, candidate_write=False
            )
            assert attn(pos, hs).shape == (2, 16)

            sel_width[0] = 4
            attn.topk_indices_buffer = torch.full((16, 8), -1, dtype=torch.int32)
            attn.indexer.indexer_op = SimpleNamespace(
                candidate_blocks=True, candidate_write=True
            )
            assert attn(pos, hs).shape == (2, 16)
            attn.indexer.indexer_op = SimpleNamespace(
                candidate_blocks=True, candidate_write=False
            )
            assert attn(pos, hs).shape == (2, 16)

        # Non-stage-cache split=True with candidate_write=True and tuple updated_cache
        swa_cache.raw_hi = torch.zeros((32, 128), dtype=torch.int32)
        attn.indexer.indexer_op = SimpleNamespace(
            candidate_blocks=True, candidate_write=True
        )
        assert attn(pos, hs).shape == (2, 16)

        # split=False non-fast index_cache offset
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)
        attn.indexer.k_cache.kv_cache = _make_paged_mock(has_row_offset=False)
        attn.indexer.indexer_op = SimpleNamespace(
            candidate_blocks=True, candidate_write=False
        )
        assert attn(pos, hs).shape == (2, 16)

        # 3. Non-tiled path (both FP8 error & BF16 non-tiled indexer None/write/filter)
        plain_kv = torch.zeros((8, 4, 8), dtype=torch.bfloat16)
        attn_fp8_nt = _base_attn(fp8_kv=True, compress_ratio=2)
        attn_fp8_nt.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=plain_kv)
        attn_fp8_nt._static_forward_context = {
            attn_fp8_nt.compressed_cache_prefix: SimpleNamespace(kv_cache=plain_kv)
        }
        with pytest.raises(NotImplementedError, match="tiled path"):
            attn_fp8_nt(pos, hs)

        attn_nt = _base_attn(fp8_kv=False, compress_ratio=2)
        attn_nt._max_model_len = 64
        attn_nt.candidate_block_buffer = torch.full((16, 2), -1, dtype=torch.int32)
        attn_nt.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=plain_kv)
        attn_nt._static_forward_context = {
            attn_nt.compressed_cache_prefix: SimpleNamespace(kv_cache=plain_kv)
        }
        attn_nt.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            wq_b=lambda qr: qr.new_zeros((qr.shape[0], 16)),
            weights_proj=lambda h: h.new_ones((h.shape[0], 2)),
            indexer_op=SimpleNamespace(candidate_blocks=None, candidate_write=False),
            k_cache=SimpleNamespace(prefix="idx", kv_cache=plain_kv, compress_ratio=2),
        )
        attn_nt._quantize_indexer = lambda x: x
        assert attn_nt(pos, hs).shape == (2, 16)
        attn_nt.indexer.indexer_op = SimpleNamespace(
            candidate_blocks=True, candidate_write=True
        )
        assert attn_nt(pos, hs).shape == (2, 16)
        attn_nt.indexer.indexer_op = SimpleNamespace(
            candidate_blocks=True, candidate_write=False
        )
        assert attn_nt(pos, hs).shape == (2, 16)
        buf_nt = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
        assert (
            attn_nt._prepare_tpu(
                hs, hs, hs, None, None, torch.ones((2, 2)), pos, buf_nt
            )
            is None
        )
