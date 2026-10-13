# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""100% statement and branch coverage tests for DeepSeek-V4.1 v1/attention.py."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

_SRC_DIR = str(Path(__file__).resolve().parents[3] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

for _pkg_name, _rel in (
    ("vllm_torchtpu.layers", "vllm_torchtpu/layers"),
    ("vllm_torchtpu.layers.adapter", "vllm_torchtpu/layers/adapter"),
    (
        "vllm_torchtpu.layers.adapter.custom_ops",
        "vllm_torchtpu/layers/adapter/custom_ops",
    ),
    ("vllm_torchtpu.models", "vllm_torchtpu/models"),
    ("vllm_torchtpu.models.vllm", "vllm_torchtpu/models/vllm"),
    ("vllm_torchtpu.models.vllm.deepseek_v4", "vllm_torchtpu/models/vllm/deepseek_v4"),
    ("vllm_torchtpu.runner", "vllm_torchtpu/runner"),
):
    if _pkg_name not in sys.modules:
        _pkg = types.ModuleType(_pkg_name)
        _pkg.__path__ = [str(Path(_SRC_DIR) / _rel)]  # type: ignore[attr-defined]
        sys.modules[_pkg_name] = _pkg


class _StubMeta(type):
    def __getattr__(cls, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return name


class _StubModule(types.ModuleType):
    __path__ = []  # type: ignore[assignment]

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name[:1].isupper() and not name.isupper():
            base = (torch.nn.Module,) if "Attention" in name else ()
            cls = _StubMeta(name, base, {})
            setattr(self, name, cls)
            return cls
        m = _StubModule(f"{self.__name__}.{name}")
        sys.modules[m.__name__] = m
        setattr(self, name, m)
        return m

    def __getitem__(self, key):
        return MagicMock()

    def __call__(self, *args, **kwargs):
        return MagicMock()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname: str, path, target=None):
        if fullname.split(".")[0] in ("vllm", "torch_tpu", "requests"):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return sys.modules.get(spec.name) or _StubModule(spec.name)

    def exec_module(self, module):
        pass


if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _StubFinder())
if "tpu" not in torch._dynamo.list_backends():
    torch._dynamo.register_backend(name="tpu", compiler_fn=lambda gm, ex: gm)

from vllm_torchtpu.kernels.deepseek_v4.csa_cache_layout import ROW_WORDS  # noqa: E402

v1_attn = importlib.import_module("vllm_torchtpu.models.vllm.deepseek_v41.v1.attention")


def _make_vllm_config(*, kv_dtype: str = "fp8", rope_dim: int = 64):
    hf_dict = {
        "qk_rope_head_dim": rope_dim,
        "rope_theta": 10000.0,
        "compress_rope_theta": 50000.0,
        "index_topk": 4,
    }
    hf_config = SimpleNamespace(
        qk_rope_head_dim=rope_dim,
        to_dict=lambda: dict(hf_dict),
    )
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda _default: kv_dtype
        ),
        model_config=SimpleNamespace(
            hf_config=hf_config,
            max_model_len=8,
        ),
    )


def test_indexer_uses_fp4_and_swa_cache_and_value_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert v1_attn.indexer_uses_fp4(_make_vllm_config(kv_dtype="fp8")) is False
    with pytest.raises(ValueError, match="requires fp8"):
        v1_attn.indexer_uses_fp4(_make_vllm_config(kv_dtype="fp4"))

    assert v1_attn._value(("first", "second")) == "first"
    assert v1_attn._value("only") == "only"

    recorded_init: dict = {}

    def _fake_swa_super_init(self, *args, **kwargs):
        recorded_init.update(kwargs)

    monkeypatch.setattr(v1_attn.DeepseekV4SWACache, "__init__", _fake_swa_super_init)
    swa = v1_attn.SWACache(cache_config=SimpleNamespace(block_size=16))
    assert recorded_init["block_size"] == 16
    assert recorded_init["backend_cls"] is v1_attn.backend_ops.AttentionBackend

    cache_t = torch.zeros((4, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires the allocator's logical layout"):
        swa.bind_kv_cache(cache_t, layout=None)

    layout = v1_attn.cache_ops.ByteLayout((4, 2, 128), torch.bfloat16, 512, 0)
    swa.bind_kv_cache(cache_t, raw_cache=None, layout=layout)
    assert not swa.has_native_cache
    assert swa.block_size == 2

    raw_pool = torch.zeros((8, ROW_WORDS), dtype=torch.int32)
    swa.bind_kv_cache(cache_t, raw_cache=raw_pool, layout=layout)
    assert swa.has_native_cache
    assert swa.kv_cache.shape == (4, 2, 128)


def test_deepseek_v41_attention_init_and_cache_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert v1_attn.DeepseekV41Attention.get_padded_num_q_heads(16) == 16

    def _make_attn(
        *,
        kv_cache_dtype=torch.bfloat16,
        compress_ratio=2,
        head_dim=512,
        rope_dim=64,
    ):
        def _fake_super_init(self, vllm_config, *args, **kwargs):
            torch.nn.Module.__init__(self)
            _ = torch.cuda.Event()
            self.kv_cache_torch_dtype = kv_cache_dtype
            self.compress_ratio = compress_ratio
            self.head_dim = head_dim
            self.rotary_emb = SimpleNamespace(
                cos_sin_cache=torch.ones((16, 8), dtype=torch.float32)
            )
            self.prefix = "model.layers.0.self_attn"
            self.compressed_cache_prefix = "model.layers.0.self_attn.comp"

        monkeypatch.setattr(
            v1_attn.upstream.DeepseekV4Attention, "__init__", _fake_super_init
        )
        return v1_attn.DeepseekV41Attention(_make_vllm_config(rope_dim=rope_dim))

    with pytest.raises(NotImplementedError, match="use --kv-cache-dtype"):
        _make_attn(kv_cache_dtype=torch.float16)

    with pytest.raises(ValueError, match="448 NoPE and 64 RoPE"):
        _make_attn(compress_ratio=2, head_dim=256)

    attn_csa = _make_attn(compress_ratio=2, head_dim=512, rope_dim=64)
    assert attn_csa.attention_kind == "csa"
    attn_csa._static_forward_context = {
        attn_csa.compressed_cache_prefix: "shared_layer"
    }
    assert attn_csa.main_cache_layer == "shared_layer"

    attn_swa = _make_attn(compress_ratio=0, head_dim=128, rope_dim=64)
    assert attn_swa.attention_kind == "swa"
    assert attn_swa.main_cache_layer is attn_swa

    with pytest.raises(ValueError, match="requires a \\(NoPE, RoPE\\) pair"):
        attn_csa.bind_kv_cache((torch.zeros(1),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires the allocator's logical layout"):
        attn_csa.bind_kv_cache(
            (torch.zeros((2, 2, 128)), torch.zeros((2, 2, 32))), layout=None
        )

    nope = torch.zeros((2, 2, ROW_WORDS), dtype=torch.int32)
    rope = torch.zeros((2, 2, 32), dtype=torch.int32)
    layout = v1_attn.cache_ops.ByteLayout((2, 2, ROW_WORDS), torch.int32, 1024, 0)
    attn_csa.bind_kv_cache((nope, rope), raw_cache=None, layout=layout)
    assert attn_csa.kv_cache == (nope, rope)

    raw_nope = torch.zeros((4, ROW_WORDS), dtype=torch.int32)
    attn_csa.bind_kv_cache((nope, rope), raw_cache=raw_nope, layout=layout)
    kv_first = attn_csa.kv_cache
    kv_second = attn_csa.kv_cache
    assert kv_first is kv_second
    assert kv_first[0].shape == (2, 2, ROW_WORDS)
    assert kv_first[1] is rope

    monkeypatch.setattr(
        v1_attn.upstream.DeepseekV4Attention,
        "get_kv_cache_spec",
        lambda self, cfg: ("super_spec", cfg),
    )
    monkeypatch.setattr(
        v1_attn,
        "deepseek_v41_cache_spec",
        lambda spec, cfg, compressed=False: (spec, cfg, compressed),
    )
    assert attn_csa.get_kv_cache_spec("vcfg") == (
        ("super_spec", "vcfg"),
        "vcfg",
        True,
    )


def test_projections_rotate_write_global_kv_and_prepare_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_super_init(self, vllm_config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.kv_cache_torch_dtype = torch.bfloat16
        self.compress_ratio = 2
        self.head_dim = 512
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.ones((16, 4), dtype=torch.float32)
        )
        self.prefix = "attn"
        self.compressed_cache_prefix = "attn"

    monkeypatch.setattr(
        v1_attn.upstream.DeepseekV4Attention, "__init__", _fake_super_init
    )
    cfg = _make_vllm_config()
    attn = v1_attn.DeepseekV41Attention(cfg)
    attn._config["qk_rope_head_dim"] = 4
    attn.head_dim = 8
    attn.n_local_heads = 2
    attn.q_lora_rank = 4
    attn.window_size = 4
    attn.scale = 0.5
    attn.attn_sink = torch.zeros((2,), dtype=torch.bfloat16)
    attn.candidate_block_size = 2

    attn._fused_wqa_wkv_gemm = lambda h: torch.ones((h.shape[0], 12), dtype=h.dtype)
    attn.q_norm = lambda x: x + 1
    attn.kv_norm = lambda x: x + 2

    class _FakeCompressor:
        def project(self, h):
            return torch.ones((h.shape[0], 9), dtype=torch.float32)

        def __call__(self, kv_s, pos):
            return torch.ones((pos.shape[0], 8), dtype=torch.bfloat16)

    attn.compressor = _FakeCompressor()

    idx_layout = v1_attn.cache_ops.RecordLayout(2, 2, 0, 8, torch.bfloat16)
    idx_pool = torch.zeros((8, ROW_WORDS), dtype=torch.int32)
    idx_scales = torch.zeros((8, 4), dtype=torch.int32)
    attn.indexer = SimpleNamespace(
        n_head=2,
        head_dim=8,
        weights_proj=lambda h: (torch.ones((h.shape[0], 2), dtype=h.dtype), None),
        wq_b=lambda qr: (torch.ones((qr.shape[0], 16), dtype=qr.dtype), None),
        wk=lambda lat: (torch.ones((lat.shape[0], 8), dtype=lat.dtype), None),
        k_norm=lambda k: k,
        k_cache=SimpleNamespace(
            prefix="idx_k",
            compress_ratio=2,
            native=lambda: SimpleNamespace(
                pool=idx_pool, scales=idx_scales, layout=idx_layout
            ),
        ),
        indexer_op=SimpleNamespace(candidate_blocks=None, candidate_write=False),
    )

    h = torch.zeros((3, 8), dtype=torch.bfloat16)
    assert attn._all_indices_fit is True
    qr_kv, kv_s, idx_w = attn._run_parallel_input_projections(h)
    assert kv_s is not None and idx_w is None

    attn._max_model_len = 16
    assert attn._all_indices_fit is False
    _, _, idx_w2 = attn._run_parallel_input_projections(h)
    assert idx_w2 is not None

    attn._ced_global_from_encoder = True
    _, kv_s_none, _ = attn._run_parallel_input_projections(h)
    assert kv_s_none is None
    attn._ced_global_from_encoder = False

    qr, _, kv = attn._split_qkv_and_norm(qr_kv)
    assert qr.shape == (3, 4) and kv.shape == (3, 8)

    pos = torch.tensor([0, 1, 2], dtype=torch.int64)
    rot_cpu = attn._rotate(kv, pos, inverse=False)
    assert rot_cpu.shape == kv.shape

    monkeypatch.setattr(
        v1_attn.DeepseekTPUAttentionMixin,
        "kv_rope_op",
        property(lambda self: lambda v, p, c: "tpu_rotated"),
    )
    fake_tpu_val = SimpleNamespace(device=SimpleNamespace(type="tpu"))
    assert attn._rotate(fake_tpu_val, pos, inverse=False) == "tpu_rotated"

    monkeypatch.setattr(
        v1_attn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    attn.write_encoder_global_kv(h, pos)

    meta_entry = SimpleNamespace(
        seq_lens=torch.tensor([3], dtype=torch.int32),
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
    )
    meta_dict = {"attn": meta_entry, "idx_k": meta_entry, "swa": meta_entry}
    monkeypatch.setattr(
        v1_attn,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=meta_dict,
            additional_kwargs={
                "deepseek_v41_replay_starts": torch.tensor([0], dtype=torch.int64)
            },
        ),
    )
    attn.kv_cache = (
        torch.zeros((2, 2, ROW_WORDS), dtype=torch.int32),
        torch.zeros((2, 2, 32), dtype=torch.int32),
    )
    attn.cache_num_states = 2

    comp_writes: list = []
    pub_writes: list = []
    monkeypatch.setattr(
        v1_attn.compressor_ops,
        "write_compressed_cache",
        lambda *a: comp_writes.append(a),
    )
    monkeypatch.setattr(
        v1_attn.indexer_ops,
        "publish_index_keys",
        lambda *a, **kw: pub_writes.append((a, kw)),
    )

    attn.write_encoder_global_kv(h, pos)
    assert len(comp_writes) == 1
    assert len(pub_writes) == 1

    deferred = attn._write_global_kv(kv_s, pos, defer_index_write=True)
    assert deferred is not None

    attn.compress_ratio = 3
    pos_i32 = torch.tensor([0, 1, 2], dtype=torch.int32)
    deferred_cr3 = attn._write_global_kv(kv_s, pos_i32, defer_index_write=True)
    assert deferred_cr3 is not None
    assert deferred_cr3[0].tolist() == [-1, -1, 1]
    attn.compress_ratio = 2

    swa_pool = torch.zeros((8, ROW_WORDS), dtype=torch.int32)
    swa_layout = v1_attn.cache_ops.RecordLayout(2, 2, 0, 8, torch.bfloat16)
    attn.swa_cache_layer = SimpleNamespace(
        prefix="swa",
        block_size=2,
        kv_cache=torch.empty((0, 2, 8), dtype=torch.bfloat16),
        native=lambda: SimpleNamespace(pool=swa_pool, layout=swa_layout),
    )
    out = torch.ones((3, 2, 8), dtype=torch.bfloat16)
    attn._prepare_tpu(h, qr, kv, None, kv_s, idx_w2, pos, out)
    assert torch.all(out == 0)

    attn.swa_cache_layer.kv_cache = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    attn.wq_b = lambda q_in: (
        torch.ones((q_in.shape[0], 16), dtype=q_in.dtype),
        None,
    )
    attn.topk_indices_buffer = torch.full((4, 4), -1, dtype=torch.int32)
    attn.candidate_block_buffer = torch.full((4, 2), -1, dtype=torch.int32)
    attn._run_mqa = lambda q, kv, p, o, topk_indices=None: torch.ones_like(o)

    monkeypatch.setattr(
        v1_attn.indexer_ops,
        "paged_indexer",
        lambda *a, **kw: (
            idx_pool,
            idx_scales,
            torch.zeros((3, 2), dtype=torch.int32),
            torch.zeros((3, 2), dtype=torch.int32),
        ),
    )
    attn._prepare_tpu(h, qr, kv, None, kv_s, idx_w2, pos, out)
    assert torch.all(out == 1)

    attn.indexer.indexer_op.candidate_blocks = 2
    attn.indexer.indexer_op.candidate_write = True
    sel_cand = attn._select_paged_indices(
        torch.ones((3, 2, 8), dtype=torch.bfloat16),
        torch.zeros((1, 2), dtype=torch.int32),
        torch.zeros((3,), dtype=torch.int64),
        4,
        torch.ones((3, 2), dtype=torch.float32),
        torch.ones((3, 1), dtype=torch.int64),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        None,
    )
    assert sel_cand.shape == (3, 2)

    attn.candidate_block_buffer = torch.full((3, 2), -1, dtype=torch.int32)
    sel_cand_exact = attn._select_paged_indices(
        torch.ones((3, 2, 8), dtype=torch.bfloat16),
        torch.zeros((1, 2), dtype=torch.int32),
        torch.zeros((3,), dtype=torch.int64),
        4,
        torch.ones((3, 2), dtype=torch.float32),
        torch.ones((3, 1), dtype=torch.int64),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        None,
    )
    assert sel_cand_exact.shape == (3, 2)

    attn._max_model_len = 8
    attn._prepare_tpu(h, qr, kv, None, kv_s, idx_w2, pos, out)
    attn.topk_indices_buffer = torch.full((3, 4), -1, dtype=torch.int32)
    attn._prepare_tpu(h, qr, kv, None, kv_s, idx_w2, pos_i32, out)
    assert attn.topk_indices_buffer.dtype == torch.int32
    assert attn.topk_indices_buffer[:3, :4].tolist() == [
        [-1, -1, -1, -1],
        [0, -1, -1, -1],
        [0, -1, -1, -1],
    ]

    attn.compress_ratio = 3
    attn.topk_indices_buffer = torch.full((3, 2), -1, dtype=torch.int32)
    attn._prepare_tpu(h, qr, kv, None, kv_s, idx_w2, pos_i32, out)
    attn.compress_ratio = 2

    saved_indexer = attn.indexer
    attn.indexer = None
    attn.compressor = None
    attn._prepare_tpu(h, qr, kv, None, kv_s, idx_w2, pos, out)

    attn.compress_ratio = 0
    attn._ced_decoder = True
    monkeypatch.setattr(
        v1_attn.ced_ops,
        "ced_attention",
        lambda q, *a, **kw: torch.full_like(q, 5.0),
    )
    attn._prepare_tpu(h, qr, kv, None, None, None, pos, out)
    assert torch.all(out == 5.0)
    attn.indexer = saved_indexer

    attn.compress_ratio = 2
    attn._static_forward_context = {"attn": attn}
    assert attn._get_window_offsets(pos.device).dtype == torch.int32
    attn.forward_mqa(
        torch.ones((3, 2, 8), dtype=torch.bfloat16),
        kv,
        pos_i32,
        out,
        topk_indices=torch.zeros((3, 2), dtype=torch.int32),
    )
    assert torch.all(out == 5.0)

    monkeypatch.setattr(
        v1_attn.DeepseekTPUAttentionMixin,
        "process_weights_after_loading",
        lambda self, dt: None,
    )
    monkeypatch.setattr(
        v1_attn.DeepseekTPUAttentionMixin,
        "o_proj_op",
        property(
            lambda self: lambda o, p, r, w, s: torch.ones(
                (o.shape[0], 8), dtype=o.dtype
            )
        ),
    )
    attn._assert_o_proj_supported = lambda: None
    attn.wo_a = SimpleNamespace(
        weight=torch.zeros((8, 16), dtype=torch.bfloat16),
        weight_scale=torch.ones((8, 1), dtype=torch.float32),
    )
    attn.wo_b = lambda proj: (torch.ones((proj.shape[0], 16), dtype=proj.dtype), None)

    attn.process_weights_after_loading(torch.bfloat16)
    attn.wo_a.weight = SimpleNamespace(device=SimpleNamespace(type="tpu"))
    attn.process_weights_after_loading(torch.bfloat16)

    proj_out = attn._o_proj(torch.ones((3, 2, 8), dtype=torch.bfloat16), pos)
    assert proj_out.shape == (3, 16)
