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
"""TD8.71: CPU coverage for the MLA / sparse-indexer custom-op bridge.

The Pallas kernels themselves are TPU-only; these tests drive the host-side
glue (op builders, scatter math, dispatch, weight post-processing and the
RoPE fallback) with the kernels replaced by recorders, and check the pure-JAX
pieces against NumPy references.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.layers.attention import mla_attention as upstream_mla
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.v1.attention.backends.mla.prefill import selector

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.layers.adapter.attention import VllmTPUDeepseekV32IndexerBackend
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

DCP = mla.DCP_AXIS_NAME
D_IDX = 4  # indexer head dim; packed row is D_IDX + 1 bytes
LKV = 8  # padded lane width of the indexer K cache
KVP = 2  # kv_packing
BSP = 2  # block_size_per_kv_packing -> page_size 4
NBLK = 4
TOPK = 3


def _mesh():
    return jax.sharding.Mesh(np.array(jax.devices()[:1]), (DCP,))


def _batch():
    # Two sequences: seq0 has 5 tokens total (2 new), seq1 has 3 (3 new);
    # token 5 is padding.
    seq_lens = np.array([5, 3], np.int32)
    cu_q_lens = np.array([0, 2, 5], np.int32)
    page_indices = np.array([1, 2, 3, 0], np.int32)  # 2 pages per seq
    num_tokens = 6
    k_packed = (np.arange(num_tokens * (D_IDX + 1)) % 251 + 1).astype(np.uint8)
    k_packed = k_packed.reshape(num_tokens, D_IDX + 1)
    return seq_lens, cu_q_lens, page_indices, k_packed


def _reference_scatter(cache, k_packed, seq_lens, page_indices, cu, *, cp, il):
    """NumPy model of the interleaved per-rank scatter for rank 0."""
    out = cache.copy()
    page_size = BSP * KVP
    pages = page_indices.reshape(seq_lens.shape[0], -1)
    for t in range(k_packed.shape[0]):
        if t >= cu[-1]:
            continue
        s = int(np.searchsorted(cu[1:], t, side="right"))
        pos = seq_lens[s] - (cu[s + 1] - cu[s]) + (t - cu[s])
        vps = page_size * cp
        offset = pos % vps
        if (offset % (cp * il)) // il != 0:
            continue
        slot = (offset // (cp * il)) * il + offset % il
        page = pages[s, pos // vps] % cache.shape[0]
        row = np.zeros((LKV,), np.uint8)
        row[: k_packed.shape[1]] = k_packed[t]
        out[page, slot // KVP, slot % KVP] = row
    return out


@pytest.mark.parametrize("cp,il", [(1, 1), (2, 1), (2, 2)])
def test_cp_scatter_indexer_k_matches_reference(cp, il):
    seq_lens, cu, page_indices, k_packed = _batch()
    cache = np.zeros((NBLK, BSP, KVP, LKV), np.uint8)
    got = mla._cp_scatter_indexer_k(
        jnp.asarray(cache),
        jnp.asarray(k_packed),
        jnp.asarray(seq_lens),
        jnp.asarray(page_indices),
        jnp.asarray(cu),
        mesh=_mesh(),
        cp_size=cp,
        interleave_size=il,
    )
    expected = _reference_scatter(
        cache, k_packed, seq_lens, page_indices, cu, cp=cp, il=il
    )
    np.testing.assert_array_equal(np.asarray(got), expected)


def test_cp_scatter_indexer_k_rejects_straddling_interleave():
    seq_lens, cu, page_indices, k_packed = _batch()
    with pytest.raises(ValueError, match="must divide the indexer"):
        mla._cp_scatter_indexer_k(
            jnp.zeros((NBLK, BSP, KVP, LKV), jnp.uint8),
            jnp.asarray(k_packed),
            jnp.asarray(seq_lens),
            jnp.asarray(page_indices),
            jnp.asarray(cu),
            mesh=_mesh(),
            cp_size=1,
            interleave_size=3,
        )


# --------------------------------------------------------------------------
# VllmTPUSparseAttnIndexer
# --------------------------------------------------------------------------


class _RecordingJaxOp:
    """Stands in for the exported op: records the JAX body and the fake."""

    def __init__(self, name, fn, **kwargs):
        self.name, self.fn, self.kwargs = name, fn, kwargs
        self.fake = None
        self.result = None

    def register_fake(self, fake):
        self.fake = fake

    def __call__(self, *args):
        return self.result


def _bare_indexer(prefix="model.layers.0.indexer.k_cache"):
    indexer = mla.VllmTPUSparseAttnIndexer.__new__(mla.VllmTPUSparseAttnIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.topk_tokens = TOPK
    indexer.k_cache = SimpleNamespace(prefix=prefix, kv_cache=torch.empty(0))
    indexer.topk_indices_buffer = torch.zeros((4, TOPK + 1), dtype=torch.int32)
    return indexer


def _pin_chunk(monkeypatch, value=None):
    monkeypatch.setitem(vars(tpu_envs), "TPU_STREAMIDX_CHUNK_TOKENS", value)


def _config(pcp, dcp, interleave=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=interleave,
        )
    )


def _patch_init(monkeypatch, config):
    monkeypatch.setattr(SparseAttnIndexer, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(mla, "get_current_vllm_config_or_none", lambda: config)
    calls = {}
    monkeypatch.setattr(
        mla.VllmTPUSparseAttnIndexer,
        "_build_streamidx_op",
        lambda self: calls.setdefault("op", "single"),
    )
    monkeypatch.setattr(
        mla.VllmTPUSparseAttnIndexer,
        "_build_streamidx_dcp_op",
        lambda self, size, il: calls.setdefault("dcp", (size, il)),
    )
    return calls


@pytest.mark.parametrize(
    "config", [None, _config(1, 1), _config("x", "y"), _config(2, 1)]
)
def test_indexer_init_without_dcp(monkeypatch, config):
    calls = _patch_init(monkeypatch, config)
    indexer = mla.VllmTPUSparseAttnIndexer()
    assert indexer.topk_op == "single"
    assert indexer.topk_op_dcp is None
    assert indexer.dcp_size == 1
    assert "dcp" not in calls


def test_indexer_init_builds_dcp_twin(monkeypatch):
    calls = _patch_init(monkeypatch, _config(1, 4, interleave=8))
    indexer = mla.VllmTPUSparseAttnIndexer()
    assert indexer.dcp_size == 4
    assert indexer.topk_op_dcp == (4, 8)
    assert calls["dcp"] == (4, 8)


def test_indexer_init_rejects_pcp_and_dcp(monkeypatch):
    _patch_init(monkeypatch, _config(2, 2))
    with pytest.raises(NotImplementedError, match="both >1"):
        mla.VllmTPUSparseAttnIndexer()


def _jax_inputs():
    seq_lens, cu, page_indices, k_packed = _batch()
    num_tokens = k_packed.shape[0]
    return dict(
        q_bytes=jnp.zeros((num_tokens, 2, D_IDX), jnp.uint8),
        weights=jnp.ones((num_tokens, 2), jnp.float32),
        cache_kv=jnp.zeros((NBLK, BSP, KVP, LKV), jnp.uint8),
        k_packed=jnp.asarray(k_packed),
        seq_lens=jnp.asarray(seq_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu),
        distribution=jnp.array([0, 0, 2], jnp.int32),
    )


def _check_op_wrapper(op, rec):
    """`op` returns the indices and writes the new cache into args[2]."""
    cache = torch.zeros((NBLK, BSP, KVP, LKV), dtype=torch.uint8)
    new_cache = torch.full_like(cache, 7)
    indices = torch.arange(6 * TOPK, dtype=torch.int32).view(6, TOPK)
    rec.result = (new_cache, indices)
    out = op(torch.zeros(6, 2, D_IDX), None, cache)
    assert out is indices
    torch.testing.assert_close(cache, new_cache)


def _check_fake(rec):
    q = torch.zeros((5, 2, D_IDX), dtype=torch.uint8)
    cache = torch.zeros((NBLK, BSP, KVP, LKV), dtype=torch.uint8)
    new_cache, idx = rec.fake(q, None, cache, None, None, None, None, None)
    assert new_cache.shape == cache.shape and new_cache.dtype == cache.dtype
    assert idx.shape == (5, TOPK) and idx.dtype == torch.int32


def test_build_streamidx_op_scatters_then_calls_kernel(monkeypatch):
    _pin_chunk(monkeypatch, 64)
    recs = []

    def jax_op(name, fn, **kwargs):
        recs.append(_RecordingJaxOp(name, fn, **kwargs))
        return recs[-1]

    seen = {}

    def fake_topk(q, weights, cache_kv, *args, **kwargs):
        seen.update(q_dtype=q.dtype, kwargs=kwargs)
        return jnp.zeros((q.shape[0], kwargs["k"]), jnp.int32)

    monkeypatch.setattr(mla, "pallas", SimpleNamespace(jax_op=jax_op))
    monkeypatch.setattr(mla, "streamindex_topk", fake_topk)
    indexer = _bare_indexer()
    op = indexer._build_streamidx_op()
    (rec,) = recs
    assert indexer._streamidx_op is op
    assert rec.name == "pallas::streamidx_topk_model_layers_0_indexer_k_cache"
    assert rec.kwargs == {"donate_argnums": (2,)}

    inputs = _jax_inputs()
    new_cache, idx = rec.fn(**inputs)
    assert idx.shape == (6, TOPK)
    assert seen["q_dtype"] == jnp.float8_e4m3fn
    assert seen["kwargs"]["chunk_tokens"] == 64
    assert seen["kwargs"]["enable_early_exit"] is mla.STREAMIDX_ENABLE_EARLY_EXIT
    seq_lens, cu, page_indices, k_packed = _batch()
    expected = _reference_scatter(
        np.zeros((NBLK, BSP, KVP, LKV), np.uint8),
        k_packed,
        seq_lens,
        page_indices,
        cu,
        cp=1,
        il=1,
    )
    np.testing.assert_array_equal(np.asarray(new_cache), expected)
    _check_fake(rec)
    _check_op_wrapper(op, rec)


def test_build_streamidx_dcp_op_scatters_then_calls_kernel(monkeypatch):
    _pin_chunk(monkeypatch)
    recs = []

    def pcp_jax_op(name, fn, **kwargs):
        recs.append(_RecordingJaxOp(name, fn, **kwargs))
        return recs[-1]

    seen = {}

    def fake_topk_dcp(q, weights, cache_kv, *args, **kwargs):
        seen.update(q_dtype=q.dtype, kwargs=kwargs)
        return jnp.full((q.shape[0], kwargs["k"]), -1, jnp.int32)

    mesh = _mesh()
    monkeypatch.setattr(mla, "get_or_create_dcp_mesh", lambda axis_name: mesh)
    monkeypatch.setattr(mla, "pcp_streaming_jax_op", pcp_jax_op)
    monkeypatch.setattr(mla, "streamindex_topk_dcp", fake_topk_dcp)
    indexer = _bare_indexer()
    op = indexer._build_streamidx_dcp_op(2, 2)
    (rec,) = recs
    assert rec.name == "pallas::streamidx_topk_dcp_model_layers_0_indexer_k_cache"
    assert rec.kwargs["mesh"] is mesh
    assert rec.kwargs["donate_argnums"] == (2,)
    sharded = jax.sharding.PartitionSpec(DCP)
    assert rec.kwargs["input_partition_specs"][2] == sharded
    assert rec.kwargs["output_partition_specs"] == (sharded, sharded)

    new_cache, idx = rec.fn(**_jax_inputs())
    assert seen["q_dtype"] == jnp.float8_e4m3fn
    assert seen["kwargs"]["dcp_size"] == 2
    assert seen["kwargs"]["interleave_size"] == 2
    assert seen["kwargs"]["mesh"] is mesh
    assert seen["kwargs"]["chunk_tokens"] is None
    assert idx.shape == (6, TOPK)
    seq_lens, cu, page_indices, k_packed = _batch()
    expected = _reference_scatter(
        np.zeros((NBLK, BSP, KVP, LKV), np.uint8),
        k_packed,
        seq_lens,
        page_indices,
        cu,
        cp=2,
        il=2,
    )
    np.testing.assert_array_equal(np.asarray(new_cache), expected)
    _check_fake(rec)
    _check_op_wrapper(op, rec)


def _forward_oot_setup(monkeypatch, attn_metadata, kv_numel=8):
    indexer = _bare_indexer(prefix="layer.k")
    indexer.k_cache.kv_cache = torch.zeros(kv_numel, dtype=torch.uint8)
    monkeypatch.setattr(
        mla,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=attn_metadata),
    )
    calls = []

    def op(*args):
        calls.append(args)
        return torch.arange(3 * TOPK, dtype=torch.int32).view(3, TOPK) + 10

    indexer.topk_op = op
    indexer.topk_op_dcp = None
    return indexer, calls


def _oot_inputs():
    hidden = torch.zeros(3, 8)
    q = torch.ones(3, 2, D_IDX).to(torch.float8_e4m3fn)
    k = torch.linspace(-1, 1, 3 * D_IDX).view(3, D_IDX)
    weights = torch.ones(3, 2)
    return hidden, q, k, weights


def _metadata():
    return AttentionMetadata(
        input_positions=torch.arange(3),
        block_tables=torch.tensor([0, 1]),
        seq_lens=torch.tensor([3]),
        query_start_loc=torch.tensor([0, 3]),
        request_distribution=torch.tensor([1, 2, 3]),
    )


@pytest.mark.parametrize(
    "attn_metadata",
    [None, {}, {"layer.k": "not-metadata"}],
    ids=["no-dict", "missing", "wrong-type"],
)
def test_forward_oot_without_metadata_returns_buffer(monkeypatch, attn_metadata):
    indexer, calls = _forward_oot_setup(monkeypatch, attn_metadata)
    before = indexer.topk_indices_buffer.clone()
    out = indexer.forward_oot(*_oot_inputs())
    assert out is indexer.topk_indices_buffer
    torch.testing.assert_close(out, before)
    assert not calls


def test_forward_oot_with_empty_cache_returns_buffer(monkeypatch):
    indexer, calls = _forward_oot_setup(
        monkeypatch, {"layer.k": _metadata()}, kv_numel=0
    )
    assert indexer.forward_oot(*_oot_inputs()) is indexer.topk_indices_buffer
    assert not calls


def test_forward_oot_rejects_non_fp8_q(monkeypatch):
    indexer, _ = _forward_oot_setup(monkeypatch, None)
    hidden, _, k, weights = _oot_inputs()
    with pytest.raises(AssertionError, match="expects fp8 q"):
        indexer.forward_oot(hidden, torch.ones(3, 2, D_IDX), k, weights)


@pytest.mark.parametrize("use_dcp", [False, True])
def test_forward_oot_dispatches_and_fills_buffer(monkeypatch, use_dcp):
    meta = _metadata()
    indexer, calls = _forward_oot_setup(monkeypatch, {"layer.k": meta})
    if use_dcp:
        indexer.topk_op_dcp, indexer.topk_op = indexer.topk_op, None
    indexer.topk_indices_buffer.fill_(99)
    hidden, q, k, weights = _oot_inputs()
    out = indexer.forward_oot(hidden, q, k, weights)

    (args,) = calls
    q_bytes, w, kv_cache, k_packed, seq_lens, pages, cu, dist = args
    assert q_bytes.dtype == torch.uint8 and q_bytes.shape == q.shape
    assert w is weights and kv_cache is indexer.k_cache.kv_cache
    assert k_packed.dtype == torch.uint8 and k_packed.shape == (3, D_IDX + 1)
    k_quant, k_scale = mla.quantize_tensor(
        k, torch.float8_e4m3fn, axis=-1, use_ue8m0=True
    )
    torch.testing.assert_close(k_packed[:, :D_IDX], k_quant.view(torch.uint8))
    torch.testing.assert_close(k_packed[:, D_IDX:], k_scale)
    assert seq_lens is meta.seq_lens and pages is meta.block_tables
    assert cu is meta.query_start_loc
    torch.testing.assert_close(dist, torch.tensor([1, 1, 3]))
    expected = torch.full((4, TOPK + 1), 99, dtype=torch.int32)
    expected[:3] = -1
    expected[:3, :TOPK] = torch.arange(3 * TOPK, dtype=torch.int32).view(3, TOPK) + 10
    torch.testing.assert_close(out, expected)


# --------------------------------------------------------------------------
# Indexer retyping and small helpers
# --------------------------------------------------------------------------


def test_indexer_cache_backend_and_rebind_identity():
    cache = mla.VllmTPUIndexerCache.__new__(mla.VllmTPUIndexerCache)
    assert cache.get_attn_backend() is VllmTPUDeepseekV32IndexerBackend
    already = mla.VllmTPUIndexer.__new__(mla.VllmTPUIndexer)
    assert mla.VllmTPUIndexer.rebind(already) is already


def test_fresh_breaks_views_and_dummy_prefill_backend():
    base = torch.arange(12.0).view(3, 4)
    view = base.t()
    fresh = mla._fresh(view)
    assert fresh.is_contiguous() and fresh.data_ptr() != base.data_ptr()
    torch.testing.assert_close(fresh, view)
    backend = mla.TPUDummyMLAPrefillBackend(1, key=2)
    assert backend.forward(3, key=4) is None


# --------------------------------------------------------------------------
# VllmTPUMLAAttention
# --------------------------------------------------------------------------


def _attn_kwargs():
    return dict(
        num_heads=2,
        scale=1.0,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=None,
        kv_lora_rank=8,
        kv_b_proj=MagicMock(),
    )


def test_mla_attention_init_swaps_and_restores_prefill_backend(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(upstream_mla, "get_mla_prefill_backend", sentinel)
    monkeypatch.setattr(selector, "get_mla_prefill_backend", sentinel)
    seen = {}

    def fake_init(self, **kwargs):
        seen["mla"] = upstream_mla.get_mla_prefill_backend(None)
        seen["selector"] = selector.get_mla_prefill_backend(None)
        seen["kwargs"] = kwargs
        self.kv_cache_dtype = "auto"

    monkeypatch.setattr(MLAAttention, "__init__", fake_init)
    attn = mla.VllmTPUMLAAttention(
        **_attn_kwargs(), dcp_q_replicate=True, topk_indices_buffer="buf"
    )
    assert seen["mla"] is mla.TPUDummyMLAPrefillBackend
    assert seen["selector"] is mla.TPUDummyMLAPrefillBackend
    assert "dcp_q_replicate" not in seen["kwargs"]
    assert upstream_mla.get_mla_prefill_backend is sentinel
    assert selector.get_mla_prefill_backend is sentinel
    assert attn.dcp_q_replicate is True
    assert attn.topk_indices_buffer == "buf"
    assert attn.kv_cache_quantized_dtype is None


def test_mla_attention_init_without_prefill_selectors(monkeypatch):
    monkeypatch.delattr(upstream_mla, "get_mla_prefill_backend", raising=False)
    monkeypatch.delattr(selector, "get_mla_prefill_backend", raising=False)

    def fake_init(self, **kwargs):
        self.kv_cache_dtype = "auto"

    monkeypatch.setattr(MLAAttention, "__init__", fake_init)
    attn = mla.VllmTPUMLAAttention(**_attn_kwargs())
    assert not hasattr(upstream_mla, "get_mla_prefill_backend")
    assert not hasattr(selector, "get_mla_prefill_backend")
    assert attn.dcp_q_replicate is False


class _TpuParam(torch.nn.Parameter):
    """A CPU parameter that reports itself as living on a TPU."""

    @property
    def device(self):
        return SimpleNamespace(type="tpu")


class _KvBProj:
    """`named_parameters` reports a direct, an attribute-only and a nested
    entry so every branch of the clearing loop is taken."""

    def __init__(self):
        self.p = torch.nn.Parameter(torch.zeros(1))
        self._parameters = {"weight": self.p}
        self.extra = self.p

    def named_parameters(self):
        return [("weight", self.p), ("extra", self.p), ("sub.weight", self.p)]


class _TorchProxy:
    """`torch` with `torch.device("tpu")` redirected to CPU."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def device(_name):
        return torch.device("cpu")


class _Group:
    def __init__(self, world_size=2):
        self.world_size = world_size
        self.gathered = []

    def all_gather(self, tensor, dim=0):
        tensor = tensor.as_subclass(torch.Tensor)
        self.gathered.append(tensor)
        return torch.cat([tensor, tensor], dim=dim)


H_, NOPE, LAT, V_ = 2, 4, 8, 4


def _pwal_attention(monkeypatch, *, quantize, tpu, dcp_q_replicate, dcp_size, group):
    attn = mla.VllmTPUMLAAttention.__new__(mla.VllmTPUMLAAttention)
    torch.nn.Module.__init__(attn)
    attn.kv_b_proj = _KvBProj()
    attn.prefix = "model.layers.0.self_attn.attn"
    attn.dcp_q_replicate = dcp_q_replicate
    impl = MagicMock()
    impl.dcp_size = dcp_size
    impl.dcp_interleave_size = 4
    impl._get_kv_scales.return_value = (1.0, 2.0, 3.0)
    attn.impl = impl
    torch.manual_seed(0)
    w_uk_t = torch.randn(H_, NOPE, LAT)
    w_uv = torch.randn(H_, LAT, V_)

    def upstream(self, act_dtype):
        self.W_UK_T = w_uk_t.clone()
        self.W_UV = w_uv.clone()

    monkeypatch.setattr(MLAAttention, "process_weights_after_loading", upstream)
    monkeypatch.setattr(mla, "torch", _TorchProxy())
    monkeypatch.setattr(mla, "should_quantize_on_load", lambda prefix: quantize)
    monkeypatch.setattr(mla, "get_dcp_group", lambda: group)
    synced = []
    monkeypatch.setattr(mla, "synchronize_tensors", synced.append)
    if tpu:
        monkeypatch.setattr(
            mla,
            "Parameter",
            lambda t, requires_grad=False: _TpuParam(
                t.as_subclass(torch.Tensor).detach(), requires_grad=requires_grad
            ),
        )
        monkeypatch.setattr(
            mla, "_fresh", lambda t: t.as_subclass(torch.Tensor).clone()
        )
    return attn, impl, synced, w_uk_t, w_uv


@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("dcp_q_replicate", [False, True])
def test_mla_process_weights_after_loading(monkeypatch, quantize, dcp_q_replicate):
    group = _Group()
    attn, impl, synced, w_uk_t, w_uv = _pwal_attention(
        monkeypatch,
        quantize=quantize,
        tpu=True,
        dcp_q_replicate=dcp_q_replicate,
        dcp_size=2,
        group=group,
    )
    attn.process_weights_after_loading(torch.bfloat16)

    if quantize:
        assert attn.W_UK_T.dtype == torch.float8_e4m3fn
        assert attn.W_UK_T_scale.dtype == torch.bfloat16
        assert attn.W_UV_scale.dtype == torch.bfloat16
        q_uk, s_uk = mla.quantize_tensor(w_uk_t, torch.float8_e4m3fn, axis=1)
        torch.testing.assert_close(
            attn.W_UK_T.as_subclass(torch.Tensor).float(), q_uk.float()
        )
        torch.testing.assert_close(
            attn.W_UK_T_scale.as_subclass(torch.Tensor), s_uk.to(torch.bfloat16)
        )
    else:
        assert not hasattr(attn, "W_UK_T_scale")
        torch.testing.assert_close(attn.W_UK_T.as_subclass(torch.Tensor), w_uk_t)
        torch.testing.assert_close(attn.W_UV.as_subclass(torch.Tensor), w_uv)

    if dcp_q_replicate:
        assert attn.W_UK_T_dcp_qrep.shape == (2 * H_, NOPE, LAT)
        assert (attn.W_UK_T_dcp_qrep_scale is not None) is quantize
        assert len(group.gathered) == (2 if quantize else 1)
    else:
        assert attn.W_UK_T_dcp_qrep is None and not group.gathered

    # kv_b_proj is emptied without breaking attribute integrity.
    assert attn.kv_b_proj._parameters == {"weight": None}
    assert not hasattr(attn.kv_b_proj, "extra")

    (tensors,) = synced
    expected_count = 2 + (2 if quantize else 0)
    expected_count += (1 + int(quantize)) if dcp_q_replicate else 0
    assert len(tensors) == expected_count

    impl._build_mla_op.assert_called_once_with(
        attn, q_scale=1.0, k_scale=2.0, v_scale=3.0
    )
    assert attn.mla_op is impl._build_mla_op.return_value
    assert attn.sparse_mla_op is impl._build_sparse_mla_op.return_value
    impl._build_sparse_mla_dcp_op.assert_called_once_with(attn, 2, 4)
    assert attn.sparse_mla_dcp_op is impl._build_sparse_mla_dcp_op.return_value


def test_mla_process_weights_after_loading_on_cpu_skips_sync(monkeypatch):
    attn, impl, synced, _, _ = _pwal_attention(
        monkeypatch,
        quantize=False,
        tpu=False,
        dcp_q_replicate=False,
        dcp_size=1,
        group=None,
    )
    attn.process_weights_after_loading(torch.float32)
    assert not synced
    assert attn.sparse_mla_dcp_op is None
    impl._build_sparse_mla_dcp_op.assert_not_called()


@pytest.mark.parametrize("group", [None, _Group(world_size=1)])
def test_mla_qrep_requires_a_dcp_group(monkeypatch, group):
    attn, _, _, _, _ = _pwal_attention(
        monkeypatch,
        quantize=False,
        tpu=False,
        dcp_q_replicate=True,
        dcp_size=1,
        group=group,
    )
    with pytest.raises(RuntimeError, match="no DCP group"):
        attn.process_weights_after_loading(torch.float32)


# --------------------------------------------------------------------------
# VllmTPUMultiHeadLatentAttentionWrapper
# --------------------------------------------------------------------------


def _modules(indexer=None, is_sparse=False, rotary_emb=None):
    return SimpleNamespace(
        fused_qkv_a_proj=None,
        kv_a_proj_with_mqa=None,
        q_a_layernorm=None,
        q_b_proj=None,
        q_proj=None,
        kv_a_layernorm=None,
        kv_b_proj=None,
        rotary_emb=rotary_emb,
        o_proj=None,
        g_proj=None,
        indexer=indexer,
        indexer_rotary_emb="idx-rope",
        is_sparse=is_sparse,
        topk_indices_buffer=None,
    )


def _construct_wrapper(monkeypatch, modules, **kwargs):
    attention = MagicMock()
    monkeypatch.setattr(mla, "VllmTPUMLAAttention", attention)
    wrapper = mla.VllmTPUMultiHeadLatentAttentionWrapper(
        hidden_size=4,
        num_heads=1,
        scale=1.0,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=2,
        q_lora_rank=None,
        kv_lora_rank=2,
        mla_modules=modules,
        **kwargs,
    )
    return wrapper, attention


@pytest.mark.parametrize("skip_topk", [False, True])
def test_wrapper_init_rebinds_indexer(monkeypatch, skip_topk):
    rebound = SimpleNamespace(topk_tokens=7)
    seen = []

    def rebind(indexer):
        seen.append(indexer)
        return rebound

    monkeypatch.setattr(mla.VllmTPUIndexer, "rebind", rebind)
    original = object()
    wrapper, attention = _construct_wrapper(
        monkeypatch, _modules(indexer=original, is_sparse=True), skip_topk=skip_topk
    )
    assert seen == [original]
    assert wrapper.indexer is rebound
    assert hasattr(wrapper, "topk_tokens") is not skip_topk
    assert attention.call_args.kwargs["indexer"] is rebound


@pytest.mark.parametrize("has_parent_hook", [False, True])
@pytest.mark.parametrize("rope", ["none", "plain", "cached"])
def test_wrapper_process_weights_after_loading(monkeypatch, has_parent_hook, rope):
    calls = []
    if has_parent_hook:
        monkeypatch.setattr(
            MultiHeadLatentAttentionWrapper,
            "process_weights_after_loading",
            lambda self, dtype: calls.append(dtype),
            raising=False,
        )
    else:
        monkeypatch.delattr(
            MultiHeadLatentAttentionWrapper,
            "process_weights_after_loading",
            raising=False,
        )
    rotary = {
        "none": None,
        "plain": SimpleNamespace(),
        "cached": SimpleNamespace(cos_sin_cache=torch.ones(2)),
    }[rope]
    wrapper, _ = _construct_wrapper(monkeypatch, _modules(rotary_emb=rotary))
    monkeypatch.setattr(mla, "torch", _TorchProxy())
    wrapper.process_weights_after_loading(torch.bfloat16)
    assert calls == ([torch.bfloat16] if has_parent_hook else [])
    if rope == "cached":
        assert rotary.cos_sin_cache.device.type == "cpu"


def _forward_wrapper(**overrides):
    wrapper = mla.VllmTPUMultiHeadLatentAttentionWrapper.__new__(
        mla.VllmTPUMultiHeadLatentAttentionWrapper
    )
    torch.nn.Module.__init__(wrapper)
    attrs = dict(
        kimi_preprocess=None,
        kimi_gate_weight="gate-w",
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_nope_head_dim=3,
        qk_rope_head_dim=2,
        qk_head_dim=5,
        num_heads=2,
        v_head_dim=2,
        kv_a_proj_with_mqa=lambda h: (torch.ones((h.shape[0], 6)), None),
        q_proj=lambda h: (torch.ones((h.shape[0], 10)), None),
        kv_a_layernorm=lambda value: value,
        rotary_emb=None,
        is_sparse=False,
        indexer=None,
        skip_topk=False,
        topk_indices_buffer=None,
        dcp_q_replicate=False,
        dcp_group_size=1,
        g_proj=None,
        gate_is_fused=False,
        mla_attn=MagicMock(return_value=torch.ones((3, 4))),
        o_proj=lambda value: (value * 2, None),
    )
    attrs.update(overrides)
    for name, value in attrs.items():
        setattr(wrapper, name, value)
    return wrapper


def test_wrapper_forward_kimi_preprocess_path():
    hidden = torch.ones((3, 8))
    seen = {}

    def kimi(h, fused_w, gate_w, q_norm_w, kv_norm_w, is_prefill_group):
        seen.update(is_prefill_group=is_prefill_group, fused_w=fused_w)
        return (
            torch.full((3, 4), 2.0),
            torch.ones((3, 4)),
            torch.ones((3, 2)),
            torch.zeros((3, 4)),
        )

    wrapper = _forward_wrapper(
        kimi_preprocess=kimi,
        q_lora_rank=4,
        fused_qkv_a_proj=SimpleNamespace(weight="fused-w"),
        q_a_layernorm=SimpleNamespace(weight="qn"),
        kv_a_layernorm=SimpleNamespace(weight="kvn"),
        q_b_proj=lambda q_c: (torch.ones((q_c.shape[0], 10)) * q_c[:, :1], None),
        kv_a_proj_with_mqa=None,
    )
    out = wrapper(torch.arange(3), hidden)
    assert seen == {"is_prefill_group": False, "fused_w": "fused-w"}
    # output_gate = 0 -> sigmoid 0.5; o_proj doubles.
    torch.testing.assert_close(out, torch.ones((3, 4)))
    (q_nope, _q_pe), kv_c_normed, k_pe = wrapper.mla_attn.call_args.args[:3]
    torch.testing.assert_close(q_nope, torch.full((3, 2, 3), 2.0))
    assert kv_c_normed.shape == (3, 4) and k_pe.shape == (3, 1, 2)


def test_wrapper_forward_sparse_indexer_scaling_and_prefill_group():
    hidden = torch.ones((3, 8))
    indexer_calls = []

    def indexer(h, q_c, positions, rope):
        indexer_calls.append((q_c, rope))
        return torch.arange(5 * 2).view(5, 2)

    prefill_group = MagicMock()
    prefill_group.project_reduce_scatter.return_value = "reduced"
    wrapper = _forward_wrapper(
        is_sparse=True, indexer=indexer, indexer_rope_emb="idx-rope"
    )
    out = wrapper(
        torch.arange(3),
        hidden,
        llama_4_scaling=torch.full((3, 1, 1), 3.0),
        prefill_group=prefill_group,
    )
    assert out == "reduced"
    assert indexer_calls == [(None, "idx-rope")]
    kwargs = wrapper.mla_attn.call_args.kwargs
    torch.testing.assert_close(kwargs["topk_indices"], torch.arange(6).view(3, 2))
    (q_nope, q_pe) = wrapper.mla_attn.call_args.args[0]
    torch.testing.assert_close(q_nope, torch.full((3, 2, 3), 3.0))
    torch.testing.assert_close(q_pe, torch.full((3, 2, 2), 3.0))
    attn_out, o_proj = prefill_group.project_reduce_scatter.call_args.args
    assert o_proj is wrapper.o_proj
    torch.testing.assert_close(attn_out, torch.ones((3, 4)))


def test_wrapper_forward_dense_qrep_uses_local_view_only():
    local = MagicMock(side_effect=lambda t: t[:, :1])
    wrapper = _forward_wrapper(
        dcp_q_replicate=True,
        dcp_group_size=1,
        q_proj=SimpleNamespace(),
    )
    wrapper.q_proj = lambda h: (torch.ones((h.shape[0], 10)), None)
    wrapper._q_proj_layer = lambda: SimpleNamespace(_local_view=local)
    wrapper(torch.arange(3), torch.ones((3, 8)))
    assert local.call_count == 2
    assert wrapper.mla_attn.call_args.kwargs["q_dcp_replicated"] is None
    (q_nope, q_pe) = wrapper.mla_attn.call_args.args[0]
    assert q_nope.shape == (3, 1, 3) and q_pe.shape == (3, 1, 2)


# --------------------------------------------------------------------------
# VllmDeepseekScalingRotaryEmbedding.forward_native
# --------------------------------------------------------------------------


def _rope(rotary_dim, head_size, neox, max_pos=8):
    rope = mla.VllmDeepseekScalingRotaryEmbedding.__new__(
        mla.VllmDeepseekScalingRotaryEmbedding
    )
    torch.nn.Module.__init__(rope)
    rope.rotary_dim = rotary_dim
    rope.head_size = head_size
    rope.is_neox_style = neox
    pos = torch.arange(max_pos, dtype=torch.float32)[:, None]
    freqs = pos * (1.0 / (10.0 ** torch.arange(rotary_dim // 2)))
    rope.cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope


def _reference_rope(x, cos, sin, rotary_dim, neox):
    rot, rest = x[..., :rotary_dim], x[..., rotary_dim:]
    cos, sin = cos[:, None, :], sin[:, None, :]
    if neox:
        x1, x2 = rot.chunk(2, dim=-1)
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:
        x1, x2 = rot[..., ::2], rot[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        out = out.flatten(-2)
    return torch.cat([out, rest], dim=-1)


@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("head_size", [4, 6])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_scaling_rope_forward_native_matches_reference(neox, head_size, with_offsets):
    rotary_dim = 4
    rope = _rope(rotary_dim, head_size, neox)
    torch.manual_seed(0)
    positions = torch.tensor([0, 2, 3])
    offsets = torch.tensor([1, 1, 2]) if with_offsets else None
    query = torch.randn(3, 2, head_size)
    key = torch.randn(3, 1, head_size)
    q_out, k_out = rope.forward_native(positions, query, key, offsets)

    pos = positions + offsets if with_offsets else positions
    cos, sin = rope.cos_sin_cache[pos].chunk(2, dim=-1)
    torch.testing.assert_close(
        q_out, _reference_rope(query, cos, sin, rotary_dim, neox)
    )
    torch.testing.assert_close(k_out, _reference_rope(key, cos, sin, rotary_dim, neox))


def test_scaling_rope_forward_native_requires_key():
    with pytest.raises(AssertionError):
        _rope(4, 4, True).forward_native(torch.tensor([0]), torch.zeros(1, 1, 4))
