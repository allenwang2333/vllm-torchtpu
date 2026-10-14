# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek kernels that call tokamax's PyTorch ops, and their fallbacks.

These call sites call ops from `tokamax.experimental.torch_tpu.ops` in place of
in-tree kernels, with `jax_op` scaffolding only for what tokamax does not do:

* DeepSeek-V4 q RMSNorm+RoPE and kv RoPE run `PallasMosaicTpuDeepseekRope`.
* DeepSeek-V4 CSA attention runs its SWA pass in a `jax_op`, then
  `PallasMosaicTpuSparseMla` for the sparse gather-and-attention pass.
* The DeepSeek-V4 indexer runs `PallasMosaicTpuDeepseekRope` (`rope_quant`),
  folds the scales into the weights in torch, then
  `PallasMosaicTpuLightningIndexer`.
* The DeepSeek-V3.2 indexer appends K rows in a `jax_op`, then
  `PallasMosaicTpuLightningIndexer`.

The routing tests fake tokamax and `jax_op` and run anywhere. The parity tests
hold each tokamax path against the in-tree op it replaces, on TPU.
"""

import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_attention_op as csa_op,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
    deepseek_v4_indexer as v4_indexer,
)
from vllm_torchtpu.models.vllm.deepseek_v4 import attention as v4_attention


@pytest.fixture
def tpu():
    """Skips off TPU. A fixture rather than a skipif: the conftest refuses test
    modules that initialize a JAX backend while being collected."""
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernels need a TPU.")
    return torch.device("tpu")


def _recording_jax_op(registered):
    """A `pallas.jax_op` stand-in: an op whose name contains a key of
    `registered` runs that function instead of its JAX one."""

    def jax_op(name, fn, **kwargs):
        (impl,) = [f for key, f in registered.items() if key in name] or [None]

        def op(*args, **kw):
            return impl(*args, **kw)

        op.register_fake = lambda fake: None
        op.kwargs = kwargs
        return op

    return jax_op


# ---------------------------------------------------------------------------
# DeepSeek-V4 CSA attention
# ---------------------------------------------------------------------------


def test_csa_unsupported_reasons(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_USE_TOKAMAX", "1")
    monkeypatch.setattr(csa_op, "tokamax_sparse_mla", object())
    assert csa_op.tokamax_csa_unsupported_reason(None) is None
    assert "mesh" in csa_op.tokamax_csa_unsupported_reason(object())
    monkeypatch.setattr(csa_op, "tokamax_sparse_mla", None)
    assert csa_op.tokamax_csa_unsupported_reason(None) == "tokamax is not importable"
    monkeypatch.setattr(csa_op, "tokamax_sparse_mla", object())
    monkeypatch.setenv("VLLM_TPU_USE_TOKAMAX", "0")
    assert csa_op.tokamax_csa_unsupported_reason(None) == "VLLM_TPU_USE_TOKAMAX=0"


def _build_csa(monkeypatch, swa, sparse_mla, in_tree, **kwargs):
    monkeypatch.setattr(csa_op.pallas, "jax_op", _recording_jax_op({"_swa": swa}))
    monkeypatch.setattr(
        csa_op,
        "tokamax_sparse_mla",
        SimpleNamespace(PallasMosaicTpuSparseMla=sparse_mla, Config=lambda: "config"),
    )
    monkeypatch.setattr(csa_op, "_tokamax_csa_op_cache", {})
    return csa_op.build_tokamax_csa_attention(
        f"pallas::test_csa_{kwargs}",
        in_tree,
        sm_scale=0.1,
        sliding_window=128,
        logical_page_size=16,
        **kwargs,
    )


@pytest.mark.parametrize("two_caches_same_buffer", [False, True])
def test_csa_runs_tokamax_sparse_mla_after_the_swa_pass(
    monkeypatch, two_caches_same_buffer
):
    tokens, heads, dim, topk = 300, 3, 448, 128
    updated_sw_cache = torch.full((2, 16, 128), 7, dtype=torch.int32)
    swa_out = torch.ones(tokens, heads, dim, dtype=torch.bfloat16)
    swa_l, swa_m = torch.ones(tokens, heads), torch.zeros(tokens, heads)
    swa_calls, sparse_calls = [], []

    def swa(*args):
        swa_calls.append(args)
        return updated_sw_cache, swa_out, swa_l, swa_m

    def sparse_mla(*args, **kwargs):
        sparse_calls.append((args, kwargs))
        return torch.full_like(args[0], 3.0)

    op = _build_csa(
        monkeypatch,
        swa,
        sparse_mla,
        lambda *args: pytest.fail("the in-tree op ran with live caches"),
        two_caches_same_buffer=two_caches_same_buffer,
    )
    q = torch.zeros(tokens, heads, dim, dtype=torch.bfloat16)
    sw_cache = torch.zeros(2, 16, 128, dtype=torch.int32)
    main_cache_kv = torch.zeros(2, 16, 128, dtype=torch.int32)
    main_cache_rope = torch.zeros(2, 4, 128, dtype=torch.int32)
    topk_indices = torch.zeros(tokens, topk, dtype=torch.int32)
    page_indices = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    sinks = torch.zeros(heads)
    output, new_sw_cache = op(
        q,
        "new_kv",
        sw_cache,
        "swa_kv_lens",
        "swa_page_indices",
        "swa_cu_q_lens",
        "swa_distribution",
        main_cache_kv,
        "main_kv_lens",
        topk_indices,
        page_indices,
        "main_cu_q_lens",
        "main_distribution",
        sinks,
        main_cache_rope,
    )

    # The SWA pass takes the SWA operands and the sinks; nothing of the main
    # caches.
    ((swa_args,),) = [swa_calls]
    assert swa_args[0] is q and swa_args[2] is sw_cache and swa_args[7] is sinks
    assert swa_args[1:7:5] == ("new_kv", "swa_distribution")
    # tokamax's sparse MLA gets the in-tree kernel's operands, with the SWA
    # partials, and reads the SWA kernel's write when both caches are one.
    ((args, kwargs),) = sparse_calls
    want_nope = updated_sw_cache if two_caches_same_buffer else main_cache_kv
    assert args[0] is q and args[1] is want_nope and args[2] is main_cache_rope
    assert args[3] is topk_indices
    assert torch.equal(args[4], page_indices.flatten())
    assert args[5:7] == ("main_cu_q_lens", "main_distribution")
    assert args[7] is sinks
    assert args[8] is swa_out and args[9] is swa_l and args[10] is swa_m
    assert kwargs == dict(sm_scale=0.1, configs=("config", None))
    assert torch.all(output == 3) and new_sw_cache is updated_sw_cache


def test_csa_profiling_pass_takes_the_in_tree_op(monkeypatch):
    sentinel = object()
    op = _build_csa(
        monkeypatch, None, None, lambda *args: sentinel, two_caches_same_buffer=False
    )
    empty = torch.zeros(0, 1, 1)
    args = [None] * 15
    args[7] = args[14] = empty
    assert op(*args) is sentinel


# ---------------------------------------------------------------------------
# DeepSeek-V4 q and kv RoPE
# ---------------------------------------------------------------------------


def _v4_attention(head_dim=512):
    layer = v4_attention.VllmDeepseekV4MLAAttention.__new__(
        v4_attention.VllmDeepseekV4MLAAttention
    )
    object.__setattr__(layer, "mla_attn", SimpleNamespace(eps=1e-6, head_dim=head_dim))
    return layer


def test_v4_rope_runs_tokamax(monkeypatch):
    calls = []

    def rope(x, positions, cos_sin_cache, **kwargs):
        calls.append((x, positions, cos_sin_cache, kwargs))
        return "rotated"

    monkeypatch.setenv("VLLM_TPU_USE_TOKAMAX", "1")
    monkeypatch.setattr(v4_attention, "get_pallas_op_mesh", lambda: None)
    monkeypatch.setattr(
        v4_attention, "tokamax_rope", SimpleNamespace(PallasMosaicTpuDeepseekRope=rope)
    )
    layer = _v4_attention()
    assert layer.qnorm_rope_op("q", "pos", "cs") == "rotated"
    assert layer.kv_rope_op("kv", "pos", "cs") == "rotated"
    assert calls == [
        ("q", "pos", "cs", dict(mode="qnorm_rope", eps=1e-6)),
        ("kv", "pos", "cs", dict(mode="rope")),
    ]


@pytest.mark.parametrize(
    "env, mesh, tokamax, head_dim",
    [
        ("0", None, True, 512),
        ("1", object(), True, 512),
        ("1", None, False, 512),
        ("1", None, True, 128),
    ],
    ids=["disabled", "mesh", "missing", "hd128"],
)
def test_v4_rope_keeps_the_in_tree_ops(monkeypatch, env, mesh, tokamax, head_dim):
    monkeypatch.setenv("VLLM_TPU_USE_TOKAMAX", env)
    monkeypatch.setattr(v4_attention, "get_pallas_op_mesh", lambda: mesh)
    monkeypatch.setattr(v4_attention, "_pallas_op_cache", {})
    monkeypatch.setattr(v4_attention, "tokamax_rope", object() if tokamax else None)
    built = []
    monkeypatch.setattr(
        v4_attention.pallas,
        "jax_op",
        lambda name, fn, **kwargs: built.append(name)
        or SimpleNamespace(register_fake=lambda f: None),
    )
    layer = _v4_attention(head_dim)
    layer.qnorm_rope_op  # noqa: B018 -- building the op is what is under test
    layer.kv_rope_op  # noqa: B018
    assert len(built) == 2
    assert built[0].startswith("pallas::deepseek_v4_qnorm_rope_e")
    assert built[1] == "pallas::deepseek_v4_kv_rope"


# ---------------------------------------------------------------------------
# DeepSeek-V4 indexer
# ---------------------------------------------------------------------------


def _v4_indexer(topk=512):
    indexer = v4_indexer.VllmDeepseekV4Indexer.__new__(v4_indexer.VllmDeepseekV4Indexer)
    object.__setattr__(indexer, "topk_tokens", topk)
    object.__setattr__(indexer, "compress_ratio", 4)
    object.__setattr__(indexer, "softmax_scale", 128**-0.5)
    object.__setattr__(indexer, "n_head", 64)
    return indexer


def test_v4_indexer_runs_tokamax_rope_then_the_indexer(monkeypatch):
    q_quant = torch.zeros(5, 64, 128)
    q_scales = torch.full((5, 64), 2.0)
    rope_calls, calls = [], []

    def rope(*args, **kwargs):
        rope_calls.append((args, kwargs))
        return q_quant, q_scales

    def lightning_indexer(*args, **kwargs):
        calls.append((args, kwargs))
        return "topk"

    monkeypatch.setenv("VLLM_TPU_USE_TOKAMAX", "1")
    monkeypatch.setattr(v4_indexer, "get_pallas_op_mesh", lambda: None)
    monkeypatch.setattr(
        v4_indexer, "tokamax_rope", SimpleNamespace(PallasMosaicTpuDeepseekRope=rope)
    )
    monkeypatch.setattr(
        v4_indexer,
        "tokamax_lightning_indexer",
        SimpleNamespace(
            PallasMosaicTpuLightningIndexer=lightning_indexer,
            Config=lambda **kwargs: kwargs,
        ),
    )
    monkeypatch.setattr(
        v4_indexer.pallas,
        "jax_op",
        lambda *args, **kwargs: pytest.fail("no jax_op on the tokamax path"),
    )
    indexer = _v4_indexer()
    op = indexer._build_indexer_op()
    weights = torch.full((5, 64), 3.0)
    block_tables = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    assert (
        op("q", "pos", "cos_sin", weights, "cache", "lens", block_tables, "loc", "dist")
        == "topk"
    )
    ((rope_args, rope_kwargs),) = rope_calls
    assert rope_args == ("q", "pos", "cos_sin")
    assert rope_kwargs == dict(mode="rope_quant", quant_dtype=torch.float8_e4m3fn)
    ((args, kwargs),) = calls
    assert args[0] is q_quant
    # The weights with the softmax scale, head count and query scales folded
    # in, as `_indexer_jax` does.
    want = weights * indexer.softmax_scale * indexer.n_head**-0.5 * q_scales
    assert torch.equal(args[1], want)
    assert args[2:4] == ("cache", "lens")
    assert torch.equal(args[4], block_tables.flatten())
    assert args[5:] == ("loc", "dist")
    assert kwargs == dict(
        k=512, compression_ratio=4, config=v4_indexer._STREAMIDX_TUNING
    )


@pytest.mark.parametrize(
    "env, mesh, tokamax",
    [("0", None, True), ("1", object(), True), ("1", None, False)],
    ids=["disabled", "mesh", "missing"],
)
def test_v4_indexer_keeps_the_in_tree_op(monkeypatch, env, mesh, tokamax):
    monkeypatch.setenv("VLLM_TPU_USE_TOKAMAX", env)
    monkeypatch.setattr(v4_indexer, "get_pallas_op_mesh", lambda: mesh)
    monkeypatch.setattr(v4_indexer, "_indexer_op_cache", {})
    monkeypatch.setattr(
        v4_indexer, "tokamax_lightning_indexer", object() if tokamax else None
    )
    built = []
    monkeypatch.setattr(
        v4_indexer.pallas,
        "jax_op",
        lambda name, fn, **kwargs: built.append(name)
        or SimpleNamespace(register_fake=lambda f: None),
    )
    _v4_indexer()._build_indexer_op()
    assert len(built) == 1 and built[0].startswith("pallas::deepseek_v4_indexer_k512")


# ---------------------------------------------------------------------------
# DeepSeek-V3.2 indexer
# ---------------------------------------------------------------------------


def _v32_indexer(topk=256):
    indexer = mla_attention_op.VllmTPUSparseAttnIndexer.__new__(
        mla_attention_op.VllmTPUSparseAttnIndexer
    )
    indexer.topk_tokens = topk
    indexer.k_cache = SimpleNamespace(prefix="model.layers.0.indexer")
    return indexer


def test_v32_indexer_appends_then_runs_tokamax(monkeypatch):
    # The head-along-sublane cache shape below. Drop a value an earlier
    # monkeypatch undo may have left, so `envs` reads the variable.
    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "has")
    if "VLLM_TPU_INDEXER_KV_CACHE_MODE" in vars(mla_attention_op.envs):
        monkeypatch.delattr(mla_attention_op.envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE")
    new_cache = torch.full((4, 32, 4, 256), 9, dtype=torch.uint8)
    calls = []

    def lightning_indexer(*args, **kwargs):
        calls.append((args, kwargs))
        return "topk"

    monkeypatch.setenv("VLLM_TPU_USE_TOKAMAX", "1")
    monkeypatch.setattr(
        mla_attention_op.pallas,
        "jax_op",
        _recording_jax_op({"k_append": lambda *args: new_cache}),
    )
    monkeypatch.setattr(
        mla_attention_op,
        "tokamax_lightning_indexer",
        SimpleNamespace(
            PallasMosaicTpuLightningIndexer=lightning_indexer,
            Config=lambda **kwargs: kwargs,
        ),
    )
    op = _v32_indexer()._build_streamidx_op()

    q_bytes = torch.zeros(6, 32, 128, dtype=torch.uint8)
    cache = torch.zeros(4, 32, 4, 256, dtype=torch.uint8)
    k_packed = torch.zeros(6, 129, dtype=torch.uint8)
    page_indices = torch.arange(4, dtype=torch.int32).reshape(2, 2)
    assert (
        op(q_bytes, "w", cache, k_packed, "lens", page_indices, "loc", "dist") == "topk"
    )
    # The appended K rows land in the runner's cache.
    assert torch.equal(cache, new_cache)
    ((args, kwargs),) = calls
    assert args[0].dtype == torch.float8_e4m3fn
    assert args[1] == "w" and args[2] is new_cache and args[3] == "lens"
    assert torch.equal(args[4], page_indices.reshape(-1))
    assert args[5:] == ("loc", "dist")
    assert kwargs["k"] == 256 and kwargs["compression_ratio"] == 1
    assert kwargs["kv_layout"] == "HEAD_ALONG_SUBLANE"
    assert kwargs["config"]["num_kv_pages_per_block"] == (2, 2, 2)


# ---------------------------------------------------------------------------
# Parity on TPU
# ---------------------------------------------------------------------------


def _to_torch(x, device):
    """A fresh device tensor with storage of its own, so it can be donated."""
    x = np.asarray(x)
    if x.dtype == jnp.bfloat16:
        return torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).to(device)
    return torch.from_numpy(x).to(device)


def _quantize_fp8(x, block):
    """Block fp8 quantization with power-of-two (e8m0) scales."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    blocked = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked / scale).astype(jnp.float8_e4m3fn).reshape(x.shape)
    return q, jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)


def _bytes(x):
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    return b.reshape(*x.shape[:-1], -1) if b.ndim > x.ndim else b


def _csa_caches(rng, num_pages, page_size):
    """DSv4 CSA caches: fp8 NoPE records and bf16 RoPE, as int32 words."""
    kv = jnp.asarray(rng.standard_normal((num_pages, page_size, 512), np.float32))
    nope = kv[..., :448]
    # One e8m0 scale per lane of a 64-lane period (sparse_mla._dequant_dsv4_fp8).
    periods = nope.reshape(num_pages, page_size, 7, 64).swapaxes(-1, -2)
    fp8, scale = _quantize_fp8(periods, 7)
    fp8 = fp8.swapaxes(-1, -2).reshape(num_pages, page_size, 448)
    scale = scale.reshape(num_pages, page_size, 64)
    slabs = jnp.concatenate([_bytes(fp8), _bytes(scale)], axis=-1)
    cache_nope = csa_cache_layout.slabs_to_words(
        slabs.reshape(num_pages, page_size, 4, 128)
    )
    rope = kv[..., 448:].astype(jnp.bfloat16).reshape(-1, 64)
    cache_rope = csa_cache_layout.encode_rope(rope).reshape(
        num_pages, page_size // 4, 128
    )
    return cache_nope, cache_rope


def _fake_swa(q, new_kv, sw_cache, *args, swa_only, **kwargs):
    """Deterministic stand-in for the SWA kernel, which both paths share."""
    del new_kv, args, kwargs
    assert not swa_only
    qf = q.astype(jnp.float32)
    bumped = sw_cache if sw_cache.dtype == jnp.int32 else sw_cache + 1
    return (
        (q * 0.25).astype(q.dtype),
        bumped,
        1.0 + jnp.mean(jnp.abs(qf), axis=-1),
        jnp.max(qf, axis=-1) * 0.1,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("two_caches_same_buffer", [False, True])
def test_csa_tokamax_matches_in_tree(tpu, monkeypatch, two_caches_same_buffer):
    pytest.importorskip("tokamax.experimental.torch_tpu.ops.sparse_mla")
    assert csa_op.tokamax_sparse_mla is not None
    # Both paths call `_run_swa` identically; stubbing it isolates the sparse
    # half, which is what tokamax's op replaces.
    monkeypatch.setattr(csa_op, "_run_swa", _fake_swa)
    monkeypatch.setattr(csa_op, "_tokamax_csa_op_cache", {})
    device = tpu
    rng = np.random.default_rng(0)

    heads, dim, topk, page_size = 16, 512, 512, 64
    q_lens = [1, 1, 140, 147]  # 2 decodes, then 2 prefills: 289 real tokens
    kv_lens = [700, 300, 900, 600]  # compressed positions per sequence
    max_tokens, max_seqs, pages_per_seq = 320, 6, 16
    num_pages = max_seqs * pages_per_seq

    cache_nope, cache_rope = _csa_caches(rng, num_pages, page_size)
    page_indices = rng.permutation(num_pages).astype(np.int32)
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cu_q_lens = np.pad(cu_q_lens, (0, max_seqs + 1 - len(cu_q_lens)), mode="edge")
    topk_indices = np.full((max_tokens, topk), -1, np.int32)
    row = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        for _ in range(q_len):
            pick = rng.permutation(kv_len)[:topk]
            topk_indices[row, : len(pick)] = pick
            row += 1

    q = (rng.standard_normal((max_tokens, heads, dim)) * 0.5).astype(np.float32)
    sinks = rng.random(heads, dtype=np.float32)
    sw_cache = (
        np.asarray(cache_nope)
        if two_caches_same_buffer
        else rng.integers(0, 200, (8, 16, 4, 128), dtype=np.uint8)
    )
    # With one buffer, the sparse pass must read the SWA kernel's output; a
    # zero main cache would show it reading the other.
    main_nope = np.zeros_like(cache_nope) if two_caches_same_buffer else cache_nope

    def operands():
        return (
            _to_torch(q, device).to(torch.bfloat16),
            torch.zeros(max_tokens, dim, dtype=torch.bfloat16, device=device),
            _to_torch(sw_cache, device),
            torch.ones(max_seqs, dtype=torch.int32, device=device),
            torch.zeros(max_seqs, 1, dtype=torch.int32, device=device),
            _to_torch(cu_q_lens, device),
            torch.tensor([2, 2, 4], dtype=torch.int32, device=device),
            _to_torch(main_nope, device),
            _to_torch(np.asarray(kv_lens + [0, 0], np.int32), device),
            _to_torch(topk_indices, device),
            _to_torch(page_indices.reshape(max_seqs, pages_per_seq), device),
            _to_torch(cu_q_lens, device),
            torch.tensor([2, 2, 4], dtype=torch.int32, device=device),
            _to_torch(sinks, device),
            _to_torch(cache_rope, device),
        )

    kwargs = dict(
        sm_scale=dim**-0.5,
        sliding_window=128,
        logical_page_size=16,
        two_caches_same_buffer=two_caches_same_buffer,
    )
    name = f"pallas::test_dsv4_csa_parity_{int(two_caches_same_buffer)}"
    in_tree = csa_op.pallas.jax_op(
        name,
        functools.partial(csa_op._attention_csa, swa_only=False, **kwargs),
        donate_argnums=(2,),
    )
    tokamax = csa_op.build_tokamax_csa_attention(f"{name}_tokamax", in_tree, **kwargs)

    want_out, want_cache = in_tree(*operands())
    want_out, want_cache = want_out.cpu(), want_cache.cpu()
    real = int(cu_q_lens[-1])

    # Eagerly, and compiled (fullgraph, so Dynamo traces every op call).
    from torch_tpu._internal import compile as tpu_compile

    compiled = torch.compile(tokamax, fullgraph=True, backend=tpu_compile.TpuBackend())
    for mode, fn in (("eager", tokamax), ("compiled", compiled)):
        got_out, got_cache = fn(*operands())
        torch.testing.assert_close(
            got_out[:real].float().cpu(),
            want_out[:real].float(),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, mode=mode: f"{mode}: {m}",
        )
        assert torch.equal(got_cache.cpu(), want_cache), mode

    # vLLM traces the model with the token count symbolic, so every op's fake
    # has to take a symbolic size.
    from torch.fx.experimental.proxy_tensor import make_fx

    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        graph = make_fx(tokamax, tracing_mode="symbolic")(*operands())
    fake_out = [n for n in graph.graph.nodes if n.op == "output"][0].args[0][0]
    assert fake_out.meta["val"].shape[1:] == (heads, dim)


def _indexer_cache(rng, num_pages, page_size, head_dim=128):
    """fp8 K records `[fp8 x D | e8m0 scale | pad]`, 4 slots to a row."""
    keys = jnp.asarray(
        rng.standard_normal((num_pages, page_size, head_dim), np.float32)
    )
    fp8, scale = _quantize_fp8(keys, head_dim)
    record = jnp.concatenate([_bytes(fp8), _bytes(scale)], axis=-1)
    record = jnp.pad(record, ((0, 0), (0, 0), (0, 256 - record.shape[-1])))
    return np.asarray(record).reshape(num_pages, page_size // 4, 4, 256)


def _reference_scores(q, weights, cache, seq_lens, block_table, q_lens, ratio):
    """Exact indexer scores per query over its causal window of cached K."""
    num_pages, page_size = cache.shape[0], cache.shape[1] * 4
    records = jnp.asarray(cache.reshape(num_pages, page_size, -1))
    keys = jax.lax.bitcast_convert_type(records[..., :128], jnp.float8_e4m3fn)
    scales = jax.lax.bitcast_convert_type(records[..., 128], jnp.float8_e8m0fnu)
    keys = np.asarray(keys.astype(jnp.float32) * scales.astype(jnp.float32)[..., None])
    q, weights = np.asarray(q, np.float32), np.asarray(weights, np.float32)
    scores, token = [], 0
    for table, q_len, seq_len in zip(block_table, q_lens, seq_lens):
        seq_keys = keys[table].reshape(-1, keys.shape[-1])
        for i in range(q_len):
            visible = min(seq_len // ratio, (seq_len - q_len + i) // ratio + 1)
            logits = np.maximum(q[token] @ seq_keys[:visible].T, 0)
            scores.append(weights[token] @ logits)
            token += 1
    return scores


def _same_selection(got, want, scores, k):
    """Both select the top-k rows per query, up to ties.

    Neither kernel promises an order, and each may break a near-tie at the
    k-th score its own way, so a row the two disagree on must score within
    rounding of the k-th largest score. The kernels round scores to about bf16
    precision (2**-8) on the way, hence the 1% bound.
    """
    got, want = got.cpu().numpy(), want.cpu().numpy()
    assert got.shape == want.shape
    same = total = 0
    for row, (g, w, s) in enumerate(zip(got, want, scores)):
        g, w = set(g[g >= 0].tolist()), set(w[w >= 0].tolist())
        assert len(g) == len(w) == min(k, len(s)), row
        kth = np.sort(s)[-len(w)]
        for idx in g ^ w:
            assert abs(s[idx] - kth) <= 1e-2 * abs(kth), (row, idx, s[idx], kth)
        same, total = same + len(g & w), total + len(w)
    print(f"top-k selection overlap: {same}/{total}")


@pytest.mark.tpu_test
def test_v4_indexer_tokamax_matches_in_tree(tpu, monkeypatch):
    pytest.importorskip("tokamax.experimental.torch_tpu.ops.lightning_indexer")
    pytest.importorskip("tokamax.experimental.torch_tpu.ops.deepseek_rope")
    assert v4_indexer.tokamax_rope is not None
    monkeypatch.setattr(v4_indexer, "get_pallas_op_mesh", lambda: None)
    device = tpu
    rng = np.random.default_rng(1)

    # Over k=512 compressed positions visible, so the top-k really selects.
    n_head, head_dim, page_size, pages_per_seq = 64, 128, 64, 20
    q_lens, seq_lens = [1, 1, 60, 62], [2600, 3300, 4100, 3000]
    max_tokens, num_seqs = 128, len(q_lens)
    num_pages = num_seqs * pages_per_seq
    positions = np.zeros(max_tokens, np.int32)
    row = 0
    for q_len, seq_len in zip(q_lens, seq_lens):
        positions[row : row + q_len] = np.arange(seq_len - q_len, seq_len)
        row += q_len
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((max_tokens, n_head, head_dim), np.float32))
    q = q.astype(jnp.bfloat16)
    # Rows for every position above: an out-of-range read would differ between
    # the two compilations.
    cos_sin_cache = rng.uniform(-1, 1, (8192, 64)).astype(np.float32)
    weights = rng.uniform(0.25, 1.75, (max_tokens, n_head)).astype(np.float32)
    cache = _indexer_cache(rng, num_pages, page_size)
    block_table = rng.permutation(num_pages).astype(np.int32)
    block_table = block_table.reshape(num_seqs, pages_per_seq)
    operands = (
        _to_torch(q, device),
        _to_torch(positions, device),
        _to_torch(cos_sin_cache, device),
        _to_torch(weights, device),
        _to_torch(cache, device),
        _to_torch(np.asarray(seq_lens, np.int32), device),
        _to_torch(block_table, device),
        _to_torch(cu_q_lens, device),
        torch.tensor([2, 2, 4], dtype=torch.int32, device=device),
    )

    indexer = _v4_indexer()
    got = indexer._build_tokamax_indexer_op()(*operands)
    monkeypatch.setattr(
        v4_indexer, "_tokamax_indexer_unsupported_reason", lambda m: "test"
    )
    want = indexer._build_indexer_op()(*operands)
    # What the in-tree kernel scores: its fp8 queries and folded weights.
    q_quant, q_scales = rope_quant(
        q, jnp.asarray(positions), jnp.asarray(cos_sin_cache)
    )
    folded = weights * indexer.softmax_scale * n_head**-0.5 * q_scales
    scores = _reference_scores(
        q_quant.astype(jnp.float32), folded, cache, seq_lens, block_table, q_lens, 4
    )
    real = int(cu_q_lens[-1])
    _same_selection(got[:real], want[:real], scores, k=indexer.topk_tokens)


@pytest.mark.tpu_test
def test_v32_indexer_tokamax_matches_in_tree(tpu, monkeypatch):
    pytest.importorskip("tokamax.experimental.torch_tpu.ops.lightning_indexer")
    assert mla_attention_op.tokamax_lightning_indexer is not None
    # The head-along-sublane cache `_indexer_cache` builds. Drop a value an
    # earlier monkeypatch undo may have left, so `envs` reads the variable.
    monkeypatch.setenv("VLLM_TPU_INDEXER_KV_CACHE_MODE", "has")
    if "VLLM_TPU_INDEXER_KV_CACHE_MODE" in vars(mla_attention_op.envs):
        monkeypatch.delattr(mla_attention_op.envs, "VLLM_TPU_INDEXER_KV_CACHE_MODE")
    device = tpu
    rng = np.random.default_rng(2)

    heads, head_dim, page_size, pages_per_seq = 32, 128, 128, 4
    q_lens, seq_lens = [1, 2, 37], [300, 410, 377]
    num_tokens, num_seqs = int(sum(q_lens)), len(q_lens)
    num_pages = num_seqs * pages_per_seq
    q_fp8, _ = _quantize_fp8(
        jnp.asarray(rng.standard_normal((num_tokens, heads, head_dim), np.float32)),
        head_dim,
    )
    k_fp8, k_scale = _quantize_fp8(
        jnp.asarray(rng.standard_normal((num_tokens, head_dim), np.float32)), head_dim
    )
    k_packed = jnp.concatenate([_bytes(k_fp8), _bytes(k_scale)], axis=-1)
    cache = _indexer_cache(rng, num_pages, page_size)
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)

    weights = rng.uniform(0.25, 1.75, (num_tokens, heads)).astype(np.float32)
    block_table = rng.permutation(num_pages).astype(np.int32)

    def operands():
        return (
            _to_torch(_bytes(q_fp8), device),
            _to_torch(weights, device),
            _to_torch(cache, device),
            _to_torch(k_packed, device),
            _to_torch(np.asarray(seq_lens, np.int32), device),
            _to_torch(block_table, device),
            _to_torch(cu_q_lens, device),
            torch.tensor([1, 1, 3], dtype=torch.int32, device=device),
        )

    got_args = operands()
    got = _v32_indexer()._build_streamidx_op()(*got_args)
    want_args = operands()
    monkeypatch.setattr(
        mla_attention_op.VllmTPUSparseAttnIndexer,
        "_tokamax_indexer_unsupported_reason",
        staticmethod(lambda: "test"),
    )
    want = _v32_indexer()._build_streamidx_op()(*want_args)

    # Both wrote this step's K rows into the cache they were handed.
    new_cache = got_args[2].cpu()
    assert torch.equal(new_cache, want_args[2].cpu())
    scores = _reference_scores(
        q_fp8.astype(jnp.float32),
        weights,
        new_cache.numpy(),
        seq_lens,
        block_table.reshape(num_seqs, pages_per_seq),
        q_lens,
        1,
    )
    _same_selection(got, want, scores, k=256)


@pytest.mark.tpu_test
def test_v4_rope_tokamax_matches_in_tree(tpu, monkeypatch):
    pytest.importorskip("tokamax.experimental.torch_tpu.ops.deepseek_rope")
    assert v4_attention.tokamax_rope is not None
    monkeypatch.setattr(v4_attention, "get_pallas_op_mesh", lambda: None)
    device = tpu
    rng = np.random.default_rng(3)
    tokens, heads, head_dim = 96, 16, 512
    q = _to_torch(rng.standard_normal((tokens, heads, head_dim), np.float32), device)
    kv = _to_torch(rng.standard_normal((tokens, head_dim), np.float32), device)
    q, kv = q.to(torch.bfloat16), kv.to(torch.bfloat16)
    positions = _to_torch(rng.integers(0, 4096, tokens).astype(np.int32), device)
    cos_sin_cache = _to_torch(rng.uniform(-1, 1, (4096, 64)).astype(np.float32), device)

    def run(layer):
        return (
            layer.qnorm_rope_op(q, positions, cos_sin_cache).float().cpu(),
            layer.kv_rope_op(kv, positions, cos_sin_cache).float().cpu(),
        )

    got = run(_v4_attention())
    monkeypatch.setattr(
        v4_attention, "_tokamax_rope_unsupported_reason", lambda *args: "test"
    )
    want = run(_v4_attention())
    for name, g, w in zip(("qnorm_rope", "rope"), got, want):
        torch.testing.assert_close(
            g, w, atol=2e-2, rtol=2e-2, msg=lambda m, name=name: f"{name}: {m}"
        )
