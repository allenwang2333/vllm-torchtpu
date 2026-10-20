# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the pure-JAX TurboQuant KV-cache path.

The numpy helpers below re-implement vLLM's Triton TurboQuant store and
decode kernels (`vllm/v1/attention/ops/triton_turboquant_{store,decode}.py`)
element by element, independently of the JAX code under test, so the byte
layout is checked against upstream's semantics rather than against itself.
"""

import math

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TQ_PRESETS,
    TurboQuantConfig,
)

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    ref_ragged_paged_attention,
)
from vllm_torchtpu.layers.core import turboquant as tq

PRESETS = sorted(TQ_PRESETS)


def _layout_and_tables(preset: str, head_dim: int):
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim)
    layout = tq.TQLayout.from_tq_config(cfg)
    centroids = get_centroids(head_dim, cfg.centroid_bits).numpy()
    return cfg, layout, tq.make_tables(layout, centroids)


# ======================================================================================
# Independent numpy reference of the Triton kernels
# ======================================================================================


def _np_pack(idx: np.ndarray, bits: int) -> np.ndarray:
    """Triton store packing: 4-bit pairs (low nibble first) / 3-bit 8-in-24."""
    idx = idx.astype(np.uint32)
    lead, d = idx.shape[:-1], idx.shape[-1]
    if bits == 4:
        pairs = idx.reshape(*lead, d // 2, 2)
        return ((pairs[..., 0] & 0xF) | ((pairs[..., 1] & 0xF) << 4)).astype(np.uint8)
    assert bits == 3
    grp = idx.reshape(*lead, d // 8, 8)
    word = np.zeros(grp.shape[:-1], np.uint32)
    for i in range(8):
        word |= (grp[..., i] & 0x7) << (3 * i)
    out = np.stack([(word >> (8 * j)) & 0xFF for j in range(3)], axis=-1)
    return out.reshape(*lead, d // 8 * 3).astype(np.uint8)


def _np_unpack(b: np.ndarray, bits: int, d: int) -> np.ndarray:
    """Triton decode unpacking: 16-bit window read at `d * bits // 8`."""
    offs = np.arange(d) * bits
    byte_idx, shift = offs // 8, offs % 8
    b = np.concatenate([b, np.zeros((*b.shape[:-1], 1), np.uint8)], axis=-1)
    raw16 = b[..., byte_idx].astype(np.int32) | (
        b[..., byte_idx + 1].astype(np.int32) << 8
    )
    return (raw16 >> shift) & ((1 << bits) - 1)


def _np_f16_bytes(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).view(np.uint8).reshape(*x.shape, 2)


def _np_quantize(k, v, cfg, centroids) -> np.ndarray:
    k = k.astype(np.float32)
    v = v.astype(np.float32)
    d = cfg.head_dim
    if cfg.key_fp8:
        key_bytes = k.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    else:
        norm = np.sqrt(np.sum(k.astype(np.float64) ** 2, -1, keepdims=True))
        h = tq.hadamard_matrix(d).astype(np.float64)
        y = (k / (norm + 1e-8)) @ h
        c = np.sort(centroids.astype(np.float64))
        mid = (c[:-1] + c[1:]) / 2
        idx = np.searchsorted(mid, y, side="right")
        key_bytes = np.concatenate(
            [_np_pack(idx, cfg.key_quant_bits), _np_f16_bytes(norm[..., 0])], -1
        )
    levels = (1 << cfg.value_quant_bits) - 1
    vmin = v.min(-1, keepdims=True)
    scale = (v.max(-1, keepdims=True) - vmin) / np.float32(levels)
    scale = np.where(scale > 1e-8, scale, np.float32(1e-8)).astype(np.float32)
    q = np.clip(((v - vmin) / scale + np.float32(0.5)).astype(np.int32), 0, levels)
    value_bytes = np.concatenate(
        [
            _np_pack(q, cfg.value_quant_bits),
            _np_f16_bytes(scale[..., 0]),
            _np_f16_bytes(vmin[..., 0]),
        ],
        -1,
    )
    slots = np.concatenate([key_bytes, value_bytes], -1)
    pad = cfg.slot_size_aligned - cfg.slot_size
    return np.pad(slots, [(0, 0)] * (slots.ndim - 1) + [(0, pad)])


def _np_dequantize(slots, cfg, centroids):
    d = cfg.head_dim
    if cfg.key_fp8:
        key = slots[..., :d].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    else:
        mse = math.ceil(d * cfg.key_quant_bits / 8)
        c = np.sort(centroids)[_np_unpack(slots[..., :mse], cfg.key_quant_bits, d)]
        if cfg.norm_correction:
            c = c / np.sqrt(np.sum(c * c, -1, keepdims=True) + 1e-16)
        norm = slots[..., mse : mse + 2].copy().view(np.float16).astype(np.float32)
        key = norm * c
    off = cfg.key_packed_size
    vdb = math.ceil(d * cfg.value_quant_bits / 8)
    q = _np_unpack(slots[..., off : off + vdb], cfg.value_quant_bits, d)
    sc = slots[..., off + vdb : off + vdb + 2].copy().view(np.float16)
    zr = slots[..., off + vdb + 2 : off + vdb + 4].copy().view(np.float16)
    value = q * sc.astype(np.float32) + zr.astype(np.float32)
    return key, value


# ======================================================================================
# Tests
# ======================================================================================


@pytest.mark.parametrize("bits", [3, 4])
def test_pack_unpack_roundtrip_and_layout(bits):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 1 << bits, size=(5, 3, 128))
    packed = np.asarray(tq.pack_bits(jnp.asarray(idx), bits))
    np.testing.assert_array_equal(packed, _np_pack(idx, bits))
    np.testing.assert_array_equal(
        np.asarray(tq.unpack_bits(jnp.asarray(packed), bits, 128)), idx
    )


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("head_dim", [128, 256])
def test_layout_matches_vllm_config(preset, head_dim):
    cfg, layout, _ = _layout_and_tables(preset, head_dim)
    assert layout.key_packed_size == cfg.key_packed_size
    assert layout.value_packed_size == cfg.value_packed_size
    assert layout.slot_size == cfg.slot_size
    assert layout.slot_size_aligned == cfg.slot_size_aligned


@pytest.mark.parametrize("preset", PRESETS)
def test_quantize_matches_triton_layout(preset):
    head_dim = 128
    cfg, layout, tables = _layout_and_tables(preset, head_dim)
    rng = np.random.default_rng(1)
    k = rng.standard_normal((64, 4, head_dim)).astype(np.float32) * 2
    v = rng.standard_normal((64, 4, head_dim)).astype(np.float32)

    got = np.asarray(
        jax.jit(tq.quantize_kv, static_argnums=(2, 3))(
            jnp.asarray(k), jnp.asarray(v), layout, tables
        )
    )
    want = _np_quantize(k, v, cfg, tables.centroids)
    assert got.shape == want.shape == (64, 4, cfg.slot_size_aligned)

    kps = cfg.key_packed_size
    # Values (and FP8 keys) are elementwise f32 math: bit exact.
    np.testing.assert_array_equal(got[..., kps:], want[..., kps:])
    if cfg.key_fp8:
        np.testing.assert_array_equal(got[..., :kps], want[..., :kps])
    else:
        mse = layout.key_mse_bytes
        # The norm is bit exact; MSE indices come from a f32 GEMM so a value
        # sitting on a midpoint may land in the neighbouring bucket.
        np.testing.assert_array_equal(got[..., mse:kps], want[..., mse:kps])
        gi = _np_unpack(got[..., :mse], cfg.key_quant_bits, head_dim)
        wi = _np_unpack(want[..., :mse], cfg.key_quant_bits, head_dim)
        assert np.abs(gi - wi).max() <= 1
        assert np.mean(gi != wi) < 1e-3


@pytest.mark.parametrize("preset", PRESETS)
def test_dequantize_matches_triton_decode(preset):
    head_dim = 128
    cfg, layout, tables = _layout_and_tables(preset, head_dim)
    rng = np.random.default_rng(2)
    k = rng.standard_normal((32, 2, head_dim)).astype(np.float32)
    v = rng.standard_normal((32, 2, head_dim)).astype(np.float32)
    slots = _np_quantize(k, v, cfg, tables.centroids)

    gk, gv = jax.jit(tq.dequantize_kv, static_argnums=(1, 2))(
        jnp.asarray(slots), layout, tables
    )
    wk, wv = _np_dequantize(slots, cfg, tables.centroids)
    np.testing.assert_allclose(np.asarray(gk), wk, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gv), wv, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("preset", PRESETS)
def test_rotated_scores_approximate_exact_scores(preset):
    """(q H) . dequant(k) approximates q . k, within the preset's budget."""
    head_dim = 128
    _, layout, tables = _layout_and_tables(preset, head_dim)
    rng = np.random.default_rng(3)
    k = rng.standard_normal((256, 1, head_dim)).astype(np.float32)
    q = rng.standard_normal((head_dim,)).astype(np.float32)
    slots = tq.quantize_kv(jnp.asarray(k), jnp.asarray(k), layout, tables)
    dk, _ = tq.dequantize_kv(slots, layout, tables)
    q_in = q if layout.key_fp8 else np.asarray(tq.rotate(q, tables.hadamard))
    approx = np.asarray(dk)[:, 0] @ q_in
    exact = k[:, 0] @ q
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    # FP8 keys are near lossless; 3-bit MSE keys are the most aggressive.
    budget = {8: 0.05, 4: 0.15, 3: 0.25}[layout.key_bits]
    assert rel < budget, rel


def test_slot_mapping_matches_python():
    page_size, pages_per_seq, max_num_seqs = 4, 3, 4
    kv_lens = np.array([5, 9, 3, 0], np.int32)
    q_lens = np.array([1, 4, 3], np.int32)
    cu = np.array([0, 1, 5, 8, 8], np.int32)
    rng = np.random.default_rng(4)
    page_indices = rng.permutation(32)[: max_num_seqs * pages_per_seq].astype(np.int32)
    got = np.asarray(
        tq.compute_slot_mapping(
            12,
            jnp.asarray(kv_lens),
            jnp.asarray(page_indices),
            jnp.asarray(cu),
            jnp.asarray([1, 1, 3], jnp.int32),
            page_size,
        )
    )
    want = []
    for s, ql in enumerate(q_lens):
        for i in range(ql):
            pos = kv_lens[s] - ql + i
            page = page_indices[s * pages_per_seq + pos // page_size]
            want.append(page * page_size + pos % page_size)
    want += [-1] * (12 - len(want))
    np.testing.assert_array_equal(got, want)


def _attention_case(preset, dtype=jnp.float32, page_size=16):
    head_dim, pages_per_seq, max_num_seqs = 128, 4, 4
    num_q_heads, num_kv_heads, num_pages, num_tokens = 4, 2, 24, 48
    cfg, layout, tables = _layout_and_tables(preset, head_dim)
    rng = np.random.default_rng(5)

    # One decode, two prefill chunks continuing cached history, one new prefill.
    kv_lens = np.array([33, 20, 40, 7], np.int32)
    q_lens = np.array([1, 12, 16, 7], np.int32)
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    distribution = np.array([1, 4, 4], np.int32)
    page_indices = rng.permutation(num_pages)[: max_num_seqs * pages_per_seq].astype(
        np.int32
    )

    # Pre-populate the history of every sequence in the TQ cache.
    hist_k = rng.standard_normal((max_num_seqs, 64, num_kv_heads, head_dim))
    hist_v = rng.standard_normal((max_num_seqs, 64, num_kv_heads, head_dim))
    cache = np.zeros(
        (num_pages, page_size, num_kv_heads, cfg.slot_size_aligned), np.uint8
    )
    for s in range(max_num_seqs):
        n_hist = kv_lens[s] - q_lens[s]
        slots = _np_quantize(
            hist_k[s, :n_hist], hist_v[s, :n_hist], cfg, tables.centroids
        )
        for p in range(n_hist):
            page = page_indices[s * pages_per_seq + p // page_size]
            cache[page, p % page_size] = slots[p]

    q = rng.standard_normal((num_tokens, num_q_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((num_tokens, num_kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((num_tokens, num_kv_heads, head_dim)).astype(np.float32)
    return dict(
        cfg=cfg,
        layout=layout,
        tables=tables,
        cache=cache,
        q=q,
        k=k,
        v=v,
        kv_lens=kv_lens,
        q_lens=q_lens,
        cu=cu,
        distribution=distribution,
        page_indices=page_indices,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        dtype=dtype,
    )


def _expected_attention(c, sm_scale):
    """History from the (numpy-dequantized) TQ cache, current tokens exact."""
    cfg, tables = c["cfg"], c["tables"]
    h = tq.hadamard_matrix(cfg.head_dim)
    group = c["q"].shape[1] // c["k"].shape[1]
    outs = []
    for s, ql in enumerate(c["q_lens"]):
        kv_len, start = c["kv_lens"][s], c["cu"][s]
        n_hist = kv_len - ql
        pages = c["page_indices"][s * c["pages_per_seq"] :][
            : math.ceil(kv_len / c["page_size"])
        ]
        flat = c["cache"][pages].reshape(-1, *c["cache"].shape[2:])[:n_hist]
        dk, dv = _np_dequantize(flat, cfg, tables.centroids)
        if not cfg.key_fp8:
            dk = dk @ h  # back to the original space (H is its own inverse)
        keys = np.concatenate([dk, c["k"][start : start + ql]], 0)
        vals = np.concatenate([dv, c["v"][start : start + ql]], 0)
        keys = np.repeat(keys, group, axis=1)
        vals = np.repeat(vals, group, axis=1)
        qs = c["q"][start : start + ql]
        logits = np.einsum("qhd,khd->hqk", qs, keys) * sm_scale
        mask = (n_hist + np.arange(ql))[:, None] >= np.arange(kv_len)[None, :]
        logits = np.where(mask[None], logits, -1e30)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        outs.append(np.einsum("hqk,khd->qhd", p, vals))
    return np.concatenate(outs, 0)


@pytest.mark.parametrize("preset", PRESETS)
def test_tq_ragged_paged_attention_matches_reference(preset):
    c = _attention_case(preset)
    sm_scale = c["cfg"].head_dim ** -0.5
    new_cache, out = tq.tq_ragged_paged_attention(
        jnp.asarray(c["q"]),
        jnp.asarray(c["k"]),
        jnp.asarray(c["v"]),
        jnp.asarray(c["cache"]),
        jnp.asarray(c["kv_lens"]),
        jnp.asarray(c["page_indices"]),
        jnp.asarray(c["cu"]),
        jnp.asarray(c["distribution"]),
        layout=c["layout"],
        tables=c["tables"],
        rpa_fn=ref_ragged_paged_attention,
        sm_scale=sm_scale,
    )
    total = int(c["cu"][-1])
    want = _expected_attention(c, sm_scale)
    np.testing.assert_allclose(np.asarray(out)[:total], want, rtol=2e-3, atol=2e-3)

    # New tokens were quantized into their slots; nothing else changed.
    new_cache = np.asarray(new_cache)
    touched = np.zeros(new_cache.shape[:2], bool)
    for s, ql in enumerate(c["q_lens"]):
        start = c["cu"][s]
        slots = np.asarray(
            tq.quantize_kv(
                jnp.asarray(c["k"][start : start + ql]),
                jnp.asarray(c["v"][start : start + ql]),
                c["layout"],
                c["tables"],
            )
        )
        for i in range(ql):
            pos = c["kv_lens"][s] - ql + i
            page = c["page_indices"][s * c["pages_per_seq"] + pos // c["page_size"]]
            np.testing.assert_array_equal(
                new_cache[page, pos % c["page_size"]], slots[i]
            )
            touched[page, pos % c["page_size"]] = True
    np.testing.assert_array_equal(new_cache[~touched], c["cache"][~touched])


def test_skip_kv_update_leaves_cache_untouched():
    c = _attention_case("turboquant_k8v4")
    new_cache, _ = tq.tq_ragged_paged_attention(
        jnp.asarray(c["q"]),
        jnp.asarray(c["k"]),
        jnp.asarray(c["v"]),
        jnp.asarray(c["cache"]),
        jnp.asarray(c["kv_lens"]),
        jnp.asarray(c["page_indices"]),
        jnp.asarray(c["cu"]),
        jnp.asarray(c["distribution"]),
        layout=c["layout"],
        tables=c["tables"],
        rpa_fn=ref_ragged_paged_attention,
        sm_scale=1.0,
        skip_kv_update=True,
    )
    np.testing.assert_array_equal(np.asarray(new_cache), c["cache"])


@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("preset", PRESETS)
def test_tq_ragged_paged_attention_128_packed_pages(preset, page_size):
    """128-byte minor-axis page layout produces identical cache and attention."""
    c = _attention_case(preset, page_size=page_size)
    sm_scale = c["cfg"].head_dim ** -0.5
    packed_cache = tq.pack_pages_128(jnp.asarray(c["cache"]), c["layout"])
    np.testing.assert_array_equal(
        np.asarray(tq.unpack_pages_128(packed_cache, c["layout"])), c["cache"]
    )
    new_packed, out_packed = tq.tq_ragged_paged_attention(
        jnp.asarray(c["q"]),
        jnp.asarray(c["k"]),
        jnp.asarray(c["v"]),
        packed_cache,
        jnp.asarray(c["kv_lens"]),
        jnp.asarray(c["page_indices"]),
        jnp.asarray(c["cu"]),
        jnp.asarray(c["distribution"]),
        layout=c["layout"],
        tables=c["tables"],
        rpa_fn=ref_ragged_paged_attention,
        sm_scale=sm_scale,
    )
    new_plain, out_plain = tq.tq_ragged_paged_attention(
        jnp.asarray(c["q"]),
        jnp.asarray(c["k"]),
        jnp.asarray(c["v"]),
        jnp.asarray(c["cache"]),
        jnp.asarray(c["kv_lens"]),
        jnp.asarray(c["page_indices"]),
        jnp.asarray(c["cu"]),
        jnp.asarray(c["distribution"]),
        layout=c["layout"],
        tables=c["tables"],
        rpa_fn=ref_ragged_paged_attention,
        sm_scale=sm_scale,
    )
    np.testing.assert_array_equal(
        np.asarray(tq.unpack_pages_128(new_packed, c["layout"])),
        np.asarray(new_plain),
    )
    np.testing.assert_allclose(np.asarray(out_packed), np.asarray(out_plain), atol=1e-6)
