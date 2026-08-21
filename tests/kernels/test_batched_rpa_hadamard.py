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
"""Tests for Hadamard-transform KV cache quantization in the batched RPA path.

The scheme: K/V are rotated by a normalized Sylvester-Hadamard matrix H along
head_dim before (FP4 per-token) quantization; the batched RPA wrapper rotates
Q by the same H before the Pallas kernel (QK^T is exactly preserved since H is
orthonormal) and applies H once more to the attention output (P @ (V H) ==
(P V) H, and H is involutory). The Pallas kernel itself is unchanged.

Covered here:
  1. Matrix properties + bitwise parity between the torch- and JAX-side
     constructions (hardware-free).
  2. Torch quantization roundtrip: rotate+quantize then dequantize+unrotate
     reconstructs the input, and beats the unrotated path on outlier-heavy
     data (hardware-free).
  3. Kernel math parity on TPU with an unquantized bf16 cache: running the
     wrapper with pre-rotated K/V and use_hadamard=True must match the
     baseline run bit-for-bit up to bf16 rounding, for both a fresh prefill
     and a decode step against the (rotated) cache written by that prefill.
  4. End-to-end FP4 per-token path on TPU: rotated-quantized K/V + scales via
     the real torch quantizer, wrapper with use_hadamard=True, compared
     against an fp32 attention reference; error must be comparable to the
     non-Hadamard FP4 run on Gaussian data and strictly better on
     outlier-heavy data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.experimental.batched_rpa import \
    hadamard as jax_hadamard
from vllm_torchtpu.kernels.experimental.batched_rpa import configs as KBC
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as KB
from vllm_torchtpu.layers.common import quantization as tq

_DECODE_BLOCKS = KBC.BlockSizes(bq_sz=1,
                                bq_c_sz=1,
                                bkv_sz=256,
                                batch_size=8,
                                n_buffer=3)
_PREFILL_BLOCKS = KBC.BlockSizes(bq_sz=256,
                                 bq_c_sz=32,
                                 bkv_sz=256,
                                 batch_size=2,
                                 n_buffer=3)

_HEAD_DIM = 128
_NUM_KV_HEADS = 2
_NUM_Q_HEADS = 8
_PAGE_SIZE = 128
_TOTAL_PAGES = 8


def _require_tpu() -> None:
    try:
        backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.skip(f"Requires TPU backend, found {backend!r}.")


# ---------------------------------------------------------------------------
# 1. Matrix construction (hardware-free)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_hadamard_matrix_properties(dim):
    h = jax_hadamard.hadamard_matrix_np(dim)
    assert h.dtype == np.float32
    # Symmetric and orthonormal => involutory.
    np.testing.assert_array_equal(h, h.T)
    np.testing.assert_allclose(h @ h.T, np.eye(dim), atol=1e-5, rtol=0)
    np.testing.assert_allclose(h @ h, np.eye(dim), atol=1e-5, rtol=0)


@pytest.mark.parametrize("dim", [0, 3, 96, 100])
def test_hadamard_matrix_rejects_non_power_of_two(dim):
    with pytest.raises(ValueError, match="power of two"):
        jax_hadamard.hadamard_matrix_np(dim)
    with pytest.raises(ValueError, match="power of two"):
        tq.hadamard_matrix(dim)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_torch_jax_hadamard_bitwise_match(dim):
    """K/V are rotated with the torch matrix but un-rotated (via Q/output)
    with the JAX matrix -- the two constructions must agree bitwise."""
    h_torch = tq.hadamard_matrix(dim).numpy()
    h_jax = jax_hadamard.hadamard_matrix_np(dim)
    np.testing.assert_array_equal(h_torch, h_jax)


# ---------------------------------------------------------------------------
# 2. Torch quantization roundtrip (hardware-free)
# ---------------------------------------------------------------------------


def _dequant_fp4(packed: torch.Tensor, scale: torch.Tensor,
                 unrotate: bool) -> torch.Tensor:
    values = tq.unpack_uint8_to_fp4(packed)  # [..., D] in float32
    values = values * scale.to(torch.float32)
    if unrotate:
        h = tq.hadamard_matrix(values.shape[-1], values.device)
        values = values @ h  # involutory: same matrix undoes the rotation
    return values


def _fp4_roundtrip_err(key: torch.Tensor, value: torch.Tensor,
                       hadamard_mode: str | None) -> tuple[float, float]:
    """Returns per-tensor (k_err, v_err) relative reconstruction errors."""
    k_packed, v_packed, k_scale, v_scale = tq.quantize_kv_fp4_per_tensor(
        key, value, scale_dtype=torch.float32, hadamard_mode=hadamard_mode)
    k_rec = _dequant_fp4(k_packed, k_scale, unrotate=hadamard_mode is not None)
    v_rec = _dequant_fp4(v_packed, v_scale, unrotate=hadamard_mode == "kv")
    err_k = float((k_rec - key).norm() / key.norm())
    err_v = float((v_rec - value).norm() / value.norm())
    return err_k, err_v


def test_quantize_kv_fp4_hadamard_roundtrip_gaussian():
    torch.manual_seed(0)
    key = torch.randn(64, _NUM_KV_HEADS, _HEAD_DIM)
    value = torch.randn(64, _NUM_KV_HEADS, _HEAD_DIM)
    err_k, err_v = _fp4_roundtrip_err(key, value, hadamard_mode="kv")
    # FP4 with per-token scales on Gaussian data reconstructs within ~10%.
    assert max(err_k, err_v) < 0.15, (err_k, err_v)


def test_quantize_kv_fp4_hadamard_improves_outliers():
    """A few large per-channel outliers blow up the per-token absmax scale;
    the Hadamard rotation spreads them across head_dim and must reduce the
    relative reconstruction error."""
    torch.manual_seed(0)
    key = torch.randn(64, _NUM_KV_HEADS, _HEAD_DIM)
    value = torch.randn(64, _NUM_KV_HEADS, _HEAD_DIM)
    # Outlier channels, RoPE-style: a couple of channels 30x larger.
    key[..., 7] *= 30.0
    key[..., 100] *= 30.0
    value[..., 3] *= 30.0

    plain = _fp4_roundtrip_err(key, value, hadamard_mode=None)
    mode_k = _fp4_roundtrip_err(key, value, hadamard_mode="k")
    mode_kv = _fp4_roundtrip_err(key, value, hadamard_mode="kv")
    # "kv" improves both tensors; "k" improves K and leaves V as-is.
    assert mode_kv[0] < plain[0], (mode_kv, plain)
    assert mode_kv[1] < plain[1], (mode_kv, plain)
    assert mode_k[0] < plain[0], (mode_k, plain)
    assert mode_k[1] == pytest.approx(plain[1]), (mode_k, plain)


# ---------------------------------------------------------------------------
# TPU helpers
# ---------------------------------------------------------------------------


def _reference_attention(q, k, v, sm_scale, kv_len_offset=0):
    """fp32 causal attention reference.

    q: [q_len, num_q_heads, head_dim]; k/v: [kv_len, num_kv_heads, head_dim].
    Query position i attends to kv positions <= i + kv_len_offset
    (kv_len_offset = kv_len - q_len).
    """
    q_len, num_q_heads, _ = q.shape
    kv_len, num_kv_heads, _ = k.shape
    group = num_q_heads // num_kv_heads
    k = np.repeat(k, group, axis=1)
    v = np.repeat(v, group, axis=1)
    scores = np.einsum("qhd,khd->hqk", q.astype(np.float64),
                       k.astype(np.float64)) * sm_scale
    q_pos = np.arange(q_len)[:, None] + kv_len_offset
    k_pos = np.arange(kv_len)[None, :]
    scores = np.where((k_pos <= q_pos)[None], scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    out = np.einsum("hqk,khd->qhd", p, v.astype(np.float64))
    return out.astype(np.float32)


def _run_batched_rpa(query, key, value, cache, kv_len, q_len, distribution,
                     sm_scale, **kwargs):
    """One wrapper call for a single sequence; returns (out, cache) numpy."""
    pages_per_seq = _TOTAL_PAGES
    out, new_cache = KB.ragged_paged_attention(
        query,
        key,
        value,
        cache,
        jnp.array([kv_len], jnp.int32),
        jnp.arange(pages_per_seq, dtype=jnp.int32),
        jnp.array([0, q_len], jnp.int32),
        jnp.asarray(distribution, jnp.int32),
        sm_scale=sm_scale,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=_PREFILL_BLOCKS,
        **kwargs,
    )
    return np.asarray(out.astype(jnp.float32)), new_cache


# ---------------------------------------------------------------------------
# 3. bf16 kernel-math parity on TPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["k", "kv"])
def test_batched_rpa_hadamard_bf16_parity(mode):
    """With an unquantized bf16 cache, hadamard_mode on pre-rotated K (and V
    for mode "kv") must reproduce the baseline attention output (the rotation
    is exactly invisible to the attention math). Checks a full prefill and
    then a decode step that reads back the rotated cache written by the
    prefill."""
    _require_tpu()
    rng = np.random.default_rng(0)
    dtype = jnp.bfloat16
    kv_len = 256
    sm_scale = _HEAD_DIM**-0.5
    h = jax_hadamard.hadamard_matrix_np(_HEAD_DIM)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    cache_shape = KB.get_kv_cache_shape(_TOTAL_PAGES, _PAGE_SIZE,
                                        _NUM_KV_HEADS, _HEAD_DIM, dtype)

    query = r(kv_len, _NUM_Q_HEADS, _HEAD_DIM)
    k_new, v_new = r(kv_len, _NUM_KV_HEADS,
                     _HEAD_DIM), r(kv_len, _NUM_KV_HEADS, _HEAD_DIM)

    def maybe_rot_v(v):
        return v @ h if mode == "kv" else v

    # Baseline prefill (fresh cache) vs Hadamard prefill on rotated K (and V
    # for mode "kv").
    out_base, cache_base = _run_batched_rpa(
        jnp.asarray(query, dtype), jnp.asarray(k_new, dtype),
        jnp.asarray(v_new, dtype), jnp.zeros(cache_shape, dtype), kv_len,
        kv_len, (0, 0, 1), sm_scale)
    out_had, cache_had = _run_batched_rpa(
        jnp.asarray(query, dtype), jnp.asarray(k_new @ h, dtype),
        jnp.asarray(maybe_rot_v(v_new), dtype),
        jnp.zeros(cache_shape, dtype), kv_len,
        kv_len, (0, 0, 1), sm_scale, hadamard_mode=mode)

    # bf16 K/V + two extra bf16 roundings (rotated K/V storage, Q rotation)
    # -> slightly looser than pure-bf16 parity.
    np.testing.assert_allclose(out_had, out_base, atol=3e-2, rtol=0)

    # Decode step: new token attends to the cache written above.
    q1 = r(1, _NUM_Q_HEADS, _HEAD_DIM)
    k1, v1 = r(1, _NUM_KV_HEADS, _HEAD_DIM), r(1, _NUM_KV_HEADS, _HEAD_DIM)
    out1_base, _ = _run_batched_rpa(jnp.asarray(q1, dtype),
                                    jnp.asarray(k1, dtype),
                                    jnp.asarray(v1,
                                                dtype), cache_base, kv_len, 1,
                                    (1, 1, 1), sm_scale)
    out1_had, _ = _run_batched_rpa(jnp.asarray(q1, dtype),
                                   jnp.asarray(k1 @ h, dtype),
                                   jnp.asarray(maybe_rot_v(v1), dtype),
                                   cache_had,
                                   kv_len,
                                   1, (1, 1, 1),
                                   sm_scale,
                                   hadamard_mode=mode)
    np.testing.assert_allclose(out1_had, out1_base, atol=3e-2, rtol=0)

    # Sanity: the rotation must NOT be invisible if the wrapper doesn't
    # compensate (i.e. the test would catch a no-op use_hadamard flag).
    out_bad, _ = _run_batched_rpa(
        jnp.asarray(query, dtype), jnp.asarray(k_new @ h, dtype),
        jnp.asarray(v_new @ h, dtype), jnp.zeros(cache_shape, dtype), kv_len,
        kv_len, (0, 0, 1), sm_scale)
    assert np.abs(out_bad - out_base).max() > 3e-2


# ---------------------------------------------------------------------------
# 4. FP4 per-token end-to-end on TPU
# ---------------------------------------------------------------------------


def _quantize_with_torch(k_np, v_np, hadamard_mode):
    k_packed, v_packed, k_scale, v_scale = tq.quantize_kv_fp4_per_tensor(
        torch.from_numpy(k_np),
        torch.from_numpy(v_np),
        scale_dtype=torch.float32,
        hadamard_mode=hadamard_mode,
    )
    to_bf16 = lambda t: jnp.asarray(t.numpy(), jnp.bfloat16)
    return (jnp.asarray(k_packed.numpy()), jnp.asarray(v_packed.numpy()),
            to_bf16(k_scale), to_bf16(v_scale))


def _run_fp4(query, k_np, v_np, kv_len, sm_scale, hadamard_mode):
    cache_shape = KB.get_kv_cache_shape(_TOTAL_PAGES,
                                        _PAGE_SIZE,
                                        _NUM_KV_HEADS,
                                        _HEAD_DIM // 2,
                                        jnp.uint8,
                                        use_per_token_scale=True,
                                        scale_dtype=jnp.bfloat16)
    k_q, v_q, k_s, v_s = _quantize_with_torch(k_np, v_np, hadamard_mode)
    out, _ = _run_batched_rpa(jnp.asarray(query, jnp.bfloat16),
                              k_q,
                              v_q,
                              jnp.zeros(cache_shape, jnp.uint8),
                              kv_len,
                              kv_len, (0, 0, 1),
                              sm_scale,
                              use_per_token_scale=True,
                              per_token_scale_dtype=jnp.bfloat16,
                              k_scale_tensor=k_s,
                              v_scale_tensor=v_s,
                              hadamard_mode=hadamard_mode)
    return out


def test_batched_rpa_hadamard_fp4_e2e():
    _require_tpu()
    rng = np.random.default_rng(0)
    kv_len = 256
    sm_scale = _HEAD_DIM**-0.5

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    query = r(kv_len, _NUM_Q_HEADS, _HEAD_DIM)
    k_new, v_new = r(kv_len, _NUM_KV_HEADS,
                     _HEAD_DIM), r(kv_len, _NUM_KV_HEADS, _HEAD_DIM)

    ref = _reference_attention(query, k_new, v_new, sm_scale)

    def rel_err(out):
        return np.linalg.norm(out - ref) / np.linalg.norm(ref)

    err_plain = rel_err(_run_fp4(query, k_new, v_new, kv_len, sm_scale,
                                 None))
    err_k = rel_err(_run_fp4(query, k_new, v_new, kv_len, sm_scale, "k"))
    err_kv = rel_err(_run_fp4(query, k_new, v_new, kv_len, sm_scale, "kv"))

    # All FP4 runs must track the fp32 reference, and Hadamard must not
    # meaningfully degrade accuracy on well-behaved data.
    assert err_plain < 0.15, err_plain
    assert err_k < 0.15, err_k
    assert err_kv < 0.15, err_kv
    assert err_k < err_plain * 1.5, (err_k, err_plain)
    assert err_kv < err_plain * 1.5, (err_kv, err_plain)


def test_batched_rpa_hadamard_fp4_outliers():
    """On outlier-heavy K/V (the motivating case), the Hadamard-rotated FP4
    path must be strictly more accurate than plain FP4."""
    _require_tpu()
    rng = np.random.default_rng(1)
    kv_len = 256
    sm_scale = _HEAD_DIM**-0.5

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    query = r(kv_len, _NUM_Q_HEADS, _HEAD_DIM)
    k_new, v_new = r(kv_len, _NUM_KV_HEADS,
                     _HEAD_DIM), r(kv_len, _NUM_KV_HEADS, _HEAD_DIM)
    k_new[..., 7] *= 30.0
    k_new[..., 100] *= 30.0
    v_new[..., 3] *= 30.0

    ref = _reference_attention(query, k_new, v_new, sm_scale)

    def rel_err(out):
        return np.linalg.norm(out - ref) / np.linalg.norm(ref)

    err_plain = rel_err(_run_fp4(query, k_new, v_new, kv_len, sm_scale,
                                 None))
    err_k = rel_err(_run_fp4(query, k_new, v_new, kv_len, sm_scale, "k"))
    err_kv = rel_err(_run_fp4(query, k_new, v_new, kv_len, sm_scale, "kv"))
    assert err_kv < err_plain, (err_kv, err_plain)
    assert err_k < err_plain, (err_k, err_plain)
