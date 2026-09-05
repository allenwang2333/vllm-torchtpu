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
"""Pooling math, the fp8/ue8m0 record layout, and the quantization gate.

Pure ``jnp``; runs on the JAX CPU backend, no TPU.

The accuracy gate at the bottom is the load-bearing part: it measures top-512
set agreement with **both** sides of the quantization applied -- the pooled key
*and* the fp8 ``q`` whose per-(token, head) scale is folded into the per-head
weights. A key-side-only measurement reads about a point higher and would set a
floor the device can never meet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.glm5_kpool import (FP8_E4M3_MAX, pack_fp8_ue8m0,
                                              pool_window,
                                              quantize_fp8_ue8m0_rows,
                                              unpack_fp8_ue8m0)

# The entry is ``tests/models/vllm`` (which exposes the package
# ``glm5_next``), never ``tests/models`` -- that would put a directory named
# ``vllm`` on sys.path and shadow the real package.
sys.path.insert(0,
                str(Path(__file__).resolve().parents[2] / "models" / "vllm"))
from glm5_next import kpool_oracle as O  # noqa: E402

PROD = O.KpoolParams()
SMALL = O.KpoolParams(index_kpool=4,
                      index_topk=8,
                      index_head_dim=16,
                      index_n_heads=4)

# ---------------------------------------------------------------------------
# pool_window
# ---------------------------------------------------------------------------


def _random_pool_inputs(num_pools, index_kpool, head_dim, seed):
    rng = np.random.default_rng(seed)
    k = rng.normal(size=(num_pools, index_kpool, head_dim))
    # `k` reaches the op already LayerNorm'd (`modeling_glm5_next.py:763`).
    k = (k - k.mean(-1, keepdims=True)) / k.std(-1, keepdims=True)
    gate = rng.normal(size=(num_pools, index_kpool, head_dim))
    ape = rng.normal(size=(index_kpool, head_dim))
    return k, gate, ape


def test_pool_window_matches_the_numpy_oracle():
    k, gate, ape = _random_pool_inputs(64, 4, 128, seed=0)
    with jax.enable_x64():
        got = np.asarray(
            pool_window(jnp.asarray(k), jnp.asarray(gate), jnp.asarray(ape)))
    want = O.pooled_keys(k.reshape(-1, 128),
                         gate.reshape(-1, 128),
                         ape,
                         index_kpool=4)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)


def test_pool_window_accumulates_in_at_least_f32_at_bf16_inputs():
    """The softmax must not run in the *input* dtype.

    The reference upcasts explicitly -- ``grouped_gate_scores.float() +
    ape.float()``, softmax, then cast the probabilities down
    (``modeling_glm5_next.py:962-966``). A bf16 softmax over 4 slots per
    channel raises no error and produces a finite, correctly shaped pooled key;
    it only perturbs the k/k+1 near-ties, which then reads as fp8 quantization
    noise rather than as a dtype bug. Every other test here feeds f32 or f64,
    so bf16 is the only input that can catch it.

    The bar is a *discriminator*, not a tolerance. An absolute tolerance would
    be fragile in both directions: pooled keys are reached by cancellation, so
    individual channels land near zero and relative error explodes (max |rel|
    exceeds 1 for both the right and the wrong code), while any absolute bound
    loose enough to be stable also admits the bug. So compute the bug variant
    explicitly and require the shipped code to be materially closer to the f64
    oracle.

    ``gate``/``ape`` are bf16 but ``k_rows`` is f32, to isolate the one thing
    under test. With *everything* bf16 the fix only improves rms by ~1.2x,
    because the error is then dominated by rounding ``k_rows`` and the result
    -- so an all-bf16 version of this test would barely discriminate and would
    overstate what the softmax dtype is worth. Isolated, the gap is 2.45x
    (8.6e-4 against 2.1e-3), which is the honest measure of this bug.
    """
    k, gate, ape = _random_pool_inputs(64, 4, 128, seed=7)
    kf = jnp.asarray(k, jnp.float32)
    gb = jnp.asarray(gate, jnp.bfloat16)
    ab = jnp.asarray(ape, jnp.bfloat16)

    want = O.pooled_keys(k.reshape(-1, 128),
                         gate.reshape(-1, 128),
                         ape,
                         index_kpool=4)
    got = np.asarray(pool_window(kf, gb, ab).astype(jnp.float32))

    # The bug, written out: softmax in the *input* dtype, upcast afterwards.
    probs_bf16 = jax.nn.softmax(gb + ab[None], axis=1)
    bugged = np.asarray(jnp.sum(probs_bf16.astype(jnp.float32) * kf, axis=1))

    def rms(a):
        return float(np.sqrt(np.mean((a - want)**2)))

    assert rms(got) < 0.7 * rms(bugged), (
        f"pool_window (rms {rms(got):.3e}) is not materially better than a "
        f"bf16-internal softmax (rms {rms(bugged):.3e}), so the probabilities "
        "are probably being computed in the input dtype rather than in at "
        "least f32 (modeling_glm5_next.py:962-966)")


_REF_CASE = O.make_case(SMALL,
                        hidden_size=32,
                        q_lora_rank=24,
                        seq_len=64,
                        weight_seed=5,
                        input_seed=4)


@pytest.fixture(scope="module")
def ref_pooled():
    """One subprocess for the module. See ``kpool_oracle.run_reference_cases``
    for why the reference cannot be imported in process."""
    if not O.glm5_next_source_available():
        pytest.skip(
            f"no glm5_next transformers overlay; set {O.TRANSFORMERS_SRC_ENV}")
    return O.run_reference_cases([_REF_CASE])[O.case_key(_REF_CASE)]


def test_pool_window_matches_get_pooled_states(ref_pooled):
    """(a) ``pool_window`` vs the reference in float64, exact to 1e-6.

    1e-6 rather than 0 because the reference hard-codes a **float32** softmax
    (``modeling_glm5_next.py:962``) whatever the module dtype, so it is the
    *less* accurate of the two. See
    ``tests/models/vllm/glm5_next/test_kpool_oracle.py::
    test_reference_softmax_is_float32_not_module_dtype``.
    """
    d, kp = SMALL.index_head_dim, SMALL.index_kpool
    with jax.enable_x64():
        got = np.asarray(
            pool_window(jnp.asarray(ref_pooled["k"].reshape(-1, kp, d)),
                        jnp.asarray(ref_pooled["gate"].reshape(-1, kp, d)),
                        jnp.asarray(ref_pooled["ape"])))
    np.testing.assert_allclose(got,
                               ref_pooled["pooled_ref"],
                               rtol=0,
                               atol=1e-6)


def test_reference_probability_downcast_is_reported_not_asserted(capsys):
    """(b) The reference casts the probabilities down to the *key* dtype before
    the 4-term sum (``modeling_glm5_next.py:964-966``) -- bf16 in the shipped
    checkpoint. This op accumulates in the input dtype and is therefore more
    accurate. Report the gap; do not assert equality, because it can never
    close."""
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    k, gate, ape = _random_pool_inputs(256, 4, 128, seed=11)
    exact = O.pooled_keys(k.reshape(-1, 128),
                          gate.reshape(-1, 128),
                          ape,
                          index_kpool=4)
    down = O.pooled_keys(k.reshape(-1, 128),
                         gate.reshape(-1, 128),
                         ape,
                         index_kpool=4,
                         prob_dtype=bf16)
    abs_err = float(np.abs(exact - down).max())
    rel_err = float(
        (np.abs(exact - down) / np.maximum(np.abs(exact), 1e-6)).max())
    with capsys.disabled():
        print(f"\n[step5b] bf16 probability downcast vs f64: "
              f"max abs {abs_err:.3e}  max rel {rel_err:.3e}")
    assert np.isfinite(abs_err)


def test_softmax_is_over_slots_per_channel():
    """A softmax over the wrong axis still returns a finite key of the right
    shape. Perturbing one channel of ``ape`` must move only that channel."""
    k, gate, ape = _random_pool_inputs(8, 4, 16, seed=2)
    base = np.asarray(
        pool_window(jnp.asarray(k), jnp.asarray(gate), jnp.asarray(ape)))
    ape2 = ape.copy()
    ape2[0, 5] += 5.0
    moved = np.asarray(
        pool_window(jnp.asarray(k), jnp.asarray(gate), jnp.asarray(ape2)))
    delta = np.abs(moved - base)
    assert delta[:, 5].max() > 1e-5
    assert np.delete(delta, 5, axis=1).max() == 0.0


def test_ape_is_indexed_by_slot_not_broadcast():
    """``index_kpool_compress_ape`` is ``[index_kpool, head_dim]``: a *different*
    vector per slot within the pool. An implementation that broadcast one row
    would be invariant to permuting the rows; this one must not be."""
    k, gate, ape = _random_pool_inputs(16, 4, 16, seed=4)
    base = np.asarray(
        pool_window(jnp.asarray(k), jnp.asarray(gate), jnp.asarray(ape)))
    rolled = np.asarray(
        pool_window(jnp.asarray(k), jnp.asarray(gate),
                    jnp.asarray(np.roll(ape, 1, axis=0))))
    assert np.abs(base - rolled).max() > 1e-3


def test_pooling_is_a_weighted_average():
    """``sum_j softmax_j == 1`` per channel, so a pool of four identical tokens
    reproduces that token exactly -- a scale bug in the softmax shows up here
    and nowhere else."""
    rng = np.random.default_rng(6)
    row = rng.normal(size=(1, 1, 32))
    k = np.repeat(row, 4, axis=1)
    gate = rng.normal(size=(1, 4, 32))
    ape = rng.normal(size=(4, 32))
    with jax.enable_x64():
        got = np.asarray(
            pool_window(jnp.asarray(k), jnp.asarray(gate), jnp.asarray(ape)))
    np.testing.assert_allclose(got, row[:, 0], rtol=0, atol=1e-13)


def test_pool_window_rejects_a_mismatched_ape():
    k, gate, ape = _random_pool_inputs(4, 4, 16, seed=1)
    with pytest.raises(AssertionError):
        pool_window(jnp.asarray(k), jnp.asarray(gate), jnp.asarray(ape[:3]))


# ---------------------------------------------------------------------------
# fp8 + inline e8m0 record layout
# ---------------------------------------------------------------------------


def _torch_reference_record(x):
    """What the GLM-5.2 producer builds today, at
    ``layers/vllm/custom_ops/mla_attention_op.py:156-166``."""
    torch = pytest.importorskip("torch")
    from vllm_torchtpu.layers.common.quantization import quantize_tensor
    q, s = quantize_tensor(torch.from_numpy(np.asarray(x)),
                           torch.float8_e4m3fn,
                           axis=-1,
                           use_ue8m0=True)
    return torch.cat([q.view(torch.uint8), s], dim=-1).numpy()


@pytest.mark.parametrize("head_dim", [16, 128])
def test_packed_record_is_bit_identical_to_the_torch_producer(head_dim):
    """The jnp transcription must not drift from
    ``layers/common/quantization.py``'s ue8m0 path -- including the
    round-the-scale-**up** repair and the all-zero row, which are exactly the
    places a re-derivation goes wrong."""
    rng = np.random.default_rng(0)
    x = (rng.normal(size=(96, head_dim)) *
         rng.choice([1e-8, 1e-3, 1.0, 1e3, 1e8], size=(96, 1))).astype(
             np.float32)
    x[0] = 0.0  # all-zero row -> log2(0) = -inf
    x[1] = FP8_E4M3_MAX  # exactly at the clamp
    x[2, 0] = 2.0**-140  # denormal-ish, below E8M0_MIN_EXP
    got = np.asarray(pack_fp8_ue8m0(jnp.asarray(x)))
    np.testing.assert_array_equal(got, _torch_reference_record(x))


def test_record_layout_is_head_dim_bytes_then_one_scale_byte():
    """``streamindex_topk.py:262-271`` reads ``flat_bkv[:, :head_dim]`` as
    fp8_e4m3 and ``flat_bkv[:, head_dim:head_dim+1]`` as e8m0."""
    rng = np.random.default_rng(3)
    d = 128
    x = rng.normal(size=(32, d)).astype(np.float32) * 7.0
    rec = np.asarray(pack_fp8_ue8m0(jnp.asarray(x)))
    assert rec.shape == (32, d + 1) and rec.dtype == np.uint8

    values, scale_bits = quantize_fp8_ue8m0_rows(jnp.asarray(x))
    np.testing.assert_array_equal(
        rec[:, :d], np.asarray(jax.lax.bitcast_convert_type(values,
                                                            jnp.uint8)))
    np.testing.assert_array_equal(rec[:, d:], np.asarray(scale_bits))

    # The scale byte is a biased e8m0 exponent: 2 ** (byte - 127).
    scale = 2.0**(rec[:, d].astype(np.int32) - 127)
    deq = np.asarray(unpack_fp8_ue8m0(jnp.asarray(rec), d))
    np.testing.assert_array_equal(
        deq,
        np.asarray(values.astype(jnp.float32)) * scale[:, None])
    assert rec[:, d].max() < 255, "255 is e8m0 NaN and must never be emitted"


def test_record_round_trip_error_is_fp8_sized():
    """Not a tolerance to be widened later: e4m3 has 3 mantissa bits, so half an
    ULP at the bottom of a binade is 6.25% of the value, and the ue8m0 scale
    rounds **up**, leaving the row max in (224, 448]. ~7% worst case relative to
    the row max is the format, not a bug -- which is why the selection gate
    below measures *set agreement* rather than score error."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=(256, 128)).astype(np.float32)
    deq = np.asarray(unpack_fp8_ue8m0(pack_fp8_ue8m0(jnp.asarray(x)), 128))
    row_max = np.abs(x).max(-1, keepdims=True)
    rel = float((np.abs(deq - x) / row_max).max())
    assert rel < 0.08, rel
    assert rel > 0.01, "suspiciously accurate for e4m3 -- check the scale"


# ---------------------------------------------------------------------------
# (c) The quantization accuracy gate -- both sides quantized.
# ---------------------------------------------------------------------------

# Measured on CPU at production geometry (32 heads, 128 dim, top-512 per query
# row), synthetic Gaussian inputs, 8 query rows. The 4096-pool row is what this
# test re-measures and prints on every run; the other two come from the same
# harness swept over context length and are recorded here because 128K is the
# committed target and this is the only rung that can price it without a chip:
#
#   pools  raw tokens  key-only         q-only           BOTH
#    1024      4 096   mean .9905       mean .9890       mean .9863  min .9824
#    4096     16 384   mean .9775       mean .9800       mean .9692  min .9609
#   32768    131 072   mean .9690       mean .9709       mean .9587  min .9551
#
# The key-only column is the optimistic reading, and it
# sits ~1 point above the number the device can actually reach, and a floor set
# from it would be one no correct implementation could meet. Every observed
# disagreement was a pool whose exact score lay within ~10% of the k/k+1
# boundary, measured against the top-1..top-k score span -- i.e. quantization
# reshuffles near-ties, it does not lose distinctly-scoring pools.
#
# The floor is set below the worst *measured* min (.9551 at 128K) with margin,
# not at it: this is a synthetic score distribution, to be re-measured on
# device against the same protocol.
GATE_AGREEMENT_FLOOR = 0.95
GATE_BOUNDARY_FRACTION = 0.15


def _quantize_q_fp8(q):
    """``quantize_tensor(q.view(-1, D), float8_e4m3fn, axis=-1,
    use_ue8m0=False)`` -- the fp32-scale path the query side uses at
    ``mla_attention_op.py:266-276``. Returns the dequantized query, which is
    what the device effectively scores with once ``q_scale`` is folded into the
    per-head weights."""
    t, h, d = q.shape
    flat = q.reshape(-1, d).astype(np.float32)
    scale = np.abs(flat).max(-1, keepdims=True) * np.float32(
        1.0 / FP8_E4M3_MAX)
    inv = np.where(scale == 0, np.float32(0.0), 1.0 / scale)
    q8 = jnp.asarray(np.clip(flat * inv, -FP8_E4M3_MAX,
                             FP8_E4M3_MAX)).astype(jnp.float8_e4m3fn)
    return (np.asarray(q8.astype(jnp.float32)) * scale).reshape(t, h, d)


def test_q_quantizer_matches_the_torch_producer():
    """The gate below is only meaningful if its ``q`` quantizer is the one the
    device uses."""
    torch = pytest.importorskip("torch")
    from vllm_torchtpu.layers.common.quantization import quantize_tensor
    rng = np.random.default_rng(1)
    q = rng.normal(size=(7, 32, 128)).astype(np.float32)
    tq, ts = quantize_tensor(torch.from_numpy(q.reshape(-1, 128)),
                             torch.float8_e4m3fn,
                             axis=-1,
                             use_ue8m0=False)
    assert ts.shape == (q.shape[0] * q.shape[1], 1
                        )  # one scale per (token, head)
    want = (tq.to(torch.float32) * ts).numpy().reshape(q.shape)
    np.testing.assert_array_equal(_quantize_q_fp8(q), want)


def _score(q, pool_key, weights, *, softmax_scale, n_heads):
    t, h, d = q.shape
    per = (q.reshape(-1, d).astype(np.float64) @ pool_key.astype(
        np.float64).T).reshape(t, h, -1)
    per = np.maximum(per * softmax_scale, 0.0)
    return np.einsum("th,thp->tp",
                     weights.astype(np.float64) * n_heads**-0.5, per)


def _agreement(s_exact, s_quant, k):
    """Top-``k`` **set** agreement, and how far each disagreeing pool sat from
    the ``k``/``k+1`` boundary as a fraction of the top-1..top-k score span."""
    fracs, worst = [], 0.0
    for t in range(s_exact.shape[0]):
        a = set(np.argpartition(-s_exact[t], k)[:k].tolist())
        b = set(np.argpartition(-s_quant[t], k)[:k].tolist())
        fracs.append(len(a & b) / k)
        srt = np.sort(s_exact[t])[::-1]
        thresh = 0.5 * (srt[k - 1] + srt[k])
        span = max(srt[0] - srt[k - 1], 1e-30)
        for j in a ^ b:
            worst = max(worst, abs(s_exact[t][j] - thresh) / span)
    return float(np.mean(fracs)), float(np.min(fracs)), worst


def test_fp8_selection_gate_quantizes_both_sides(capsys):
    n_heads, d, kp = PROD.index_n_heads, PROD.index_head_dim, PROD.index_kpool
    num_pools, num_q, k = 4096, 8, PROD.select_k
    rng = np.random.default_rng(0)

    kr, gate, ape = _random_pool_inputs(num_pools, kp, d, seed=0)
    pooled = np.asarray(
        pool_window(jnp.asarray(kr.astype(np.float32)),
                    jnp.asarray(gate.astype(np.float32)),
                    jnp.asarray(ape.astype(np.float32))))
    q = rng.normal(size=(num_q, n_heads, d)).astype(np.float32)
    w = rng.normal(size=(num_q, n_heads)).astype(np.float32)

    pooled_q = np.asarray(
        unpack_fp8_ue8m0(pack_fp8_ue8m0(jnp.asarray(pooled)), d))
    q_deq = _quantize_q_fp8(q)

    def sc(qa, ka):
        return _score(qa,
                      ka,
                      w,
                      softmax_scale=PROD.softmax_scale,
                      n_heads=n_heads)

    exact = sc(q, pooled)
    key_only = _agreement(exact, sc(q, pooled_q), k)
    q_only = _agreement(exact, sc(q_deq, pooled), k)
    both = _agreement(exact, sc(q_deq, pooled_q), k)

    with capsys.disabled():
        print(f"\n[step5c] {num_pools} pools ({num_pools * kp} raw tokens), "
              f"top-{k} of {num_pools}, {num_q} query rows")
        for name, (mean, lo, worst) in (("key only", key_only),
                                        ("q only", q_only), ("BOTH", both)):
            print(f"  {name:9s} agreement mean {mean:.4f} min {lo:.4f}  "
                  f"worst boundary distance {worst:.3f} of span")

    assert both[1] >= GATE_AGREEMENT_FLOOR, both
    assert both[2] <= GATE_BOUNDARY_FRACTION, both
    # A key-side-only measurement is optimistic, so a floor derived from it
    # would be one the device can never meet. Asserted, not assumed.
    assert both[0] < key_only[0], (both[0], key_only[0])
    assert both[0] < q_only[0], (both[0], q_only[0])


def test_quantization_gate_is_not_measuring_a_no_op():
    """If the fp8 round trip were accidentally exact, the gate above would pass
    vacuously."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(64, 128)).astype(np.float32)
    deq = np.asarray(unpack_fp8_ue8m0(pack_fp8_ue8m0(jnp.asarray(x)), 128))
    assert not np.array_equal(deq, x)
    assert float(np.abs(deq - x).max()) > 1e-3
