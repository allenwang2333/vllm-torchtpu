"""Checks for the DeepSeek V4.1 delayed mHC shared by the v1 and v2 paths.

The oracle is vLLM's ``mhc_pre_delayed_torch``. ``delayed_pre_flat_jax``
and ``delayed_post_pre_flat_jax`` take the Pallas pass for the token counts
`_pallas_tile` accepts and the XLA path otherwise; the Pallas cases need TPU
hardware.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_delayed_torch

from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc

MULT = 4
MIXES = MULT * (MULT + 2)
CONSTANTS = (1e-6, 1e-6, 1e-6, 2.0, 20)  # rms, pre, sinkhorn eps; post mult; repeats
GATE_RTOL, GATE_ATOL = 5e-3, 1e-5
BF16_RTOL, BF16_ATOL = 2e-2, 2e-2


def _inputs(tokens, dim, seed=0):
    rng = np.random.default_rng(seed)
    residual = torch.from_numpy(rng.standard_normal((tokens, MULT, dim))).bfloat16()
    fn = torch.from_numpy(0.02 * rng.standard_normal((MIXES, MULT * dim))).float()
    scale = torch.from_numpy(1.0 + 0.1 * rng.standard_normal(3)).float()
    base = torch.from_numpy(0.5 * rng.standard_normal(MIXES)).float()
    previous_pre = torch.from_numpy(rng.uniform(0.0, 1.0, (tokens, MULT))).float()
    return residual, fn, scale, base, previous_pre


def _post_inputs(tokens, dim, seed=1):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal((tokens, dim))).bfloat16()
    post_mix = torch.from_numpy(rng.uniform(0.0, 2.0, (tokens, MULT, 1))).float()
    comb_mix = torch.from_numpy(rng.uniform(0.0, 1.0, (tokens, MULT, MULT))).float()
    comb_mix = comb_mix / comb_mix.sum(dim=-1, keepdim=True)
    return x, post_mix, comb_mix


def _jax(t: torch.Tensor) -> jax.Array:
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.float().numpy()
    return np.asarray(jnp.asarray(a).astype(jnp.float32))


def _check(tokens, dim, packed):
    residual, fn, scale, base, previous_pre = _inputs(tokens, dim)
    weight = mhc.pack_mix_weight(fn) if packed else fn
    got = mhc.delayed_pre_flat_jax(
        _jax(residual.flatten(1)),
        _jax(weight),
        _jax(scale),
        _jax(base),
        _jax(previous_pre),
        *CONSTANTS,
    )
    want = mhc_pre_delayed_torch(
        residual, fn, scale, base, *CONSTANTS, pre_mix=previous_pre
    )
    post, comb, value, new_pre = got
    assert post.shape == tuple(want[0].shape)
    assert value.dtype == jnp.bfloat16
    for g, w in ((post, want[0]), (comb, want[1]), (new_pre, want[3])):
        np.testing.assert_allclose(_np(g), _np(w), rtol=GATE_RTOL, atol=GATE_ATOL)
    np.testing.assert_allclose(_np(value), _np(want[2]), rtol=BF16_RTOL, atol=BF16_ATOL)


def _check_post_pre(tokens, dim, packed):
    residual, fn, scale, base, previous_pre = _inputs(tokens, dim)
    x, post_mix, comb_mix = _post_inputs(tokens, dim)
    weight = mhc.pack_mix_weight(fn) if packed else fn
    got = mhc.delayed_post_pre_flat_jax(
        _jax(x),
        _jax(residual.flatten(1)),
        _jax(post_mix),
        _jax(comb_mix),
        _jax(weight),
        _jax(scale),
        _jax(base),
        _jax(previous_pre),
        *CONSTANTS,
    )
    # 1. Exact bitwise parity against two-step (_post_jax -> delayed_pre_flat_jax)
    step_flat = mhc._post_jax(
        _jax(x), _jax(residual.flatten(1)), _jax(post_mix), _jax(comb_mix)
    )
    step_pre = mhc.delayed_pre_flat_jax(
        step_flat,
        _jax(weight),
        _jax(scale),
        _jax(base),
        _jax(previous_pre),
        *CONSTANTS,
    )
    #    when both take the same path. From `_POST_PRE_PALLAS_MIN_TOKENS` the
    #    fused pass may take Pallas while the pre-only pass stays on XLA; the
    #    two differ only in accumulation order.
    flat_new, post, comb, value, new_pre = got
    np.testing.assert_array_equal(_np(flat_new), _np(step_flat))
    same_path = mhc._pallas_tile(tokens, MIXES) == mhc._pallas_tile(
        tokens, MIXES, fused_post=True
    )
    #    ``value`` is BF16, so a one-ulp rounding flip needs the BF16 tolerance.
    tolerances = ((GATE_RTOL, GATE_ATOL),) * 2 + ((BF16_RTOL, BF16_ATOL),)
    tolerances += ((GATE_RTOL, GATE_ATOL),)
    for g, s, (rtol, atol) in zip((post, comb, value, new_pre), step_pre, tolerances):
        if same_path:
            np.testing.assert_array_equal(_np(g), _np(s))
        else:
            np.testing.assert_allclose(_np(g), _np(s), rtol=rtol, atol=atol)

    # 2. Close agreement with vLLM PyTorch reference
    want_res = mhc_post_torch(x, residual, post_mix, comb_mix)
    want = mhc_pre_delayed_torch(
        want_res, fn, scale, base, *CONSTANTS, pre_mix=previous_pre
    )
    np.testing.assert_allclose(
        _np(flat_new.reshape(tokens, MULT, dim)),
        _np(want_res),
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    for g, w in ((post, want[0]), (comb, want[1]), (new_pre, want[3])):
        np.testing.assert_allclose(_np(g), _np(w), rtol=GATE_RTOL, atol=GATE_ATOL)
    np.testing.assert_allclose(_np(value), _np(want[2]), rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.cpu_test
def test_packed_weight_is_exact_split():
    _, fn, *_ = _inputs(1, 256)
    packed = mhc.pack_mix_weight(fn).float()  # [K, 128]: hi | mid | lo | 0
    parts = packed[:, : 3 * MIXES].T.reshape(3, MIXES, -1)
    torch.testing.assert_close(parts.sum(0), fn, rtol=0, atol=0)
    assert torch.count_nonzero(packed[:, 3 * MIXES :]) == 0


@pytest.mark.cpu_test
def test_torch_and_jax_packing_agree():
    _, fn, *_ = _inputs(1, 256)
    want = mhc._pack_mix_weight(_jax(fn))
    np.testing.assert_array_equal(_np(mhc.pack_mix_weight(fn)), _np(want))


@pytest.mark.cpu_test
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "tokens", [1, 8, 16, 24, 32, 48, 63, 100, 144, 192, 200, 288, 520]
)
def test_xla_path_matches_torch_reference(tokens, packed):
    """Token counts neither Pallas pass takes."""
    assert mhc._pallas_tile(tokens, MIXES) is None
    assert mhc._pallas_tile(tokens, MIXES, fused_post=True) is None
    _check(tokens, 256, packed)
    _check_post_pre(tokens, 256, packed)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens,pre_tile,post_tile",
    [
        (63, None, None),
        (64, 64, 64),
        (96, 96, 96),
        (100, None, None),
        (128, 128, 128),
        (144, None, None),
        (192, None, None),
        (256, 128, 128),
        (288, None, None),
        (320, None, None),
        (384, None, 96),
        (400, None, 80),  # TD8.237: C=80 DSpark draft (C * K)
        (512, None, 64),
        (520, None, None),
        (560, None, 80),  # TD8.237: C=112 DSpark draft
        (576, None, 96),
        (640, None, 64),  # tile 64 beat 80 here (Job 1314)
        (768, None, 96),
        (960, None, 96),
        (1024, 64, 64),
        (1056, None, 96),
        (1152, 64, 96),  # TD8.237: C=192 verify, was 64 (18 -> 12 grid steps)
        (2048, 64, 64),
        (4096, 64, 64),
    ],
)
def test_pallas_tile(tokens, pre_tile, post_tile):
    assert mhc._pallas_tile(tokens, MIXES) == pre_tile
    assert mhc._pallas_tile(tokens, MIXES, fused_post=True) == post_tile
    assert mhc._pallas_tile(tokens, 3 * 128, fused_post=True) is None


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize(
    "tokens,dim",
    [
        (64, 256),
        (96, 256),
        (256, 256),
        (384, 256),
        (512, 256),
        (576, 256),
        (960, 256),
        (1024, 256),
        (400, 256),
        (640, 256),
        (1152, 256),
        (128, 5120),
        (384, 5120),
        (768, 5120),
        (1152, 5120),
    ],
)
def test_pallas_path_matches_torch_reference(tokens, dim, packed):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel needs TPU hardware")
    _check(tokens, dim, packed)
    _check_post_pre(tokens, dim, packed)


@pytest.mark.parametrize("tokens", [960, 1152])
def test_fused_post_pre_tile_is_bitwise_invariant(tokens, monkeypatch):
    """TD8.237: the fused pass gives the same bits at tiles 96, 80 and 64."""
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel needs TPU hardware")
    residual, fn, scale, base, previous_pre = _inputs(tokens, 5120)
    x, post_mix, comb_mix = _post_inputs(tokens, 5120)
    args = (
        _jax(x),
        _jax(residual.flatten(1)),
        _jax(post_mix),
        _jax(comb_mix),
        _jax(mhc.pack_mix_weight(fn)),
        _jax(scale),
        _jax(base),
        _jax(previous_pre),
        *CONSTANTS,
    )
    outs = []
    for tile in (96, 80, 64):
        if tokens % tile:
            continue
        monkeypatch.setattr(mhc, "_pallas_tile", lambda *a, t=tile, **k: t)
        outs.append(mhc.delayed_post_pre_flat_jax(*args))
    assert len(outs) >= 2
    for got in outs[1:]:
        for g, w in zip(got, outs[0]):
            np.testing.assert_array_equal(_np(g), _np(w))


@pytest.mark.cpu_test
@pytest.mark.parametrize("mult", [2, 4])
def test_collapse_flat_matches_view_sum(mult):
    """Slice collapse equals the ``[T, M, D]`` view sum.

    Bitwise on TPU, where ``(s0 + s2) + (s1 + s3)`` is XLA's reduce order for
    the view; CPU XLA may sum in another order, so it only checks closeness.
    """
    rng = np.random.default_rng(2)
    tokens, dim = 16, 256
    flat = jnp.asarray(rng.standard_normal((tokens, mult * dim)) * 4, jnp.bfloat16)
    pre = jnp.asarray(rng.uniform(0.01, 1.5, (tokens, mult)), jnp.float32)

    def view_sum(f, p):
        view = f.reshape(tokens, mult, dim).astype(jnp.float32)
        return (p[..., None] * view).sum(1)

    want = jax.jit(view_sum)(flat, pre)
    got = jax.jit(mhc._collapse_flat)(flat, pre)
    if jax.default_backend() == "tpu":
        np.testing.assert_array_equal(_np(got), _np(want))
    else:
        np.testing.assert_allclose(_np(got), _np(want), rtol=1e-6, atol=1e-5)


@pytest.mark.cpu_test
@pytest.mark.parametrize("mult", [2, 4])
def test_mean_hc_streams_matches_mean(mult):
    """TD8.251: slice-sum stream mean; bitwise vs ``mean(1)`` is a TPU check."""
    rng = np.random.default_rng(5)
    residual = torch.from_numpy(rng.standard_normal((24, mult, 256)) * 8).bfloat16()
    got = mhc._mean_hc_streams(residual)
    want = residual.mean(1)
    assert got.shape == want.shape and got.dtype == torch.bfloat16
    if mult != 4:
        assert torch.equal(got, want)
    else:
        torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tokens", [768, 1152])
@pytest.mark.parametrize("compiled", [False, True])
def test_mean_hc_streams_bitwise_on_tpu(tokens, compiled, device):
    """TD8.251: on TPU the slice sum equals BF16 ``residual.mean(1)`` bitwise."""
    if device.type != "tpu":
        pytest.skip("bitwise parity is defined by the TPU reduce order")
    rng = np.random.default_rng(tokens)
    residual = (
        torch.from_numpy(rng.standard_normal((tokens, 4, 5120)) * 8)
        .bfloat16()
        .to(device)
    )
    mean = (
        (lambda r: r.mean(1))
        if not compiled
        else torch.compile(lambda r: r.mean(1), backend="tpu", dynamic=False)
    )
    streams = (
        mhc._mean_hc_streams
        if not compiled
        else torch.compile(mhc._mean_hc_streams, backend="tpu", dynamic=False)
    )
    assert torch.equal(streams(residual).cpu(), mean(residual).cpu())
