"""D-sharded mHC residual: the 1/tp-width seam against the replicated one.

The JAX halves are checked on CPU (XLA path) by splitting the hidden
dimension into emulated TP ranks and summing their partial sums the way the
seam's all-reduce does. The torch wiring is checked with stand-in ops and
collectives.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc

MULT, MIXES = 4, 24
CONSTANTS = (1e-6, 1e-6, 1e-6, 2.0, 20)
NORM_EPS = 1e-6


def _inputs(tokens, dim, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((tokens, dim)), jnp.bfloat16)
    flat = jnp.asarray(rng.standard_normal((tokens, MULT * dim)), jnp.bfloat16)
    post = jnp.asarray(rng.uniform(0.0, 2.0, (tokens, MULT, 1)), jnp.float32)
    comb = rng.uniform(0.0, 1.0, (tokens, MULT, MULT))
    comb = jnp.asarray(comb / comb.sum(-1, keepdims=True), jnp.float32)
    fn = torch.from_numpy(0.02 * rng.standard_normal((MIXES, MULT * dim))).float()
    packed = jnp.asarray(mhc.pack_mix_weight(fn).float().numpy(), jnp.bfloat16)
    scale = jnp.asarray(1.0 + 0.1 * rng.standard_normal(3), jnp.float32)
    base = jnp.asarray(0.5 * rng.standard_normal(MIXES), jnp.float32)
    pre = jnp.asarray(rng.uniform(0.0, 1.0, (tokens, MULT)), jnp.float32)
    norm_w = jnp.asarray(1.0 + 0.1 * rng.standard_normal(dim), jnp.bfloat16)
    return x, flat, post, comb, packed, fn, scale, base, pre, norm_w


def _rank_cols(dim, tp, r):
    width = dim // tp
    return slice(r * width, (r + 1) * width)


def _stream_cols(a, dim, cols):
    return jnp.concatenate(
        [a[..., s * dim + cols.start : s * dim + cols.stop] for s in range(MULT)], -1
    )


def _f32(a):
    return np.asarray(jnp.asarray(a).astype(jnp.float32))


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [8, 40])
def test_sharded_seam_matches_replicated(tokens):
    """Two emulated ranks reproduce `delayed_post_pre_flat_jax` + rms_norm."""
    dim, tp = 256, 2
    x, flat, post, comb, packed, _, scale, base, pre, norm_w = _inputs(tokens, dim)
    assert mhc._sharded_tile(tokens) is None  # CPU: XLA path
    want = mhc.delayed_post_pre_flat_jax(
        x, flat, post, comb, packed, scale, base, pre, *CONSTANTS
    )
    flat_ref, post_ref, comb_ref, value_ref, pre_ref = want
    v32 = value_ref.astype(jnp.float32)
    value_ref = (
        v32
        * jax.lax.rsqrt(jnp.mean(v32 * v32, -1, keepdims=True) + NORM_EPS)
        * norm_w.astype(jnp.float32)
    ).astype(jnp.bfloat16)

    parts = []
    for r in range(tp):
        cols = _rank_cols(dim, tp, r)
        weight = jnp.concatenate(
            [packed[s * dim + cols.start : s * dim + cols.stop] for s in range(MULT)]
        )
        parts.append(
            mhc.delayed_post_pre_partial_flat_jax(
                x[:, cols], _stream_cols(flat, dim, cols), post, comb, weight, pre
            )
        )
    stats = sum(p[2] for p in parts)
    assert stats.shape == (tokens, MIXES + 2)
    for r, (flat_new, value, _) in enumerate(parts):
        cols = _rank_cols(dim, tp, r)
        np.testing.assert_array_equal(
            _f32(flat_new), _f32(_stream_cols(flat_ref, dim, cols))
        )
        got = mhc.sharded_gates_norm_jax(
            stats, value, scale, base, norm_w[cols], *CONSTANTS, dim, NORM_EPS
        )
        g_post, g_comb, g_value, g_pre = got
        for g, w in ((g_post, post_ref), (g_comb, comb_ref), (g_pre, pre_ref)):
            np.testing.assert_allclose(_f32(g), _f32(w), rtol=1e-5, atol=1e-6)
        # One BF16 ulp: the norm's square sum adds the ranks' partials.
        np.testing.assert_allclose(
            _f32(g_value), _f32(value_ref[:, cols]), rtol=1e-2, atol=1e-2
        )


@pytest.mark.cpu_test
def test_partial_pass_unpacked_weight_matches_packed():
    tokens, dim = 8, 128
    x, flat, post, comb, packed, fn, *_rest = _inputs(tokens, dim, seed=3)
    pre = _rest[-2]
    a = mhc.delayed_post_pre_partial_flat_jax(x, flat, post, comb, packed, pre)
    b = mhc.delayed_post_pre_partial_flat_jax(
        x, flat, post, comb, jnp.asarray(fn.numpy()), pre
    )
    for got, want in zip(a, b):
        np.testing.assert_array_equal(_f32(got), _f32(want))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens,tile",
    [(1152, 384), (768, 384), (512, 256), (640, 128), (96, 96), (40, None)],
)
def test_sharded_tile(tokens, tile):
    assert mhc._sharded_tile(tokens) == tile


@pytest.mark.cpu_test
def test_sharded_tile_takes_pallas_pass(monkeypatch):
    calls = []

    def fake_sums(sub_x, flat, post, comb, weight, pre, mixes, tile):
        calls.append((sub_x.shape, flat.shape, post.shape, comb.shape, tile))
        tokens, dim = sub_x.shape
        value = jnp.ones((tokens, dim), flat.dtype)
        return flat, jnp.zeros((tokens, mixes)), jnp.ones((tokens, 1)), value

    monkeypatch.setattr(mhc, "_flat_post_pre_sums", fake_sums)
    tokens, dim = 96, 128
    x, flat, post, comb, packed, *_rest = _inputs(tokens, dim)
    pre = _rest[-2]
    _, value, stats = mhc.delayed_post_pre_partial_flat_jax(
        x, flat, post, comb, packed, pre
    )
    assert calls == [((96, 128), (96, 512), (96, 4), (96, 16), 96)]
    np.testing.assert_array_equal(_f32(stats[:, MIXES:]), [[1.0, 128.0]] * tokens)
    assert value.shape == (tokens, dim)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "flag,tp,expected", [(1, 8, True), (0, 8, False), (1, 1, False)]
)
def test_sharded_mhc_enabled(monkeypatch, flag, tp, expected):
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_SHARDED_MHC", bool(flag))
    assert mhc.sharded_mhc_enabled(tp) is expected
    monkeypatch.setattr(mhc, "get_tensor_model_parallel_world_size", lambda: tp)
    assert mhc.sharded_mhc_enabled() is expected


def _fake_tp(monkeypatch, rank=1, tp=4):
    gathered = []

    def all_gather(t, dim=-1):
        gathered.append(tuple(t.shape))
        return torch.cat([t] * tp, dim=dim)

    monkeypatch.setattr(mhc, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(mhc, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(mhc, "tensor_model_parallel_all_gather", all_gather)
    return gathered


@pytest.mark.cpu_test
def test_residual_helpers(monkeypatch):
    gathered = _fake_tp(monkeypatch)
    full = torch.arange(2 * MULT * 16, dtype=torch.float32).reshape(2, MULT, 16)
    assert mhc._rank_columns(16) == slice(4, 8)
    local = mhc._local_residual(full, 16)
    torch.testing.assert_close(local, full[..., 4:8])
    assert mhc._local_residual(local, 16) is local
    assert mhc.full_residual(full, 16) is full
    assert not gathered
    assert mhc.full_residual(local, 16).shape == (2, MULT, 16)
    assert gathered == [(2, MULT, 4)]


@pytest.mark.cpu_test
def test_local_mix_weight(monkeypatch):
    _fake_tp(monkeypatch, rank=1, tp=2)
    dim = 4
    packed = torch.arange(MULT * dim * 2, dtype=torch.bfloat16).reshape(MULT * dim, 2)
    rows = mhc._local_mix_weight(packed, dim, MULT)
    want_rows = [s * dim + c for s in range(MULT) for c in (2, 3)]
    torch.testing.assert_close(rows, packed[want_rows])
    unpacked = torch.arange(3 * MULT * dim, dtype=torch.float32).reshape(3, MULT * dim)
    cols = mhc._local_mix_weight(unpacked, dim, MULT)
    torch.testing.assert_close(cols, unpacked[:, want_rows])


def _sharded_call_stubs(monkeypatch, tokens, dim, tp, rank):
    events = []
    width = dim // tp

    def reduce_scatter(x, dim=-1):
        events.append(("rs", tuple(x.shape)))
        return x[..., rank * width : (rank + 1) * width]

    def all_reduce(t):
        events.append(("ar", tuple(t.shape)))
        return t * tp

    def all_gather(t, dim=-1):
        events.append(("ag", tuple(t.shape)))
        return torch.cat([t] * tp, dim=dim)

    def partial_op(x, flat, post, comb, fn, pre):
        events.append(("seam", tuple(x.shape), tuple(flat.shape), tuple(fn.shape)))
        value = x + 1
        stats = torch.ones(tokens, MIXES + 2)
        return flat + 1, value, stats

    def gates_op(stats, value, scale, base, weight, *args):
        events.append(("gates", tuple(weight.shape), args[-2:]))
        return (
            torch.zeros(tokens, MULT, 1),
            torch.zeros(tokens, MULT, MULT),
            value * 2,
            torch.zeros(tokens, MULT),
        )

    monkeypatch.setattr(mhc, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(mhc, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(mhc, "tensor_model_parallel_reduce_scatter", reduce_scatter)
    monkeypatch.setattr(mhc, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(mhc, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.setattr(mhc, "_post_pre_partial_flat_op", partial_op)
    monkeypatch.setattr(mhc, "_sharded_gates_op", gates_op)
    return events


@pytest.mark.cpu_test
@pytest.mark.parametrize("full_input", [True, False])
@pytest.mark.parametrize("with_norm", [True, False])
@pytest.mark.parametrize("capture_aux", [True, False])
def test_sharded_post_pre_wiring(monkeypatch, full_input, with_norm, capture_aux):
    tokens, dim, tp, rank = 4, 16, 4, 2
    events = _sharded_call_stubs(monkeypatch, tokens, dim, tp, rank)
    width = dim // tp
    x = torch.randn(tokens, dim)
    residual = torch.randn(tokens, MULT, dim if full_input else width)
    packed = torch.zeros(MULT * dim, 128, dtype=torch.bfloat16)
    norm_w = torch.arange(dim, dtype=torch.float32) if with_norm else None
    out = mhc._sharded_post_pre(
        x,
        residual,
        None,
        None,
        packed,
        None,
        None,
        CONSTANTS,
        torch.zeros(tokens, MULT),
        norm_w,
        NORM_EPS,
        capture_aux,
    )
    new_residual, post, comb, value, next_pre, aux = out
    local = residual[..., rank * width : (rank + 1) * width] if full_input else residual
    torch.testing.assert_close(new_residual, local + 1)
    expect_value = (x[:, rank * width : (rank + 1) * width] + 1) * (
        2 if with_norm else 1
    )
    torch.testing.assert_close(value, torch.cat([expect_value] * tp, -1))
    assert aux.shape == ((tokens, dim) if capture_aux else (0, dim))
    assert events[:4] == [
        ("rs", (tokens, dim)),
        ("seam", (tokens, width), (tokens, MULT * width), (MULT * width, 128)),
        ("ar", (tokens, MIXES + 2)),
        ("gates", (width,), (dim, NORM_EPS)),
    ]
    assert events[4] == ("ag", (tokens, width))
    assert len(events) == (6 if capture_aux else 5)


def _seam_args(tokens, dim):
    return (
        torch.randn(tokens, dim),
        torch.randn(tokens, MULT, dim),
        torch.zeros(tokens, MULT, 1),
        torch.zeros(tokens, MULT, MULT),
        torch.zeros(MULT * dim, 128, dtype=torch.bfloat16),
        torch.ones(3),
        torch.zeros(MIXES),
        *CONSTANTS,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("sharded", [True, False])
def test_shifted_post_pre_dispatch(monkeypatch, sharded):
    """reduce_results takes the sharded seam only with the flag on."""
    calls = []
    monkeypatch.setattr(mhc, "_post_pre_flat_op", object())
    monkeypatch.setattr(mhc, "sharded_mhc_enabled", lambda: sharded)
    monkeypatch.setattr(
        mhc, "_sharded_post_pre", lambda *a: calls.append(("sharded", a)) or "s"
    )
    monkeypatch.setattr(
        mhc,
        "tensor_model_parallel_all_reduce",
        lambda x: calls.append(("ar", x.shape)) or x,
    )

    def fake_delayed_post_pre(x, residual, *args, **kw):
        calls.append(("replicated", x.shape))
        t = x.shape[0]
        return residual, None, None, torch.zeros(t, x.shape[1]), None

    monkeypatch.setattr(mhc, "delayed_post_pre", fake_delayed_post_pre)
    args = list(_seam_args(4, 8))
    args[1] = SimpleNamespace(
        device=SimpleNamespace(type="tpu"), shape=(4, MULT, 8), dtype=torch.bfloat16
    )
    monkeypatch.setattr(mhc, "_default_pre_mix", lambda r, d: torch.zeros(4, MULT))
    monkeypatch.setattr(mhc, "rms_norm", lambda v, w, e: v)
    out = mhc.shifted_post_pre(*args, reduce_results=True)
    if sharded:
        assert out == "s" and calls[0][0] == "sharded"
        assert calls[0][1][7] == CONSTANTS
    else:
        assert [c[0] for c in calls] == ["ar", "replicated"]


@pytest.mark.cpu_test
def test_post_and_pre_gather_sharded_residual(monkeypatch):
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_SHARDED_MHC", True)
    gathered = _fake_tp(monkeypatch, rank=0, tp=2)
    tokens, dim = 2, 8
    x = torch.randn(tokens, dim)
    local = torch.randn(tokens, MULT, dim // 2)
    out = mhc.post(
        x,
        local,
        torch.ones(tokens, MULT, 1),
        torch.eye(MULT).expand(tokens, MULT, MULT),
    )
    assert out.shape == (tokens, MULT, dim) and gathered == [(tokens, MULT, dim // 2)]

    seen = []

    def fake_pre_delayed(residual, *args, **kwargs):
        seen.append(tuple(residual.shape))
        return None, None, torch.zeros(tokens, dim), None

    monkeypatch.setattr(mhc, "mhc_pre_delayed_torch", fake_pre_delayed)
    mhc.pre(local, None, None, None, *CONSTANTS, norm_weight=torch.ones(dim))
    assert seen == [(tokens, MULT, dim)]


@pytest.mark.cpu_test
def test_sharded_ops_registered_with_fakes(monkeypatch):
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as ctx

    class _Op:
        def __init__(self, name, fn, **kwargs):
            self.name, self.fn, self.kwargs = name, fn, kwargs

        def register_fake(self, fake):
            self.fake = fake

    monkeypatch.setattr(pallas, "jax_op", _Op)
    monkeypatch.setattr(
        ctx, "get_vllm_model_wrapper_context", lambda: SimpleNamespace(mesh=None)
    )
    for name in (
        "_pre_op",
        "_pre_flat_op",
        "_post_pre_flat_op",
        "_post_pre_partial_flat_op",
        "_sharded_gates_op",
    ):
        monkeypatch.setattr(mhc, name, None)
    mhc.initialize_delayed_mhc()
    partial, gates = mhc._post_pre_partial_flat_op, mhc._sharded_gates_op
    assert partial.fn is mhc.delayed_post_pre_partial_flat_jax
    assert gates.fn is mhc.sharded_gates_norm_jax
    assert len(gates.kwargs["input_partition_specs"]) == 12
    tokens, width = 6, 32
    flat_new, value, stats = partial.fake(
        torch.empty(tokens, width // MULT),
        torch.empty(tokens, width),
        None,
        None,
        None,
        torch.empty(tokens, MULT),
    )
    assert flat_new.shape == (tokens, width)
    assert value.shape == (tokens, width // MULT)
    assert stats.shape == (tokens, MIXES + 2)
    post, comb, normed, nxt = gates.fake(stats, value, None, None, None)
    assert post.shape == (tokens, MULT, 1) and comb.shape == (tokens, MULT, MULT)
    assert normed.shape == value.shape and nxt.shape == (tokens, MULT)
