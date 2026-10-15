# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The V-space DCP merge operand (`vspace_weighted_heads_major`,
TPU_DCP_MERGE_VSPACE) on one TPU core: `where(w > 0, (o_h @ W_UV_h) * w, 0)`,
heads-major.

* f32 output against an f64 reference of the same bf16 inputs: f32-accurate
  (the MXU takes the bf16 operands exactly, accumulates in f32).
* bf16 output = the f32 output rounded once, bit for bit.
* NaN partials of zero weight come back 0; the per-channel W_UV scale is
  applied.
* Positive controls: reversed W_UV heads and a skipped weight both break it.
* The torch bridge (eager and torch.compile) returns the jax op's bits.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.mla import dcp_weighted_hm as k  # noqa: E402

H, C, V = 64, 512, 256


def _inputs(t, seed=0):
    r = np.random.default_rng(seed + t)
    o = jnp.asarray(r.standard_normal((t, H, C)), jnp.bfloat16)
    w = r.random((t, H)).astype(np.float32)
    w[r.random((t, H)) < 0.1] = 0.0
    # zero-weight rows may be NaN
    o = jnp.where(jnp.asarray(w == 0.0)[:, :, None], jnp.nan, o).astype(jnp.bfloat16)
    wuv = jnp.asarray(r.standard_normal((H, C, V)) / np.sqrt(C), jnp.bfloat16)
    return o, jnp.asarray(w), wuv


def _ref(o, w, wuv, scale=None):
    of = np.nan_to_num(np.asarray(o.astype(jnp.float32), np.float64))
    wf = np.asarray(wuv.astype(jnp.float32), np.float64)
    v = np.matmul(of.transpose(1, 0, 2), wf)  # [H, T, V]
    if scale is not None:
        v = v * np.asarray(scale, np.float64).reshape(H, 1, V)
    ww = np.asarray(w, np.float64).T[:, :, None]
    return np.where(ww > 0, v * ww, 0.0)


def _rel(got, ref):
    got = np.asarray(jnp.asarray(got, jnp.float32), np.float64)
    return float(np.abs(got - ref).max() / np.abs(ref).max())


@pytest.mark.parametrize("t", [128, 256, 512])
def test_f32_matches_f64(t):
    o, w, wuv = _inputs(t)
    got = k.vspace_weighted_heads_major(o, w, wuv, out_dtype=jnp.float32)
    assert got.shape == (H, t, V)
    assert bool(jnp.isfinite(got).all())
    assert _rel(got, _ref(o, w, wuv)) < 1e-5


@pytest.mark.parametrize("t", [128, 512])
def test_bf16_is_f32_rounded_once(t):
    o, w, wuv = _inputs(t)
    f32 = k.vspace_weighted_heads_major(o, w, wuv, out_dtype=jnp.float32)
    b16 = k.vspace_weighted_heads_major(o, w, wuv, out_dtype=jnp.bfloat16)
    assert b16.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(b16.view(jnp.int16)), np.asarray(f32.astype(jnp.bfloat16).view(jnp.int16))
    )


def test_scale_applied():
    o, w, wuv = _inputs(128)
    scale = jnp.asarray(np.random.default_rng(3).random((H, 1, V)) + 0.5, jnp.float32)
    got = k.vspace_weighted_heads_major(o, w, wuv, scale, out_dtype=jnp.float32)
    assert _rel(got, _ref(o, w, wuv, scale)) < 1e-5


@pytest.mark.parametrize("fault", ["swap_heads", "unweighted"])
def test_positive_controls(fault):
    o, w, wuv = _inputs(256)
    got = k.vspace_weighted_heads_major(o, w, wuv, out_dtype=jnp.float32, fault=fault)
    assert _rel(got, _ref(o, w, wuv)) > 1e-2


@pytest.mark.parametrize("compiled", [False, True])
def test_torch_bridge(compiled):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.layers.adapter import dcp_weighted_hm as hw

    o, w, wuv = _inputs(256)
    want = k.vspace_weighted_heads_major(o, w, wuv, out_dtype=jnp.bfloat16)

    def to_t(x):
        a = np.asarray(x.astype(jnp.float32))
        return torch.from_numpy(a).to(torch.bfloat16 if x.dtype == jnp.bfloat16 else torch.float32)

    ot, wt, wuvt = (to_t(x).to("tpu") for x in (o, w, wuv))
    fn = hw.vspace_weighted_heads_major
    if compiled:
        from torch_tpu._internal import compile as tpu_compile

        torch._dynamo.reset()
        fn = torch.compile(fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend())
    got = fn(ot, wt, wuvt).cpu()
    assert tuple(got.shape) == (H, 256, V) and got.dtype == torch.bfloat16
    np.testing.assert_array_equal(
        got.view(torch.int16).numpy(), np.asarray(want.view(jnp.int16))
    )
