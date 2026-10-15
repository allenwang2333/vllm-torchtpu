# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""FUSED_MOE_PREFILL_KERNEL_V2 (stage A) against the default build, one device:
the same op bit for bit -- weight ring depth, explicit-MXU matmuls with the
per-N-tile down epilogue, the inline row gather, 256-row tiles and the flat
combine pointer all restructure the issue, not the arithmetic. Also the
norm-side quantization (fused_add_rms_norm_quant_moe) against the op's own
_quantize_rows, and the op fed from it against the op fed y."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels import fused_add_rmsnorm as fnorm
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import fused_moe_prefill
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel as v1,
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel_v2 as v2,
)

# GLM-5.2's per-rank MoE slice at TP16 (i = i_s = 128, top-8, routed scale
# 2.5, ungated shared expert) with fewer experts and a smaller hidden size.
H, INTER, INTER_S, E, TOPK = 1024, 128, 128, 64, 8
ROUTED_SCALE = 2.5
F8 = jnp.float8_e4m3fn


@pytest.fixture(scope="module")
def setup():
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    tpu = jax.devices()[0]
    if "7" not in tpu.device_kind:
        pytest.skip("the explicit-MXU path needs MXU accumulators (v7x)")
    ks = jax.random.split(jax.random.key(5), 4)

    def q(w, axis):
        s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
        return (w / s).astype(F8), s

    w1, w1s = q(jax.random.normal(ks[0], (E, H, 2 * INTER)) * 0.02, 1)
    w2, w2s = q(jax.random.normal(ks[1], (E, INTER, H)) * 0.02, 1)
    sw1, sw1s = q(jax.random.normal(ks[2], (H, 2 * INTER_S)) * 0.02, 0)
    sw2, sw2s = q(jax.random.normal(ks[3], (INTER_S, H)) * 0.02, 0)
    weights = (w1, w2, w1s[:, :, None, :], w2s[:, :, None, :], sw1, sw1s.reshape(-1), sw2, sw2s.reshape(-1))
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("d",))
    return weights, mesh


def _routing(m, seed, hot):
    rng = np.random.default_rng(seed)
    g = rng.random((m, E))
    if hot:  # one expert in every token: multi-tile visits and tails
        g[:, 7] = 2.0
    ids = np.argsort(-g, axis=1)[:, :TOPK].astype(np.int32)
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    tw[-3:] = 0.0  # padded rows
    return jnp.asarray(tw, jnp.bfloat16), jnp.asarray(ids)


def _run(mesh, fn, *args):
    return jax.jit(
        jax.shard_map(fn, mesh=mesh, in_specs=(P(),) * len(args), out_specs=P(), check_vma=False)
    )(*args)


@pytest.mark.parametrize("m,hot", [(256, False), (1024, True), (4096, True)])
def test_v2_matches_v1_bit_for_bit(setup, m, hot):
    weights, mesh = setup
    x = jax.random.normal(jax.random.key(m), (m, H)).astype(jnp.bfloat16)
    tw, ti = _routing(m, m, hot)

    def body(op, **kw):
        def f(x, tw, ti, *w):
            w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
            return op(x, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, None, tw, ti, axis_name="d",
                      routed_scale=ROUTED_SCALE, **kw)

        return f

    ref = _run(mesh, body(fused_moe_prefill), x, tw, ti, *weights)
    got = _run(mesh, body(v2.fused_moe_prefill, **v2.STAGE_A), x, tw, ti, *weights)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


@pytest.mark.parametrize("m", [512, 4096])
def test_norm_quant_moe_is_quantize_rows(setup, m):
    weights, mesh = setup
    x0 = jax.random.normal(jax.random.key(1), (m, H)).astype(jnp.bfloat16)
    r0 = (jax.random.normal(jax.random.key(2), (m, H)) * 0.5).astype(jnp.bfloat16)
    nw = (1.0 + 0.05 * jax.random.normal(jax.random.key(3), (H,))).astype(jnp.bfloat16)
    y_ref, res_ref = fnorm.fused_add_rms_norm(x0, r0, nw, eps=1e-5)
    y, x_q, x_scale, res = fnorm.fused_add_rms_norm_quant_moe(x0, r0, nw, eps=1e-5)
    q_ref, s_ref = jax.jit(lambda y: v1._quantize_rows(y, F8))(y_ref)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(y_ref))
    np.testing.assert_array_equal(np.asarray(res), np.asarray(res_ref))
    np.testing.assert_array_equal(np.asarray(x_q).view(np.uint8), np.asarray(q_ref).view(np.uint8))
    np.testing.assert_array_equal(np.asarray(x_scale), np.asarray(s_ref))

    tw, ti = _routing(m, 9, True)

    def fed(x_q, x_scale, tw, ti, *w):
        w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
        return v2.fused_moe_prefill(None, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, None, tw, ti, axis_name="d",
                                    routed_scale=ROUTED_SCALE, x_q=x_q, x_scale=x_scale, **v2.STAGE_A)

    def own(y, tw, ti, *w):
        w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
        return fused_moe_prefill(y, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, None, tw, ti, axis_name="d",
                                 routed_scale=ROUTED_SCALE)

    np.testing.assert_array_equal(
        np.asarray(_run(mesh, fed, x_q, x_scale, tw, ti, *weights)),
        np.asarray(_run(mesh, own, y_ref, tw, ti, *weights)),
    )


@pytest.mark.parametrize("v2_flag", [False, True])
def test_prefill_layer_on_norm_rows(setup, monkeypatch, v2_flag):
    """TPU_FUSED_ADD_RMSNORM_QUANT_MOE's op path: prefill_layer fed the norm's
    (x_q, x_scale) equals prefill_layer quantizing y itself, with
    FUSED_MOE_PREFILL_KERNEL_V2 off (v2 at v1's levers vs v1) and on."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import fused_moe_prefill as adapter

    monkeypatch.setattr(envs, "FUSED_MOE_PREFILL_KERNEL_V2", v2_flag, raising=False)
    weights, mesh = setup
    m = 1024
    x0 = jax.random.normal(jax.random.key(11), (m, H)).astype(jnp.bfloat16)
    r0 = (jax.random.normal(jax.random.key(12), (m, H)) * 0.5).astype(jnp.bfloat16)
    nw = (1.0 + 0.05 * jax.random.normal(jax.random.key(13), (H,))).astype(jnp.bfloat16)
    y, x_q, x_scale, _ = fnorm.fused_add_rms_norm_quant_moe(x0, r0, nw, eps=1e-5)
    tw, ti = _routing(m, 21, True)

    def own(y, tw, ti, *w):
        w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
        return adapter.prefill_layer(y, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, None, tw, ti, None, ROUTED_SCALE)

    def fed(y, x_q, x_scale, tw, ti, *w):
        w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s = w
        return adapter.prefill_layer(y, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, None, tw, ti, None, ROUTED_SCALE,
                                     x_q=x_q, x_scale=x_scale)

    np.testing.assert_array_equal(
        np.asarray(_run(mesh, fed, y, x_q, x_scale, tw, ti, *weights)),
        np.asarray(_run(mesh, own, y, tw, ti, *weights)),
    )
