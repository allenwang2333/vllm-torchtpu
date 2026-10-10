# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""8-device parity for the fused MoE + shared expert + in-kernel all-reduce,
with the experts TP-sliced (every rank holds all E) or EP-sharded (rank r holds
global experts [r*E, (r+1)*E)); the shared expert is always a TP slice."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental import moe_fused_tp as kernel

N, H, INTER, INTER_S, E, TOPK = 8, 1024, 256, 128, 64, 10
F8 = jnp.float8_e4m3fn


@pytest.fixture(scope="module")
def setup():
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("requires 8 TPU devices")
    mesh = Mesh(np.asarray(jax.devices()[:N]), ("d",))
    sh = NamedSharding(mesh, P("d"))
    rank = jax.device_put(np.arange(N, dtype=np.int32).reshape(N, 1), sh)

    def gen(r):
        ks = jax.random.split(jax.random.fold_in(jax.random.key(3), r[0, 0]), 4)

        def q(w, axis):
            s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
            return (w / s).astype(F8), s

        w1, s1 = q(jax.random.normal(ks[0], (E, H, 2 * INTER)) * 0.02, 1)
        w2, s2 = q(jax.random.normal(ks[1], (E, INTER, H)) * 0.02, 1)
        w1s, s1s = q(jax.random.normal(ks[2], (H, 2 * INTER_S)) * 0.02, 0)
        w2s, s2s = q(jax.random.normal(ks[3], (INTER_S, H)) * 0.02, 0)
        return w1, w2, s1[:, :, None, :], s2[:, :, None, :], w1s, s1s, w2s, s2s

    weights = jax.jit(
        jax.shard_map(
            gen, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
        )
    )(rank)
    wg = jax.device_put(
        jnp.asarray(np.random.default_rng(1).normal(size=(H, 1)) * 0.02, jnp.bfloat16),
        NamedSharding(mesh, P()),
    )
    return mesh, sh, rank, weights, wg


def _ffn(x, w1, s1, w2, s2):
    i = w2.shape[0]
    gu = jnp.dot(x, w1.astype(jnp.float32) * s1)
    return jnp.dot(jax.nn.silu(gu[:, :i]) * gu[:, i:], w2.astype(jnp.float32) * s2)


def _routed(x, w1, w2, s1, s2, tw, ti, *, skip_expert=None):
    out = jnp.zeros((x.shape[0], H), jnp.float32)
    for e in range(E):
        coeff = jnp.sum(
            jnp.where(ti == e, tw.astype(jnp.float32), 0.0), 1, keepdims=True
        )
        if e != skip_expert:
            out = out + coeff * _ffn(x, w1[e], s1[e, 0], w2[e], s2[e, 0])
    return out


def _reference(
    rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti, skip_expert=None, ep=False
):
    """The decomposed layer in f32: routed experts + gated shared expert + psum."""
    if ep:
        ti = ti - rank.reshape(()) * E
    xf = x.astype(jnp.float32)
    shared = _ffn(xf, w1s, s1s, w2s, s2s)
    shared = jax.nn.sigmoid(jnp.dot(x, wg, preferred_element_type=jnp.float32)) * shared
    routed = _routed(xf, w1, w2, s1, s2, tw, ti, skip_expert=skip_expert)
    return jax.lax.psum(routed + shared, "d").astype(x.dtype)


def _fused(rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti, ep=False):
    return kernel.fused_moe_tp_ar(
        rank.reshape(-1),
        x,
        w1,
        w2,
        s1,
        s2,
        w1s,
        s1s,
        w2s,
        s2s,
        wg,
        tw,
        ti,
        n_ranks=N,
        expert_start=rank.reshape(()) * E if ep else None,
    )


@pytest.mark.parametrize("m", [8, 32])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("ep", [False, True])
def test_matches_decomposed_layer(setup, m, padded, ep):
    mesh, sh, rank, weights, wg = setup
    rng = np.random.default_rng(m)
    x = rng.normal(size=(m, H)).astype(np.float32)
    n_global = N * E if ep else E
    ids = np.stack(
        [rng.choice(n_global, TOPK, replace=False) for _ in range(m)]
    ).astype(np.int32)
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    if padded:
        tw[m - 3 :] = 0
    args = (
        rank,
        jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
        *weights,
        wg,
        jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh),
        jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh),
    )
    specs = (P("d"),) * 10 + (P(), P("d"), P("d"))

    def run(fn):
        out = jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )(*args)
        return np.asarray(out.astype(jnp.float32)).reshape(N, m, H)

    def max_rel(a, ref):
        return (
            np.linalg.norm(a - ref, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        ).max()

    got = run(functools.partial(_fused, ep=ep))
    ref = run(functools.partial(_reference, ep=ep))
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    assert np.isfinite(got).all()
    # FP8 activations and a BF16 all-reduce against f32 math: a few percent per row.
    assert max_rel(got[0], ref[0]) < 0.08
    # Positive control: a reference missing one routed expert must fail the bound.
    # In EP mode the local id of expert ids[0, 0] is only that on its owner rank.
    broken = run(functools.partial(_reference, skip_expert=int(ids[0, 0]) % E, ep=ep))
    assert max_rel(got[0], broken[0]) > 0.08


def _fused_blocked(rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti, ep, ib):
    return kernel.fused_moe_tp_ar(
        rank.reshape(-1),
        x,
        w1,
        w2,
        s1,
        s2,
        w1s,
        s1s,
        w2s,
        s2s,
        wg,
        tw,
        ti,
        n_ranks=N,
        expert_start=rank.reshape(()) * E if ep else None,
        i_block=ib,
    )


@pytest.mark.parametrize("m", [8, 32])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("ib", [128, 256])
def test_blocked_matches_decomposed_layer(setup, m, ep, ib):
    """Experts streamed in intermediate blocks (the mode GLM-5.2's EP experts
    take because whole-expert slots do not fit VMEM), forced here by i_block
    at a small shape: ib 128 is two blocks per expert, ib 256 one."""
    mesh, sh, rank, weights, wg = setup
    rng = np.random.default_rng(100 + m)
    x = rng.normal(size=(m, H)).astype(np.float32)
    n_global = N * E if ep else E
    ids = np.stack(
        [rng.choice(n_global, TOPK, replace=False) for _ in range(m)]
    ).astype(np.int32)
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    tw[m - 3 :] = 0
    args = (
        rank,
        jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
        *weights,
        wg,
        jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh),
        jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh),
    )
    specs = (P("d"),) * 10 + (P(), P("d"), P("d"))

    def run(fn):
        out = jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )(*args)
        return np.asarray(out.astype(jnp.float32)).reshape(N, m, H)

    def max_rel(a, ref):
        return (
            np.linalg.norm(a - ref, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        ).max()

    got = run(functools.partial(_fused_blocked, ep=ep, ib=ib))
    ref = run(functools.partial(_reference, ep=ep))
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    assert np.isfinite(got).all()
    assert max_rel(got[0], ref[0]) < 0.08
    # Positive control: drop row 0's heaviest route (Dirichlet weights can make
    # an arbitrary route too light for the bound to see).
    heavy = int(ids[0, int(np.argmax(tw[0]))]) % E
    broken = run(functools.partial(_reference, skip_expert=heavy, ep=ep))
    assert max_rel(got[0], broken[0]) > 0.08


def test_ring_plan_blocks_only_oversized_experts():
    """Whole-expert slots wherever three of them fit (Qwen3.5-397B at TP8 and
    EP8, GLM-5.2 at TP16, up to the 128-token decode cap and beyond); blocks
    only for GLM-5.2's whole EP experts."""
    if jax.default_backend() != "tpu":
        pytest.skip("reads the TPU's VMEM capacity")
    for m in (16, 128, 256):
        assert kernel.plan_weight_ring(m, 4096, 128, 128, 8)[0] is None
        assert kernel.plan_weight_ring(m, 4096, 1024, 128, 8)[0] is None
        assert kernel.plan_weight_ring(m, 6144, 128, 128, 16)[0] is None
        ib, nb = kernel.plan_weight_ring(m, 6144, 2048, 128, 16)
        assert ib is not None and 2048 % ib == 0 and nb >= 3
    assert kernel.plan_weight_ring(16, 4096, 1024, 128, 8, i_block=256)[0] == 256


def _reference_glm(
    rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti, scale_shared=False
):
    """GLM-5.2's layer with the kernel off: vLLM's MoERunner scales the routed
    output by routed_scaling_factor (2.5), adds the ungated shared expert, then
    all-reduces. scale_shared=True is the wrong placement (scaling the sum)."""
    ti = ti - rank.reshape(()) * E
    xf = x.astype(jnp.float32)
    shared = _ffn(xf, w1s, s1s, w2s, s2s)
    routed = GLM_SCALE * _routed(xf, w1, w2, s1, s2, tw, ti)
    total = GLM_SCALE * shared + routed if scale_shared else shared + routed
    return jax.lax.psum(total, "d").astype(x.dtype)


GLM_SCALE = 2.5


def _fused_glm(rank, x, w1, w2, s1, s2, w1s, s1s, w2s, s2s, wg, tw, ti, ib):
    del wg  # GLM-5.2's shared expert has no gate
    return kernel.fused_moe_tp_ar(
        rank.reshape(-1),
        x,
        w1,
        w2,
        s1,
        s2,
        w1s,
        s1s,
        w2s,
        s2s,
        None,
        tw,
        ti,
        n_ranks=N,
        expert_start=rank.reshape(()) * E,
        i_block=ib,
        routed_scale=GLM_SCALE,
    )


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("ib", [None, 128])
def test_glm_semantics_ungated_shared_routed_scale(setup, m, ib):
    """GLM-5.2 semantics under EP: no shared-expert gate (coefficient 1) and
    routed_scaling_factor 2.5 on the routed experts only, in the blocked mode
    (ib 128 = two blocks per expert) and the default plan, against the
    kernel-off composition. Must-fail controls: the scale applied to the
    shared expert too, and a sigmoid-gated shared expert (Qwen's convention)."""
    mesh, sh, rank, weights, wg = setup
    rng = np.random.default_rng(200 + m)
    x = rng.normal(size=(m, H)).astype(np.float32)
    ids = np.stack(
        [rng.choice(N * E, TOPK, replace=False) for _ in range(m)]
    ).astype(np.int32)
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    tw[m - 2 :] = 0  # padded bucket rows: zero routing weight
    args = (
        rank,
        jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
        *weights,
        wg,
        jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh),
        jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh),
    )
    specs = (P("d"),) * 10 + (P(), P("d"), P("d"))

    def run(fn):
        out = jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )(*args)
        return np.asarray(out.astype(jnp.float32)).reshape(N, m, H)

    def max_rel(a, ref):
        return (
            np.linalg.norm(a - ref, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        ).max()

    got = run(functools.partial(_fused_glm, ib=ib))
    ref = run(_reference_glm)
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    assert np.isfinite(got).all()
    assert max_rel(got[0], ref[0]) < 0.08
    wrong_scale = run(functools.partial(_reference_glm, scale_shared=True))
    assert max_rel(got[0], wrong_scale[0]) > 0.08
    gated = run(functools.partial(_reference, ep=True))  # gated, unscaled
    assert max_rel(got[0], gated[0]) > 0.08


def test_ring_plan_never_asserts_and_gates_on_expert_size():
    """No block width narrower than i (Qwen3.5 TP8, i 128) keeps whole experts
    at every token count instead of raising; small experts stay whole even
    when the token-side buffers alone would not fit."""
    if jax.default_backend() != "tpu":
        pytest.skip("reads the TPU's VMEM capacity")
    for m in (2048, 4096, 8192):
        assert kernel.plan_weight_ring(m, 4096, 128, 128, 8)[0] is None
        assert kernel.plan_weight_ring(m, 4096, 1024, 128, 8)[0] is None
        assert kernel.plan_weight_ring(m, 6144, 128, 128, 16)[0] is None
