# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""8-device parity for the fused MoE decode kernel (routed + shared expert +
in-kernel all-reduce, or its reduce-scatter-only DP mode), with the experts
TP-sliced (every rank holds all E) or EP-sharded (rank r holds global experts
[r*E, (r+1)*E)); the shared expert is always a TP slice."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
    fused_moe_decode,
    vmem_bytes,
)
from vllm_torchtpu.layers.adapter import fused_moe_decode as adapter

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

        w1, w1_scale = q(jax.random.normal(ks[0], (E, H, 2 * INTER)) * 0.02, 1)
        w2, w2_scale = q(jax.random.normal(ks[1], (E, INTER, H)) * 0.02, 1)
        shared_w1, shared_w1_scale = q(
            jax.random.normal(ks[2], (H, 2 * INTER_S)) * 0.02, 0
        )
        shared_w2, shared_w2_scale = q(jax.random.normal(ks[3], (INTER_S, H)) * 0.02, 0)
        return (
            w1,
            w2,
            w1_scale[:, :, None, :],
            w2_scale[:, :, None, :],
            shared_w1,
            shared_w1_scale,
            shared_w2,
            shared_w2_scale,
        )

    weights = jax.jit(
        jax.shard_map(
            gen, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
        )
    )(rank)
    shared_gate_weight = jax.device_put(
        jnp.asarray(np.random.default_rng(1).normal(size=(H, 1)) * 0.02, jnp.bfloat16),
        NamedSharding(mesh, P()),
    )
    return mesh, sh, rank, weights, shared_gate_weight


def _ffn(x, w1, w1_scale, w2, w2_scale):
    i = w2.shape[0]
    gu = jnp.dot(x, w1.astype(jnp.float32) * w1_scale)
    return jnp.dot(
        jax.nn.silu(gu[:, :i]) * gu[:, i:], w2.astype(jnp.float32) * w2_scale
    )


def _routed(x, w1, w2, w1_scale, w2_scale, tw, ti, *, skip_expert=None):
    out = jnp.zeros((x.shape[0], H), jnp.float32)
    for e in range(E):
        coeff = jnp.sum(
            jnp.where(ti == e, tw.astype(jnp.float32), 0.0), 1, keepdims=True
        )
        if e != skip_expert:
            out = out + coeff * _ffn(x, w1[e], w1_scale[e, 0], w2[e], w2_scale[e, 0])
    return out


def _reference(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    skip_expert=None,
    ep=False,
):
    """The decomposed layer in f32: routed experts + gated shared expert + psum."""
    if ep:
        ti = ti - rank.reshape(()) * E
    xf = x.astype(jnp.float32)
    shared = _ffn(xf, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale)
    shared = (
        jax.nn.sigmoid(
            jnp.dot(x, shared_gate_weight, preferred_element_type=jnp.float32)
        )
        * shared
    )
    routed = _routed(xf, w1, w2, w1_scale, w2_scale, tw, ti, skip_expert=skip_expert)
    return jax.lax.psum(routed + shared, "d").astype(x.dtype)


def _fused(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    ep=False,
):
    return fused_moe_decode(
        rank.reshape(-1),
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate_weight,
        tw,
        ti,
        n_ranks=N,
        expert_start=rank.reshape(()) * E if ep else None,
    )


def _fallback(
    rank,
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    ep=False,
):
    """The adapter's per-rank body with every token count above its cap: the GMM
    MoE + shared expert + psum path that replaces the kernel above it."""
    return adapter._layer(
        jnp.concatenate([rank, rank * E], axis=1),  # [[mesh index, first expert]]
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate_weight,
        tw,
        ti,
        n_ranks=N,
        topk=TOPK,
        expert_parallel=ep,
        max_tokens=1,
    )


@pytest.mark.parametrize("m", [8, 32])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("path", ["kernel", "gmm_fallback"])
def test_matches_decomposed_layer(setup, m, padded, ep, path):
    if path == "gmm_fallback" and (m != 32 or padded):
        pytest.skip("one fallback case per mode")
    mesh, sh, rank, weights, shared_gate_weight = setup
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
        shared_gate_weight,
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

    got = run(functools.partial(_fused if path == "kernel" else _fallback, ep=ep))
    ref = run(functools.partial(_reference, ep=ep))
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    assert np.isfinite(got).all()
    # FP8 activations and a BF16 all-reduce against f32 math: a few percent per row.
    assert max_rel(got[0], ref[0]) < 0.08
    # Positive control: a reference missing one routed expert must fail the bound.
    # In EP mode the local id of expert ids[0, 0] is only that on its owner rank.
    broken = run(functools.partial(_reference, skip_expert=int(ids[0, 0]) % E, ep=ep))
    assert max_rel(got[0], broken[0]) > 0.08


def test_vmem_bound_matches_build_limits():
    """The arming gate's estimate against the Qwen3.5-397B-A17B-FP8 sweep at 8
    ranks: TP (512 experts, i 128) builds at 512 tokens and not 1024; EP (64
    experts, i 1024) builds at 256 and not 512."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires a TPU")
    from jax.experimental.pallas import tpu as pltpu

    have = pltpu.get_tpu_info().vmem_capacity_bytes
    assert vmem_bytes(512, 4096, 128, 128, 512, 8, jnp.float8_e4m3fn) <= have
    assert vmem_bytes(1024, 4096, 128, 128, 512, 8, jnp.float8_e4m3fn) > have
    assert vmem_bytes(256, 4096, 1024, 128, 64, 8, jnp.float8_e4m3fn) <= have
    assert vmem_bytes(512, 4096, 1024, 128, 64, 8, jnp.float8_e4m3fn) > have
    # gather_rows (DP attention, global tokens) at the adapter's default cap.
    for i, e in ((128, 512), (1024, 64)):
        assert vmem_bytes(256, 4096, i, 128, e, 8, jnp.float8_e4m3fn, True) <= have


@pytest.mark.parametrize("m", [8, 32, 128])
@pytest.mark.parametrize("ep", [False, True])
def test_scatter_rows_is_own_slice_of_all_reduce(setup, m, ep):
    """scatter_rows=True (DP attention: reduce-scatter only, no all-gather)
    returns on rank r exactly rows [r*m/N, (r+1)*m/N) of the all-reduce mode's
    output. The BF16 wire (m/N % 16 == 0) is bit-identical; the FP32 wire used
    for small row counts differs only by the all-reduce's BF16 rounding."""
    mesh, sh, rank, weights, shared_gate_weight = setup
    rng = np.random.default_rng(100 + m)
    x = rng.normal(size=(m, H)).astype(np.float32)
    n_global = N * E if ep else E
    ids = np.stack(
        [rng.choice(n_global, TOPK, replace=False) for _ in range(m)]
    ).astype(np.int32)
    tw = rng.dirichlet(np.ones(TOPK), size=m).astype(np.float32)
    args = (
        rank,
        jax.device_put(jnp.tile(jnp.asarray(x, jnp.bfloat16), (N, 1)), sh),
        *weights,
        shared_gate_weight,
        jax.device_put(jnp.tile(jnp.asarray(tw, jnp.bfloat16), (N, 1)), sh),
        jax.device_put(jnp.tile(jnp.asarray(ids), (N, 1)), sh),
    )
    specs = (P("d"),) * 10 + (P(), P("d"), P("d"))

    def call(rank, x, *rest, scatter_rows):
        return fused_moe_decode(
            rank.reshape(-1),
            x,
            *rest,
            n_ranks=N,
            expert_start=rank.reshape(()) * E if ep else None,
            scatter_rows=scatter_rows,
        )

    def run(scatter_rows):
        out = jax.jit(
            jax.shard_map(
                functools.partial(call, scatter_rows=scatter_rows),
                mesh=mesh,
                in_specs=specs,
                out_specs=P("d"),
                check_vma=False,
            )
        )(*args)
        return np.asarray(out.astype(jnp.float32))

    full = run(False).reshape(N, m, H)[0]
    own = run(True)
    assert own.shape == (m, H)
    assert np.isfinite(own).all()
    if (m // N) % 16 == 0:
        np.testing.assert_array_equal(own, full)
    else:
        rel = np.linalg.norm(own - full, axis=1) / np.maximum(
            np.linalg.norm(full, axis=1), 1e-30
        )
        assert rel.max() < 1e-2
    # Positive control: a one-row shift of the slices must not match.
    assert not np.allclose(np.roll(own, 1, axis=0), full, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16, 24])
@pytest.mark.parametrize("ep", [False, True])
def test_gather_rows_matches_scatter_rows_on_gathered_inputs(setup, rows, ep):
    """gather_rows=True on each rank's own rows (the kernel all-gathers them)
    returns exactly what scatter_rows=True returns when XLA all-gathered the
    same rows rank-major first: bit-identical, on every gather wire (BF16 rows
    for 2/4/16, FP32 for 1/8/12/24) and reduce-scatter wire (rows % 16 ==
    0 is BF16, the others FP32)."""
    mesh, sh, rank, weights, shared_gate_weight = setup
    m = rows * N
    rng = np.random.default_rng(200 + m)
    x = jnp.asarray(rng.normal(size=(m, H)), jnp.bfloat16)
    n_global = N * E if ep else E
    ids = jnp.asarray(
        np.stack([rng.choice(n_global, TOPK, replace=False) for _ in range(m)]),
        jnp.int32,
    )
    tw = jnp.asarray(rng.dirichlet(np.ones(TOPK), size=m), jnp.bfloat16)
    specs = (P("d"),) * 10 + (P(), P("d"), P("d"))

    def call(rank, x, *rest, gather_rows):
        return fused_moe_decode(
            rank.reshape(-1),
            x,
            *rest,
            n_ranks=N,
            expert_start=rank.reshape(()) * E if ep else None,
            scatter_rows=True,
            gather_rows=gather_rows,
        )

    def run(gather_rows, x, tw, ids):
        if not gather_rows:  # every rank holds the whole gathered batch
            x, tw, ids = (jnp.tile(a, (N, 1)) for a in (x, tw, ids))
        out = jax.jit(
            jax.shard_map(
                functools.partial(call, gather_rows=gather_rows),
                mesh=mesh,
                in_specs=specs,
                out_specs=P("d"),
                check_vma=False,
            )
        )(
            rank,
            jax.device_put(x, sh),
            *weights,
            shared_gate_weight,
            jax.device_put(tw, sh),
            jax.device_put(ids, sh),
        )
        return np.asarray(out.astype(jnp.float32))

    want = run(False, x, tw, ids)
    got = run(True, x, tw, ids)
    assert got.shape == (m, H)
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got, want)
    # Must-fail control: ranks 0 and 1 swap their own rows, so the gathered
    # batch is not rank-major; rows of both blocks must then differ.
    perm = np.r_[rows : 2 * rows, 0:rows, 2 * rows : m]
    swapped = run(True, x[perm], tw[perm], ids[perm])
    for block in (slice(0, rows), slice(rows, 2 * rows)):
        assert not np.allclose(swapped[block], want[block], rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("rows", [4, 64])
def test_sequence_parallel_layer_matches_kernel_off_sp_path(setup, rows):
    """The adapter's sequence-parallel body on each rank's own rows against
    the kernel-off SP path computed independently in f32: all-gather, every
    expert on every row, reduce-scatter, plus the whole replicated shared
    expert on the rank's own rows. rows=4 (32 global tokens) is a decode
    bucket served by the kernel (gather_rows); rows=64 (512 global) is above
    the cap of 256 and takes the in-program GMM path."""
    import torch

    mesh, sh, rank, weights, shared_gate_weight = setup
    w1, w2, w1_scale, w2_scale = weights[:4]
    m = rows * N
    i_full = N * INTER_S
    rng = np.random.default_rng(300 + m)

    def q(w, axis):
        s = np.max(np.abs(w), axis=axis, keepdims=True) / 448.0
        return jnp.asarray(w / s, F8), s.astype(np.float32)

    full_w1, full_s1 = q(rng.normal(size=(H, 2 * i_full)) * 0.02, 0)
    full_w2, full_s2 = q(rng.normal(size=(i_full, H)) * 0.02, 0)
    full = (
        np.asarray(full_w1.astype(jnp.float32)),
        full_s1.reshape(-1),
        np.asarray(full_w2.astype(jnp.float32)),
        full_s2.reshape(-1),
        np.zeros(1, np.float32),
    )
    # Each rank's kernel slice, cut by the adapter's own slicing.
    slices = [
        [
            t.numpy()
            for t in adapter._shared_tp_slice(
                tuple(torch.from_numpy(a) for a in full), r, N
            )[:4]
        ]
        for r in range(N)
    ]
    per_rank = [np.concatenate([s[k] for s in slices]) for k in range(4)]
    per_rank[0] = jnp.asarray(per_rank[0], F8)
    per_rank[2] = jnp.asarray(per_rank[2], F8)
    rep = [
        jnp.asarray(np.tile(a, (N,) + (1,) * (a.ndim - 1)), d)
        for a, d in zip(full[:4], (F8, jnp.float32, F8, jnp.float32))
    ]
    x = jnp.asarray(rng.normal(size=(m, H)), jnp.bfloat16)
    ids = jnp.asarray(
        np.stack([rng.choice(N * E, TOPK, replace=False) for _ in range(m)]),
        jnp.int32,
    )
    tw = jnp.asarray(rng.dirichlet(np.ones(TOPK), size=m), jnp.bfloat16)
    put = functools.partial(jax.device_put, device=sh)
    args = (
        rank,
        put(x),
        w1,
        w2,
        w1_scale,
        w2_scale,
        put(per_rank[0]),
        put(per_rank[1]),
        put(per_rank[2]),
        put(np.tile(full[3], N)),
        shared_gate_weight,
        put(tw),
        put(ids),
        *(put(a) for a in rep),
    )
    specs = (P("d"),) * 10 + (P(), P("d"), P("d")) + (P("d"),) * 4

    def sp(rank, *a):
        return adapter._sp_layer(
            jnp.concatenate([rank, rank * E], axis=1),
            *a,
            n_ranks=N,
            topk=TOPK,
            max_tokens=256,
        )

    def reference(rank, x, w1, w2, w1_scale, w2_scale, *rest, skip_expert=None):
        _, _, _, _, gate_w, tw, ti, fw1, fs1, fw2, fs2 = rest
        xg = jax.lax.all_gather(x, "d", axis=0, tiled=True).astype(jnp.float32)
        tig = jax.lax.all_gather(ti, "d", axis=0, tiled=True) - rank.reshape(()) * E
        twg = jax.lax.all_gather(tw, "d", axis=0, tiled=True)
        routed = _routed(
            xg, w1, w2, w1_scale, w2_scale, twg, tig, skip_expert=skip_expert
        )
        routed = jax.lax.psum_scatter(routed, "d", scatter_dimension=0, tiled=True)
        shared = _ffn(x.astype(jnp.float32), fw1, fs1.reshape(1, -1), fw2, fs2)
        gate = jax.nn.sigmoid(jnp.dot(x, gate_w, preferred_element_type=jnp.float32))
        return (routed + gate * shared).astype(x.dtype)

    def run(fn):
        f = jax.jit(
            jax.shard_map(
                fn, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
            )
        )
        return f, np.asarray(f(*args).astype(jnp.float32))

    f, got = run(sp)
    _, ref = run(reference)
    hlo = f.lower(*args).as_text()
    assert ("fused_moe_decode_ag" in hlo) == (m <= 256), "wrong branch"
    assert got.shape == (m, H) and np.isfinite(got).all()

    def max_rel(a, b):
        return (
            np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-30)
        ).max()

    assert max_rel(got, ref) < 0.08
    # Positive control: a reference missing one routed expert fails the bound.
    _, broken = run(functools.partial(reference, skip_expert=int(ids[0, 0]) % E))
    assert max_rel(got, broken) > 0.08
