# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TEST branch: the fused-MoE-decode op's exported program with
FUSED_MOE_DECODE_PREFILL_IMPL=v3, at Qwen3.5-397B-A17B-FP8's MoE geometry on
8 devices (EP8, 64 experts a rank, TP-sliced shared expert).

Up to the cap the program must be the decode kernel; above it, fused EP MoE
v3 sequence parallel. Both are held to the GMM MoE path (the program with the
cap at 0 and no v3: GMM + shared expert + all-reduce, what serving runs with
the kernel off), with a must-fail control: the same GMM with every token's
expert ids shifted by one."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.layers.adapter import fused_moe_decode as adapter

# T=1004 is padded to 1008 (126 rows a rank); an odd T is not a served bucket
# (the GMM reference's SparseCore gather needs an even row count).

N, H, INTER, INTER_S, EL, TOPK = 8, 4096, 1024, 128, 64, 10
E = N * EL
F8 = jnp.float8_e4m3fn
CAP = 256
# Batch-wide relative L2 against the GMM path. The worst single row of a
# 2048-token batch is a tail statistic of two fp8 paths (v3's e4m3 wire vs
# GMM's bf16 activations): measured 0.082 at T=2048, so it gets its own band.
BOUND = 0.08
ROW_BOUND = 0.12


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("requires 8 TPU devices")
    return Mesh(np.asarray(jax.devices()[:N]), ("d",))


def _weights(mesh, rank_order):
    """Mesh index i holds EP rank rank_order[i]'s 64 experts (weights keyed by
    global expert id) and its TP slice of the shared expert."""
    sh = NamedSharding(mesh, P("d"))
    owner = jax.device_put(np.asarray(rank_order, np.int32).reshape(N, 1), sh)

    def gen(r):
        ks = jax.random.split(jax.random.fold_in(jax.random.key(7), r[0, 0]), 4)

        def q(w, axis):
            s = jnp.max(jnp.abs(w), axis=axis, keepdims=True) / 448.0
            return (w / s).astype(F8), s

        w1, s1 = q(jax.random.normal(ks[0], (EL, H, 2 * INTER)) * 0.02, 1)
        w2, s2 = q(jax.random.normal(ks[1], (EL, INTER, H)) * 0.03, 1)
        sw1, ss1 = q(jax.random.normal(ks[2], (H, 2 * INTER_S)) * 0.02, 0)
        sw2, ss2 = q(jax.random.normal(ks[3], (INTER_S, H)) * 0.03, 0)
        return (w1, w2, s1[:, :, None, :], s2[:, :, None, :], sw1, ss1, sw2, ss2)

    weights = jax.jit(
        jax.shard_map(
            gen, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
        )
    )(owner)
    gate = jax.device_put(
        jnp.asarray(np.random.default_rng(1).normal(size=(H, 1)) * 0.02, jnp.bfloat16),
        NamedSharding(mesh, P()),
    )
    rank = jax.device_put(
        np.asarray([[i, rank_order[i] * EL] for i in range(N)], np.int32), sh
    )
    return weights, gate, rank


def _inputs(mesh, t, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(t, H)).astype(np.float32)
    logits = (rng.normal(size=(t, E)) * 2).astype(np.float32)
    # vLLM's softmax top-k with renormalize (Qwen3.5 norm_topk_prob=True).
    ti = np.argsort(-logits, axis=1)[:, :TOPK].astype(np.int32)
    sel = np.take_along_axis(logits, ti, 1)
    tw = np.exp(sel - sel.max(1, keepdims=True))
    tw = (tw / tw.sum(1, keepdims=True)).astype(np.float32)
    sh = NamedSharding(mesh, P("d"))

    def rep(a, dtype=None):
        a = jnp.asarray(a) if dtype is None else jnp.asarray(a, dtype)
        return jax.device_put(jnp.tile(a, (N, 1)), sh)

    return rep(x, jnp.bfloat16), rep(tw), rep(ti), rep(logits), ti


def _run(mesh, weights, gate, rank, x, tw, ti, logits, *, max_tokens, v3):
    fn = functools.partial(
        adapter._program,
        mesh=mesh,
        n_ranks=N,
        topk=TOPK,
        expert_parallel=True,
        max_tokens=max_tokens,
        v3=v3,
    )
    out = jax.jit(fn)(x, *weights, gate, tw, ti, logits, rank)
    t = x.shape[0] // N
    return np.asarray(out.astype(jnp.float32)).reshape(N, t, H)


def _rel(a, ref):
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def _max_row_rel(a, ref):
    return float(
        (
            np.linalg.norm(a - ref, axis=1)
            / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
        ).max()
    )


@pytest.mark.parametrize(
    "t,permuted",
    [
        (64, False),
        (256, False),
        (1024, False),
        (2048, False),
        (1004, False),
        (1024, True),
    ],
)
def test_combined_op_matches_gmm(mesh, t, permuted, monkeypatch):
    traced = []
    for name in ("_v3_prefill", "_kernel_body", "_gmm_layer"):
        real = getattr(adapter, name)

        def spy(*a, _real=real, _name=name, **k):
            traced.append(_name)
            return _real(*a, **k)

        monkeypatch.setattr(adapter, name, spy)
    order = (3, 0, 6, 1, 7, 2, 5, 4) if permuted else tuple(range(N))
    weights, gate, rank = _weights(mesh, order)
    v3 = adapter.V3Prefill(
        rank_order=order, local_experts=EL, hidden=H, inter=INTER, renormalize=True
    )
    x, tw, ti, logits, ti_np = _inputs(mesh, t, seed=t + permuted)
    run = functools.partial(_run, mesh, weights, gate, rank, x, tw)
    got = run(ti, logits, max_tokens=CAP, v3=v3)
    want = "_kernel_body" if t <= CAP else "_v3_prefill"
    assert set(traced) == {want}, traced
    ref = run(ti, logits, max_tokens=0, v3=None)
    shifted = jax.device_put(
        jnp.tile(jnp.asarray((ti_np + 1) % E), (N, 1)), NamedSharding(mesh, P("d"))
    )
    broken = run(shifted, logits, max_tokens=0, v3=None)
    assert np.isfinite(got).all()
    assert all(np.array_equal(got[0], got[r]) for r in range(N)), "ranks disagree"
    rel, row = _rel(got[0], ref[0]), _max_row_rel(got[0], ref[0])
    ctl = _rel(got[0], broken[0])
    print(
        f"t={t} permuted={permuted} path={'kernel' if t <= CAP else 'v3'} "
        f"rel={rel:.4f} max_row={row:.4f} control={ctl:.4f}"
    )
    assert rel < BOUND and row < ROW_BOUND
    assert ctl > BOUND, "control: shifted expert ids must fail the bound"
