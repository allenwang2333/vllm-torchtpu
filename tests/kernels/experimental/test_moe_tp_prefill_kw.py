# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Probe-01 variants (moe_tp_prefill_kw) against the incumbent moe_tp_prefill.

L1 (routing_plan_v2) and L2 (combine_w) must be bit-identical to the
incumbent; removal arms only have to build and run (timing-only).
Runs in the Pallas TPU interpreter on CPU (small shapes) and compiled on TPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import moe_tp_prefill as mtp
from vllm_torchtpu.kernels.experimental import moe_tp_prefill_kw as kw

from test_moe_tp_prefill import _on_tpu, make_case  # noqa: E402 (rootdir-prepend import)


def _reference_plan(topk_ids, E, row_scale, route_valid, n_valid):
    """The 3-array sort + [E, R] broadcast reference of
    test_moe_tp_prefill.test_routing_plan_matches_reference."""
    T, K = topk_ids.shape
    n_routes = T * K
    e_flat = topk_ids.reshape(-1).astype(jnp.int32)
    iota = jnp.arange(n_routes, dtype=jnp.int32)
    if n_valid is not None:
        e_flat = jnp.where(iota // K < n_valid, e_flat, E)
    if route_valid is not None:
        e_flat = jnp.where(route_valid.reshape(-1), e_flat, E)
    s_rep = jnp.broadcast_to(row_scale[:, None], (T, K)).reshape(-1)
    _, order, s_sorted = jax.lax.sort((e_flat, iota, s_rep), num_keys=1, is_stable=True)
    _, pos = jax.lax.sort((order, iota), num_keys=1, is_stable=False)
    counts = jnp.sum(
        (e_flat[None, :] == jnp.arange(E, dtype=jnp.int32)[:, None]).astype(jnp.int32),
        axis=1,
    )
    base = jnp.cumsum(counts) - counts
    active = counts > 0
    (visit_e,) = jnp.nonzero(active, size=E + mtp.VISIT_SLACK, fill_value=E)
    visit_e = visit_e.astype(jnp.int32)
    n_rv = jnp.sum(active).astype(jnp.int32)
    counts_x = jnp.concatenate([counts, jnp.zeros((1,), jnp.int32)])
    base_x = jnp.concatenate([base, jnp.zeros((1,), jnp.int32)])
    live = (e_flat < E).reshape(T, K)
    return (
        order // K,
        pos.reshape(T, K),
        s_sorted,
        visit_e,
        base_x[visit_e],
        counts_x[visit_e],
        n_rv,
    ), live


def _ids(T, K, E, key, distinct=True):
    if distinct:  # top-k of uniform logits: distinct per token, as the router
        return jax.lax.top_k(jax.random.uniform(key, (T, E)), K)[1].astype(jnp.int32)
    return jax.random.randint(key, (T, K), 0, E, dtype=jnp.int32)


# (T, E, K, n_valid, p_valid, distinct, exact_dead)
PLAN_CASES = [
    (256, 16, 4, 131, 0.9, False, True),  # the incumbent test's cases
    (8192, 512, 10, None, 0.9, False, True),
    (16384, 512, 10, None, 1.0, True, True),  # production: all routes live
    (16384, 512, 10, None, 0.95, True, True),  # ~8k dead routes < 2**14: exact
    (16384, 512, 10, 8000, 0.9, True, False),  # dead routes > 2**14
    (16384, 512, 10, 1000, 1.0, True, False),  # dead routes >> 2**14
    (3072, 512, 10, 2695, 1.0, True, True),  # c8k bucket (<= 2**16 routes)
    (1024, 512, 10, 686, 1.0, True, True),  # c2k bucket
]


@pytest.mark.parametrize("T,E,K,n_valid,p_valid,distinct,exact_dead", PLAN_CASES)
def test_routing_plan_v2_matches_reference(T, E, K, n_valid, p_valid, distinct, exact_dead):
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    topk_ids = _ids(T, K, E, k1, distinct)
    row_scale = jax.random.uniform(k2, (T,), dtype=jnp.float32)
    route_valid = jax.random.bernoulli(k3, p_valid, (T, K))
    got = kw.routing_plan_v2(topk_ids, E, row_scale, route_valid=route_valid, n_valid=n_valid)
    inc = mtp.routing_plan(topk_ids, E, row_scale, route_valid=route_valid, n_valid=n_valid)
    ref, live = _reference_plan(topk_ids, E, row_scale, route_valid, n_valid)
    live = np.asarray(live)
    n_dead = int((~live).sum())
    if not exact_dead:
        assert n_dead > (1 << 14)
    for i, (g, r, c) in enumerate(zip(got, ref, inc)):
        g, r, c = map(np.asarray, (g, r, c))
        if i == 1 and not exact_dead:
            # pos: exact on live routes; dead routes are redirected by the
            # caller (shared row), so compare the redirected tables.
            assert np.array_equal(g[live], r[live])
            assert np.array_equal(g[live], c[live])
            shared = 10**9 + np.arange(T)[:, None]
            assert np.array_equal(np.where(live, g, shared), np.where(live, c, shared))
        else:
            assert np.array_equal(g, r), i
            assert np.array_equal(g, c), i


def test_routing_plan_v2_nondistinct_falls_back_exact():
    """Every route on one expert (the SP f32-bitcast bug's routing): a
    count > 2**14 takes the two-operand branch and stays exact."""
    T, K, E = 16384, 10, 512
    topk_ids = jnp.zeros((T, K), jnp.int32)
    row_scale = jax.random.uniform(jax.random.key(1), (T,), dtype=jnp.float32)
    got = kw.routing_plan_v2(topk_ids, E, row_scale)
    inc = mtp.routing_plan(topk_ids, E, row_scale)
    for g, c in zip(got, inc):
        assert np.array_equal(np.asarray(g), np.asarray(c))


def test_combine_w_kernel_matches_incumbent_bit_exact():
    k1, k2, k3, k4 = jax.random.split(jax.random.key(7), 4)
    tc, topk, hidden, slab = 256, 4, 256, 1024
    part = jax.random.normal(k1, (slab, hidden // mtp.LANES, mtp.LANES)).astype(mtp.FP8)
    pscl = jax.random.uniform(k2, (slab,), dtype=jnp.float32)
    coef = jax.random.uniform(k3, (tc, topk), dtype=jnp.float32)
    pos = jax.random.randint(k4, (tc, topk), 0, slab, dtype=jnp.int32)
    common = dict(t_local=tc, topk=topk, hidden=hidden, lane_blocks=hidden // mtp.LANES,
                  wire_dtype=mtp.FP8, out_dtype=jnp.bfloat16, interpret=not _on_tpu())
    inc = mtp.build_tp_combine_kernel(scale_len=slab, **common)
    new = kw.build_tp_combine_kernel_w(**common)
    a = np.asarray(inc(part, pscl, coef, pos)).view(np.uint16)
    b = np.asarray(new(part, coef * pscl[pos], pos)).view(np.uint16)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("routing,combine_impl", [("v2", "v1"), ("v1", "v2"), ("v2", "v2")])
def test_kw_matches_incumbent_bit_exact(routing, combine_impl):
    c = make_case(300, 16, 256, 128, 4, skew=True)
    args = (c["x"], c["w1"], c["w2"], c["s1"], c["s2"], c["w1s"], c["w2s"], c["s1s"],
            c["s2s"], c["wg"], c["tw"], c["ti"])
    ref = mtp.moe_tp_prefill(*args, interpret=not _on_tpu())
    got = kw.moe_tp_prefill_kw(*args, interpret=not _on_tpu(), routing=routing,
                               combine_impl=combine_impl)
    assert np.array_equal(np.asarray(ref).view(np.uint16), np.asarray(got).view(np.uint16))


@pytest.mark.parametrize("fault", ["ids0", "drop_shared", "slab_off1"])
def test_kw_faults_are_caught(fault):
    c = make_case(300, 16, 256, 128, 4, skew=True)
    args = (c["x"], c["w1"], c["w2"], c["s1"], c["s2"], c["w1s"], c["w2s"], c["s1s"],
            c["s2s"], c["wg"], c["tw"], c["ti"])
    ref = mtp.moe_tp_prefill(*args, interpret=not _on_tpu())
    got = kw.moe_tp_prefill_kw(*args, interpret=not _on_tpu(), routing="v2",
                               combine_impl="v2", fault=fault)
    assert not np.array_equal(np.asarray(ref).view(np.uint16), np.asarray(got).view(np.uint16))


@pytest.mark.parametrize("removal", ["epi", "gather", "wdma"])
def test_removal_arms_build_and_run(removal):
    c = make_case(300, 16, 256, 128, 4, skew=True)
    args = (c["x"], c["w1"], c["w2"], c["s1"], c["s2"], c["w1s"], c["w2s"], c["s1s"],
            c["s2s"], c["wg"], c["tw"], c["ti"])
    y = kw.moe_tp_prefill_kw(*args, interpret=not _on_tpu(), removal=removal)
    assert y.shape == c["x"].shape
