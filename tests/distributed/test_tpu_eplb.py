# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Host-side checks of the online EPLB placement and exchange plan (no TPU)."""

import numpy as np
import pytest

from vllm_torchtpu.distributed import tpu_eplb as T

E, EP = 256, 16
G = E // EP


def _zipf_load(seed, s=1.1):
    rng = np.random.default_rng(seed)
    w = 1.0 / np.arange(1, E + 1) ** s
    return rng.multinomial(200_000, rng.permutation(w / w.sum())).astype(np.int64)


def _rank_max(load, phys2log):
    per_slot = load[phys2log]
    return per_slot.reshape(EP, G).sum(1).max()


@pytest.mark.parametrize("order", [None, tuple(np.random.default_rng(3).permutation(EP))])
def test_plan_is_a_balanced_permutation_in_delivery_order(order):
    p2l = T.initial_phys2log_mesh(E, EP, order)
    for seed in range(4):
        load = _zipf_load(seed)
        new, info = T.plan_layer(load, p2l, EP, 0.0)
        assert new is not None
        assert sorted(new.tolist()) == list(range(E))
        assert _rank_max(load, new) < _rank_max(load, p2l)
        assert info["new_max_over_mean"] < info["old_max_over_mean"]
        xp = T.exchange_plan(p2l, new, EP)  # asserts delivery order
        # the simulated collective puts exactly the planned expert in every slot
        old_local = [list(p2l[r * G:(r + 1) * G]) for r in range(EP)]
        got = T.simulate_exchange(old_local, xp)
        assert np.array_equal(np.concatenate(got), new)
        assert xp.moved_experts == int(info["moved"])
        assert np.array_equal(xp.send_sz.sum(0), np.full(EP, G))
        assert np.array_equal(xp.send_sz.sum(1), np.full(EP, G))
        p2l = new


def test_plan_is_deterministic_and_keeps_stayers():
    load = _zipf_load(7)
    p2l = T.initial_phys2log_mesh(E, EP, None)
    a, _ = T.plan_layer(load, p2l, EP, 0.0)
    b, _ = T.plan_layer(load.copy(), p2l.copy(), EP, 0.0)
    assert np.array_equal(a, b)
    # replanning on the same load moves nothing (packs already match ranks)
    c, info = T.plan_layer(load, a, EP, 0.0)
    assert c is None or np.array_equal(
        np.sort(c.reshape(EP, G), 1), np.sort(a.reshape(EP, G), 1)
    )


def test_min_gain_skips_balanced_layer():
    load = np.full(E, 100, np.int64)
    p2l = T.initial_phys2log_mesh(E, EP, None)
    new, info = T.plan_layer(load, p2l, EP, 0.05)
    assert new is None and info["moved"] == 0


def test_ep_ids_round_trip_through_mesh_relabel():
    order = tuple(np.random.default_rng(5).permutation(EP))
    p2l = T.initial_phys2log_mesh(E, EP, order)
    new, _ = T.plan_layer(_zipf_load(1), p2l, EP, 0.0)
    l2p = np.empty(E, np.int64)
    l2p[new] = np.arange(E)
    ep_ids = T.ep_ids_from_mesh(l2p, order, G)
    # the kernel's relabel: EP rank r -> mesh index m with order[m] == r
    mesh_of_rank = np.empty(EP, np.int64)
    mesh_of_rank[np.asarray(order)] = np.arange(EP)
    mesh_slot = mesh_of_rank[ep_ids // G] * G + ep_ids % G
    assert np.array_equal(new[mesh_slot], np.arange(E))
    # the identity placement is the identity map
    l2p0 = np.empty(E, np.int64)
    l2p0[p2l] = np.arange(E)
    assert np.array_equal(T.ep_ids_from_mesh(l2p0, order, G), np.arange(E))


def test_exchange_fn_signature_resolves_for_sharded_jax_callable():
    # sharded_jax_callable evaluates the closure's string annotations in the
    # module's globals (eval_str=True); a function-local `import jax` is not
    # enough (served run j-2b772a7a died with NameError at the first rebalance).
    import inspect

    import jax
    from jax.sharding import Mesh

    mesh = Mesh(np.array(jax.devices()[:1]), ("ep",))
    fn = T.make_exchange_fn(mesh, "ep")
    sig = inspect.signature(inspect.unwrap(fn), eval_str=True)
    assert len(sig.parameters) == 6
