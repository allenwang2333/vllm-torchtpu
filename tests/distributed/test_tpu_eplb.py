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


def test_step_trigger_first_then_every_interval():
    fired, nxt = [], 200
    for count in range(0, 800, 7):  # counts advance in uneven jumps
        due, nxt = T.step_trigger(count, nxt, 96)
        if due:
            fired.append(count)
    assert fired[0] == 203  # first step at or past FIRST
    assert all(b - a >= 96 for a, b in zip(fired, fired[1:]))
    assert T.step_trigger(0, 0, 96) == (False, 0)  # never before any forward


def _fake_rebalancer(first, interval):
    r = object.__new__(T.EplbRebalancer)
    r.first, r.interval, r.count, r.next_at, r.armed = first, interval, 0, first, True
    r.fired = []
    r.rebalance = lambda at, r=r: r.fired.append(at)
    return r


def _dp_step(ranks, local):
    """One engine step as tpu_runner._dp_coordinated_step runs it: one MAX
    all-reduce of [tokens, chunks, reqs, count, -count], then on_step on every
    rank from the reduced values only. local[i] = (tokens, chunks); an idle rank
    (execute_dummy_batch) contributes (0, 0)."""
    red = [max(x) for x in zip(*[(tok, ch, 0, r.count, -r.count, -tok)
                                 for r, (tok, ch) in zip(ranks, local)])]
    mk_decode = red[0] <= 1 and red[1] == 1
    for r in ranks:
        r.on_step(red[3], -red[4], 0 if mk_decode else red[1], min_tokens=-red[5])


def test_trigger_agrees_across_busy_and_idle_ranks():
    # j-36147a83 hung at the first rebalance that fell in a 1-busy/15-idle
    # step. Replay that mix of c16 prefill, c1 prefill (one busy rank), c1
    # megakernel decode and fully idle waves, and require every rank -- busy or
    # idle -- to fire at the same steps with the same count.
    rng = np.random.default_rng(0)
    ranks = [_fake_rebalancer(first=200, interval=96) for _ in range(EP)]
    ranks[5].count = 3  # even a drifted local counter must follow the agreed max
    fired_in_c1 = False
    for step in range(3000):
        phase = (step // 150) % 4
        if phase == 0:  # c16 prefill: every rank busy, 1..2 chunks
            local = [(1024, int(rng.integers(1, 3))) for _ in range(EP)]
        elif phase == 1:  # c1 prefill: one rank busy, 15 idle
            busy = int(rng.integers(EP))
            local = [(1024, 1) if i == busy else (0, 0) for i in range(EP)]
        elif phase == 2:  # c1 decode on the megakernel: not counted
            busy = int(rng.integers(EP))
            local = [(1, 1) if i == busy else (0, 0) for i in range(EP)]
        else:
            local = [(0, 0)] * EP
        before = len(ranks[0].fired)
        _dp_step(ranks, local)
        if len(ranks[0].fired) > before and phase == 1:
            fired_in_c1 = True
    assert len(ranks[0].fired) >= 5
    assert fired_in_c1
    assert all(r.fired == ranks[0].fired for r in ranks)
    assert all(r.count == ranks[0].count for r in ranks)


def test_trace_arms_post_rebalance_forwards_and_steps():
    r = _fake_rebalancer(first=4, interval=4)
    r.trace, r.epoch = True, 0
    for _ in range(5):  # counts 0..4; the step that sees count 4 fires
        r.on_step(r.count, r.count, 1, allreduce_ms=0.5)
    assert r.fired == [4] and r.trace_fwd_left == 3 and r.trace_steps_left == 12
    r.on_step(r.count, r.count, 1)
    assert r.trace_steps_left == 11


def test_async_save_writes_npz(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setenv("VLLM_TPU_EPLB_ASYNC_SAVE", "1")
    r = _fake_rebalancer(first=4, interval=4)
    path = str(tmp_path / "sub" / "e1.npz")
    r._save_npz(path, epoch=1, x=np.arange(3))
    r._saver.shutdown(wait=True)
    assert np.load(path)["x"].tolist() == [0, 1, 2]


def test_busy_gate_defers_to_first_all_busy_step():
    assert T.step_trigger(10, 8, 4, all_busy=False) == (False, 8)
    assert T.step_trigger(12, 8, 4, all_busy=True) == (True, 16)
    EP = 16
    ranks = [_fake_rebalancer(first=8, interval=8) for _ in range(EP)]
    for r in ranks:
        r.busy_gate = True
    busy_at = []
    for step in range(200):
        if (step // 20) % 2:  # c1: rank 0 busy, the rest idle
            local = [(1024, 1)] + [(0, 0)] * (EP - 1)
        else:  # c16
            local = [(1024, 1)] * EP
        before = [len(r.fired) for r in ranks]
        _dp_step(ranks, local)
        if any(len(r.fired) > b for r, b in zip(ranks, before)):
            busy_at.append(all(tok > 0 for tok, _ in local))
    fired = [r.fired for r in ranks]
    assert all(f == fired[0] for f in fired) and len(fired[0]) >= 3
    assert all(busy_at)  # never fired with an idle rank
    # Ungated: fires in c1 steps too.
    ranks = [_fake_rebalancer(first=8, interval=8) for _ in range(EP)]
    idle_fire = False
    for step in range(200):
        c1 = (step // 20) % 2
        local = ([(1024, 1)] + [(0, 0)] * (EP - 1)) if c1 else [(1024, 1)] * EP
        before = len(ranks[0].fired)
        _dp_step(ranks, local)
        idle_fire |= bool(c1 and len(ranks[0].fired) > before)
    assert idle_fire
