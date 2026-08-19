# SPDX-License-Identifier: Apache-2.0
"""The routing plan's per-pair lookups, against the one-hot spelling they replace.

``build_routing_tables`` runs in XLA once per MoE layer per call and is v2's
whole fixed cost. Three of its four ``[n_blocks, block, e_total]`` passes are a
table lookup spelled as a one-hot select-and-sum, which costs ``e_total`` times
what the lookup does; the obvious rewrite is ``take_along_axis``.

**The rewrite is correct and slower**, and this module holds both halves of that
statement so the next reader does not have to rediscover either. The equality is
asserted here, bit for bit on every field the kernel reads, across routing
distributions including the degenerate ones. The cost is recorded in
``host._lookup_per_pair``'s docstring: AOT-compiled for tpu7x at the serving
shape, the gather cuts XLA's flop count 2.3x and multiplies bytes accessed by
36 (63.4 MB -> 2283.3 MB), because a TPU gather over the lane axis does not fuse.

What is left after that is the real optimization, and it is an algebra change
rather than a respelling -- see the same docstring.
"""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402


def _gather_plan(topk_idx, *, e_total, ep, t_local, block, tile_m,
                 shard_stride):
    """``build_routing_tables`` with the three lookups spelled as gathers."""
    original = host._lookup_per_pair

    def gather(table, expert_blocks, _bins):
        return jnp.take_along_axis(table, expert_blocks, axis=1, mode="clip")

    host._lookup_per_pair = gather
    try:
        return host.build_routing_tables(topk_idx,
                                         e_total=e_total,
                                         ep=ep,
                                         t_local=t_local,
                                         block=block,
                                         tile_m=tile_m,
                                         shard_stride=shard_stride)
    finally:
        host._lookup_per_pair = original


def _routing(idx, *, e_total, ep, t_local, capacity=128):
    kw = dict(e_total=e_total,
              ep=ep,
              t_local=t_local,
              block=host.routing_block(t_local, idx.shape[1]),
              tile_m=capacity,
              shard_stride=host.ragged_stride_bound(idx.shape[0], idx.shape[1],
                                                    e_total, capacity))
    return host.build_routing_tables(idx, **kw), _gather_plan(idx, **kw)


# (e_total, ep, t_local, topk, "how the tokens route")
CASES = [
    (512, 8, 256, 10, "uniform"),  # the serving shape, scaled down in T
    (512, 8, 64, 10, "uniform"),
    (64, 8, 32, 4, "uniform"),
    (64, 8, 32, 4, "one_expert"),  # every pair on one expert: max run
    (64, 8, 32, 4, "one_per_block"),  # every pair distinct: no run at all
    (16, 2, 128, 2, "skewed"),  # a heavy tail on the low experts
]


def _indices(rng, T, topk, e_total, how):
    if how == "one_expert":
        return jnp.zeros((T, topk), jnp.int32)
    if how == "one_per_block":
        flat = np.arange(T * topk, dtype=np.int32) % e_total
        return jnp.asarray(flat.reshape(T, topk))
    if how == "skewed":
        p = 1.0 / (1 + np.arange(e_total))
        p /= p.sum()
        return jnp.asarray(
            np.stack([
                rng.choice(e_total, size=topk, replace=False, p=p)
                for _ in range(T)
            ]).astype(np.int32))
    return jnp.asarray(
        np.stack([
            rng.choice(e_total, size=topk, replace=False) for _ in range(T)
        ]).astype(np.int32))


@pytest.mark.parametrize("e_total,ep,t_local,topk,how", CASES)
def test_the_one_hot_and_the_gather_are_the_same_integers(
        e_total, ep, t_local, topk, how):
    rng = np.random.default_rng(0)
    idx = _indices(rng, t_local * ep, topk, e_total, how)
    got, want = _routing(idx, e_total=e_total, ep=ep, t_local=t_local)
    for field in got._fields:
        a, b = getattr(got, field), getattr(want, field)
        if isinstance(a, int):
            assert a == b, field
            continue
        np.testing.assert_array_equal(np.asarray(a),
                                      np.asarray(b),
                                      err_msg=f"{field} differs")


def test_every_routed_pair_lands_on_a_distinct_slab_row():
    """What the plan is for, checked directly rather than through the kernel.

    Two pairs sharing a slab row is the silent-wrong-answer failure this
    function can have: the second overwrites the first and one token loses an
    expert with nothing raising anywhere.
    """
    rng = np.random.default_rng(7)
    e_total, ep, t_local, topk = 64, 8, 32, 4
    idx = _indices(rng, t_local * ep, topk, e_total, "skewed")
    routing, _ = _routing(idx, e_total=e_total, ep=ep, t_local=t_local)
    rows = np.asarray(routing.slab_row)
    assert len(np.unique(rows)) == rows.size
    assert rows.min() >= 0 and rows.max() < routing.rows_alloc


def test_the_capacity_slab_masks_exactly_what_it_drops():
    """The other half of the slab trade, at the shape it is for.

    A slab below the drop-free bound loses the pairs that do not fit -- which is
    a defensible trade for the 8.3x the smaller slab is worth on device. What is
    not defensible is those pairs still carrying a router weight at the
    destination: their arrival rows were never written, so the combine would
    weight uninitialized HBM. This checks the mask against the definition
    rather than against the implementation: a pair is kept exactly when its slab
    row lies inside its own owner's slab.
    """
    e_total, ep, t_local, topk, cap = 512, 8, 2048, 10, 128
    T = t_local * ep
    rng = np.random.default_rng(1)
    idx = _indices(rng, T, topk, e_total, "uniform")
    g_local = e_total // ep
    owner = np.asarray(idx) // g_local

    seen = {}
    for factor in (0.5, 1.0, 1.5, 16.0):
        stride = host.capacity_slab_rows(T, topk, e_total, ep, cap, factor)
        routing = host.build_routing_tables(idx,
                                            e_total=e_total,
                                            ep=ep,
                                            t_local=t_local,
                                            block=host.routing_block(
                                                t_local, topk),
                                            tile_m=cap,
                                            shard_stride=stride)
        slab = np.asarray(routing.slab_row).reshape(T, topk)
        kept = (slab - owner * stride) < stride
        assert (slab - owner * stride
                >= 0).all(), "a row landed before its slab"
        seen[factor] = (stride, int((~kept).sum()))

    # A margin big enough to reach the drop-free bound must drop nothing, and
    # dropping must be monotone in the margin.
    assert seen[16.0][0] == host.ragged_stride_bound(T, topk, e_total, cap)
    assert seen[16.0][1] == 0, seen
    drops = [seen[f][1] for f in (0.5, 1.0, 1.5, 16.0)]
    assert drops == sorted(drops, reverse=True), seen
    # W5's law with no extra margin already covers a uniform draw at the
    # serving shape -- that is what the sqrt term is for.
    assert seen[1.0][1] == 0, seen
    print(f"[capacity slab] stride/drops by skew margin: {seen}")
