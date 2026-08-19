# SPDX-License-Identifier: Apache-2.0
"""The decomposed routing plan against the whole one, at the kernel's operands.

``build_routing_tables`` is v2's dominant fixed cost -- 1049.5 M vector flops
per MoE layer per call at E=512, ep=8, T=16384, four one-hot passes over the
whole ``T x K x e_total`` product -- and every shard computes the identical
tables.
``build_routing_tables_sharded`` computes the two complementary 1/ep slices its
consumers actually read, plus one 2 kB all-gather of the per-(shard, expert)
counts.

What has to hold is not that the two plans agree field for field -- the
decomposed one deliberately does not build most of the fields whole -- but that
**the operands the kernel receives are the same integers**. So this compares
the interface: everything ``layer.py`` hands to ``kfn``, per shard, for every
shard, including the two scatter-built tables and the activation scale slab.

The one documented difference is the scale slab's overhang row, which is past
the slab it mirrors and which the whole plan fills with an accidental sum of
off-shard pairs' scale bits; see ``_assert_same_operands``.

The plan is deterministic integer arithmetic, so randomized testing is the
strongest evidence available and most of this file is seeds.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax import lax  # noqa: E402

from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402

CAPACITY = 128
LANES = host.HIDDEN_LANE_BLOCK
# Devices for the one test that runs the plan through a real mesh. Asked for at
# IMPORT: the knob only takes before the backend initializes, and the first
# jnp call in any test here would initialize it. Four, which is what
# test_fused_ep_v2_numerics.py asks for, so the two files can share a process.
REAL_EP = 4
try:
    jax.config.update("jax_num_cpu_devices", REAL_EP)
except Exception:  # already initialized by another module, or the knob moved
    pass


def _kw(idx, e_total, ep, t_local, capacity):
    T, K = idx.shape
    return dict(e_total=e_total,
                ep=ep,
                t_local=t_local,
                block=host.routing_block(t_local, K),
                tile_m=capacity,
                shard_stride=host.ragged_stride_bound(T, K, e_total, capacity))


def _count_table(idx, ep, t_local, e_total):
    """[ep, E] per-(source shard, expert) pair counts, by an independent route.

    What the decomposed plan all-gathers, computed here with numpy rather than
    with any of the code under test, so the emulated collective is a reference
    and not an echo.
    """
    flat = np.asarray(idx, np.int64)
    return np.stack([
        np.bincount(flat[d * t_local:(d + 1) * t_local].reshape(-1),
                    minlength=e_total) for d in range(ep)
    ]).astype(np.int32)


def _sharded(idx, me, kw, counts, prefix_rank=False):
    """One shard's plan, with the all-gather emulated and its input checked."""
    seen = []

    def all_gather_rows(row):
        seen.append(np.asarray(row))
        return jnp.asarray(counts)

    plan = host.build_routing_tables_sharded(idx,
                                             me,
                                             all_gather_rows=all_gather_rows,
                                             prefix_rank=prefix_rank,
                                             **kw)
    assert len(seen) == 1, "the plan takes exactly one collective"
    np.testing.assert_array_equal(
        seen[0].reshape(-1), counts[me],
        "the count row a shard contributes to the all-gather is not its own")
    return plan


def _scale_slab(slab_row, shard_stride, scale_bits):
    """``layer.py``'s activation scale slab, as int32 bit patterns."""
    rows = host.act_scale_slab_rows(shard_stride)
    return jnp.zeros(
        (rows * LANES, ),
        jnp.int32).at[slab_row].add(scale_bits, mode="drop").reshape(rows, -1)


def _operands(routing, me, *, e_total, ep, t_local, topk, shard_stride,
              scale_bits):
    """Everything ``local_fn`` builds and hands to the kernel, by name.

    Left as jax arrays rather than numpy, so that the same builder runs inside
    a shard_map trace as well as eagerly.
    """
    out = {}
    block_names = ("run_start", "aligned_rows", "contrib_off", "push_src",
                   "push_len", "push_dst", "totals")
    for name, table in zip(
            block_names,
            host.shard_transport_tables_in_blocks(routing,
                                                  me,
                                                  e_total=e_total,
                                                  ep=ep)):
        out[f"blocks.{name}"] = table
    for name, table in zip(("true_rows", "recv_row_off", "totals"),
                           host.shard_push_tables_in_rows(routing,
                                                          me,
                                                          e_total=e_total,
                                                          ep=ep)):
        out[f"rows.{name}"] = table
    slab_row = host.local_slab_rows(routing, me, shard_stride=shard_stride)
    out["local_slab_rows"] = slab_row
    out["token_gather"] = host.shard_token_gather(routing,
                                                  me,
                                                  shard_stride=shard_stride)
    expert_rows, slab_base = host.shard_expert_slabs(routing,
                                                     me,
                                                     e_total=e_total,
                                                     ep=ep)
    out["expert_rows"], out["slab_base"] = expert_rows, slab_base
    visit, n_visit = host.expert_visit_list(expert_rows, e_total // ep)
    out["visit"], out["n_visit"] = visit, n_visit
    out["counts"] = host.shard_count_vector(routing,
                                            expert_rows,
                                            me,
                                            e_total=e_total,
                                            ep=ep)
    out["pos"] = (routing.pos if isinstance(routing, host.ShardRoutingTables)
                  else lax.dynamic_slice(routing.arrival_row,
                                         (me * t_local, 0), (t_local, topk)))
    out["scale_slab"] = _scale_slab(slab_row, shard_stride, scale_bits)
    return out


def _assert_same_operands(whole, shard, *, shard_stride, ep, me):
    """Every kernel operand, one plan against the other."""
    assert whole.keys() == shard.keys()
    view = host.act_scale_slab_rows(shard_stride) * LANES
    for name in whole:
        a, b = np.asarray(whole[name]), np.asarray(shard[name])
        if name == "local_slab_rows":
            # A pair on another shard's expert is dropped by both scatters in
            # either plan, and the two plans drop it at different indices: the
            # whole plan knows the row it really lands on and the decomposed
            # one deliberately does not. So the claim is that the same pairs
            # land on the same rows OF THIS SHARD'S SLAB, and that everything
            # else is out of range of every table this index reaches.
            live_a, live_b = a < shard_stride, b < shard_stride
            np.testing.assert_array_equal(live_a, live_b,
                                          f"[me={me}] different pairs land")
            np.testing.assert_array_equal(a[live_a], b[live_b],
                                          f"[me={me}] different slab rows")
            assert (b[~live_b] >= view).all(), (
                f"[me={me}] the off-shard sentinel {b[~live_b].min()} is "
                f"inside the {view}-element scale slab view")
            continue
        if name == "scale_slab":
            # The slab proper is bit-identical. Its LAST row is the overhang
            # act_scale_slab_rows adds for the final tile's sublane read, past
            # the slab's last row and selected by nothing: the whole plan lands
            # off-shard pairs on it (their rebased rows are the next shard's
            # first rows, which are inside the view), the decomposed plan drops
            # them. Asserted rather than assumed, both ways.
            flat_a, flat_b = a.reshape(-1), b.reshape(-1)
            np.testing.assert_array_equal(flat_a[:shard_stride],
                                          flat_b[:shard_stride],
                                          f"[me={me}] scale slab differs")
            assert not flat_b[shard_stride:].any(), (
                f"[me={me}] overhang is not clean")
            continue
        np.testing.assert_array_equal(a, b, f"[me={me}] {name} differs")


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
    if how == "half_idle":
        # Only the first half of the experts is ever selected, so half of every
        # shard's local experts have no rows and drop out of the visit list.
        return jnp.asarray(
            np.stack([
                rng.choice(max(e_total // 2, topk), size=topk, replace=False)
                for _ in range(T)
            ]).astype(np.int32))
    return jnp.asarray(
        np.stack([
            rng.choice(e_total, size=topk, replace=False) for _ in range(T)
        ]).astype(np.int32))


def _compare_every_shard(e_total, ep, t_local, topk, how, seed,
                         capacity=CAPACITY, prefix_rank=False):
    rng = np.random.default_rng(seed)
    T = t_local * ep
    idx = _indices(rng, T, topk, e_total, how)
    kw = _kw(idx, e_total, ep, t_local, capacity)
    stride = kw["shard_stride"]
    whole_plan = host.build_routing_tables(idx, **kw)
    counts = _count_table(idx, ep, t_local, e_total)
    # One row scale per token, as layer.py repeats it over the token's
    # selections; the bit patterns are what the slab carries.
    scale_bits = np.asarray(
        lax.bitcast_convert_type(
            jnp.asarray(rng.normal(size=T).astype(np.float32)), jnp.int32))
    scale_bits = jnp.asarray(np.repeat(scale_bits, topk))
    common = dict(e_total=e_total,
                  ep=ep,
                  t_local=t_local,
                  topk=topk,
                  shard_stride=stride,
                  scale_bits=scale_bits)
    for me in range(ep):
        shard_plan = _sharded(idx, me, kw, counts, prefix_rank)
        _assert_same_operands(_operands(whole_plan, me, **common),
                              _operands(shard_plan, me, **common),
                              shard_stride=stride,
                              ep=ep,
                              me=me)
    return whole_plan


# (e_total, ep, t_local, topk, how)
CASES = [
    (512, 8, 64, 10, "uniform"),  # the serving shape, scaled down in T
    (512, 8, 64, 10, "skewed"),
    (64, 8, 32, 4, "uniform"),  # the small shape
    (64, 8, 32, 4, "one_expert"),  # every pair on one expert: max run
    (64, 8, 32, 4, "one_per_block"),  # every pair distinct: no run at all
    (64, 8, 32, 4, "skewed"),
    (64, 8, 32, 4, "half_idle"),  # half the local experts never visited
    (16, 2, 128, 2, "skewed"),  # ep=2
    (64, 2, 64, 4, "uniform"),
    (64, 2, 64, 4, "one_expert"),
    (128, 4, 32, 6, "uniform"),  # ep=4
    (128, 4, 32, 6, "one_per_block"),
    (32, 4, 16, 3, "skewed"),
    (8, 8, 16, 2, "uniform"),  # one expert per shard
    (512, 8, 8, 10, "uniform"),  # more experts than routed pairs
]


@pytest.mark.parametrize("prefix_rank", [False, True], ids=["rank", "prefix"])
@pytest.mark.parametrize("e_total,ep,t_local,topk,how", CASES)
def test_the_two_plans_hand_the_kernel_the_same_operands(
        e_total, ep, t_local, topk, how, prefix_rank):
    """Both spellings of the within-block rank, since they are one function.

    prefix_rank takes the rank out of the lookup's own one-hot grid instead of
    a quadratic pass; it has to produce the same integers or it is not a
    spelling, and it is the same operand comparison either way.
    """
    _compare_every_shard(e_total, ep, t_local, topk, how, seed=0,
                         prefix_rank=prefix_rank)


@pytest.mark.parametrize("e_total,ep,t_local,topk,how", [
    (128, 8, 32, 4, "uniform"),
    (128, 4, 32, 6, "skewed"),
    (16, 2, 128, 2, "one_expert"),
])
def test_the_same_operands_at_a_stride_that_is_not_a_whole_lane_block(
        e_total, ep, t_local, topk, how):
    """capacity=64 rounds the slab to a stride the scale slab's lane-block
    view does not divide, which moves the boundary the sentinel has to clear
    and the one the scale slab is compared on."""
    idx = _indices(np.random.default_rng(5), t_local * ep, topk, e_total, how)
    assert _kw(idx, e_total, ep, t_local, 64)["shard_stride"] % LANES, (
        "this case is here for a stride that is not a whole lane block")
    _compare_every_shard(e_total, ep, t_local, topk, how, seed=5, capacity=64)


SEED_SHAPES = [
    (64, 8, 32, 4),
    (128, 4, 32, 6),
    (32, 4, 16, 3),
    (16, 2, 64, 2),
]
FLAVOURS = ["uniform", "skewed", "half_idle"]


@pytest.mark.parametrize("seed", range(24))
def test_the_same_operands_under_random_routing(seed):
    """The plan is deterministic integer arithmetic, so seeds are the evidence.

    Each seed takes a different shape and a different routing distribution, so
    the 24 cover every (shape, flavour) pair twice.
    """
    e_total, ep, t_local, topk = SEED_SHAPES[seed % len(SEED_SHAPES)]
    how = FLAVOURS[(seed // len(SEED_SHAPES)) % len(FLAVOURS)]
    _compare_every_shard(e_total, ep, t_local, topk, how, seed=1000 + seed,
                         prefix_rank=bool(seed % 2))


@pytest.mark.parametrize("seed", range(4))
def test_the_same_operands_at_the_serving_shape(seed):
    """E=512, top_k=10, ep=8 -- the shape the optimization is aimed at."""
    _compare_every_shard(512, 8, 64, 10, "skewed", seed=7000 + seed)


def test_every_pair_on_my_experts_lands_on_a_distinct_row_of_my_slab():
    """The silent-corruption failure, checked on the decomposed plan directly.

    Two pairs sharing a slab row lose a token an expert with nothing raising.
    The whole plan's version of this is in test_fused_ep_v2_routing_plan.py;
    the decomposed plan needs its own because its rows come from a different
    prefix sum and because the pairs that are NOT its own have to be gone.
    """
    e_total, ep, t_local, topk = 64, 8, 32, 4
    rng = np.random.default_rng(11)
    idx = _indices(rng, t_local * ep, topk, e_total, "skewed")
    kw = _kw(idx, e_total, ep, t_local, CAPACITY)
    stride = kw["shard_stride"]
    counts = _count_table(idx, ep, t_local, e_total)
    g_local = e_total // ep
    flat = np.asarray(idx).reshape(-1)
    for me in range(ep):
        plan = _sharded(idx, me, kw, counts)
        rows = np.asarray(host.local_slab_rows(plan, me, shard_stride=stride))
        mine = (flat >= me * g_local) & (flat < (me + 1) * g_local)
        assert (rows[mine] < stride).all()
        assert len(np.unique(rows[mine])) == int(mine.sum())
        assert (rows[~mine] >= host.act_scale_slab_rows(stride) * LANES).all()


BAD_ARGS = [
    dict(e_total=64, ep=7),  # experts not divisible by ep
    dict(ep=16),  # alignment slots past ALIGNMENT_SLOT_FIELD
    dict(block=48),  # block does not divide the pair count
    dict(tile_m=12),  # tile height not a whole number of row blocks
    dict(shard_stride=1000),  # stride not a whole number of tiles
]


@pytest.mark.parametrize("bad", BAD_ARGS)
def test_both_plans_refuse_the_same_arguments_with_the_same_words(bad):
    """The refusals are the plan's only guard; bounds checks are off below.

    A decomposed plan that accepted an argument the whole one refuses would be
    a silently different contract, so the messages are compared, not just the
    exception type.
    """
    e_total, ep, t_local, topk = 64, 8, 32, 4
    idx = _indices(np.random.default_rng(0), t_local * ep, topk, e_total,
                   "uniform")
    kw = _kw(idx, e_total, ep, t_local, CAPACITY) | bad
    with pytest.raises(ValueError) as whole:
        host.build_routing_tables(idx, **kw)
    with pytest.raises(ValueError) as shard:
        host.build_routing_tables_sharded(idx,
                                          0,
                                          all_gather_rows=lambda row: row,
                                          **kw)
    assert str(whole.value) == str(shard.value)


def test_the_off_shard_sentinel_clears_every_table_it_could_index():
    """The sentinel is a value, not a hope: check it against the two lengths.

    ``local_slab_rows`` rebases by ``me * shard_stride``, so the sentinel is
    smallest for the last shard, and the scale slab's view -- one lane block
    longer than the slab -- is the longer of the two tables it must clear.
    """
    for ep in (2, 4, 8):
        for stride in (128, 1664, 8832, 167552):
            row = host.off_shard_slab_row(stride, ep)
            local = row - (ep - 1) * stride
            assert local >= stride
            assert local >= host.act_scale_slab_rows(stride) * LANES
            assert row < 2**31


@pytest.fixture(scope="module")
def cpu_devices():
    """``REAL_EP`` addressable devices, on a host that has one CPU."""
    devs = jax.devices()
    if len(devs) < REAL_EP:
        pytest.skip(f"{len(devs)} devices, need {REAL_EP}")
    return tuple(devs[:REAL_EP])


@pytest.mark.parametrize("e_total,topk,t_local", [(64, 4, 16), (32, 3, 32)])
def test_the_real_collective_gives_the_same_operands(cpu_devices, e_total,
                                                     topk, t_local):
    """The same comparison, with jax's own all-gather over four devices.

    Everything above emulates the collective and calls the plan with a python
    `me`, which leaves two things unchecked: that
    ``lax.all_gather(row, axis=0, tiled=True)`` on a [1, e_total] row returns
    the [ep, e_total] table ordered by shard index, which is what the plan
    assumes when it indexes it by destination; and that a TRACED `me` -- every
    slice in the plan is dynamic under shard_map -- cuts the same tables. Here
    the plan runs inside a real shard_map and every operand comes back through
    the mesh, to be compared by the same function as everywhere else.
    """
    from jax.sharding import AxisType, Mesh
    from jax.sharding import PartitionSpec as P

    ep = REAL_EP
    T = t_local * ep
    mesh = Mesh(np.asarray(cpu_devices).reshape(ep), ("d", ),
                axis_types=(AxisType.Explicit, ))
    rng = np.random.default_rng(2)
    idx = _indices(rng, T, topk, e_total, "skewed")
    kw = _kw(idx, e_total, ep, t_local, CAPACITY)
    stride = kw["shard_stride"]
    scale = jnp.asarray(rng.normal(size=(T, 1)).astype(np.float32))
    common = dict(e_total=e_total,
                  ep=ep,
                  t_local=t_local,
                  topk=topk,
                  shard_stride=stride)
    whole = host.build_routing_tables(idx, **kw)
    bits = jnp.asarray(
        np.repeat(
            np.asarray(
                lax.bitcast_convert_type(jnp.asarray(np.asarray(scale)[:, 0]),
                                         jnp.int32)), topk))
    want = [
        _operands(whole, me, scale_bits=bits, **common) for me in range(ep)
    ]
    names = sorted(want[0])

    def body(idx_l, scale_l):
        me = lax.axis_index("d")
        idx_g = lax.all_gather(idx_l, "d", axis=0, tiled=True)
        scale_g = lax.all_gather(scale_l, "d", axis=0, tiled=True)
        plan = host.build_routing_tables_sharded(
            idx_g,
            me,
            all_gather_rows=lambda row: lax.all_gather(
                row, "d", axis=0, tiled=True),
            **kw)
        ops = _operands(plan,
                        me,
                        scale_bits=lax.bitcast_convert_type(
                            jnp.repeat(scale_g[:, 0], topk), jnp.int32),
                        **common)
        return tuple(
            jnp.asarray(ops[k], jnp.int32).reshape(1, -1) for k in names)

    sharded = jax.sharding.NamedSharding(mesh, P("d"))
    got = jax.jit(
        jax.shard_map(body,
                      mesh=mesh,
                      in_specs=(P("d"), P("d")),
                      out_specs=P("d"),
                      check_vma=False))(jax.device_put(idx, sharded),
                                        jax.device_put(scale, sharded))
    for me in range(ep):
        _assert_same_operands(want[me], {
            k: np.asarray(got[i])[me].reshape(np.shape(want[me][k]))
            for i, k in enumerate(names)
        },
                              shard_stride=stride,
                              ep=ep,
                              me=me)
