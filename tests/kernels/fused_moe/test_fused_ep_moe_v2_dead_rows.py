# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dead rows in the fused EP MoE routing plan.

A row whose gate logits are all -inf (TPU_MOE_SKIP_PADDED_TOKENS padding, an
idle DP rank's `_dummy_run`) gets the sentinel expert id `e_total` for every
selection. These tests hold the plan builders to what that has to mean: the
live pairs' slab rows, arrival rows and mirror positions are exactly what a
plan over the live pairs alone assigns, every count and transport table
excludes the dead pairs, and no slab scatter writes a dead pair anywhere.

The reference is a direct numpy statement of the plan (rank of a pair within
its (expert, destination) run, offset by that run's bases). It is checked
against the builder on an all-live batch first, so a reference that is wrong
cannot pass the dead-row cases by agreeing with a wrong builder.

The whole-layer check -- live rows bit-identical to the all-live run, dead
rows zero -- needs the kernel and is in test_fused_ep_moe_v2.py.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402
from vllm_torchtpu.kernels.fused_moe.v2.layer import _dead_rows  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_backend():
    """Keep every array in this file off the accelerator."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


GEOMETRIES = [
    pytest.param(8, 64, 32, 8, id="t64_ep8_g32_topk8"),
    pytest.param(16, 128, 16, 8, id="glm_t128_ep16_g16_topk8"),
    pytest.param(4, 128, 32, 8, id="ep4_g32_topk8"),
]

# Which rows are dead: every row of every shard but 0 (the idle-DP-rank
# prefill), a scattered subset on every shard, and everything.
DEAD_PATTERNS = ["idle_ranks", "scattered", "all"]


def _inputs(ep, t_local, g_local, topk, pattern, seed=0):
    e_total = g_local * ep
    T = t_local * ep
    rng = np.random.default_rng(seed)
    idx = np.stack([rng.permutation(e_total)[:topk] for _ in range(T)]).astype(np.int32)
    if pattern == "idle_ranks":
        dead = np.arange(T) >= t_local
    elif pattern == "scattered":
        dead = rng.random(T) < 0.4
    elif pattern == "all":
        dead = np.ones(T, bool)
    else:
        dead = np.zeros(T, bool)
    capacity = 128
    kwargs = dict(
        e_total=e_total,
        ep=ep,
        t_local=t_local,
        block=host.routing_block(t_local, topk),
        tile_m=capacity,
        shard_stride=host.ragged_stride_bound(T, topk, e_total, capacity),
    )
    return idx, dead, kwargs


def _with_sentinel(idx, dead, e_total):
    return np.where(dead[:, None], np.int32(e_total), idx).astype(np.int32)


def _reference(idx, dead, *, e_total, ep, t_local, tile_m, shard_stride, **_):
    """Counts and live-pair positions of a plan over the live pairs only."""
    T, K = idx.shape
    g_local = e_total // ep
    live = ~dead
    run_rows = np.zeros((e_total, ep), np.int64)
    rank = np.full((T, K), -1, np.int64)
    for t in range(T):
        if not live[t]:
            continue
        d = t // t_local
        for k in range(K):
            e = idx[t, k]
            rank[t, k] = run_rows[e, d]
            run_rows[e, d] += 1
    aligned = (run_rows + host.ROWBLK - 1) // host.ROWBLK * host.ROWBLK
    run_start_aligned = np.cumsum(aligned, axis=1) - aligned
    expert_rows = aligned.sum(axis=1)
    by_shard = expert_rows.reshape(ep, g_local)
    expert_base = (
        (np.cumsum(by_shard, axis=1) - by_shard) + np.arange(ep)[:, None] * shard_stride
    ).reshape(e_total)
    region = aligned.reshape(ep, g_local, ep)  # [s, g, d]
    per_dest = region.transpose(2, 0, 1).reshape(ep, e_total)
    recv_base = (np.cumsum(per_dest, axis=1) - per_dest).reshape(ep, ep, g_local)

    slab = np.full((T, K), -1, np.int64)
    arrival = np.full((T, K), -1, np.int64)
    mirror = np.full((T, K), -1, np.int64)
    for t in range(T):
        if not live[t]:
            continue
        d = t // t_local
        for k in range(K):
            e = idx[t, k]
            s, g = divmod(e, g_local)
            r = rank[t, k]
            slab[t, k] = expert_base[e] + run_start_aligned[e, d] + r
            arrival[t, k] = recv_base[d, s, g] + r
            mirror[t, k] = (
                r
                + ((recv_base[d, s, g] // tile_m) + 2 * e) * tile_m
                + (run_start_aligned[e, d] % tile_m)
            )
    return dict(
        run_rows=run_rows,
        run_rows_aligned=aligned,
        expert_rows_aligned=expert_rows,
        expert_base=expert_base,
        recv_base=recv_base,
        slab=slab,
        arrival=arrival,
        mirror=mirror,
    )


def _check_plan_against_reference(routing, idx, dead, kwargs):
    ref = _reference(idx, dead, **kwargs)
    live = ~dead
    for name in (
        "run_rows",
        "run_rows_aligned",
        "expert_rows_aligned",
        "expert_base",
        "recv_base",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(routing, name)), ref[name], err_msg=name
        )
    slab = np.asarray(routing.slab_row).reshape(idx.shape)
    np.testing.assert_array_equal(slab[live], ref["slab"][live])
    np.testing.assert_array_equal(
        np.asarray(routing.arrival_row)[live], ref["arrival"][live]
    )
    np.testing.assert_array_equal(
        np.asarray(routing.mirror_row)[live], ref["mirror"][live]
    )
    return ref


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
def test_reference_matches_the_builder_on_an_all_live_batch(ep, t_local, g_local, topk):
    """The positive control: the reference is the builder's plan."""
    idx, dead, kwargs = _inputs(ep, t_local, g_local, topk, "none")
    routing = host.build_routing_tables(jnp.asarray(idx), **kwargs)
    _check_plan_against_reference(routing, idx, dead, kwargs)


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("pattern", DEAD_PATTERNS)
def test_dead_pairs_leave_the_live_plan_unchanged(ep, t_local, g_local, topk, pattern):
    idx, dead, kwargs = _inputs(ep, t_local, g_local, topk, pattern)
    e_total, stride = kwargs["e_total"], kwargs["shard_stride"]
    sentinel = jnp.asarray(_with_sentinel(idx, dead, e_total))
    routing = host.build_routing_tables(sentinel, **kwargs)
    ref = _check_plan_against_reference(routing, idx, dead, kwargs)

    # Counts exclude the dead pairs.
    assert int(np.asarray(routing.run_rows).sum()) == int((~dead).sum()) * topk

    # A dead pair's slab row is off every shard's slab, so both slab
    # scatters (token gather and activation scale) drop it on every shard.
    slab = np.asarray(routing.slab_row).reshape(idx.shape)
    off = host.off_shard_slab_row(stride, ep)
    assert (slab[dead] == off).all()
    scale_len = host.act_scale_slab_rows(stride) * host.HIDDEN_LANE_BLOCK
    for me in range(ep):
        rows = np.asarray(host.local_slab_rows(routing, me, shard_stride=stride))
        rows = rows.reshape(idx.shape)
        assert (rows[dead] >= max(stride, scale_len)).all()
        # Token gather: each live pair of this shard's experts writes its own
        # token number on its own row; nothing else is written.
        want = np.zeros(stride, np.int64)
        mine = (~dead[:, None]) & (idx // g_local == me)
        tok = np.broadcast_to(np.arange(idx.shape[0])[:, None], idx.shape)
        want[ref["slab"][mine] - me * stride] = tok[mine]
        got = np.asarray(host.shard_token_gather(routing, me, shard_stride=stride))
        np.testing.assert_array_equal(got, np.clip(want, 0, idx.shape[0] - 1))

        # Transport and count tables are functions of the live counts only.
        expert_rows, _ = host.shard_expert_slabs(routing, me, e_total=e_total, ep=ep)
        np.testing.assert_array_equal(
            np.asarray(expert_rows),
            ref["expert_rows_aligned"][me * g_local : (me + 1) * g_local],
        )


@pytest.mark.parametrize("ep,t_local,g_local,topk", GEOMETRIES)
@pytest.mark.parametrize("pattern", DEAD_PATTERNS)
def test_sharded_plan_agrees_on_dead_rows(ep, t_local, g_local, topk, pattern):
    """The sharded builder needs no change: the sentinel is on no shard."""
    idx, dead, kwargs = _inputs(ep, t_local, g_local, topk, pattern, seed=3)
    e_total, stride = kwargs["e_total"], kwargs["shard_stride"]
    sentinel = _with_sentinel(idx, dead, e_total)
    whole = host.build_routing_tables(jnp.asarray(sentinel), **kwargs)
    counts = np.stack(
        [
            np.bincount(
                sentinel[d * t_local : (d + 1) * t_local].reshape(-1),
                minlength=e_total + 1,
            )[:e_total]
            for d in range(ep)
        ]
    ).astype(np.int32)
    live = ~dead
    for me in range(ep):
        shard = host.build_routing_tables_sharded(
            jnp.asarray(sentinel),
            jnp.int32(me),
            all_gather_rows=lambda row: jnp.asarray(counts),
            **kwargs,
        )
        mine = slice(me * t_local, (me + 1) * t_local)
        np.testing.assert_array_equal(
            np.asarray(shard.pos)[live[mine]],
            np.asarray(whole.arrival_row)[mine][live[mine]],
        )
        # Off-shard rows differ in value between the builders (both are past
        # this shard's slab); the rows this shard writes must be identical.
        shard_rows = np.asarray(host.local_slab_rows(shard, me, shard_stride=stride))
        whole_rows = np.asarray(host.local_slab_rows(whole, me, shard_stride=stride))
        np.testing.assert_array_equal(shard_rows < stride, whole_rows < stride)
        on_me = whole_rows < stride
        np.testing.assert_array_equal(shard_rows[on_me], whole_rows[on_me])
        assert not on_me.reshape(idx.shape)[dead].any()
        for builder in (
            host.shard_transport_tables_in_blocks,
            host.shard_push_tables_in_rows,
            host.shard_expert_slabs,
        ):
            for a, b in zip(
                builder(whole, me, e_total=e_total, ep=ep),
                builder(shard, me, e_total=e_total, ep=ep),
            ):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_dead_rows_are_exactly_the_all_minus_inf_rows():
    inf = np.inf
    gating = jnp.asarray(
        [
            [-inf, -inf, -inf, -inf],  # padding
            [-inf, -inf, -inf, 0.0],  # one live logit: routes
            [-inf, -inf, -inf, -3e38],  # finite, however small: routes
            [np.nan, np.nan, np.nan, np.nan],  # NaN is not padding
            [0.1, 0.2, 0.3, 0.4],
        ],
        jnp.float32,
    )
    np.testing.assert_array_equal(
        np.asarray(_dead_rows(gating)), [True, False, False, False, False]
    )
    np.testing.assert_array_equal(
        np.asarray(_dead_rows(gating.astype(jnp.bfloat16))),
        [True, False, False, False, False],
    )


def test_padding_mask_writes_minus_inf_rows():
    torch = pytest.importorskip("torch")
    import importlib.util
    from pathlib import Path

    import vllm_torchtpu

    # By path: the module needs only torch, and importing it through its
    # package pulls in the vLLM attention stack.
    path = Path(vllm_torchtpu.__file__).parent / "layers/adapter/token_padding.py"
    spec = importlib.util.spec_from_file_location("_token_padding_under_test", path)
    token_padding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(token_padding)

    logits = torch.randn(6, 4)
    assert token_padding.mask_router_logits_for_padding(logits) is logits
    state = token_padding.TokenPaddingState.create(8, "cpu")
    state.update(local_valid_tokens=4, local_padded_tokens=8)
    token_padding.set_padding_state(state)
    out = token_padding.mask_router_logits_for_padding(logits)
    assert torch.equal(out[:4], logits[:4])
    assert torch.isneginf(out[4:]).all()
    np.testing.assert_array_equal(
        np.asarray(_dead_rows(jnp.asarray(out.numpy()))), [False] * 4 + [True] * 2
    )
