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
"""Tests for DCP (decode context parallel) StreamIndex Top-K.

The DCP variant differs from PCP in what it delivers, not in how it selects:
queries are replicated and the KV is sharded, so each rank must come away with
the slice of *every* token's global top-k that it happens to own. These tests
pin the two properties that makes usable -- the reassembled result is exactly
the unsharded global top-k, and each rank is handed only positions it holds.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME, _partition_winners_by_owner, pcp_global_to_local,
    pcp_local_to_global, pcp_owner_rank, streamindex_topk,
    streamindex_topk_dcp)

P = jax.sharding.PartitionSpec

# ===================================================================== #
# Coordinate algebra. `pcp_local_to_global` is already covered by the PCP
# tests; what is new here is the *inverse*, which DCP needs in order to
# hand a rank a position in that rank's own coordinate space while
# running on some other rank.
# ===================================================================== #

_LAYOUTS = [(d, ic) for d in (2, 4, 8, 16) for ic in (1, 2, 4, 8, 64)]


@pytest.mark.parametrize("dcp_size,interleave_c", _LAYOUTS)
def test_global_to_local_inverts_local_to_global(dcp_size, interleave_c):
    """Round-trip on every rank, over several interleave cycles."""
    for rank in range(dcp_size):
        for local in range(3 * interleave_c * dcp_size):
            g = int(pcp_local_to_global(local, rank, dcp_size, interleave_c))
            assert int(pcp_owner_rank(jnp.int32(g), dcp_size,
                                      interleave_c)) == rank
            assert int(pcp_global_to_local(jnp.int32(g), dcp_size,
                                           interleave_c)) == local


@pytest.mark.parametrize("dcp_size,interleave_c", _LAYOUTS)
def test_global_to_local_needs_no_rank_argument(dcp_size, interleave_c):
    """The whole point: the inverse is rank-free, so any rank can compute it.

    Evaluated blind on a contiguous global range, `(owner, local)` must still
    be a bijection onto the shards -- that is what lets the merge rank address
    a winner into its owner's cache without being that owner.
    """
    n = 5 * dcp_size * interleave_c + 3
    gs = jnp.arange(n, dtype=jnp.int32)
    owners = np.asarray(pcp_owner_rank(gs, dcp_size, interleave_c))
    locals_ = np.asarray(pcp_global_to_local(gs, dcp_size, interleave_c))
    assert len(set(zip(owners.tolist(), locals_.tolist()))) == n
    for g in range(n):
        assert int(
            pcp_local_to_global(int(locals_[g]), int(owners[g]), dcp_size,
                                interleave_c)) == g


# ===================================================================== #
# The partition step, on its own. This is where a global top-k list is
# split into one padded per-owner list, and where capacity overflow is
# either reported or silently corrupts the result.
# ===================================================================== #


@pytest.mark.parametrize("dcp_size,interleave_c", [(2, 1), (4, 4), (8, 4),
                                                   (8, 64)])
def test_partition_is_lossless_when_capacity_suffices(dcp_size, interleave_c):
    """Reassembling the per-owner lists recovers the input set exactly."""
    rng = np.random.default_rng(20260906)
    num_rows, k, span = 16, 128, 4096
    merged = np.stack([
        rng.choice(span, size=k, replace=False).astype(np.int32)
        for _ in range(num_rows)
    ])
    # A ragged tail, to exercise the `-1` padding of the input as well.
    merged[0, -5:] = -1

    lists, dropped = _partition_winners_by_owner(jnp.asarray(merged), dcp_size,
                                                 interleave_c, capacity=k)
    lists = np.asarray(lists)
    assert int(dropped) == 0

    for t in range(num_rows):
        want = set(merged[t][merged[t] >= 0].tolist())
        got = set()
        for r in range(dcp_size):
            row = lists[t, r]
            for local in row[row >= 0].tolist():
                g = int(
                    pcp_local_to_global(local, r, dcp_size, interleave_c))
                # A rank must only ever be handed positions it actually owns.
                assert (g // interleave_c) % dcp_size == r
                got.add(g)
        assert got == want
        assert len(got) == int((merged[t] >= 0).sum())


def test_partition_reports_overflow_rather_than_hiding_it():
    """Undersized capacity must surface as a count, not a quiet truncation."""
    dcp_size, interleave_c, capacity = 4, 1, 3
    # All 16 positions on one row; with interleave_c=1 each rank owns 4 of
    # them, so every rank overflows a capacity of 3 by exactly one.
    merged = jnp.arange(16, dtype=jnp.int32)[None, :]
    lists, dropped = _partition_winners_by_owner(merged, dcp_size,
                                                 interleave_c, capacity)
    assert int(dropped) == dcp_size
    assert int((np.asarray(lists) >= 0).sum()) == dcp_size * capacity


def test_partition_leaves_unowned_ranks_empty():
    """A rank owning none of a token's winners gets an all -1 row.

    The attention side keys its LSE mask off exactly this, so it has to be
    distinguishable from "owns position 0".
    """
    # interleave_c=1, dcp_size=4: these are all owned by rank 1.
    merged = jnp.asarray([[1, 5, 9, 13]], jnp.int32)
    lists = np.asarray(
        _partition_winners_by_owner(merged, 4, 1, capacity=8)[0])
    assert (lists[0, 1] >= 0).sum() == 4
    for r in (0, 2, 3):
        assert (lists[0, r] == -1).all()


# ===================================================================== #
# Argument validation.
# ===================================================================== #


def _mesh(size):
    return jax.sharding.Mesh(np.array(jax.devices()[:size]), (DCP_AXIS_NAME, ))


def _dcp_kwargs(**over):
    kw = dict(mesh=None, k=16, compression_ratio=1, dcp_size=2,
              interleave_size=1, capacity=8)
    kw.update(over)
    return kw


def _dummy_args(num_tokens=8, num_heads=4, head_dim=128, pages=2,
                page_size=16):
    width = 256
    return (
        jnp.zeros((num_tokens, num_heads, head_dim), jnp.float8_e4m3fn),
        jnp.zeros((num_tokens, num_heads), jnp.bfloat16),
        jnp.zeros((pages, page_size // 4, 4, width), jnp.uint8),
        jnp.array([num_tokens], jnp.int32),
        jnp.arange(pages, dtype=jnp.int32),
        jnp.array([0, num_tokens], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
    )


def test_dcp_size_one_is_rejected():
    mesh = _mesh(1)
    with pytest.raises(ValueError, match="requires dcp_size > 1"):
        streamindex_topk_dcp(*_dummy_args(),
                             **_dcp_kwargs(mesh=mesh, dcp_size=1))


def test_capacity_above_k_is_rejected():
    if len(jax.devices()) < 2:
        pytest.skip("needs 2 devices")
    with pytest.raises(ValueError, match="exceeds k"):
        streamindex_topk_dcp(*_dummy_args(),
                             **_dcp_kwargs(mesh=_mesh(2), k=16, capacity=32))


def test_token_count_must_divide_over_the_dcp_axis():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 devices")
    with pytest.raises(ValueError, match="divisible by"):
        streamindex_topk_dcp(*_dummy_args(num_tokens=6),
                             **_dcp_kwargs(mesh=_mesh(4), dcp_size=4))


# ===================================================================== #
# End-to-end against the unsharded kernel.
# ===================================================================== #

H_IDX, D_IDX, K = 32, 128, 2048
Q_LEN, KV_LEN, PAGE_SIZE = 256, 32768, 1024
WIDTH = 256


def _capacity(dcp_size):
    """2x the `k / dcp_size` mean occupancy, capped at `k`.

    Headroom has to scale with the shard count, not be a constant: mean
    occupancy is `k / dcp_size`, so a fixed width that is generous at
    dcp_size=8 is a hard truncation at dcp_size=2.
    """
    return min(K, 2 * K // dcp_size)


def _global_records(kv_len, seed=7):
    """One record array; both paths are views of it, so any diff is a real bug."""
    vals = jax.random.normal(jax.random.key(seed), (kv_len, D_IDX),
                             jnp.float32)
    fp8 = jax.lax.bitcast_convert_type(vals.astype(jnp.float8_e4m3fn),
                                       jnp.uint8).reshape(kv_len, D_IDX)
    # 127 is the e8m0 exponent bias, i.e. a scale of exactly 1.0.
    rec = jnp.concatenate([fp8, jnp.full((kv_len, 1), 127, jnp.uint8)], -1)
    return jnp.pad(rec, ((0, 0), (0, WIDTH - rec.shape[-1])))


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_dcp_reassembles_the_unsharded_global_topk(dcp_size):
    """Union the per-rank lists back together; it must be the global top-k.

    Exactness is the requirement, not an approximation target: DCP partitions
    the KV, so a global winner is beaten by at most `k - 1` positions overall
    and hence by at most `k - 1` on its own shard, and always survives that
    shard's local top-k.
    """
    devices = jax.devices()
    if len(devices) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(devices)}")
    capacity = _capacity(dcp_size)

    rec = _global_records(KV_LEN)
    kq, kw = jax.random.split(jax.random.key(0))
    q = (jax.random.normal(kq, (Q_LEN, H_IDX, D_IDX), jnp.float32) *
         0.5).astype(jnp.float8_e4m3fn)
    weights = jax.random.normal(kw, (Q_LEN, H_IDX),
                                jnp.float32).astype(jnp.bfloat16)
    seq_lens = jnp.array([KV_LEN], jnp.int32)
    cu_q_lens = jnp.array([0, Q_LEN], jnp.int32)
    distribution = jnp.array([0, 0, 1], jnp.int32)

    pages = KV_LEN // PAGE_SIZE
    reference = np.asarray(
        streamindex_topk(q,
                         weights,
                         rec.reshape(pages, PAGE_SIZE // 4, 4, WIDTH),
                         seq_lens,
                         jnp.arange(pages, dtype=jnp.int32),
                         cu_q_lens,
                         distribution,
                         k=K,
                         compression_ratio=1,
                         num_kv_pages_per_block=2,
                         num_queries_per_block=128))

    # interleave_size=1: global position g lives on rank g % dcp_size at local
    # index g // dcp_size, so the shards are a strided de-interleave.
    vpages = KV_LEN // (PAGE_SIZE * dcp_size)
    shards = jnp.concatenate([
        rec[r::dcp_size].reshape(vpages, PAGE_SIZE // 4, 4, WIDTH)
        for r in range(dcp_size)
    ], 0)
    mesh = jax.sharding.Mesh(np.array(devices[:dcp_size]), (DCP_AXIS_NAME, ))
    shards = jax.device_put(shards,
                            jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME)))

    local_topk, dropped = jax.jit(
        functools.partial(streamindex_topk_dcp,
                          mesh=mesh,
                          k=K,
                          compression_ratio=1,
                          dcp_size=dcp_size,
                          interleave_size=1,
                          capacity=capacity,
                          num_kv_pages_per_block=2,
                          num_queries_per_block=128))(
                              q, weights, shards, seq_lens,
                              jnp.arange(vpages, dtype=jnp.int32), cu_q_lens,
                              distribution)

    assert int(np.asarray(dropped).sum()) == 0, (
        f"capacity {capacity} overflowed: {np.asarray(dropped).tolist()}")

    # Global leading dim is dcp_size * Q_LEN, rank-major.
    local_topk = np.asarray(local_topk).reshape(dcp_size, Q_LEN, capacity)
    assert local_topk.max() < KV_LEN // dcp_size, "a rank got a foreign index"

    for t in range(Q_LEN):
        got = set()
        for r in range(dcp_size):
            row = local_topk[r, t]
            for local in row[row >= 0].tolist():
                got.add(int(pcp_local_to_global(local, r, dcp_size, 1)))
        want = set(reference[t][reference[t] >= 0].tolist())
        assert got == want, (f"token {t}: {len(want - got)} missing, "
                             f"{len(got - want)} spurious")


@pytest.mark.multichip
def test_dcp_ranks_partition_the_winners_without_duplication(dcp_size=4):
    """Each winner lands on exactly one rank -- else the LSE merge double-counts.

    Set equality above would still pass if a position were delivered twice, so
    this checks the multiplicity the attention merge actually depends on.
    """
    devices = jax.devices()
    if len(devices) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(devices)}")
    capacity = _capacity(dcp_size)

    rec = _global_records(KV_LEN)
    kq, kw = jax.random.split(jax.random.key(3))
    q = (jax.random.normal(kq, (Q_LEN, H_IDX, D_IDX), jnp.float32) *
         0.5).astype(jnp.float8_e4m3fn)
    weights = jax.random.normal(kw, (Q_LEN, H_IDX),
                                jnp.float32).astype(jnp.bfloat16)

    vpages = KV_LEN // (PAGE_SIZE * dcp_size)
    shards = jnp.concatenate([
        rec[r::dcp_size].reshape(vpages, PAGE_SIZE // 4, 4, WIDTH)
        for r in range(dcp_size)
    ], 0)
    mesh = jax.sharding.Mesh(np.array(devices[:dcp_size]), (DCP_AXIS_NAME, ))
    shards = jax.device_put(shards,
                            jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME)))

    local_topk, _ = jax.jit(
        functools.partial(streamindex_topk_dcp,
                          mesh=mesh,
                          k=K,
                          compression_ratio=1,
                          dcp_size=dcp_size,
                          interleave_size=1,
                          capacity=capacity,
                          num_kv_pages_per_block=2,
                          num_queries_per_block=128))(
                              q, weights, shards,
                              jnp.array([KV_LEN], jnp.int32),
                              jnp.arange(vpages, dtype=jnp.int32),
                              jnp.array([0, Q_LEN], jnp.int32),
                              jnp.array([0, 0, 1], jnp.int32))
    local_topk = np.asarray(local_topk).reshape(dcp_size, Q_LEN, capacity)

    for t in range(Q_LEN):
        globals_ = [
            int(pcp_local_to_global(local, r, dcp_size, 1))
            for r in range(dcp_size) for local in
            local_topk[r, t][local_topk[r, t] >= 0].tolist()
        ]
        assert len(globals_) == len(set(globals_)), f"token {t} has duplicates"
        # Every row is full here: the query tokens are the last Q_LEN of a
        # KV_LEN context, so even the earliest sees far more than K positions.
        assert len(globals_) == K
