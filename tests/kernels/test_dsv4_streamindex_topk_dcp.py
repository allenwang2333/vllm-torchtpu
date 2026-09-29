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
"""Tests for DCP (decode context parallel) StreamIndex Top-K"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    cp_local_to_global,
    cp_rank_as_data,
    dcp_sc_compact,
    metadata,
    streamindex_topk,
    streamindex_topk_dcp,
)

P = jax.sharding.PartitionSpec


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_candidate_all_gather_is_rank_major(dcp_size):
    """The merge reads a winner's position in its own list as `slot % k`.

    The source rank now travels as its own lane, so the chunk *order* no longer
    matters -- but the chunk *width* still does, and nothing else in the merge
    would notice it changing: every rank would resolve the wrong candidate and
    still return a full, plausible, silently wrong list. Asserting the full
    rank-major arange pins width and order together.
    """
    devices = jax.devices()
    if len(devices) < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {len(devices)}")

    rows, k = 4, 8
    mesh = jax.sharding.Mesh(np.array(devices[:dcp_size]), (DCP_AXIS_NAME,))

    def _local(_):
        rank = cp_rank_as_data(DCP_AXIS_NAME, dcp_size)
        # Rank r writes r*k + j, so a rank-major gather reads back as arange.
        mine = rank * k + jnp.arange(k, dtype=jnp.int32)
        return jax.lax.all_gather(
            jnp.broadcast_to(mine, (rows, k)), DCP_AXIS_NAME, axis=1, tiled=True
        )

    gathered = jax.jit(
        jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=P(DCP_AXIS_NAME),
            check_vma=False,
        )
    )(jnp.zeros((rows, 1), jnp.int32))

    got = np.asarray(gathered).reshape(dcp_size, rows, dcp_size * k)
    want = np.broadcast_to(
        np.arange(dcp_size * k, dtype=np.int32), (rows, dcp_size * k)
    )
    for r in range(dcp_size):
        np.testing.assert_array_equal(
            got[r], want, err_msg=f"rank {r} saw a non-rank-major candidate axis"
        )


def _require_sparsecore():
    """Probe in a test body: a chip touch at collection breaks spawned workers."""
    try:
        if pltpu.get_tpu_info().sparse_core is not None:
            return
    except Exception:  # no TPU at all
        pass
    pytest.skip("needs SparseCore")


def _xla_pack_reference(merged, dcp_size, interleave_c, dcp_rank):
    """The scatter the kernel replaced, unreachable in `src`, kept as an oracle."""
    num_rows, width = merged.shape
    mine = (merged >= 0) & (
        metadata.cp_owner_rank(merged, dcp_size, interleave_c) == dcp_rank
    )
    local = metadata.cp_global_to_local(jnp.maximum(merged, 0), dcp_size, interleave_c)

    # Exclusive prefix count of owned winners gives each entry's destination slot.
    # At most `width` entries can be `mine`, so `slot <= width - 1`.
    slot = jnp.cumsum(mine, axis=1, dtype=jnp.int32) - mine
    rows = jnp.arange(num_rows, dtype=jnp.int32)[:, None]

    # An extra column past the end (`width`) absorbs entries owned by other ranks;
    # they all write -1 to the dump column, which is sliced off at the end.
    out = jnp.full((num_rows, width + 1), -1, dtype=jnp.int32)
    out = out.at[rows, jnp.where(mine, slot, width)].set(jnp.where(mine, local, -1))
    return out[:, :width]


def _merged_fixture(num_rows=16, k=128, span=4096, seed=20260906):
    """A global winner list: distinct positions per row, `-1` padded in row 0."""
    rng = np.random.default_rng(seed)
    merged = np.stack(
        [
            rng.choice(span, size=k, replace=False).astype(np.int32)
            for _ in range(num_rows)
        ]
    )
    merged[0, -5:] = -1
    return merged


def _filter_all_ranks(
    merged, dcp_size, interleave_c, fn=dcp_sc_compact.pack_owned_winners
):
    """What the `dcp_size` ranks each keep, stacked as [dcp_size, rows, k]."""
    return np.stack(
        [
            np.asarray(fn(jnp.asarray(merged), dcp_size, interleave_c, jnp.int32(r)))
            for r in range(dcp_size)
        ]
    )


# The XLA scatter is the reference, and the only pack that runs off a
# SparseCore-bearing chip.
@pytest.mark.parametrize(
    "pack_fn,needs_sc",
    [(_xla_pack_reference, False), (dcp_sc_compact.pack_owned_winners, True)],
    ids=["xla", "sc"],
)
@pytest.mark.parametrize("dcp_size,interleave_c", [(2, 1), (4, 4), (8, 4), (8, 64)])
def test_owner_filters_partition_the_list(pack_fn, needs_sc, dcp_size, interleave_c):
    if needs_sc:
        _require_sparsecore()
    merged = _merged_fixture()
    num_rows, k = merged.shape

    lists = _filter_all_ranks(merged, dcp_size, interleave_c, pack_fn)
    assert lists.shape == (dcp_size, num_rows, k)

    for t in range(num_rows):
        want = set(merged[t][merged[t] >= 0].tolist())
        got = []
        for r in range(dcp_size):
            row = lists[r, t]
            for local in row[row >= 0].tolist():
                g = int(cp_local_to_global(local, r, dcp_size, interleave_c))
                # A rank must only ever keep positions it actually owns.
                assert (g // interleave_c) % dcp_size == r
                got.append(g)
        # Multiplicity, not just set equality: the LSE merge double-counts a
        # position delivered to two ranks.
        assert len(got) == len(set(got))
        assert set(got) == want


@pytest.mark.parametrize("dcp_size,interleave_c", [(2, 1), (4, 4), (8, 4), (8, 64)])
def test_sparsecore_pack_matches_xla(dcp_size, interleave_c):
    """Elementwise, not just as a set: both packs must emit in ascending order."""
    _require_sparsecore()
    merged = _merged_fixture()

    want = _filter_all_ranks(merged, dcp_size, interleave_c, _xla_pack_reference)
    got = _filter_all_ranks(
        merged, dcp_size, interleave_c, dcp_sc_compact.pack_owned_winners
    )

    np.testing.assert_array_equal(got, want)


# ===================================================================== #
# End-to-end against the unsharded kernel.
# ===================================================================== #

H_IDX, D_IDX, K = 32, 128, 2048
Q_LEN, KV_LEN, PAGE_SIZE = 256, 32768, 1024
WIDTH = 256


def _global_records(kv_len, seed=7):
    """One record array; both paths are views of it, so any diff is a real bug."""
    vals = jax.random.normal(jax.random.key(seed), (kv_len, D_IDX), jnp.float32)
    fp8 = jax.lax.bitcast_convert_type(
        vals.astype(jnp.float8_e4m3fn), jnp.uint8
    ).reshape(kv_len, D_IDX)
    # 127 is the e8m0 exponent bias, i.e. a scale of exactly 1.0.
    rec = jnp.concatenate([fp8, jnp.full((kv_len, 1), 127, jnp.uint8)], -1)
    return jnp.pad(rec, ((0, 0), (0, WIDTH - rec.shape[-1])))


def _need_devices(dcp_size):
    n = len(jax.devices())
    if n < dcp_size:
        pytest.skip(f"needs {dcp_size} devices, have {n}")


def _dcp_case(dcp_size, seed):
    """Records plus the positional args and kwargs of one sharded DCP call."""
    rec = _global_records(KV_LEN)
    kq, kw = jax.random.split(jax.random.key(seed))
    q = (jax.random.normal(kq, (Q_LEN, H_IDX, D_IDX), jnp.float32) * 0.5).astype(
        jnp.float8_e4m3fn
    )
    weights = jax.random.normal(kw, (Q_LEN, H_IDX), jnp.float32).astype(jnp.bfloat16)

    # interleave_size=1: global position g lives on rank g % dcp_size at local
    # index g // dcp_size, so the shards are a strided de-interleave.
    vpages = KV_LEN // (PAGE_SIZE * dcp_size)
    shards = jnp.concatenate(
        [
            rec[r::dcp_size].reshape(vpages, PAGE_SIZE // 4, 4, WIDTH)
            for r in range(dcp_size)
        ],
        0,
    )
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp_size]), (DCP_AXIS_NAME,))
    shards = jax.device_put(shards, jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME)))

    args = (
        q,
        weights,
        shards,
        jnp.array([KV_LEN], jnp.int32),
        jnp.arange(vpages, dtype=jnp.int32),
        jnp.array([0, Q_LEN], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
    )
    kwargs = dict(
        mesh=mesh,
        k=K,
        compression_ratio=1,
        dcp_size=dcp_size,
        interleave_size=1,
        num_kv_pages_per_block=2,
        num_queries_per_block=128,
    )
    return rec, args, kwargs


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
def test_dcp_reassembles_the_unsharded_global_topk(dcp_size):
    """Union the per-rank lists back together; it must be the global top-k"""
    _need_devices(dcp_size)
    rec, args, kwargs = _dcp_case(dcp_size, 0)
    q, weights = args[0], args[1]

    pages = KV_LEN // PAGE_SIZE
    reference = np.asarray(
        streamindex_topk(
            q,
            weights,
            rec.reshape(pages, PAGE_SIZE // 4, 4, WIDTH),
            jnp.array([KV_LEN], jnp.int32),
            jnp.arange(pages, dtype=jnp.int32),
            jnp.array([0, Q_LEN], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32),
            k=K,
            compression_ratio=1,
            num_kv_pages_per_block=2,
            num_queries_per_block=128,
        )
    )

    # Global leading dim is dcp_size * Q_LEN, rank-major.
    local_topk = np.asarray(streamindex_topk_dcp(*args, **kwargs)).reshape(
        dcp_size, Q_LEN, K
    )
    assert local_topk.max() < KV_LEN // dcp_size, "a rank got a foreign index"

    for t in range(Q_LEN):
        got = set()
        for r in range(dcp_size):
            row = local_topk[r, t]
            for local in row[row >= 0].tolist():
                got.add(int(cp_local_to_global(local, r, dcp_size, 1)))
        want = set(reference[t][reference[t] >= 0].tolist())
        assert got == want, (
            f"token {t}: {len(want - got)} missing, {len(got - want)} spurious"
        )


@pytest.mark.multichip
@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("chunk_tokens", [32, 64, 128])
def test_dcp_chunking_does_not_change_the_result(chunk_tokens, dcp_size):
    _need_devices(dcp_size)
    _, args, kwargs = _dcp_case(dcp_size, 11)

    hlo = streamindex_topk_dcp.lower(
        *args, **kwargs, chunk_tokens=chunk_tokens
    ).as_text()
    assert "_scheduling_group_id" in hlo, (
        "no scheduling group annotation: the DCP path ran a single unchunked"
        " pass, so this comparison would pass trivially"
    )

    unchunked = np.asarray(streamindex_topk_dcp(*args, **kwargs))
    chunked = np.asarray(
        streamindex_topk_dcp(*args, **kwargs, chunk_tokens=chunk_tokens)
    )
    np.testing.assert_array_equal(chunked, unchunked)
