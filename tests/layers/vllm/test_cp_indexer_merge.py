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
"""The torch ownership filter keeps the winners this rank owns, and no more."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    cp_local_to_global
from vllm_torchtpu.layers.vllm.cp_indexer_merge import (
    check_positional_owner, select_owned_winners,
    select_owned_winners_gathered)

pytestmark = pytest.mark.cpu_test

# (dcp_size, interleave_c, k, num_rows)
_LAYOUTS = [
    (2, 4, 8, 5),
    (4, 4, 8, 7),
    (4, 32, 16, 3),
    (8, 4, 16, 9),
    (16, 1, 8, 4),
]


def _candidates(rng, dcp_size, interleave_c, k, num_rows, empty_frac):
    """Per-rank `idxs`: global positions that rank owns, `-1` padded."""
    idxs = np.full((dcp_size, num_rows, k), -1, np.int32)
    for r in range(dcp_size):
        for t in range(num_rows):
            n = rng.integers(0, k + 1) if rng.random() < empty_frac else k
            local = rng.choice(4 * k, size=int(n), replace=False)
            idxs[r, t, :n] = np.asarray(
                cp_local_to_global(jnp.asarray(local, jnp.int32), r, dcp_size,
                                   interleave_c))
    return idxs


def _slots(rng, dcp_size, k, num_rows, empty_frac):
    """`sparsecore_topk` output: `k` distinct columns per row, `-1` padded."""
    out = np.full((num_rows, k), -1, np.int32)
    for t in range(num_rows):
        n = rng.integers(0, k + 1) if rng.random() < empty_frac else k
        out[t, :n] = rng.choice(dcp_size * k, size=int(n), replace=False)
    return out


def _gather(idxs, slots):
    """The global position each winning column carries, `-1` padded."""
    gathered = np.concatenate(list(idxs), axis=1)
    rows = np.arange(slots.shape[0])[:, None]
    return np.where(slots >= 0, gathered[rows, np.maximum(slots, 0)], -1)


def _reference(merged, dcp_size, interleave_c, rank, k):
    """What rank `rank` should keep, spelled out one position at a time."""
    out = np.full((merged.shape[0], k), -1, np.int32)
    cycle = dcp_size * interleave_c
    for t, row in enumerate(merged):
        mine = [
            g for g in row.tolist()
            if g >= 0 and (g // interleave_c) % dcp_size == rank
        ]
        local = [(g // cycle) * interleave_c + g % interleave_c for g in mine]
        out[t, :len(local)] = local
    return out


def _run(slots, idxs_r, *, k, dcp_size, dcp_rank, interleave_c):
    return select_owned_winners(torch.from_numpy(slots),
                                torch.from_numpy(idxs_r),
                                k=k,
                                dcp_size=dcp_size,
                                dcp_rank=dcp_rank,
                                interleave_c=interleave_c).numpy()


@pytest.mark.parametrize("dcp_size,interleave_c,k,num_rows", _LAYOUTS)
@pytest.mark.parametrize("empty_frac", [0.0, 0.5])
def test_matches_value_based_filter(dcp_size, interleave_c, k, num_rows,
                                    empty_frac):
    rng = np.random.default_rng(0xC0FFEE + dcp_size * 31 + k)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, empty_frac)
    slots = _slots(rng, dcp_size, k, num_rows, empty_frac)

    for r in range(dcp_size):
        held = idxs[r][idxs[r] >= 0]
        assert np.all((held // interleave_c) % dcp_size == r)

    merged = _gather(idxs, slots)
    for r in range(dcp_size):
        want = _reference(merged, dcp_size, interleave_c, r, k)
        got = _run(slots,
                   idxs[r],
                   k=k,
                   dcp_size=dcp_size,
                   dcp_rank=r,
                   interleave_c=interleave_c)
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dcp_size,interleave_c,k,num_rows", _LAYOUTS)
def test_ranks_partition_the_winners(dcp_size, interleave_c, k, num_rows):
    """Every selected winner is claimed by exactly one rank."""
    rng = np.random.default_rng(dcp_size * 7 + k)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, 0.5)
    slots = _slots(rng, dcp_size, k, num_rows, 0.5)

    kept = np.zeros((num_rows, ), np.int64)
    for r in range(dcp_size):
        got = _run(slots,
                   idxs[r],
                   k=k,
                   dcp_size=dcp_size,
                   dcp_rank=r,
                   interleave_c=interleave_c)
        for t in range(num_rows):
            row = got[t]
            filled = int((row >= 0).sum())
            assert np.all(row[:filled] >= 0) and np.all(row[filled:] == -1)
        kept += (got >= 0).sum(axis=1)

    selected = _gather(idxs, slots)
    np.testing.assert_array_equal(kept, (selected >= 0).sum(axis=1))


def test_a_rank_owning_every_winner_loses_nothing():
    """A rank owning the whole winner list still fits it at width `k`."""
    dcp_size, interleave_c, k = 4, 1, 16
    slots = np.arange(k, dtype=np.int32)[None, :]
    idxs = np.arange(0, k * dcp_size, dcp_size, dtype=np.int32)[None, :]

    got = _run(slots,
               idxs,
               k=k,
               dcp_size=dcp_size,
               dcp_rank=0,
               interleave_c=interleave_c)
    assert (got[0] >= 0).sum() == k
    assert sorted(got[0].tolist()) == list(range(k))


def _fake_gather(idxs, order):
    """Stand-in for the DCP all-gather that lands blocks in `order`."""

    def gather(x):
        if tuple(x.shape) == (1, 1):
            return torch.as_tensor([list(order)], dtype=torch.int32)
        return torch.from_numpy(
            np.concatenate([idxs[o] for o in order], axis=1))

    return gather


def _merged_in_order(idxs, slots, order):
    """The global position each winning column carries, blocks in `order`."""
    g = np.concatenate([idxs[o] for o in order], axis=1)
    rows = np.arange(slots.shape[0])[:, None]
    return np.where(slots >= 0, g[rows, np.maximum(slots, 0)], -1)


def _run_gathered(slots, idxs, order, *, dcp_size, dcp_rank, interleave_c):
    return select_owned_winners_gathered(torch.from_numpy(slots),
                                         torch.from_numpy(idxs[dcp_rank]),
                                         dcp_size=dcp_size,
                                         dcp_rank=dcp_rank,
                                         interleave_c=interleave_c,
                                         gather=_fake_gather(idxs,
                                                             order)).numpy()


@pytest.mark.parametrize("dcp_size,interleave_c,k,num_rows", _LAYOUTS)
@pytest.mark.parametrize("shuffle", [False, True])
def test_gathered_path_is_order_independent(dcp_size, interleave_c, k,
                                            num_rows, shuffle):
    """Ownership rides in the value, so any concat order gives the same answer."""
    rng = np.random.default_rng(dcp_size * 23 + k)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, 0.5)
    slots = _slots(rng, dcp_size, k, num_rows, 0.5)
    order = list(range(dcp_size))
    if shuffle:
        rng.shuffle(order)

    merged = _merged_in_order(idxs, slots, order)
    for r in range(dcp_size):
        want = _reference(merged, dcp_size, interleave_c, r, k)
        got = _run_gathered(slots,
                            idxs,
                            order,
                            dcp_size=dcp_size,
                            dcp_rank=r,
                            interleave_c=interleave_c)
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dcp_size,interleave_c,k,num_rows", _LAYOUTS)
def test_positional_path_breaks_where_the_gathered_one_holds(
        dcp_size, interleave_c, k, num_rows):
    """Why the second all-gather matters: only the value-carried owner survives
    a gather that does not land in rank order."""
    rng = np.random.default_rng(dcp_size * 29 + k)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, 0.0)
    slots = _slots(rng, dcp_size, k, num_rows, 0.0)
    order = list(range(dcp_size))
    order[0], order[1] = order[1], order[0]

    merged = _merged_in_order(idxs, slots, order)
    diverged = False
    for r in range(dcp_size):
        want = _reference(merged, dcp_size, interleave_c, r, k)
        np.testing.assert_array_equal(
            _run_gathered(slots,
                          idxs,
                          order,
                          dcp_size=dcp_size,
                          dcp_rank=r,
                          interleave_c=interleave_c), want)
        positional = _run(slots,
                          idxs[r],
                          k=k,
                          dcp_size=dcp_size,
                          dcp_rank=r,
                          interleave_c=interleave_c)
        diverged = diverged or not np.array_equal(positional, want)
    assert diverged


@pytest.mark.parametrize("dcp_size,interleave_c,k,num_rows", _LAYOUTS)
def test_owner_check_passes_in_rank_order(dcp_size, interleave_c, k, num_rows):
    rng = np.random.default_rng(dcp_size * 13 + k)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, 0.5)
    slots = _slots(rng, dcp_size, k, num_rows, 0.5)

    for r in range(dcp_size):
        check_positional_owner(torch.from_numpy(slots),
                               torch.from_numpy(idxs[r]),
                               k=k,
                               dcp_size=dcp_size,
                               dcp_rank=r,
                               interleave_c=interleave_c,
                               gather=_fake_gather(idxs, range(dcp_size)))


@pytest.mark.parametrize("dcp_size,interleave_c,k,num_rows", _LAYOUTS)
def test_owner_check_catches_a_permuted_gather(dcp_size, interleave_c, k,
                                               num_rows):
    rng = np.random.default_rng(dcp_size * 17 + k)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, 0.0)
    slots = _slots(rng, dcp_size, k, num_rows, 0.0)
    order = list(range(dcp_size))
    order[0], order[1] = order[1], order[0]

    with pytest.raises(RuntimeError, match="column order"):
        check_positional_owner(torch.from_numpy(slots),
                               torch.from_numpy(idxs[0]),
                               k=k,
                               dcp_size=dcp_size,
                               dcp_rank=0,
                               interleave_c=interleave_c,
                               gather=_fake_gather(idxs, order))


def test_owner_check_catches_a_wrong_interleave():
    """Order is right but the ownership arithmetic is not."""
    dcp_size, interleave_c, k, num_rows = 4, 4, 8, 6
    rng = np.random.default_rng(99)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, 0.0)
    slots = _slots(rng, dcp_size, k, num_rows, 0.0)

    with pytest.raises(RuntimeError, match="owner mismatch"):
        check_positional_owner(torch.from_numpy(slots),
                               torch.from_numpy(idxs[0]),
                               k=k,
                               dcp_size=dcp_size,
                               dcp_rank=0,
                               interleave_c=interleave_c * 2,
                               gather=_fake_gather(idxs, range(dcp_size)))


def test_unowned_rank_gets_an_all_minus_one_row():
    """A rank owning none of a token's winners gets an all -1 row."""
    dcp_size, interleave_c, k = 4, 1, 4
    slots = np.arange(k, 2 * k, dtype=np.int32)[None, :]
    idxs = np.asarray([[1, 5, 9, 13]], np.int32)

    for r in (0, 2, 3):
        got = _run(slots,
                   idxs,
                   k=k,
                   dcp_size=dcp_size,
                   dcp_rank=r,
                   interleave_c=interleave_c)
        assert (got == -1).all()

    got = _run(slots,
               idxs,
               k=k,
               dcp_size=dcp_size,
               dcp_rank=1,
               interleave_c=interleave_c)
    assert (got[0] >= 0).sum() == k
