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

import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
    cp_local_to_global
from vllm_torchtpu.layers.vllm.cp_indexer_merge import select_owned_winners

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
            idxs[r, t, :n] = cp_local_to_global(local.astype(np.int32), r,
                                                dcp_size, interleave_c)
    return idxs


def _one_per_rank(dcp_size, k):
    """`idxs` for one row; rank `r` owns positions `r + dcp_size * j`."""
    return np.stack([(r + dcp_size * np.arange(k)).astype(np.int32)
                     for r in range(dcp_size)])[:, None, :]


def _slots(rng, dcp_size, k, num_rows, empty_frac):
    """`sparsecore_topk` output: `k` distinct columns per row, `-1` padded."""
    out = np.full((num_rows, k), -1, np.int32)
    for t in range(num_rows):
        n = rng.integers(0, k + 1) if rng.random() < empty_frac else k
        out[t, :n] = rng.choice(dcp_size * k, size=int(n), replace=False)
    return out


def _merged(idxs, winner_cols, order):
    """The global position each winning column carries, blocks in `order`."""
    g = np.concatenate([idxs[o] for o in order], axis=1)
    rows = np.arange(winner_cols.shape[0])[:, None]
    return np.where(winner_cols >= 0, g[rows, np.maximum(winner_cols, 0)], -1)


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


def _run(winner_cols, idxs, order, *, dcp_size, dcp_rank, interleave_c):
    """Run the filter with the all-gather landing blocks in `order`."""

    def gather(x):
        del x
        return torch.from_numpy(
            np.concatenate([idxs[o] for o in order], axis=1))

    return select_owned_winners(torch.from_numpy(winner_cols),
                                torch.from_numpy(idxs[dcp_rank]),
                                dcp_size=dcp_size,
                                dcp_rank=dcp_rank,
                                interleave_c=interleave_c,
                                gather=gather).numpy()


@pytest.mark.parametrize("dcp_size,interleave_c,k,num_rows", _LAYOUTS)
@pytest.mark.parametrize("empty_frac", [0.0, 0.5])
def test_matches_value_based_filter(dcp_size, interleave_c, k, num_rows,
                                    empty_frac):
    rng = np.random.default_rng(0xC0FFEE + dcp_size * 31 + k)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, empty_frac)
    winner_cols = _slots(rng, dcp_size, k, num_rows, empty_frac)
    order = range(dcp_size)

    for r in range(dcp_size):
        held = idxs[r][idxs[r] >= 0]
        assert np.all((held // interleave_c) % dcp_size == r)

    merged = _merged(idxs, winner_cols, order)
    for r in range(dcp_size):
        want = _reference(merged, dcp_size, interleave_c, r, k)
        got = _run(winner_cols,
                   idxs,
                   order,
                   dcp_size=dcp_size,
                   dcp_rank=r,
                   interleave_c=interleave_c)
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dcp_size,interleave_c,k,num_rows", _LAYOUTS)
@pytest.mark.parametrize("shuffle", [False, True])
def test_is_order_independent(dcp_size, interleave_c, k, num_rows, shuffle):
    """Ownership rides in the value, so any gather order gives the same answer.

    The device all-gather is not rank-major, and reading the owner off the
    column instead halted the 128K DCP=16 server.
    """
    rng = np.random.default_rng(dcp_size * 23 + k)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, 0.5)
    winner_cols = _slots(rng, dcp_size, k, num_rows, 0.5)
    order = list(range(dcp_size))
    if shuffle:
        rng.shuffle(order)

    merged = _merged(idxs, winner_cols, order)
    for r in range(dcp_size):
        want = _reference(merged, dcp_size, interleave_c, r, k)
        got = _run(winner_cols,
                   idxs,
                   order,
                   dcp_size=dcp_size,
                   dcp_rank=r,
                   interleave_c=interleave_c)
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dcp_size,interleave_c,k,num_rows", _LAYOUTS)
def test_ranks_partition_the_winners(dcp_size, interleave_c, k, num_rows):
    """Every selected winner is claimed by exactly one rank."""
    rng = np.random.default_rng(dcp_size * 7 + k)
    idxs = _candidates(rng, dcp_size, interleave_c, k, num_rows, 0.5)
    winner_cols = _slots(rng, dcp_size, k, num_rows, 0.5)
    order = range(dcp_size)

    kept = np.zeros((num_rows, ), np.int64)
    for r in range(dcp_size):
        got = _run(winner_cols,
                   idxs,
                   order,
                   dcp_size=dcp_size,
                   dcp_rank=r,
                   interleave_c=interleave_c)
        for t in range(num_rows):
            row = got[t]
            filled = int((row >= 0).sum())
            assert np.all(row[:filled] >= 0) and np.all(row[filled:] == -1)
        kept += (got >= 0).sum(axis=1)

    selected = _merged(idxs, winner_cols, order)
    np.testing.assert_array_equal(kept, (selected >= 0).sum(axis=1))


def test_a_rank_owning_every_winner_loses_nothing():
    """A rank owning the whole winner list still fits it at width `k`."""
    dcp_size, interleave_c, k = 4, 1, 16
    idxs = _one_per_rank(dcp_size, k)
    winner_cols = np.arange(k, dtype=np.int32)[None, :]

    got = _run(winner_cols,
               idxs,
               range(dcp_size),
               dcp_size=dcp_size,
               dcp_rank=0,
               interleave_c=interleave_c)
    assert (got[0] >= 0).sum() == k
    assert sorted(got[0].tolist()) == list(range(k))


def test_unowned_rank_gets_an_all_minus_one_row():
    """A rank owning none of a token's winners gets an all -1 row."""
    dcp_size, interleave_c, k = 4, 1, 4
    idxs = _one_per_rank(dcp_size, k)
    winner_cols = np.arange(k, 2 * k, dtype=np.int32)[None, :]

    for r in (0, 2, 3):
        got = _run(winner_cols,
                   idxs,
                   range(dcp_size),
                   dcp_size=dcp_size,
                   dcp_rank=r,
                   interleave_c=interleave_c)
        assert (got == -1).all()

    got = _run(winner_cols,
               idxs,
               range(dcp_size),
               dcp_size=dcp_size,
               dcp_rank=1,
               interleave_c=interleave_c)
    assert (got[0] >= 0).sum() == k
