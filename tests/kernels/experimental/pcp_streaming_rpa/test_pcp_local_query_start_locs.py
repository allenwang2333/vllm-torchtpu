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
"""Per-rank query_start_loc must agree with the rank-major token packing.

Under PCP the sparse-MLA layer runs attention on this rank's own queries, so
it needs a `query_start_loc` over *this rank's* slice of the packed token
buffer -- while the replicated KV-cache write still uses the global one. The
two must describe the same packing or a rank attends over the wrong rows, so
this cross-checks the closed-form cumsum against the packing itself.
"""

import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order, pcp_local_query_start_locs,
    pcp_local_token_counts)


def _reference_from_packing(query_lens, pcp_size, interleave_size):
    """Derive the per-rank cumsum from `build_pcp_rank_major_token_order`.

    Rebuilds what the packing actually did: for each rank, count how many of
    its packed slots belong to each request, in request order.
    """
    counts = pcp_local_token_counts(query_lens, pcp_size, interleave_size)
    padded_local = int(counts.max())
    order, _ = build_pcp_rank_major_token_order(query_lens, pcp_size,
                                                interleave_size,
                                                padded_local * pcp_size)
    request_of_token = np.repeat(np.arange(len(query_lens)),
                                 np.asarray(query_lens, dtype=np.int64))

    expected = np.zeros((pcp_size, len(query_lens) + 1), dtype=np.int32)
    for rank in range(pcp_size):
        slots = order[rank * padded_local:(rank + 1) * padded_local]
        per_req = np.zeros(len(query_lens), dtype=np.int32)
        for slot in slots:
            if slot >= 0:
                per_req[request_of_token[slot]] += 1
        expected[rank, 1:] = np.cumsum(per_req)
    return expected


@pytest.mark.parametrize("pcp_size", [2, 4, 8])
@pytest.mark.parametrize("interleave_size", [1, 2, 3, 128])
def test_matches_the_rank_major_packing(pcp_size, interleave_size):
    rng = np.random.default_rng(pcp_size * 1000 + interleave_size)
    for _ in range(25):
        num_reqs = int(rng.integers(1, 9))
        # Deliberately include lengths that are not multiples of
        # `pcp_size * interleave_size`: that is exactly when ranks disagree on
        # a request's local token count, which is why this array is stacked.
        query_lens = rng.integers(1,
                                  4 * pcp_size * interleave_size + 1,
                                  size=num_reqs).astype(np.int32)

        got = pcp_local_query_start_locs(query_lens, pcp_size, interleave_size)
        expected = _reference_from_packing(query_lens, pcp_size,
                                           interleave_size)
        np.testing.assert_array_equal(got, expected)


def test_row_totals_match_local_token_counts():
    query_lens = np.array([5, 17, 1, 64], dtype=np.int32)
    pcp_size, interleave_size = 4, 2

    qsl = pcp_local_query_start_locs(query_lens, pcp_size, interleave_size)
    counts = pcp_local_token_counts(query_lens, pcp_size, interleave_size)

    np.testing.assert_array_equal(qsl[:, -1], counts)
    assert int(qsl[:, -1].sum()) == int(query_lens.sum())
    # Monotone non-decreasing within every rank's row.
    assert (np.diff(qsl, axis=1) >= 0).all()


def test_padded_request_slots_repeat_the_last_offset():
    query_lens = np.array([9, 3], dtype=np.int32)
    qsl = pcp_local_query_start_locs(query_lens,
                                     pcp_size=2,
                                     interleave_size=1,
                                     num_reqs_padded=5)

    assert qsl.shape == (2, 6)
    # Slots 2..4 are padding: no tokens, so the offset must not advance.
    np.testing.assert_array_equal(qsl[:, 2], qsl[:, -1])


def test_pcp_size_one_is_the_global_cumsum():
    query_lens = np.array([4, 0, 7], dtype=np.int32)
    qsl = pcp_local_query_start_locs(query_lens, pcp_size=1, interleave_size=1)

    np.testing.assert_array_equal(qsl, np.array([[0, 4, 4, 11]],
                                                dtype=np.int32))


def test_empty_batch():
    qsl = pcp_local_query_start_locs(np.zeros(0, dtype=np.int32),
                                     pcp_size=4,
                                     interleave_size=1)
    np.testing.assert_array_equal(qsl, np.zeros((4, 1), dtype=np.int32))
