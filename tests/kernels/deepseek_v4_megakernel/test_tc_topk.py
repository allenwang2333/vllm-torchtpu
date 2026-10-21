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
"""TensorCore top-k vs a NumPy model of the SparseCore contract, index for index.

Runs the kernel in Pallas interpret mode on CPU. On device the same cases are
compared against `sparsecore_topk` itself (see the module docstring of
`tc_topk`).
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from topk_reference import KINDS, make_case, reference_topk, reference_union  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel.tc_topk import (  # noqa: E402
    tc_topk,
    tc_topk_union_grouped,
    to_group_layout,
)

K = 1024
N = 16384  # smallest supported row width; 128k context is n = 32768


def _run(x, lens, threshold="stack"):
    return np.asarray(
        tc_topk(jnp.asarray(x), K, jnp.asarray(lens), interpret=True, threshold=threshold)
    )


@pytest.mark.parametrize("threshold", ["stack", "radix"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rows", [7, 14])
def test_matches_reference(kind, rows, threshold):
    x, lens = make_case(kind, rows, N, np.random.default_rng(rows))
    np.testing.assert_array_equal(_run(x, lens, threshold), reference_topk(x, K, lens))


def test_56_rows_mixed_lengths():
    rng = np.random.default_rng(56)
    x, _ = make_case("random", 56, N, rng)
    lens = rng.integers(0, N + 1, size=56).astype(np.int32)
    np.testing.assert_array_equal(_run(x, lens), reference_topk(x, K, lens))


def test_check_bites():
    """Positive control: the comparison rejects a plausible wrong answer.

    Ties broken to the highest column instead of the lowest is the error a
    kernel that gets everything else right would make.
    """
    x, lens = make_case("ties", 7, N, np.random.default_rng(0))
    got = _run(x, lens)
    assert (reference_topk(x, K, lens, ties_to_lowest=False) != got).any()
    np.testing.assert_array_equal(got, reference_topk(x, K, lens))


def _check_union(x, lens, lists_too):
    rows = x.shape[0]
    lists, union, ulen = tc_topk_union_grouped(
        to_group_layout(jnp.asarray(x)), jnp.asarray(lens), k=K, interpret=True, lists=lists_too
    )
    ref = reference_topk(x, K, lens)
    if lists_too:
        np.testing.assert_array_equal(np.asarray(lists)[:rows], ref)
    union, ulen = np.asarray(union), np.asarray(ulen)
    for g in range(union.shape[0]):
        want = reference_union(ref[g * 8:(g + 1) * 8])
        got = union[g].reshape(-1)
        assert ulen[g] == want.size
        np.testing.assert_array_equal(got[: want.size], want)
        assert (got[want.size:] == -1).all()


@pytest.mark.parametrize("kind", KINDS)
def test_union_matches_reference(kind):
    """14 rows: one full 8-row group (member bits up to 255) and one with 6 rows."""
    x, lens = make_case(kind, 14, N, np.random.default_rng(14))
    _check_union(x, lens, lists_too=True)


@pytest.mark.parametrize("kind", ["random", "ties", "garbage_tail"])
def test_union_without_lists(kind):
    x, lens = make_case(kind, 7, N, np.random.default_rng(7))
    lens[3] = 0
    _check_union(x, lens, lists_too=False)
