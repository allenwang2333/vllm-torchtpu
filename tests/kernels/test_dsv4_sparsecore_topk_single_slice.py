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
"""`sparsecore_topk(single_slice=True)` must return the default arrays exactly.

The indexer's per-rank score row is sized for max_model_len (65,536 wide at
1M / DCP16), so the partitioned (P > 1, cooperative subcore group) path cuts
it into slices whose tail is empty for every real request. `single_slice`
returns the first slice's top-k from one subcore per row when every row fits
it and falls back to the cooperative path otherwise. The bar is `array_equal`
on indices *and* scores -- slot order and padding included. (The fused-payload
arms of the original tests are gone with the payload API.)

`test_first_slice_alone_is_wrong_on_overflow` is the positive control: on a
row that overflows the slice, the first-slice answer differs from the exact
one, so the overflow cases below exercise the fallback, not a lucky match.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    _pick_partition,
    _sc_topk_direct,
    sparsecore_topk,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SEED = 20261002
K = 2048


def _require_sparsecore():
    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("needs a TPU with SparseCore")


def _scores(kind, b, n, rng):
    if kind == "random":
        return rng.standard_normal((b, n), dtype=np.float32)
    if kind == "heavy_ties":
        return rng.integers(0, 17, size=(b, n), dtype=np.int8).astype(np.float32)
    if kind == "neg_inf_tail":
        # Like the indexer: real scores up to the row length, -inf after.
        return rng.standard_normal((b, n), dtype=np.float32)
    raise ValueError(kind)


def _np(x):
    return np.asarray(jax.block_until_ready(x))


def _lengths(case, b, n_p, rng):
    if case == "fit":
        # 0, tiny, < k, exactly k, mid, exactly the slice.
        base = np.minimum(np.array([0, 5, K - 1, K, 18117, n_p], np.int32), n_p)
        return np.resize(np.concatenate([base, rng.integers(0, n_p + 1, b)]), b)
    if case == "chunk":
        return rng.integers(768, 1025, b).astype(np.int32)
    if case == "overflow_one":
        lens = rng.integers(0, n_p + 1, b).astype(np.int32)
        lens[b // 2] = n_p + 1
        return lens
    if case == "overflow_all":
        return np.full(b, 2 * n_p, np.int32)
    raise ValueError(case)


# (b, n, p): decode local (T=16), mixed-step chunk (T=256), DCP merge (T=16).
SHAPES = [
    pytest.param(16, 65536, 2, id="decode-local-b16-n65536"),
    pytest.param(256, 65536, 2, id="chunk-local-b256-n65536"),
    pytest.param(16, 16 * K, 2, id="merge-b16-n32768"),
    pytest.param(8, 131072, 4, id="p4-b8-n131072"),
]


@pytest.mark.parametrize("b,n,p", SHAPES)
@pytest.mark.parametrize("kind", ["random", "heavy_ties", "neg_inf_tail"])
@pytest.mark.parametrize("case", ["fit", "chunk", "overflow_one", "overflow_all"])
def test_single_slice_equals_two_stage(b, n, p, kind, case):
    _require_sparsecore()
    assert _pick_partition(b, n, K) == p
    rng = np.random.default_rng(SEED + b + n)
    n_p = n // p
    scores = _scores(kind, b, n, rng)
    lengths = _lengths(case, b, n_p, rng)
    if kind == "neg_inf_tail":
        cols = np.arange(n)[None, :]
        scores[cols >= np.minimum(lengths, n)[:, None]] = -np.inf
    scores, lengths = jnp.asarray(scores), jnp.asarray(lengths)
    kw = dict(return_scores=True)
    want_i, want_s = sparsecore_topk(scores, K, lengths, **kw)
    got_i, got_s = sparsecore_topk(scores, K, lengths, single_slice=True, **kw)
    np.testing.assert_array_equal(_np(got_i), _np(want_i))
    np.testing.assert_array_equal(_np(got_s), _np(want_s))
    assert (_np(got_i) >= 0).any()


def test_strided_read_equals_sliced_copy():
    """`row_stride` reads the first slice in place, same as slicing it out."""
    _require_sparsecore()
    for b in (16, 40, 256):
        rng = np.random.default_rng(SEED + b)
        scores = jnp.asarray(_scores("random", b, 65536, rng))
        lengths = jnp.asarray(rng.integers(0, 32769, b).astype(np.int32))
        want = _sc_topk_direct(scores[:, :32768], K, lengths)
        got = _sc_topk_direct(scores, K, lengths, row_stride=2)
        np.testing.assert_array_equal(_np(got[0]), _np(want[0]), err_msg=f"{b=}")
        np.testing.assert_array_equal(_np(got[1]), _np(want[1]), err_msg=f"{b=}")


def test_first_slice_alone_is_wrong_on_overflow():
    _require_sparsecore()
    b, n = 16, 65536
    rng = np.random.default_rng(SEED)
    scores = rng.standard_normal((b, n), dtype=np.float32)
    scores[:, n // 2 :] += 10.0  # the winners live in the second slice
    scores = jnp.asarray(scores)
    lengths = jnp.full((b,), n, jnp.int32)
    exact = _np(sparsecore_topk(scores, K, lengths))
    first = _np(_sc_topk_direct(scores, K, lengths, row_stride=2)[0])
    assert not np.array_equal(first, exact)
    got = _np(sparsecore_topk(scores, K, lengths, single_slice=True))
    np.testing.assert_array_equal(got, exact)
    assert (got >= n // 2).all()
