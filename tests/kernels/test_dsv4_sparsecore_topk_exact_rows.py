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
"""`exact_rows=True` must return the padded path's arrays exactly.

The SparseCore top-k and pack kernels work a row per subcore, 32 rows a wave.
Without `exact_rows` a decode batch of 16 rows is padded to 32 in front of the
kernel and sliced back to 16 behind it -- two copies per array per call, five
SparseCore calls deep in every indexed layer (TPU_INDEXER_TOPK_POST_FUSED).
With it the inactive subcores re-read the last real row and write nothing.
The bar is `array_equal` on indices *and* scores, slot order and padding
included, on every path the indexer takes: plain, payload (fused or not),
single slice, forced partition, two-stage.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import dcp_sc_compact

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SEED = 20261009
K = 2048
DCP = 16


def _require_sparsecore():
    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("needs a TPU with SparseCore")


def _np(x):
    return jax.tree.map(lambda a: np.asarray(jax.block_until_ready(a)), x)


def _assert_same(got, want):
    for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        assert g.shape == w.shape
        np.testing.assert_array_equal(g, w)


def _scores(b, n, lens, rng):
    s = rng.standard_normal((b, n), dtype=np.float32)
    s[np.arange(n)[None, :] >= np.asarray(lens)[:, None]] = -np.inf
    return jnp.asarray(s)


@pytest.mark.parametrize("b", [1, 5, 16, 31, 33, 50])
@pytest.mark.parametrize("payload", ["none", "plain", "fused"])
def test_global_merge_exact_rows(b, payload):
    """The DCP merge (rows DCP x K wide), P = 1 as the served override."""
    _require_sparsecore()
    rng = np.random.default_rng(SEED + b)
    n = DCP * K
    lens = np.full(b, n, np.int32)
    lens[b // 2 :] = rng.integers(K, n, b - b // 2)
    scores = _scores(b, n, lens, rng)
    kw = dict(return_scores=True, partition=1)
    if payload != "none":
        kw.update(
            payload=jnp.asarray(rng.integers(0, 1 << 20, (b, K)).astype(np.int32)),
            payload_offset=jnp.full((b,), 3 * K, jnp.int32),
            fused_payload=payload == "fused",
        )
    want = _np(sparsecore_topk(scores, K, jnp.asarray(lens), **kw))
    got = _np(sparsecore_topk(scores, K, jnp.asarray(lens), exact_rows=True, **kw))
    _assert_same(got, want)
    assert (got[0] >= 0).any()


@pytest.mark.parametrize("b", [16, 20, 48])
@pytest.mark.parametrize("single_slice", [False, True])
@pytest.mark.parametrize("case", ["fit", "overflow"])
def test_local_topk_exact_rows(b, single_slice, case):
    """The per-rank shape (rows 32,768 wide): single slice and two-stage."""
    _require_sparsecore()
    rng = np.random.default_rng(SEED + b + 7)
    n = 32768
    lens = np.zeros(b, np.int32)
    lens[: b // 2] = 5226 if case == "fit" else 20000
    scores = _scores(b, n, lens, rng)
    kw = dict(return_scores=True, fused_payload=True, single_slice=single_slice)
    want = _np(sparsecore_topk(scores, K, jnp.asarray(lens), **kw))
    got = _np(sparsecore_topk(scores, K, jnp.asarray(lens), exact_rows=True, **kw))
    _assert_same(got, want)
    assert (got[0] >= 0).any()


@pytest.mark.parametrize("b", [1, 16, 33])
def test_exact_rows_drops_the_pad_and_slice(b):
    """The point of the flag: no pad in front, no slice behind, at b % 32."""
    _require_sparsecore()
    n = DCP * K
    scores = jnp.zeros((b, n), jnp.float32)
    lens = jnp.full((b,), n, jnp.int32)

    def hlo(**kw):
        return (
            jax.jit(lambda s, m: sparsecore_topk(s, K, m, partition=1, **kw))
            .lower(scores, lens)
            .as_text()
        )

    padded, exact = hlo(), hlo(exact_rows=True)
    assert "stablehlo.slice" in padded
    assert "stablehlo.slice" not in exact
    assert exact.count("stablehlo.pad") < padded.count("stablehlo.pad")


@pytest.mark.parametrize("b,width", [(1, 128), (16, 2048), (33, 256), (64, 2048)])
@pytest.mark.parametrize("keep_rate", [0.0, 0.1, 0.9])
def test_pack_nonnegative_exact_rows(b, width, keep_rate):
    _require_sparsecore()
    rng = np.random.default_rng(SEED + b + width)
    vals = rng.integers(0, 1 << 20, size=(b, width)).astype(np.int32)
    vals[rng.random((b, width)) >= keep_rate] = -1
    vals = jnp.asarray(vals)
    want = _np(dcp_sc_compact.pack_nonnegative(vals))
    got = _np(dcp_sc_compact.pack_nonnegative(vals, exact_rows=True))
    _assert_same(got, want)
