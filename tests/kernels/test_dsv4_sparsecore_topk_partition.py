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
"""`sparsecore_topk(partition=P)` must return the heuristic's arrays exactly.

`partition` forces the partition factor that `_pick_partition` picks for
utilisation: on this tree, the size of the cooperative subcore group that
shares one row through a `VMEM_SHARED` histogram (P = 1: one subcore per row).
The bar is `array_equal` on indices *and* scores, slot order and padding
included, on the single-slice path and off it. (The fused-payload arms of the
original tests are gone with the payload API: the single-stage top-k leaves no
candidate gather to fuse.)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc_mod
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    DEFAULT_PARTITION_OVERRIDE,
    _pick_partition,
    parse_partition_override,
    sparsecore_topk,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SEED = 20261003
K = 2048
DCP = 16


def _require_sparsecore():
    if jax.default_backend() != "tpu" or pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("needs a TPU with SparseCore")


def _np(x):
    return np.asarray(jax.block_until_ready(x))


@pytest.mark.parametrize(
    "spec,want",
    [
        ("", (None, None)),
        ("0", (None, None)),
        ("off", (None, None)),
        ("1", (None, 1)),
        ("on", (None, 1)),
        ("global:1", (None, 1)),
        ("local:4", (4, None)),
        ("local:4,global:2", (4, 2)),
        (" Global:1 , local:2 ", (2, 1)),
    ],
)
def test_parse_partition_override(spec, want):
    assert parse_partition_override(spec) == want


def test_default_override_is_global_p1():
    assert parse_partition_override(DEFAULT_PARTITION_OVERRIDE) == (None, 1)


@pytest.mark.parametrize("spec", ["global", "global:3", "global:0", "rank:1", "x"])
def test_parse_partition_override_rejects(spec):
    with pytest.raises(ValueError):
        parse_partition_override(spec)


def test_pick_partition_override():
    # Heuristic: decode merge b16 over 32768 -> P = 2.
    assert _pick_partition(16, DCP * K, K) == 2
    assert _pick_partition(16, DCP * K, K, 1) == 1
    assert _pick_partition(16, DCP * K, K, 8) == 8  # several waves, allowed
    # Raised to the tile-memory minimum: 65,536-wide rows need P >= 2.
    assert _pick_partition(16, 65536, K, 1) == 2
    with pytest.raises(ValueError):
        _pick_partition(16, DCP * K, K, 3)
    with pytest.raises(ValueError):
        _pick_partition(16, 64, K, 8)  # 8-word slices


def _global_scores(b, real, valid, rng):
    """Rank-major gather of DCP x K candidate scores, -inf past each valid prefix."""
    s = np.full((b, DCP * K), -np.inf, np.float32)
    for r in range(real):
        for d in range(DCP):
            v = min(int(valid[r]), K)
            s[r, d * K : d * K + v] = rng.standard_normal(v)
    return s


@pytest.mark.parametrize(
    "b,real", [(16, 4), (16, 8), (16, 16), (8, 8), (4, 4), (64, 64)]
)
@pytest.mark.parametrize("valid", ["full", "chunk"])
@pytest.mark.parametrize("partition", [1, 4, 8])
def test_global_merge_partition_is_bit_identical(b, real, valid, partition):
    """The DCP merge shape (16 x 2048 gathered candidates per row)."""
    _require_sparsecore()
    rng = np.random.default_rng(SEED + b + real)
    v = np.full(b, K) if valid == "full" else rng.integers(768, 1025, b)
    scores = jnp.asarray(_global_scores(b, real, v, rng))
    lens = jnp.full((b,), DCP * K, jnp.int32)
    kw = dict(return_scores=True)
    want = _np(sparsecore_topk(scores, K, lens, **kw))
    got = _np(sparsecore_topk(scores, K, lens, partition=partition, **kw))
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    assert (got[0][:real] >= 0).any()


@pytest.mark.parametrize("single_slice", [False, True])
@pytest.mark.parametrize("partition", [2, 4, 8])
@pytest.mark.parametrize("case", ["decode_fit", "decode_long", "chunk", "overflow"])
def test_local_partition_is_bit_identical(single_slice, partition, case):
    """The per-rank shape (rows 65,536 wide); single_slice falls back exactly."""
    _require_sparsecore()
    rng = np.random.default_rng(SEED + partition)
    b = 256 if case == "chunk" else 16
    n = 65536
    if case == "decode_fit":
        lens = np.zeros(b, np.int32)
        lens[:4] = 5226
    elif case == "decode_long":
        lens = np.zeros(b, np.int32)
        lens[:8] = 18117  # past the P = 4 window -> the fallback
    elif case == "chunk":
        lens = rng.integers(768, 1025, b).astype(np.int32)
        lens[:8] = 8808
    else:
        lens = np.full(b, 18117, np.int32)
        lens[3] = 40000
    s = rng.standard_normal((b, n), dtype=np.float32)
    s[np.arange(n)[None, :] >= lens[:, None]] = -np.inf
    scores, lens = jnp.asarray(s), jnp.asarray(lens)
    kw = dict(return_scores=True, single_slice=single_slice)
    want = _np(sparsecore_topk(scores, K, lens, return_scores=True))
    got = _np(sparsecore_topk(scores, K, lens, partition=partition, **kw))
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_global_p1_takes_the_one_subcore_kernel(monkeypatch):
    """The override reaches the kernel: P = 1 (one subcore per row) instead of
    the heuristic's cooperative P = 2 group. Either is one SparseCore call."""
    _require_sparsecore()
    scores = jnp.zeros((16, DCP * K), jnp.float32)
    lens = jnp.full((16,), DCP * K, jnp.int32)
    seen = []
    real = sc_mod._sc_topk_direct

    def spy(*args, **kwargs):
        seen.append(kwargs.get("p", 1))
        return real(*args, **kwargs)

    monkeypatch.setattr(sc_mod, "_sc_topk_direct", spy)
    unjitted = sparsecore_topk.__wrapped__
    for kw in ({}, {"partition": 1}):
        jax.make_jaxpr(lambda s, n, kw=kw: unjitted(s, K, n, **kw))(scores, lens)
    assert seen == [2, 1], seen
