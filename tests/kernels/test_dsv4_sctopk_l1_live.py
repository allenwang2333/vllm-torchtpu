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
"""Tests for the live-slice stage-1 DMA (`TPU_STREAMIDX_L1_LIVE=1`).

`live=True` changes only which score words stage 1 copies into VMEM
(align16(slice length), none for an empty slice), so it must return the SAME
bits as `live=False`: the same indices in the same slots and the same score
bits, on every tail and on the fused chain. Production shapes: b=128 rows
(TPU_STREAMIDX_CHUNK_TOKENS=128), n=264192 (257 pages), k=2048, causal row
lengths C+1..C+128 at C in {0, 32K, 128K, 262016}.

Positive control: `live_fault="shift"` reads every slice 16 words early;
the same comparison must go red.
"""

import os
import re
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_dsv4_sparsecore_topk_l1 import (  # noqa: E402
    B,
    K,
    N_PROD,
    N_SHORT,
    SEED,
    _as_bits,
    _call_kwargs,
    _mismatch,
    _ragged_lengths,
    _scores,
)

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    streamindex_topk,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    sctopk_l1 as l1,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

CONTEXTS = [0, 32768, 131072, 262016]


def _causal_lengths(c, b=B):
    return (c + 1 + np.arange(b)).astype(np.int32)


def _impl(scores, lens, k=K, **kw):
    idx, keys = l1._sparsecore_topk_l1_impl(
        _as_bits(scores),
        k,
        jnp.asarray(lens),
        write_empty_rows=True,
        scheduling_group_id=None,
        stage2_scheduling_group_id=None,
        threshold_scheduling_group_id=None,
        **kw,
    )
    return np.asarray(idx), np.asarray(keys)


def _first_diff(a, b):
    bad = np.argwhere(a != b)
    return None if bad.size == 0 else (bad.shape[0], tuple(bad[0]))


@pytest.mark.parametrize("tail", ["", "seg", "segc", "segc2"])
@pytest.mark.parametrize("kind", ["iid", "hot_recent", "heavy_ties", "sparse_neg_inf"])
@pytest.mark.parametrize("c", CONTEXTS)
def test_live_bit_exact_at_production_contexts(c, kind, tail):
    rng = np.random.default_rng(SEED + 60 + c % 97)
    lens = _causal_lengths(c)
    scores = _scores(kind, B, N_PROD, rng, lens)
    off_i, off_s = _impl(scores, lens, topk_tail=tail)
    on_i, on_s = _impl(scores, lens, topk_tail=tail, live=True)
    assert (err := _mismatch(scores, on_i, K, lens)) is None, err
    assert _first_diff(on_i, off_i) is None, ("idx", _first_diff(on_i, off_i))
    assert _first_diff(on_s, off_s) is None, ("scores", _first_diff(on_s, off_s))


@pytest.mark.parametrize(
    "b,n,k,lens_kind",
    [
        pytest.param(B, N_PROD, K, "ragged", id="b128-ragged"),
        pytest.param(B, N_PROD, K, "full", id="b128-full"),
        pytest.param(B, N_PROD, K, "empty", id="b128-all-empty"),
        pytest.param(40, N_PROD, 512, "ragged", id="b40-k512-padded-waves"),
        pytest.param(16, N_SHORT, K, "ragged", id="p2-b16-one-wave"),
    ],
)
def test_live_bit_exact_edge_lengths(b, n, k, lens_kind):
    assert l1.l1_supported(b, n, k)
    rng = np.random.default_rng(SEED + 61)
    if lens_kind == "ragged":
        lens = _ragged_lengths(b, n, k)
    elif lens_kind == "full":
        lens = np.full(b, n, np.int32)
    else:
        lens = np.zeros(b, np.int32)
    scores = _scores("iid", b, n, rng, lens)
    for tail in ("", "segc2"):
        off_i, off_s = _impl(scores, lens, k, topk_tail=tail)
        on_i, on_s = _impl(scores, lens, k, topk_tail=tail, live=True)
        assert _first_diff(on_i, off_i) is None, (tail, _first_diff(on_i, off_i))
        assert _first_diff(on_s, off_s) is None, (tail, _first_diff(on_s, off_s))


def _chain(chunks, **kw):
    return [
        (np.asarray(i), np.asarray(s))
        for i, s in l1.sparsecore_topk_l1_chain(
            [(bits, jnp.asarray(lens), t) for _, lens, bits, t in chunks],
            K,
            topk_tail="segc2",
            **kw,
        )
    ]


def _chain_inputs(seed):
    # One prefill's 8 chunks of 128 rows at C = 32K (T=1024 step).
    rng = np.random.default_rng(seed)
    chunks = []
    for m in range(8):
        lens = _causal_lengths(32768 + 128 * m)
        scores = _scores("hot_recent" if m % 2 else "iid", B, N_PROD, rng, lens)
        bits = _as_bits(scores)
        t_rows = l1.row_threshold_rows(bits, K, jnp.asarray(lens))
        chunks.append((scores, lens, bits, t_rows))
    return chunks


def test_live_chain_bit_exact():
    """segcD: the fused chain with live stage 1 == without, every chunk."""
    chunks = _chain_inputs(SEED + 62)
    off = _chain(chunks)
    on = _chain(chunks, live=True)
    for m, ((scores, lens, _, _), (oi, os_), (ni, ns)) in enumerate(
        zip(chunks, off, on)
    ):
        assert (err := _mismatch(scores, ni, K, lens)) is None, (m, err)
        assert _first_diff(ni, oi) is None, (m, "idx", _first_diff(ni, oi))
        assert _first_diff(ns, os_) is None, (m, "scores", _first_diff(ns, os_))


@pytest.mark.parametrize("c", CONTEXTS)
def test_control_live_shift_fault_goes_red(c):
    """A stage 1 that reads each slice 16 words early must fail both the
    oracle and the bit comparison."""
    rng = np.random.default_rng(SEED + 63)
    lens = _causal_lengths(c)
    scores = _scores("iid", B, N_PROD, rng, lens)
    off_i, _ = _impl(scores, lens, topk_tail="segc2")
    bad_i, _ = _impl(scores, lens, topk_tail="segc2", live=True, live_fault="shift")
    red = sum(
        _mismatch(scores[r : r + 1], bad_i[r : r + 1], K, lens[r : r + 1]) is not None
        for r in range(B)
    )
    print(f"CONTROL live shift C={c}: red rows {red} of {B}")
    assert _first_diff(bad_i, off_i) is not None
    assert red > 0


def test_control_live_shift_fault_goes_red_on_chain():
    chunks = _chain_inputs(SEED + 64)[:3]
    off = _chain(chunks)
    bad = _chain(chunks, live=True, live_fault="shift")
    for m, ((oi, _), (bi, _)) in enumerate(zip(off, bad)):
        assert _first_diff(bi, oi) is not None, m


@pytest.mark.parametrize(
    "T_list,S_list,hot",
    [
        ((1024,), (1024,), False),
        ((1024,), (33792,), True),
        ((1024,), (132096,), False),
        ((1024,), (263040,), True),
        ((1, 1, 1, 1, 600, 300), (250000, 1500, 64, 200000, 230000, 3000), True),
    ],
    ids=["C0", "C32k-hot", "C128k", "C262016-hot", "mixed-padded"],
)
@pytest.mark.parametrize("tail", ["", "segcD"])
def test_call_site_live_bit_exact(T_list, S_list, hot, tail):
    kw = _call_kwargs(T_list, S_list, hot=hot)
    kw.update(sctopk_l1=True, topk_tail=tail)
    hlo = streamindex_topk.lower(**kw, l1_live=True).as_text()
    assert re.search(r"sc_topk_l1\w*_live", hlo), "l1_live did not reach stage 1"
    off = np.asarray(streamindex_topk(**kw))
    on = np.asarray(streamindex_topk(**kw, l1_live=True))
    assert _first_diff(on, off) is None, _first_diff(on, off)
