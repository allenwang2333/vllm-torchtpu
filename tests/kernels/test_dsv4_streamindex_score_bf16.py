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
"""Tests for the grouped bf16 head sum (`TPU_STREAMIDX_SCORE_BF16_GROUP`).

`score_bf16_group=G` changes the prefill scores' rounding, so the bar is not
identity but the output that matters, the top-k set:

* **G = 8, 16.** Per row with more than k live columns, the selected index set
  overlaps the f32 default's in >= 99% of k slots, with and without
  `sctopk_l1`. With `sctopk_l1`, the scoring epilogue's T equals
  `row_threshold_keys` of the kernel's own (bf16-grouped) scores on every live
  row, so T stays a valid lower bound on v_k. Decode rows are unchanged, and
  the scores really differ from f32 (the flag reached the kernel).
* **G = 0.** Lowers to the default program and selects the same indices.

The recall metric is asserted red on a reference with one head dropped.
"""

import difflib

import jax
import numpy as np
import pytest

from test_dsv4_sparsecore_topk_l1 import (  # sibling test module
    K,
    _call_kwargs,
    _capture_t,
    _normalized,
    sim,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

RECALL_MIN = 0.99

CASES = [
    pytest.param((1024,), (230000,), False, id="prefill-230k"),
    pytest.param((1024,), (185000,), True, id="prefill-185k-hot"),
    pytest.param((1024,), (30000,), False, id="prefill-30k"),
    # Decodes, long and short prefills, and padded rows.
    pytest.param(
        (1, 1, 1, 1, 600, 300),
        (250000, 1500, 64, 200000, 230000, 3000),
        True,
        id="mixed-padded",
    ),
]


def _row_len(T_list, S_list):
    """Live columns per token row (compression 1), 0 on padding rows."""
    out = []
    for t, s in zip(T_list, S_list):
        out.extend(s - t + 1 + np.arange(t))
    return np.array(out, np.int64)


def _recall(a_idx, b_idx, rows):
    return np.array(
        [len(set(a_idx[r].tolist()) & set(b_idx[r].tolist())) / K for r in rows]
    )


@pytest.mark.parametrize("sctopk_l1", [True, False])
@pytest.mark.parametrize("group", [8, 16])
@pytest.mark.parametrize("T_list,S_list,hot", CASES)
def test_grouped_bf16_recall(T_list, S_list, hot, group, sctopk_l1):
    kw = _call_kwargs(T_list, S_list, hot=hot)
    on_i, on_s = streamindex_topk(
        **kw, sctopk_l1=sctopk_l1, score_bf16_group=group, return_scores=True
    )
    off_i, off_s = streamindex_topk(**kw, sctopk_l1=sctopk_l1, return_scores=True)
    on_i, off_i = np.asarray(on_i), np.asarray(off_i)
    on_s, off_s = np.asarray(on_s), np.asarray(off_s)
    lens = _row_len(T_list, S_list)
    n_real = len(lens)
    # Padding rows stay empty.
    assert (on_i[n_real:] == -1).all()
    # Decode rows (q_len 1) run the unchanged decode path: bit-identical.
    dec = [i for i, t in enumerate(T_list) if t == 1]
    np.testing.assert_array_equal(on_i[dec], off_i[dec])
    np.testing.assert_array_equal(on_s[dec], off_s[dec])
    # Rows with <= k live columns select all of them either way.
    short = np.flatnonzero(lens <= K)
    for r in short:
        assert set(on_i[r][on_i[r] >= 0].tolist()) == set(range(lens[r])), r
    long_rows = np.flatnonzero(lens > K)
    long_rows = np.setdiff1d(long_rows, dec)
    assert long_rows.size, "no prefill row longer than k"
    assert (on_i[long_rows] >= 0).all(), "a long row left top-k slots unfilled"
    rec = _recall(on_i, off_i, long_rows)
    print(
        f"G={group} l1={sctopk_l1}: recall min {rec.min():.5f} "
        f"mean {rec.mean():.5f} over {rec.size} rows"
    )
    assert rec.min() >= RECALL_MIN, (
        f"recall_min {rec.min():.5f} at row {long_rows[rec.argmin()]}"
    )
    # The flag reached the kernel: bf16 grouping changes some selected score.
    assert not np.array_equal(on_s[long_rows], off_s[long_rows]), (
        "scores are bit-identical to f32; the group flag did not take effect"
    )


@pytest.mark.parametrize("group", [8, 16])
def test_grouped_bf16_epilogue_t_is_a_valid_bound(monkeypatch, group):
    """Epilogue T == row_threshold_keys(kernel's own scores) on live rows."""
    jax.clear_caches()
    store = []
    monkeypatch.setattr(sim, "sparsecore_topk_l1", _capture_t(store))
    T_list, S_list = (1, 1, 698, 324), (70001, 99, 150001, 3333)
    kw = _call_kwargs(T_list, S_list, hot=True)
    try:
        jax.block_until_ready(
            streamindex_topk(**kw, sctopk_l1=True, score_bf16_group=group)
        )
        jax.effects_barrier()
    finally:
        monkeypatch.undo()
        jax.clear_caches()
    assert store, "no chunk reached sparsecore_topk_l1"
    live = 0
    for chunk, (t_epi, t_ref, lens) in enumerate(store):
        m = lens > 0
        live += int(m.sum())
        bad = np.flatnonzero(m & (t_epi != t_ref))
        assert not bad.size, f"chunk {chunk}: {bad.size} rows with T != reference"
    assert live == sum(T_list)


def test_group0_is_the_default_program():
    kw = _call_kwargs((1024,), (230000,))
    for l1 in (False, True):
        default = streamindex_topk.lower(**kw, sctopk_l1=l1).as_text()
        off = streamindex_topk.lower(**kw, sctopk_l1=l1, score_bf16_group=0).as_text()
        a, b = _normalized(default), _normalized(off)
        diff = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""))
        assert not diff, "\n".join(diff[:60])
        g16 = _normalized(
            streamindex_topk.lower(**kw, sctopk_l1=l1, score_bf16_group=16).as_text()
        )
        assert g16 != a, "score_bf16_group=16 lowered to the default program"
    on = np.asarray(streamindex_topk(**kw, score_bf16_group=0))
    off = np.asarray(streamindex_topk(**kw))
    np.testing.assert_array_equal(on, off)


def test_group_must_divide_heads():
    kw = _call_kwargs((1024,), (30000,))
    with pytest.raises(ValueError, match="must divide"):
        streamindex_topk.lower(**kw, score_bf16_group=12)
    with pytest.raises(ValueError, match=">= 0"):
        streamindex_topk.lower(**kw, score_bf16_group=-1)


def test_recall_check_goes_red_on_a_wrong_reference():
    """Positive control: the recall metric rejects one head dropped."""
    kw = _call_kwargs((1024,), (30000,))
    w = kw["indexer_weights"]
    bad = dict(kw, indexer_weights=w.at[:, -1].set(0.0))
    a = np.asarray(streamindex_topk(**kw))
    b = np.asarray(streamindex_topk(**bad))
    rows = np.flatnonzero(_row_len((1024,), (30000,)) > K)
    rec = _recall(a, b, rows)
    print(f"drop-head control: recall min {rec.min():.5f}")
    assert rec.min() < RECALL_MIN, "recall gate cannot see a dropped head"
