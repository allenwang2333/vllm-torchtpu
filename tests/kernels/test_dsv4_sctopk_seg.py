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
"""Tests for the segmented stage 2 (`TPU_STREAMIDX_TOPK_TAIL=seg`).

`topk_tail="seg"` must return exactly what the incumbent stage 2 returns:
the same indices in the same slots and the same score bits, not merely the
same set. Every case is checked against the exact score-multiset oracle and
for bit equality with `topk_tail=""` on the same inputs.

Positive controls: a stage 2 that walks `cnt // 16` chunks per segment
("floor") or one chunk fewer than needed ("droplast") loses candidates; the
same comparison must go red, and the catch rate per input class is printed.
"""

import difflib
import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_dsv4_sparsecore_topk_l1 import (  # noqa: E402
    B,
    K,
    N_128K,
    N_PROD,
    N_SHORT,
    SEED,
    _as_bits,
    _call_kwargs,
    _mismatch,
    _normalized,
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


SHAPES = [
    pytest.param(B, N_PROD, K, id="prod-b128-n264192"),
    pytest.param(B, N_128K, K, id="b128-n133120"),
    pytest.param(16, N_SHORT, K, id="p2-b16-n10240"),
    pytest.param(40, N_PROD, 512, id="b40-n264192-k512"),
    pytest.param(32, N_PROD, K, id="b32-n264192"),
]
KINDS = ["iid", "hot_recent", "heavy_ties", "strict_desc", "all_neg_inf",
         "sparse_neg_inf"]


TAILS = ["seg", "segc"]


@pytest.mark.parametrize("tail", TAILS)
@pytest.mark.parametrize("ragged", [False, True], ids=["full", "ragged"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("b,n,k", SHAPES)
def test_seg_is_bit_exact_to_incumbent(b, n, k, kind, ragged, tail):
    assert l1.l1_supported(b, n, k)
    rng = np.random.default_rng(SEED + 40)
    lens = _ragged_lengths(b, n, k) if ragged else np.full(b, n, np.int32)
    scores = _scores(kind, b, n, rng, lens)
    inc_i, inc_s = _impl(scores, lens, k)
    seg_i, seg_s = _impl(scores, lens, k, topk_tail=tail)
    assert (err := _mismatch(scores, seg_i, k, lens)) is None, err
    assert _first_diff(seg_i, inc_i) is None, ("idx", _first_diff(seg_i, inc_i))
    assert _first_diff(seg_s, inc_s) is None, ("scores", _first_diff(seg_s, inc_s))


@pytest.mark.parametrize("tail", TAILS)
@pytest.mark.parametrize("kind", ["iid", "hot_recent", "heavy_ties"])
def test_seg_python_unrolled_segments_match(kind, tail):
    rng = np.random.default_rng(SEED + 41)
    lens = _ragged_lengths(B, N_PROD, K)
    scores = _scores(kind, B, N_PROD, rng, lens)
    inc_i, inc_s = _impl(scores, lens)
    seg_i, seg_s = _impl(scores, lens, topk_tail=tail, seg_loop="py")
    assert _first_diff(seg_i, inc_i) is None
    assert _first_diff(seg_s, inc_s) is None


# --------------------------------------------------------------------------- #
# Positive controls.
# --------------------------------------------------------------------------- #

CONTROL_KINDS = ["iid", "hot_recent", "heavy_ties", "sparse_neg_inf"]


@pytest.mark.parametrize("tail", TAILS)
@pytest.mark.parametrize("fault", ["floor", "droplast"])
def test_control_faulty_segment_walk_goes_red(fault, tail):
    red = {}
    for kind in CONTROL_KINDS:
        for ragged in (False, True):
            rng = np.random.default_rng(SEED + 42)
            lens = _ragged_lengths(B, N_PROD, K) if ragged else np.full(
                B, N_PROD, np.int32
            )
            scores = _scores(kind, B, N_PROD, rng, lens)
            inc_i, _ = _impl(scores, lens)
            bad_i, _ = _impl(scores, lens, topk_tail=tail, seg_fault=fault)
            rows = sum(
                _mismatch(scores[r : r + 1], bad_i[r : r + 1], K, lens[r : r + 1])
                is not None
                for r in range(B)
            )
            bit_rows = int((bad_i != inc_i).any(1).sum())
            red[(kind, ragged)] = (rows, bit_rows)
    print(f"CONTROL {tail} {fault}: " + " ".join(
        f"{k}{'/rag' if r else ''}={o}/{bb}of{B}" for (k, r), (o, bb) in red.items()
    ))
    # droplast drops a live chunk of every segment with a candidate: every live
    # iid row must lose a winner. floor only bites where cnt % 16 != 0.
    assert red[("iid", False)][0] > 0
    if fault == "droplast":
        assert red[("iid", False)][0] == B


# --------------------------------------------------------------------------- #
# Call site.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "T_list,S_list,hot,extra",
    [
        ((1024,), (98304,), False, {}),
        ((1024,), (230000,), True, {}),
        ((1024,), (4096,), False, {}),
        ((1, 1, 1, 1, 600, 300), (250000, 1500, 64, 200000, 230000, 3000), True, {}),
        ((1024,), (230000,), False, {"chunk_tokens": None}),
    ],
    ids=["98k", "230k-hot", "4k", "mixed-padded", "unchunked"],
)
def test_call_site_seg_bit_exact(T_list, S_list, hot, extra):
    kw = _call_kwargs(T_list, S_list, hot=hot, **extra)
    inc = np.asarray(streamindex_topk(**kw, sctopk_l1=True))
    for tail in TAILS:
        seg = np.asarray(streamindex_topk(**kw, sctopk_l1=True, topk_tail=tail))
        assert _first_diff(seg, inc) is None, (tail, _first_diff(seg, inc))


def test_tail_default_lowers_to_the_default_program():
    kw = _call_kwargs((1024,), (230000,))
    default = streamindex_topk.lower(**kw, sctopk_l1=True).as_text()
    off = streamindex_topk.lower(**kw, sctopk_l1=True, topk_tail="").as_text()
    seg = streamindex_topk.lower(**kw, sctopk_l1=True, topk_tail="seg").as_text()
    assert "sc_topk_seg" not in default and "sc_topk_seg" in seg
    a, b = _normalized(default), _normalized(off)
    diff = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""))
    assert not diff, "\n".join(diff[:60])
