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


TAILS = ["seg", "segc", "segc2"]
# Call-site: "segcD" (chain) and the barrier modes change the chunk loop.
CALL_TAILS = TAILS + [
    "segcD", "segcD:segc", "segcD:segc2g256", "segcB2", "segcB3:segc",
    "segcL1", "segcL2:segc",
]


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


@pytest.mark.parametrize("granule", [16, 128, 512, 2048])
@pytest.mark.parametrize("kind", ["iid", "hot_recent", "heavy_ties", "sparse_neg_inf"])
def test_segc2_granules_bit_exact(kind, granule):
    rng = np.random.default_rng(SEED + 43)
    lens = _ragged_lengths(B, N_PROD, K)
    scores = _scores(kind, B, N_PROD, rng, lens)
    inc_i, inc_s = _impl(scores, lens)
    g_i, g_s = _impl(scores, lens, topk_tail="segc2", seg_granule=granule)
    assert _first_diff(g_i, inc_i) is None, _first_diff(g_i, inc_i)
    assert _first_diff(g_s, inc_s) is None, _first_diff(g_s, inc_s)


def _chain_inputs(kinds, seed):
    rng = np.random.default_rng(seed)
    chunks = []
    for i, kind in enumerate(kinds):
        lens = _ragged_lengths(B, N_PROD, K) if i % 2 else np.full(B, N_PROD, np.int32)
        scores = _scores(kind, B, N_PROD, rng, lens)
        bits = _as_bits(scores)
        t_rows = l1.row_threshold_rows(bits, K, jnp.asarray(lens))
        chunks.append((scores, lens, bits, t_rows))
    return chunks


CHAIN_KINDS = ["iid", "hot_recent", "heavy_ties", "sparse_neg_inf"]


@pytest.mark.parametrize("tail", ["segc", "segc2", "segc2g2048"])
def test_chain_fused_stage2_bit_exact(tail):
    """Chained (chunk m-1's stage 2 + chunk m's stage 1 in one kernel) ==
    the per-chunk segc path == the incumbent, on every chunk."""
    chunks = _chain_inputs(CHAIN_KINDS, SEED + 44)
    got = l1.sparsecore_topk_l1_chain(
        [(bits, jnp.asarray(lens), t) for _, lens, bits, t in chunks], K, topk_tail=tail
    )
    assert len(got) == len(chunks)
    for m, ((scores, lens, _, t), (gi, gs)) in enumerate(zip(chunks, got)):
        inc_i, inc_s = _impl(scores, lens, t_rows=t)
        gi, gs = np.asarray(gi), np.asarray(gs)
        assert (err := _mismatch(scores, gi, K, lens)) is None, (m, err)
        assert _first_diff(gi, inc_i) is None, (m, "idx", _first_diff(gi, inc_i))
        assert _first_diff(gs, inc_s) is None, (m, "scores", _first_diff(gs, inc_s))


def test_control_chain_faulty_stage2_goes_red():
    """The chain comparison must catch a stage 2 that drops a live chunk, in
    the fused part (chunks 0..n-2) and in the lone last stage 2 alike."""
    chunks = _chain_inputs(["iid"] * 3, SEED + 45)
    got = l1.sparsecore_topk_l1_chain(
        [(bits, jnp.asarray(lens), t) for _, lens, bits, t in chunks], K,
        topk_tail="segc", seg_fault="droplast",
    )
    red = []
    for (scores, lens, _, t), (gi, _) in zip(chunks, got):
        gi = np.asarray(gi)
        red.append(sum(
            _mismatch(scores[r : r + 1], gi[r : r + 1], K, lens[r : r + 1]) is not None
            for r in range(B) if lens[r] > 0
        ))
    live = [int((c[1] > 0).sum()) for c in chunks]
    print(f"CONTROL chain droplast: red rows per chunk {red} of live {live}")
    assert red[0] == live[0] and red[-1] == live[-1] and all(r > 0 for r in red)


@pytest.mark.parametrize("tail", [""] + TAILS)
def test_write_empty_rows_is_inert(tail):
    # Carried trap: seg/segc/segc2 never see `write_empty_rows`. It is also a
    # no-op on the incumbent (`_sc_topk_direct` deletes it), so every path
    # must give the same bits either way, empty rows included.
    rng = np.random.default_rng(SEED + 44)
    lens = _ragged_lengths(B, N_PROD, K)
    assert (lens == 0).any()
    scores = _scores("iid", B, N_PROD, rng, lens)
    kw = dict(topk_tail=tail) if tail else {}
    on_i, on_s = _impl(scores, lens, **kw)
    off = l1._sparsecore_topk_l1_impl(
        _as_bits(scores), K, jnp.asarray(lens), write_empty_rows=False,
        scheduling_group_id=None, stage2_scheduling_group_id=None,
        threshold_scheduling_group_id=None, **kw,
    )
    assert _first_diff(np.asarray(off[0]), on_i) is None
    assert _first_diff(np.asarray(off[1]), on_s) is None
    assert (on_i[lens == 0] == -1).all()


# --------------------------------------------------------------------------- #
# Positive controls.
# --------------------------------------------------------------------------- #

CONTROL_KINDS = ["iid", "hot_recent", "heavy_ties", "sparse_neg_inf"]


@pytest.mark.parametrize(
    "tail,fault",
    [(t, f) for t in TAILS for f in ("floor", "droplast")] + [("segc2", "dmashort")],
)
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
    # dmashort leaves the last live granule of each segment stale.
    assert red[("iid", False)][0] > 0
    if fault in ("droplast", "dmashort"):
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
    for tail in CALL_TAILS:
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
