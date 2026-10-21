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
"""Read-modify-write granules of the attention phase's record stores (interpret).

A 7-row window closes up to two CSA compressed rows k0, k0 + 1. Their RoPE
words share one 4-record row, and their indexer records one 128-record SAL
lane tile, unless k0 sits at the row's / tile's / page's end. The phase
reads each granule once per boundary and must write it back once with both
records. One call here holds every alignment, each in its own request, and
its caches must equal the serialized per-record stores of
`kv_write.compress_store` bit for bit. The control writes the two copies of
a shared granule as two unmerged runs (one record lost) and must fail.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_phase_harness as H  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP  # noqa: E402

Q = 7


def _seq(k0, first):
    """seq_len whose window closes compressed row k0 at row `first` (< 3: and k0 + 1)."""
    return 4 * k0 + 3 - first + Q


# (case, seq_len): granules of the first boundary's records vs the second's
CASES = [
    ("rope_row_and_tile_shared", _seq(500, 0)),
    ("rope_row_and_tile_shared_page0", _seq(0, 0)),
    ("rope_rows_differ_tile_shared", _seq(503, 1)),
    ("tiles_differ_page_shared", _seq(383, 2)),
    ("pages_differ", _seq(511, 0)),
    ("one_boundary", 2007),
]
SEQ = [s for _, s in CASES]


def _boundaries(seq):
    p0 = seq - Q
    return [(p + 1) // 4 - 1 for p in range(p0, seq) if (p + 1) % 4 == 0]


def test_cases_cover_every_alignment():
    ks = {name: _boundaries(s) for name, s in CASES}
    assert all(len(v) == 2 for n, v in ks.items() if n != "one_boundary")
    assert len(ks["one_boundary"]) == 1
    same = lambda k, unit: k[0] // unit == k[1] // unit  # noqa: E731
    assert same(ks["rope_row_and_tile_shared"], 4) and same(ks["rope_row_and_tile_shared"], 128)
    assert not same(ks["rope_rows_differ_tile_shared"], 4)
    assert same(ks["rope_rows_differ_tile_shared"], 128)
    assert not same(ks["tiles_differ_page_shared"], 128)
    assert same(ks["tiles_differ_page_shared"], 256)
    assert not same(ks["pages_differ"], 256)


@pytest.fixture(scope="module")
def layer_set():
    rng = np.random.default_rng(7)
    ls = H.build(rng, ("hca", "csa"), SEQ, 4096)
    return ls, {k: H.make_inputs(rng, ls, k) for k in ("hca", "csa")}


def _diff_requests(ls, layer, role, caches, exp):
    """Requests owning a page of `role` (layer) whose words differ from `exp`."""
    a = ls.role_arr[role][ls.role_layers[role].index(layer)]
    bad = np.flatnonzero(np.any(np.asarray(caches[a]) != exp[a], axis=(1, 2)))
    table = ls.tables[(role, layer)]
    return {r for r in range(table.shape[0]) if np.isin(table[r], bad).any()}


@pytest.mark.parametrize("kind,layer", [("hca", 0), ("csa", 1)])
def test_stores_bit_exact(layer_set, kind, layer):
    ls, inp = layer_set
    caches, _ = H.run_phase(ls, kind, layer, inp[kind])
    exp, _, _ = H.reference(ls, kind, layer, inp[kind])
    for c, x in zip(caches, exp):
        assert np.array_equal(np.asarray(c), x)


def test_positive_control_two_runs(layer_set, monkeypatch):
    ls, inp = layer_set
    monkeypatch.setattr(AP, "STORE_FAULT", "two_runs")
    caches, _ = H.run_phase(ls, "csa", 1, inp["csa"])
    exp, _, _ = H.reference(ls, "csa", 1, inp["csa"])
    # exactly the requests whose two records share a RoPE row / a SAL lane tile
    rows_shared = {i for i, (n, _) in enumerate(CASES) if n.startswith("rope_row_and")}
    tiles_shared = rows_shared | {CASES.index(c) for c in CASES
                                  if c[0] == "rope_rows_differ_tile_shared"}
    assert _diff_requests(ls, 1, "kv_rope", caches, exp) == rows_shared
    assert _diff_requests(ls, 1, "indexer_k", caches, exp) == tiles_shared
    assert _diff_requests(ls, 1, "kv", caches, exp) == set()
