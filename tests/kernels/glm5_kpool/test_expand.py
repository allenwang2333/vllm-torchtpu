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
"""``expand_pools_with_tail`` vs the step-2 CPU oracle. Rung 1, no TPU.

Everything is compared as **sets** of selected ids, never as ordered vectors:
``streamindex_topk`` returns pool ids in unspecified order and torch / XLA /
SparseCore break top-k ties differently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.glm5_kpool import (align_to, derive_out_width,
                                              expand_pools_with_tail)

# The oracle lives under tests/models/vllm/glm5_next; make it importable
# regardless of which subset of the suite pytest collected. The entry is
# ``tests/models/vllm`` (which exposes the package ``glm5_next``), never
# ``tests/models`` -- that would put a directory named ``vllm`` on sys.path and
# shadow the real package.
sys.path.insert(0,
                str(Path(__file__).resolve().parents[2] / "models" / "vllm"))
from glm5_next import kpool_oracle as O  # noqa: E402

PROD = O.KpoolParams()
PROD_WIDTH = derive_out_width(PROD.index_topk, PROD.index_kpool)  # 2176

SMALL = O.KpoolParams(index_kpool=4,
                      index_topk=8,
                      index_head_dim=16,
                      index_n_heads=4)
MID = O.KpoolParams(index_kpool=4,
                    index_topk=64,
                    index_head_dim=32,
                    index_n_heads=8)


def _pad_pool_ids(rows: list[list[int]], select_k: int) -> np.ndarray:
    """Pack per-row pool ids into ``[T, select_k]`` with a ``-1`` **suffix**,
    which is ``sparsecore_topk``'s documented output contract
    (``sparsecore_topk.py:25-28``)."""
    out = np.full((len(rows), select_k), -1, dtype=np.int32)
    for t, ids in enumerate(rows):
        assert len(ids) <= select_k
        out[t, :len(ids)] = ids
    return out


def _synthetic_oracle(params, seq_len, seed):
    """Oracle rows plus the pool ids that produced them, from random tensors.

    No transformers dependency: ``test_kpool_oracle.py`` is what ties the oracle
    to the reference; this file's job is to tie the op to the oracle.
    """
    rng = np.random.default_rng(seed)
    d, h = params.index_head_dim, params.index_n_heads
    k = rng.normal(size=(seq_len, d))
    gate = rng.normal(size=(seq_len, d))
    ape = rng.normal(size=(params.index_kpool, d))
    q = rng.normal(size=(seq_len, h, d))
    w = rng.normal(size=(seq_len, h))
    pk = O.pooled_keys(k, gate, ape, index_kpool=params.index_kpool)
    scores = O.pool_scores(q,
                           pk,
                           w,
                           softmax_scale=params.softmax_scale,
                           n_heads=h)
    rows = O.select_pool_ids(scores,
                             seq_len=seq_len,
                             index_kpool=params.index_kpool,
                             select_k=params.select_k)
    expected = O.expand_with_tail(rows,
                                  index_kpool=params.index_kpool,
                                  out_width=params.reference_out_width)
    return rows, expected


def _run(pool_ids, positions, valid, params, out_width):
    return np.asarray(
        jax.jit(expand_pools_with_tail,
                static_argnames=("index_kpool",
                                 "out_width"))(jnp.asarray(pool_ids),
                                               jnp.asarray(positions),
                                               jnp.asarray(valid),
                                               index_kpool=params.index_kpool,
                                               out_width=out_width))


# ---------------------------------------------------------------------------
# out_width is derived, not chosen.
# ---------------------------------------------------------------------------


def test_derive_out_width_is_2176():
    assert align_to(2051, 128) == 2176
    assert derive_out_width(2048, 4) == 2176
    assert PROD_WIDTH == 2176
    assert PROD_WIDTH >= PROD.index_topk + PROD.index_kpool - 1
    assert PROD_WIDTH % PROD.index_kpool == 0
    # The rationale that was grafted on for 2052 was simply false.
    assert 2052 % 128 != 0


def test_out_width_2048_is_rejected_statically():
    """Trimming to ``index_topk`` clips the always-selected tail. That is
    invisible at ``L % index_kpool == 0`` and wrong at the other three
    residues, so it has to fail at trace time, not at inference time.

    ``ValueError`` and not ``AssertionError`` on purpose: this is a
    construction-time configuration guard over plain ints, and ``python -O``
    deletes ``assert``."""
    pool_ids = np.full((2, 512), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="clips the always-selected tail"):
        expand_pools_with_tail(jnp.asarray(pool_ids),
                               jnp.zeros(2, jnp.int32),
                               jnp.ones(2, bool),
                               index_kpool=4,
                               out_width=2048)
    # 2052 satisfies the correctness bound (it is only VMEM-suboptimal), so it
    # must NOT raise -- the assertion has to discriminate, not just refuse.
    expand_pools_with_tail(jnp.asarray(pool_ids),
                           jnp.zeros(2, jnp.int32),
                           jnp.ones(2, bool),
                           index_kpool=4,
                           out_width=2052)


def test_out_width_must_be_divisible_by_index_kpool():
    pool_ids = np.full((1, 2), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="multiple of index_kpool"):
        expand_pools_with_tail(jnp.asarray(pool_ids),
                               jnp.zeros(1, jnp.int32),
                               jnp.ones(1, bool),
                               index_kpool=4,
                               out_width=11)


# ---------------------------------------------------------------------------
# (i) Set equality against the oracle, at every residue of L mod index_kpool.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params,seq_len", [
    (SMALL, 13),
    (SMALL, 14),
    (SMALL, 15),
    (SMALL, 16),
    (MID, 13),
    (MID, 14),
    (MID, 15),
    (MID, 16),
    (MID, 190),
    (MID, 191),
    (MID, 192),
    (MID, 193),
])
def test_sets_match_oracle(params, seq_len):
    """``L % 4 in {1, 2, 3}`` is mandatory: at ``L % 4 == 0`` the last query's
    tail is empty and a dropped tail cannot fire, which is exactly why the
    existing 192-token fixture is green today."""
    rows, expected = _synthetic_oracle(params, seq_len, seed=seq_len * 31)
    width = derive_out_width(params.index_topk, params.index_kpool)
    got = _run(_pad_pool_ids(rows, params.select_k),
               np.arange(seq_len, dtype=np.int32),
               np.ones(seq_len, dtype=bool), params, width)
    assert got.shape == (seq_len, width)
    for p in range(seq_len):
        assert O.row_id_set(got[p]) == O.row_id_set(expected[p]), (seq_len, p)


REF_LENS = (13, 14, 15, 16, 190, 191, 192, 193)


def _ref_case(seq_len):
    return O.make_case(MID,
                       hidden_size=64,
                       q_lora_rank=48,
                       seq_len=seq_len,
                       weight_seed=17,
                       input_seed=seq_len)


@pytest.fixture(scope="module")
def ref_all():
    """One subprocess for the module. See ``kpool_oracle.run_reference_cases``
    for why the reference cannot be imported in process."""
    if not O.glm5_next_source_available():
        pytest.skip(
            f"no glm5_next transformers overlay; set {O.TRANSFORMERS_SRC_ENV}")
    return O.run_reference_cases([_ref_case(n) for n in REF_LENS])


@pytest.mark.parametrize("seq_len", REF_LENS)
def test_sets_match_the_transformers_reference(ref_all, seq_len):
    """The op, end to end against ``Glm5NextTextIndexer`` itself.

    ``test_kpool_oracle.py`` proves oracle == reference and the test above
    proves op == oracle; this closes the triangle so a shared mistake in the
    two numpy paths cannot pass.
    """
    res = ref_all[O.case_key(_ref_case(seq_len))]
    pk = O.pooled_keys(res["k"],
                       res["gate"],
                       res["ape"],
                       index_kpool=MID.index_kpool)
    scores = O.pool_scores(res["q"],
                           pk,
                           res["weights_raw"],
                           softmax_scale=MID.softmax_scale,
                           n_heads=MID.index_n_heads)
    rows = O.select_pool_ids(scores,
                             seq_len=seq_len,
                             index_kpool=MID.index_kpool,
                             select_k=MID.select_k)
    width = derive_out_width(MID.index_topk, MID.index_kpool)
    got = _run(_pad_pool_ids(rows, MID.select_k),
               np.arange(seq_len, dtype=np.int32),
               np.ones(seq_len, dtype=bool), MID, width)
    ref = res["ref"]
    for p in range(seq_len):
        assert O.row_id_set(got[p]) == O.row_id_set(ref[p]), (seq_len, p)


# ---------------------------------------------------------------------------
# (ii) The prefix property: the reference fails it, this op passes.
# ---------------------------------------------------------------------------


def test_reference_fails_prefix_property_and_the_op_passes():
    """The inverted assertion documents why the op exists.

    ``kernels/mla/sparse/kernel.py:209`` / ``:339`` mask by column position
    against ``kv_lens`` (``:622``), so an interior ``-1`` gathers a decoy row at
    full weight. If the reference ever becomes prefix-contiguous this test
    fails and the op's justification has to be re-derived -- not deleted.
    """
    seq_len = 14
    rows, _ = _synthetic_oracle(SMALL, seq_len, seed=99)
    # Reproduce the reference's layout: tail at the fixed column
    # index_kpool * select_k, holes in between.
    kp, sk = SMALL.index_kpool, SMALL.select_k
    ref_like = np.full((seq_len, sk * kp + kp - 1), -1, dtype=np.int32)
    for p, ids in enumerate(rows):
        for n, j in enumerate(ids):
            ref_like[p, kp * n:kp * n + kp] = kp * j + np.arange(kp)
        start, count = O.tail_span(p, kp)
        ref_like[p, sk * kp:sk * kp + count] = start + np.arange(count)

    assert any(not O.is_prefix_contiguous(r) for r in ref_like)

    got = _run(_pad_pool_ids(rows, sk), np.arange(seq_len, dtype=np.int32),
               np.ones(seq_len, dtype=bool), SMALL,
               derive_out_width(SMALL.index_topk, kp))
    for p in range(seq_len):
        assert O.is_prefix_contiguous(got[p]), p
        assert O.row_id_set(got[p]) == O.row_id_set(ref_like[p]), p


@pytest.mark.parametrize("params,seq_len", [(SMALL, 15), (MID, 191)])
def test_no_duplicates_and_no_future_ids(params, seq_len):
    """Pool ids and tail ids cannot collide: ``kp*j + kp - 1 <= p`` implies
    ``kp*(j+1) <= kp*floor((p+1)/kp) == tail_start``."""
    rows, _ = _synthetic_oracle(params, seq_len, seed=7)
    width = derive_out_width(params.index_topk, params.index_kpool)
    got = _run(_pad_pool_ids(rows, params.select_k),
               np.arange(seq_len, dtype=np.int32),
               np.ones(seq_len, dtype=bool), params, width)
    for p in range(seq_len):
        ids = [int(v) for v in got[p] if int(v) != -1]
        assert len(ids) == len(set(ids)), p
        assert all(0 <= i <= p for i in ids), p
        n_valid = params.index_kpool * len(rows[p])
        assert len(ids) == n_valid + O.tail_span(p, params.index_kpool)[1]


# ---------------------------------------------------------------------------
# (iii) Identity regime, at true production geometry.
# ---------------------------------------------------------------------------


def test_identity_regime_at_production_geometry():
    """``p <= 2050`` -> set == ``range(p + 1)``, at ``select_k = 512`` and
    ``out_width = 2176``.

    In the identity regime every candidate pool fits the budget, so the input
    is the candidate prefix itself; no scores are needed and no HF module is
    built. This is the theorem the 2048 cap rests on, checked at the exact
    widths the device will use.
    """
    kp, sk = PROD.index_kpool, PROD.select_k
    positions = np.arange(PROD.identity_bound + 1, dtype=np.int32)
    seq_len = int(positions[-1]) + 1
    pool_ids = np.full((seq_len, sk), -1, dtype=np.int32)
    for p in range(seq_len):
        n = O.num_candidate_pools(p, seq_len, kp)
        assert n <= sk
        pool_ids[p, :n] = np.arange(n)

    got = _run(pool_ids, positions, np.ones(seq_len, dtype=bool), PROD,
               PROD_WIDTH)
    assert got.shape == (seq_len, PROD_WIDTH)
    for p in range(seq_len):
        assert O.row_id_set(got[p]) == set(range(p + 1)), p
        assert O.is_prefix_contiguous(got[p]), p


def test_identity_breaks_one_past_the_bound():
    """Positive proof that 2050 is the real edge, not an arbitrary cap."""
    kp, sk = PROD.index_kpool, PROD.select_k
    p = PROD.identity_bound + 1  # 2051
    n = O.num_candidate_pools(p, p + 1, kp)
    assert n == sk + 1 > sk
    pool_ids = np.full((1, sk), -1, dtype=np.int32)
    pool_ids[0] = np.arange(sk)  # budget-limited: drop the oldest candidate
    got = _run(pool_ids, np.array([p], dtype=np.int32), np.ones(1, dtype=bool),
               PROD, PROD_WIDTH)
    ids = O.row_id_set(got[0])
    assert len(ids) == PROD.index_topk == 2048
    assert ids != set(range(p + 1))


# ---------------------------------------------------------------------------
# (iv) The full row with no -1 at all.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("residue", [0, 1, 2, 3])
def test_full_budget_row_places_the_tail_at_column_2048(residue):
    """When ``total >= k`` the ``-1`` prefill at ``sparsecore_topk.py:262-267``
    never runs, so every column up to ``n_valid`` is real. At residue 3 the row
    is 2051 long inside a 2176 buffer -- the case that a 2048-wide buffer
    silently truncates."""
    kp, sk = PROD.index_kpool, PROD.select_k
    # A query far enough along that the budget is saturated, with
    # (p + 1) % 4 == residue.
    p = 100000 + ((residue - (100000 + 1) % kp) % kp)
    assert (p + 1) % kp == residue
    tail_start, tail_count = O.tail_span(p, kp)
    assert tail_count == residue

    base = (p + 1) // kp - sk  # the oldest pool still in budget
    pool_ids = (base + np.arange(sk, dtype=np.int32))[None, :]
    got = _run(pool_ids, np.array([p], dtype=np.int32), np.ones(1, dtype=bool),
               PROD, PROD_WIDTH)[0]

    n_valid = kp * sk
    assert n_valid == 2048
    assert (got[:n_valid] != -1).all()
    np.testing.assert_array_equal(
        got[n_valid:n_valid + tail_count],
        tail_start + np.arange(tail_count, dtype=np.int32))
    assert (got[n_valid + tail_count:] == -1).all()
    assert int((got != -1).sum()) == 2048 + residue
    if residue == 3:
        assert int((got != -1).sum()) == 2051
        assert got[2050] != -1 and got[2051] == -1


# ---------------------------------------------------------------------------
# Rows that must be all -1, and positions that do not start at 0.
# ---------------------------------------------------------------------------


def test_invalid_tokens_are_forced_to_all_minus_one():
    """``attention_interface.py:788-789`` documents "every token must have at
    least one valid entry"; a padded row deliberately violates that and the
    sparse kernel is expected to mask it (``kernels/mla/sparse/kernel.py:206-209``
    and ``:337-339``). Whether the consumer really tolerates ``kv_len == 0`` is
    a device-rung question -- what is pinned here is that
    this op emits the all-``-1`` row rather than something addressable."""
    rows, _ = _synthetic_oracle(SMALL, 12, seed=3)
    valid = np.ones(12, dtype=bool)
    valid[[0, 5, 11]] = False
    got = _run(_pad_pool_ids(rows, SMALL.select_k),
               np.arange(12, dtype=np.int32), valid, SMALL,
               derive_out_width(SMALL.index_topk, SMALL.index_kpool))
    for p in range(12):
        if valid[p]:
            assert O.row_id_set(got[p])
        else:
            assert (got[p] == -1).all(), p


@pytest.mark.parametrize("start", [0, 1, 2, 3, 4, 1000, 100001])
def test_decode_positions_do_not_have_to_start_at_zero(start):
    """A decode step supplies one absolute position per row; the tail must be
    computed from that position, not from the row index."""
    kp, sk = MID.index_kpool, MID.select_k
    positions = np.array([start], dtype=np.int32)
    n = min(sk, (start + 1) // kp)
    pool_ids = np.full((1, sk), -1, dtype=np.int32)
    pool_ids[0, :n] = np.arange((start + 1) // kp - n, (start + 1) // kp)
    got = _run(pool_ids, positions, np.ones(1, dtype=bool), MID,
               derive_out_width(MID.index_topk, kp))[0]
    tail_start, tail_count = O.tail_span(start, kp)
    ids = O.row_id_set(got)
    assert len(ids) == kp * n + tail_count
    assert max(ids, default=-1) <= start
    for m in range(tail_count):
        assert tail_start + m in ids
    assert O.is_prefix_contiguous(got)


def test_zero_pools_selected_still_emits_the_tail():
    """The first ``index_kpool - 1`` positions of a sequence have no complete
    pool at all; the tail is the only thing they can attend to."""
    kp = PROD.index_kpool
    positions = np.arange(kp - 1, dtype=np.int32)
    pool_ids = np.full((kp - 1, PROD.select_k), -1, dtype=np.int32)
    got = _run(pool_ids, positions, np.ones(kp - 1, dtype=bool), PROD,
               PROD_WIDTH)
    for p in range(kp - 1):
        assert O.row_id_set(got[p]) == set(range(p + 1)), p


def test_interior_minus_one_never_produces_a_negative_id():
    """The suffix contract violated on purpose.

    ``n_valid`` is a *count*, so an interior ``-1`` makes the row wrong no
    matter what. What must not happen is a **negative** id reaching
    ``dsa_gather``, which would address a page from a negative index. Pinning
    this keeps the guard in ``expand.py`` from being "simplified" away.
    """
    pool_ids = np.array([[3, -1, 5, -1]], dtype=np.int32)
    got = _run(pool_ids, np.array([100], dtype=np.int32), np.ones(1,
                                                                  dtype=bool),
               SMALL, 32)[0]
    assert got.min() >= -1
