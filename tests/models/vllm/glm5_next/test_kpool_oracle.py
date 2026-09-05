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
"""``kpool_oracle`` vs the real ``Glm5NextTextIndexer``.

Nothing here compares the oracle to prose: every expectation is either
derived from the tensors the reference actually used, or asserted against the
reference's own output. Pure CPU, no TPU, no checkpoint download.

Comparisons are on **sets** of selected ids, never on ordered vectors: the
reference selects pools in descending-score order and the device path gets them
back from ``sparsecore_topk`` in unspecified order (``sparsecore_topk.py:25-28``),
and top-k tie-breaking differs between torch, XLA and SparseCore.

The reference runs in one subprocess for the whole module -- see the comment
above ``run_reference_cases`` in ``kpool_oracle.py`` for why an in-process
import is not merely slower but silently wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import kpool_oracle as O

# Small enough to run in float64 in milliseconds, large enough that the pool
# budget actually bites: at index_topk=8 the budget is 2 pools.
SMALL = O.KpoolParams(index_kpool=4,
                      index_topk=8,
                      index_head_dim=16,
                      index_n_heads=4)
SMALL_HIDDEN, SMALL_QLORA = 32, 24

# select_k = 16, identity bound = 66. Long enough that ``L % 4`` residues and
# genuine dropping both show up.
MID = O.KpoolParams(index_kpool=4,
                    index_topk=64,
                    index_head_dim=32,
                    index_n_heads=8)
MID_HIDDEN, MID_QLORA = 64, 48

SMALL_LENS = (1, 2, 3, 4, 12, 13, 14, 15, 16, 67)
MID_LENS = (63, 64, 65, 66, 67, 68, 70, 190, 191, 192, 193)
PAD_LENS = (13, 14, 15, 16)
PADS = (1, 2, 4, 5)


def _small(seq_len, n_pad=0):
    return O.make_case(SMALL,
                       hidden_size=SMALL_HIDDEN,
                       q_lora_rank=SMALL_QLORA,
                       seq_len=seq_len,
                       weight_seed=0,
                       input_seed=1000 + seq_len,
                       n_pad=n_pad)


def _mid(seq_len):
    return O.make_case(MID,
                       hidden_size=MID_HIDDEN,
                       q_lora_rank=MID_QLORA,
                       seq_len=seq_len,
                       weight_seed=3,
                       input_seed=2000 + seq_len)


ALL_CASES = ([_small(n) for n in SMALL_LENS] + [_mid(n) for n in MID_LENS] +
             [_small(n, p) for n in PAD_LENS for p in PADS])


@pytest.fixture(scope="module")
def ref_all():
    if not O.glm5_next_source_available():
        pytest.skip("transformers has no glm5_next; set "
                    f"${O.TRANSFORMERS_SRC_ENV} to a source overlay")
    return O.run_reference_cases(ALL_CASES)


def _oracle(res, params, out_width=None):
    return O.kpool_oracle_rows(res["k"],
                               res["gate"],
                               res["ape"],
                               res["q"],
                               res["weights_raw"],
                               params=params,
                               out_width=out_width)


# ---------------------------------------------------------------------------
# The config contract, recorded from the real zai-org/GLM-5.3-Flash config.
# ---------------------------------------------------------------------------


def test_real_config_constants():
    p = O.KpoolParams()
    assert (p.index_kpool, p.index_topk, p.index_head_dim,
            p.index_n_heads) == (4, 2048, 128, 32)
    assert p.always_select_tail is True
    assert p.select_k == 512
    assert p.reference_out_width == 2051
    assert p.identity_bound == 2050 == O.REAL_IDENTITY_BOUND


def test_index_topk_must_be_a_whole_number_of_pools():
    with pytest.raises(AssertionError):
        O.KpoolParams(index_kpool=4, index_topk=2050)


# ---------------------------------------------------------------------------
# A worked case, reproduced rather than copied.
# ---------------------------------------------------------------------------


def test_worked_small_case_structure(ref_all):
    """T=14, kpool=4, index_topk=8: shape (14, 11), identity through p=10,
    first real drop at p=11, exactly six non-prefix-contiguous reference rows.

    The plan records these structural facts and explicitly warns that the
    study's *derived* selected-pool vector is not reproducible (it depends on
    RNG consumption during module construction). Only the structure transfers,
    so only the structure is asserted.
    """
    res = ref_all[O.case_key(_small(14))]
    ref, orc = res["ref"], _oracle(res, SMALL)
    assert ref.shape == (14, 11) == orc.shape

    for p in range(SMALL.identity_bound + 1):  # p <= 10
        assert O.row_id_set(ref[p]) == set(range(p + 1)), p
        assert O.row_id_set(orc[p]) == set(range(p + 1)), p

    # p = 11 is the first position where the budget (2 pools = 8 tokens, plus a
    # 0-token tail) cannot cover all 12 causal tokens.
    assert len(O.row_id_set(ref[11])) == 8
    assert O.row_id_set(ref[11]) != set(range(12))

    non_prefix = [p for p in range(14) if not O.is_prefix_contiguous(ref[p])]
    assert non_prefix == [0, 1, 2, 4, 5, 6]


def test_reference_is_not_prefix_contiguous_but_the_oracle_is(ref_all):
    """The positive assertion that pins the bug the design avoids.

    ``kernels/mla/sparse/kernel.py:209`` / ``:339`` mask by column position
    against ``kv_lens = sum(topk_indices != -1)`` (``:622``), so an interior
    ``-1`` makes the kernel gather a decoy row at full weight. The reference
    writes its tail at the fixed column ``index_kpool * select_k``, leaving
    holes whenever fewer than ``select_k`` pools are valid; the TPU-shaped
    oracle compacts. If this test ever starts passing "both prefix-contiguous",
    the reference changed and ``expand_pools_with_tail`` needs re-deriving,
    not deleting.
    """
    res = ref_all[O.case_key(_small(14))]
    assert any(not O.is_prefix_contiguous(row) for row in res["ref"])
    assert all(O.is_prefix_contiguous(row) for row in _oracle(res, SMALL))


# ---------------------------------------------------------------------------
# Oracle == reference, on sets, at every residue of L mod index_kpool.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", SMALL_LENS)
def test_oracle_matches_reference_small(ref_all, seq_len):
    res = ref_all[O.case_key(_small(seq_len))]
    ref, orc = res["ref"], _oracle(res, SMALL)
    assert ref.shape == orc.shape
    for p in range(seq_len):
        assert O.row_id_set(ref[p]) == O.row_id_set(orc[p]), (seq_len, p)


@pytest.mark.parametrize("seq_len", MID_LENS)
def test_oracle_matches_reference_mid(ref_all, seq_len):
    res = ref_all[O.case_key(_mid(seq_len))]
    ref, orc = res["ref"], _oracle(res, MID)
    for p in range(seq_len):
        assert O.row_id_set(ref[p]) == O.row_id_set(orc[p]), (seq_len, p)


def test_mid_fixture_actually_drops_tokens(ref_all):
    """Guard against the trap that made a wrong port read PASS: a fixture too
    short to select sparsely proves nothing."""
    ref = ref_all[O.case_key(_mid(193))]["ref"]
    dropped = [
        p for p in range(193) if O.row_id_set(ref[p]) != set(range(p + 1))
    ]
    assert dropped, "fixture never drops a causal token"
    assert min(dropped) == MID.identity_bound + 1 == 67


# ---------------------------------------------------------------------------
# The causal convention: floor((p + 1) / index_kpool). Confirmed, not assumed.
# ---------------------------------------------------------------------------


def _pool_id_count(row, p, index_kpool):
    """Non-tail ids in a reference row.

    Pool ids and tail ids never collide: ``index_kpool*j + index_kpool - 1 <= p``
    implies ``index_kpool*(j+1) <= index_kpool * floor((p+1)/index_kpool) ==
    tail_start``.
    """
    _, tail_count = O.tail_span(p, index_kpool)
    return len(O.row_id_set(row)) - tail_count


@pytest.mark.parametrize("seq_len", [12, 13, 14, 15, 16, 67])
def test_candidate_count_is_floor_p_plus_1(ref_all, seq_len):
    ref = ref_all[O.case_key(_small(seq_len))]["ref"]
    kp, sk = SMALL.index_kpool, SMALL.select_k
    complete = O.num_complete_pools(seq_len, kp)

    discriminating = 0
    for p in range(seq_len):
        want = kp * min(sk, min(complete, (p + 1) // kp))
        # The convention this design was at risk of using instead.
        wrong = kp * min(sk, min(complete, p // kp + 1))
        assert _pool_id_count(ref[p], p, kp) == want, p
        discriminating += want != wrong
    assert discriminating > 0, (
        "this seq_len cannot tell floor((p+1)/kpool) from floor(p/kpool)+1")


@pytest.mark.parametrize("seq_len", SMALL_LENS)
def test_reference_never_selects_a_future_token(ref_all, seq_len):
    ref = ref_all[O.case_key(_small(seq_len))]["ref"]
    for p in range(seq_len):
        ids = O.row_id_set(ref[p])
        assert not ids or max(ids) <= p, (p, sorted(ids))


def test_num_candidate_pools_helper_agrees_with_the_mask():
    for seq_len in (0, 1, 3, 4, 7, 8, 13, 16, 193):
        mask = O.candidate_mask(seq_len, seq_len, index_kpool=4)
        for p in range(seq_len):
            n = int(mask[p].sum())
            assert n == O.num_candidate_pools(p, seq_len, 4)
            # The candidate set is a prefix, which is what lets
            # ``sparsecore_topk``'s scalar ``row_lengths`` encode it.
            assert mask[p, :n].all() and not mask[p, n:].any()


# ---------------------------------------------------------------------------
# Identity regime.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [65, 66, 67, 70])
def test_identity_regime_is_exact(ref_all, seq_len):
    """``sorted(non-(-1)) == range(p + 1)`` for ``p <= index_topk + kpool - 2``.

    This is the theorem the 2048 cap rests on. It is asserted on the reference
    *and* the oracle so neither can drift into agreeing on something wrong.
    """
    res = ref_all[O.case_key(_mid(seq_len))]
    ref, orc = res["ref"], _oracle(res, MID)
    for p in range(min(seq_len, MID.identity_bound + 1)):
        assert O.row_id_set(ref[p]) == set(range(p + 1)), p
        assert O.row_id_set(orc[p]) == set(range(p + 1)), p


def test_identity_bound_is_tight(ref_all):
    """One past the bound, the identity must actually break -- otherwise the
    bound is untested and the cap could be anywhere."""
    ref = ref_all[O.case_key(_mid(MID.identity_bound + 2))]["ref"]
    p = MID.identity_bound + 1
    assert O.row_id_set(ref[p]) != set(range(p + 1))
    assert len(O.row_id_set(ref[p])) == MID.index_topk  # full budget, no tail


# ---------------------------------------------------------------------------
# Left padding: pools anchor at ``first_key``, not at raw slot 0.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", PAD_LENS)
@pytest.mark.parametrize("n_pad", PADS)
def test_left_padding_shifts_every_id(ref_all, seq_len, n_pad):
    """``get_pooled_states`` anchors pooling at ``first_key``
    (``modeling_glm5_next.py:940-944``), so ``n_pad`` left pads shift every
    selected id by exactly ``n_pad`` and leave the selection otherwise
    identical. An implementation that anchored at raw slot 0 would regroup the
    tokens and fail this for every ``n_pad % index_kpool != 0``.

    The TPU path never sees left padding -- vLLM's layout is unpadded and ragged
    -- so what this pins is the *mapping*: absolute position ``p`` with
    ``first_key == 0`` is the right ground truth for the device.
    """
    padded = ref_all[O.case_key(_small(seq_len, n_pad))]
    # The unpadded case shares weights and `input_seed`, so its projections are
    # the same tensors the padded run saw for its real tokens.
    plain = ref_all[O.case_key(_small(seq_len))]
    np.testing.assert_array_equal(padded["k"], plain["k"])

    orc = _oracle(plain, SMALL)
    ref = padded["ref"]
    assert ref.shape[0] == n_pad + seq_len
    for i in range(n_pad):
        assert O.row_id_set(ref[i]) == set(), f"pad query row {i} not empty"
    for i in range(seq_len):
        assert O.row_id_set(ref[n_pad + i]) == {
            x + n_pad
            for x in O.row_id_set(orc[i])
        }, (seq_len, n_pad, i)


# ---------------------------------------------------------------------------
# Pooling math, straight against ``get_pooled_states``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len", [12, 13, 14, 15, 16])
def test_pooled_keys_match_get_pooled_states(ref_all, seq_len):
    res = ref_all[O.case_key(_small(seq_len))]
    complete = O.num_complete_pools(seq_len, SMALL.index_kpool)
    assert res["pooled_ref"].shape[0] == complete
    assert bool(res["pool_valid_ref"].all())
    # Pool j is exactly raw tokens [kpool*j, kpool*j + kpool).
    want_idx = (SMALL.index_kpool * np.arange(complete)[:, None] +
                np.arange(SMALL.index_kpool)[None, :])
    np.testing.assert_array_equal(res["pool_indices_ref"], want_idx)

    # The reference hard-codes a **float32** softmax via ``.float()`` at
    # ``modeling_glm5_next.py:962`` even when the module is float64, so a
    # float64 oracle cannot agree to better than float32 epsilon no matter how
    # correct it is. 1e-6 absolute is the bar, and the keys are
    # O(1); the measured worst case on this fixture is 9.8e-8.
    ours = O.pooled_keys(res["k"],
                         res["gate"],
                         res["ape"],
                         index_kpool=SMALL.index_kpool)
    np.testing.assert_allclose(ours, res["pooled_ref"], rtol=0, atol=1e-6)


def test_reference_softmax_is_float32_not_module_dtype(ref_all):
    """Pin the numerics fact so nobody "fixes" the float64 oracle into a
    difference, and so it is on record which side is more accurate.

    ``logits = grouped_gate_scores.float() + index_kpool_compress_ape.float()``
    (``modeling_glm5_next.py:962``) forces float32 even when the module is
    float64. Two independent checks, because a purely numerical one cannot
    distinguish "reference rounds to fp32" from "oracle has a small bug":

    1. the source line still says ``.float()``;
    2. switching the oracle's softmax to float32 moves it by the same order of
       magnitude as its disagreement with the reference -- i.e. the residual is
       float32 rounding and nothing larger is hiding under it.
    """
    with open(O.glm5_next_source_path()) as fh:
        src = fh.read()
    assert ("logits = grouped_gate_scores.float() + "
            "self.index_kpool_compress_ape.float()[None, None]") in src, (
                "the reference's pooling-softmax precision changed; re-derive "
                "the tolerance in test_pooled_keys_match_get_pooled_states "
                "instead of widening it")

    res = ref_all[O.case_key(_small(16))]
    f64 = O.pooled_keys(res["k"],
                        res["gate"],
                        res["ape"],
                        index_kpool=SMALL.index_kpool)
    f32 = O.pooled_keys(res["k"],
                        res["gate"],
                        res["ape"],
                        index_kpool=SMALL.index_kpool,
                        softmax_dtype=np.float32)
    gap_ref = float(np.abs(f64 - res["pooled_ref"]).max())
    gap_precision = float(np.abs(f64 - f32).max())
    assert 0.0 < gap_ref < 1e-6, gap_ref
    assert gap_ref < 10 * gap_precision, (gap_ref, gap_precision)

    # bf16 probabilities -- what the shipped checkpoint actually does at
    # ``:964-966`` -- are orders of magnitude coarser than either.
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    down = O.pooled_keys(res["k"],
                         res["gate"],
                         res["ape"],
                         index_kpool=SMALL.index_kpool,
                         prob_dtype=bf16)
    assert float(np.abs(f64 - down).max()) > 100 * gap_ref


def test_pooling_softmax_is_per_channel_over_slots(ref_all):
    """A softmax over the wrong axis still produces a finite pooled key.

    Perturb one channel of ``ape`` and assert only that channel of the pooled
    key moves. A channel-axis softmax (or a shared scalar weight) would couple
    all 16 channels.
    """
    res = ref_all[O.case_key(_small(16))]
    base = O.pooled_keys(res["k"],
                         res["gate"],
                         res["ape"],
                         index_kpool=SMALL.index_kpool)
    ape2 = res["ape"].copy()
    ape2[0, 3] += 5.0
    moved = O.pooled_keys(res["k"],
                          res["gate"],
                          ape2,
                          index_kpool=SMALL.index_kpool)
    delta = np.abs(moved - base)
    assert delta[:, 3].max() > 1e-6
    assert np.delete(delta, 3, axis=1).max() == 0.0


def test_pool_key_weights_sum_to_one_per_channel():
    """``sum_m softmax_m == 1``: pooling is a weighted *average*, so a pool of
    identical tokens must reproduce that token exactly."""
    rng = np.random.default_rng(0)
    kp, d = SMALL.index_kpool, SMALL.index_head_dim
    row = rng.normal(size=(1, d))
    k = np.repeat(row, 2 * kp, axis=0)
    gate = rng.normal(size=(2 * kp, d))
    ape = rng.normal(size=(kp, d))
    pooled = O.pooled_keys(k, gate, ape, index_kpool=kp)
    np.testing.assert_allclose(pooled, np.repeat(row, 2, axis=0), atol=1e-13)


# ---------------------------------------------------------------------------
# Tail.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_kpool", [1, 2, 4, 8])
def test_tail_span(index_kpool):
    for p in range(40):
        start, count = O.tail_span(p, index_kpool)
        assert count == (p + 1) % index_kpool
        assert start + count == p + 1
        assert start % index_kpool == 0


def test_the_most_recent_token_is_only_reachable_through_the_tail(ref_all):
    """The residue-dependent case. At ``(p + 1) % 4 == 0`` the tail is empty and
    a dropped tail is invisible, which is why every set comparison above is
    parameterized over all four residues."""
    ref = ref_all[O.case_key(_small(16))]["ref"]
    for p in range(16):
        ids = O.row_id_set(ref[p])
        if (p + 1) % SMALL.index_kpool != 0:
            assert p in ids, p
