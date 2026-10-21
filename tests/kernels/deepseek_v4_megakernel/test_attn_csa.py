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
"""CSA megakernel steps vs the plain-JAX reference (interpret mode).

Indexer scores: tolerance 1e-6 relative (f32 sums of exact fp8 products in a
different order), identical -inf pattern and row lengths. Sparse attention:
the same tolerance tier as the SWA/HCA step, against the reference fed the
same top-k lists.
"""

import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cases as C  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import (  # noqa: E402
    attn_csa,
    attn_ref,
)

MAX_REL = 1e-2
MEAN_ABS = 1e-3


def _close(out, ref):
    err = np.abs(out - ref)
    return err.max() / np.abs(ref).max() <= MAX_REL and err.mean() <= MEAN_ABS


def _scores(case, max_len):
    return attn_csa.indexer_scores(
        case.q_idx, case.weights, case.idx_words, case.csa_table, case.win.seq_lens,
        max_len=max_len, blk_pages=2, interpret=True)


def test_indexer_scores_match_reference():
    seq = [700, 3000, 9000]
    case = C.csa_case(np.random.default_rng(0), seq, heads=8)
    sc, rl = _scores(case, max(seq))
    ref, rl_ref = attn_ref.indexer_scores_reference(
        case.q_idx.astype(jnp.float32), case.weights, case.idx_words, case.csa_table,
        case.win.seq_lens)
    assert np.array_equal(np.asarray(rl), np.asarray(rl_ref))
    sc, ref = np.asarray(sc), np.asarray(ref)
    for r, s in enumerate(seq):
        a, b = sc[r, :, : s // 4], ref[r, :, : s // 4]
        fin = np.isfinite(b)
        assert np.array_equal(np.isfinite(a), fin)
        assert np.abs(a[fin] - b[fin]).max() <= 1e-6 * np.abs(b[fin]).max()


def test_union_placeholder():
    rng = np.random.default_rng(1)
    scores = jnp.asarray(rng.standard_normal((2, 7, 300), dtype=np.float32))
    row_lens = jnp.asarray([[290, 291, 292, 293, 294, 295, 296], [5, 6, 7, 8, 9, 10, 11]], jnp.int32)
    union, ulen, idx = attn_csa.topk_union_placeholder(scores, row_lens, 16)
    idx, union, ulen = np.asarray(idx), np.asarray(union), np.asarray(ulen)
    for r in range(2):
        sets = [set(i for i in idx[r, t] if i >= 0) for t in range(7)]
        u = union[r, : ulen[r]]
        assert list(u >> 8) == sorted(set().union(*sets))
        for e in u:
            assert (e & 0xFF) == sum(1 << t for t in range(7) if (e >> 8) in sets[t])
    assert list(idx[1, 0]) == [0, 1, 2, 3, 4] + [-1] * 11  # short row: all visible


@pytest.mark.parametrize("heads,seq,k", [(8, [300, 3000, 9000], 64), (128, [2100, 600], 128)])
def test_csa_attention_matches_reference(heads, seq, k):
    case = C.csa_case(np.random.default_rng(heads), seq, heads=heads)
    w = case.win
    sc, rl = _scores(case, max(seq))
    union, ulen, idx = attn_csa.topk_union_placeholder(sc, rl, k)
    out = np.asarray(attn_csa.csa_attention(
        w.q, w.new_kv, w.swa_words, w.swa_table, case.nope_words, case.rope_words,
        case.csa_table, union, ulen, w.seq_lens, w.sinks, sm_scale=w.sm_scale,
        interpret=True), np.float32)
    ref = np.asarray(attn_ref.csa_reference(
        w.q, w.new_kv, w.swa_words, w.swa_table, case.nope_words, case.rope_words,
        case.csa_table, idx, w.seq_lens, w.sinks, sm_scale=w.sm_scale))
    assert _close(out, ref), (np.abs(out - ref).max(), np.abs(ref).max())

    # Positive controls: dropping row 0's selection of one entry, or one wrong
    # RoPE segment in the reference's gather, must fail the same check.
    bad = np.asarray(union).copy()
    bad[1, np.flatnonzero(bad[1] & 1)[0]] &= ~1
    out_bad = np.asarray(attn_csa.csa_attention(
        w.q, w.new_kv, w.swa_words, w.swa_table, case.nope_words, case.rope_words,
        case.csa_table, jnp.asarray(bad), ulen, w.seq_lens, w.sinks, sm_scale=w.sm_scale,
        interpret=True), np.float32)
    err = np.abs(out_bad[1, 0] - ref[1, 0])
    assert err.max() > 10 * np.abs(out[1, 0] - ref[1, 0]).max()
    rope_bad = np.asarray(case.rope_words).copy()
    rope_bad = np.roll(rope_bad.reshape(*rope_bad.shape[:2], 4, 32), 1, axis=2).reshape(
        rope_bad.shape)
    ref_bad = np.asarray(attn_ref.csa_reference(
        w.q, w.new_kv, w.swa_words, w.swa_table, case.nope_words, jnp.asarray(rope_bad),
        case.csa_table, idx, w.seq_lens, w.sinks, sm_scale=w.sm_scale))
    assert not _close(out, ref_bad)


@pytest.mark.parametrize("heads", [8, 128])
def test_csa_hybrid_gather_matches_reference(heads):
    """All-dense (k=1), mixed and all-per-row (huge k) splits of the union."""
    case = C.csa_case(np.random.default_rng(5), [3000, 9000, 300], heads=heads)
    w = case.win
    sc, rl = _scores(case, 9000)
    union, ulen, idx = attn_csa.topk_union_placeholder(sc, rl, 300)
    ref = np.asarray(attn_ref.csa_reference(
        w.q, w.new_kv, w.swa_words, w.swa_table, case.nope_words, case.rope_words,
        case.csa_table, idx, w.seq_lens, w.sinks, sm_scale=w.sm_scale))
    for k_dense in (1, 40, 10**6):
        out = np.asarray(attn_csa.csa_attention(
            w.q, w.new_kv, w.swa_words, w.swa_table, case.nope_words, case.rope_words,
            case.csa_table, union, ulen, w.seq_lens, w.sinks, sm_scale=w.sm_scale,
            gather="hybrid", k_dense=k_dense, blk_pages=2, interpret=True), np.float32)
        assert _close(out, ref), (k_dense, np.abs(out - ref).max())
