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
"""Correctness gate for D1-lite (small-shape grid).

D1-lite must not change a single output value -- only *where* the metadata is
computed.  Every assertion here is zero-tolerance `np.testing.assert_array_equal`
against a separately-constructed device call to the incumbent.

The grid deliberately contains both regimes:

  * shapes where the early-exit predicate `max(seq_lens) // compression_ratio
    <= k` is TRUE, so `_fast_path` fires and no Pallas kernel runs at all;
  * shapes where it is FALSE, so `_common_path` runs.

`test_fast_path_actually_fires` is the guard that the first group is not
silently the second: it checks the incumbent's early-exit output against the
closed form `_fast_path` computes, which the scoring path does *not* reproduce
row-for-row (SparseCore returns an unsorted top-k).  Without it a bug that
disabled the fast path everywhere would leave every other test in this file
passing.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as _sc  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    _effective_row_lengths, align_to, streamindex_topk)

import streamindex_topk_d1lite as DL  # noqa: E402

PAGE_SIZE = 128
PAGES_PER_SEQ = 32
BKV_P = 1
BQ_SZ = 4
H_I = 4
HEAD_DIM = 128
K = 64
DRBS = 4
CR = 1

HOISTS = ("none", "plain", "cond")


def _to_byte_lane(x):
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def _quantize_fp8_ue8m0(x, block_size):
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4,
                    None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    return q, jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)


def _pack_cache(keys):
    num_pages, page_size, head_dim = keys.shape
    fp8, scale = _quantize_fp8_ue8m0(jnp.asarray(keys), head_dim)
    record = jnp.concatenate([_to_byte_lane(fp8),
                              _to_byte_lane(scale)], axis=-1)
    width = -(-record.shape[-1] // 128) * 128
    record = jnp.pad(record, ((0, 0), (0, 0), (0, width - record.shape[-1])))
    return np.asarray(record).reshape(num_pages, page_size // 4, 4, width)


def _inputs(q_lens, seq_lens, seed, declared=None):
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    num_tokens = int(sum(q_lens))
    num_pages = num_seqs * PAGES_PER_SEQ
    block_table = rng.permutation(num_pages).astype(np.int32).reshape(
        num_seqs, PAGES_PER_SEQ)
    keys = rng.standard_normal((num_pages, PAGE_SIZE, HEAD_DIM),
                               dtype=np.float32)
    num_decodes = 0
    while num_decodes < num_seqs and q_lens[num_decodes] == 1:
        num_decodes += 1
    n_declared = num_seqs if declared is None else declared
    return {
        "q":
        jnp.asarray(
            rng.standard_normal((num_tokens, H_I, HEAD_DIM),
                                dtype=np.float32)),
        "indexer_weights":
        jnp.asarray(
            rng.uniform(0.25, 1.75, (num_tokens, H_I)).astype(np.float32)),
        "cache_kv":
        jnp.asarray(_pack_cache(keys)),
        "seq_lens":
        jnp.asarray(np.asarray(seq_lens, np.int32)),
        "page_indices":
        jnp.asarray(block_table.reshape(-1)),
        "cu_q_lens":
        jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)),
        "distribution":
        jnp.asarray(np.array([num_decodes, num_decodes, n_declared],
                             np.int32)),
    }


# `q_lens`, `seq_lens`, label.  `fires` == the early-exit predicate at K=64.
CASES = [
    ([1] * 8, [700] * 8, "decode_b8_homog"),
    ([1] * 8, [64, 700, 129, 4000, 1, 2500, 33, 900], "decode_b8_ragged"),
    ([1] * 4, [3000, 3000, 3000, 3000], "decode_b4_exact_group"),
    ([1] * 6, [3000] * 6, "decode_b6_ragged_group"),
    ([1] * 8, [30] * 8, "decode_b8_fastpath_fires"),
    ([1] * 8, [1, 7, 30, 64, 12, 3, 50, 64], "decode_b8_fastpath_fires_ragged"),
    ([1, 1, 1, 1, 64, 128], [900, 900, 900, 900, 900, 900], "mixed_b6"),
    ([16, 32], [1000, 2000], "prefill_b2"),
]
IDS = [c[2] for c in CASES]


def _fires(seq_lens) -> bool:
    return max(seq_lens) // CR <= K


def _num_positions(page_indices_len, max_num_seqs):
    pages_per_seq = page_indices_len // max_num_seqs
    num_sublanes_total = max(
        align_to(pages_per_seq, b) * PAGE_SIZE // 128
        for b in (BKV_P, BKV_P, BKV_P))
    return max(num_sublanes_total * 128, K)


def _run_incumbent(inp, enable_early_exit):
    return np.asarray(
        streamindex_topk(
            **inp,
            k=K,
            compression_ratio=CR,
            num_kv_pages_per_block=BKV_P,
            num_queries_per_block=BQ_SZ,
            decode_req_batch_size=DRBS,
            enable_early_exit=enable_early_exit,
        ))


def _run_dl(inp, hoist, enable_early_exit=True):
    return np.asarray(
        DL.streamindex_topk_dl(
            **inp,
            k=K,
            compression_ratio=CR,
            num_kv_pages_per_block=BKV_P,
            num_queries_per_block=BQ_SZ,
            decode_req_batch_size=DRBS,
            enable_early_exit=enable_early_exit,
            hoist=hoist,
        ))


# ---------------------------------------------------------------------------
# 1. What is hoisted equals what the incumbent rebuilds, value for value.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_lens,seq_lens,label", CASES, ids=IDS)
def test_hoisted_metadata_equals_per_layer(q_lens, seq_lens, label):
    inp = _inputs(q_lens, seq_lens, seed=7)
    num_tokens = int(sum(q_lens))
    max_num_seqs = len(q_lens)
    num_positions = _num_positions(int(inp["page_indices"].shape[0]),
                                   max_num_seqs)

    meta = jax.jit(lambda s, c, d: DL.precompute(
        s, c, d,
        num_tokens=num_tokens,
        num_positions=num_positions,
        compression_ratio=CR,
        decode_req_batch_size=DRBS,
    ))(inp["seq_lens"], inp["cu_q_lens"], inp["distribution"])

    eff_ref = jax.jit(lambda s, c, d: _effective_row_lengths(
        seq_lens=s,
        cu_q_lens=c,
        distribution=d,
        num_tokens=num_tokens,
        num_positions=num_positions,
        compression_ratio=CR,
    ))(inp["seq_lens"], inp["cu_q_lens"], inp["distribution"])
    b_pad = _sc._align_to(num_tokens, _sc.LANES)
    eff_ref_padded = np.pad(np.asarray(eff_ref), (0, b_pad - num_tokens))

    np.testing.assert_array_equal(np.asarray(meta.eff_lengths_padded),
                                  eff_ref_padded)
    assert meta.eff_lengths_padded.dtype == jnp.int32

    dist = np.asarray(inp["distribution"])
    decode_batch_end = dist[0] // DRBS * DRBS
    expected_se = [
        np.array([0, decode_batch_end], np.int32),
        np.array([decode_batch_end, dist[1]], np.int32),
        np.array([dist[1], dist[2]], np.int32),
    ]
    for i, exp in enumerate(expected_se):
        np.testing.assert_array_equal(np.asarray(meta.start_end[i]), exp)

    np.testing.assert_array_equal(np.asarray(meta.sem_ids),
                                  np.zeros(3, np.int32))
    np.testing.assert_array_equal(np.asarray(meta.bo_sz),
                                  np.full(2, -1, np.int32))

    for i, exp in enumerate([decode_batch_end // DRBS,
                             dist[1] - decode_batch_end,
                             dist[2] - dist[1]]):
        assert int(meta.grid[i]) == int(exp), (i, meta.grid[i], exp)


# ---------------------------------------------------------------------------
# 2. End to end: every hoist mode is index-identical to the incumbent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hoist", HOISTS, ids=[f"hoist_{h}" for h in HOISTS])
@pytest.mark.parametrize("q_lens,seq_lens,label", CASES, ids=IDS)
def test_d1lite_matches_incumbent_exact(q_lens, seq_lens, label, hoist):
    """Element for element, against a separately-constructed device call."""
    inp = _inputs(q_lens, seq_lens, seed=13)
    base = _run_incumbent(inp, enable_early_exit=True)
    got = _run_dl(inp, hoist, enable_early_exit=True)
    assert got.shape == base.shape, (got.shape, base.shape)
    assert np.array_equal(got, base), (
        f"{label}/{hoist}: {int((got != base).sum())} of {base.size} "
        f"entries differ; fast_path_fires={_fires(seq_lens)}")


@pytest.mark.parametrize("q_lens,seq_lens,label", CASES, ids=IDS)
def test_d1lite_no_early_exit_matches_incumbent_no_early_exit(
        q_lens, seq_lens, label):
    """The D1 arm's own control: with the cond off on both sides the two
    modules must agree exactly, on every shape including the firing ones."""
    inp = _inputs(q_lens, seq_lens, seed=13)
    base = _run_incumbent(inp, enable_early_exit=False)
    got = _run_dl(inp, hoist="plain", enable_early_exit=False)
    assert np.array_equal(got, base)


# ---------------------------------------------------------------------------
# 3. The fast path really is being taken where we claim it is.
# ---------------------------------------------------------------------------


def _fast_path_closed_form(inp, num_tokens):
    """`streamindex_topk.py:840-856`, recomputed in numpy."""
    seq_lens = np.asarray(inp["seq_lens"])
    cu = np.asarray(inp["cu_q_lens"])
    token_idx = np.arange(num_tokens)
    seq_idx = np.minimum(np.searchsorted(cu[1:], token_idx, side="right"),
                         seq_lens.shape[0] - 1)
    seq_len = seq_lens[seq_idx]
    q_len = cu[seq_idx + 1] - cu[seq_idx]
    q_start = cu[seq_idx]
    q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
    max_valid = np.minimum(seq_len // CR - 1, q_abs_pos // CR)
    max_valid = np.where(token_idx < cu[-1], max_valid, -1)
    s_idx = np.arange(K, dtype=np.int32)[None, :]
    return np.where(s_idx <= max_valid[:, None], s_idx, -1).astype(np.int32)


FIRING = [c for c in CASES if _fires(c[1])]
NOT_FIRING = [c for c in CASES if not _fires(c[1])]
assert FIRING and NOT_FIRING, "the grid must contain both regimes"


@pytest.mark.parametrize("q_lens,seq_lens,label", FIRING,
                         ids=[c[2] for c in FIRING])
def test_fast_path_actually_fires(q_lens, seq_lens, label):
    """On a firing shape the incumbent and every D1-lite hoist mode must return
    `_fast_path`'s closed form -- an ascending run then -1 padding.  The
    scoring path returns the same *set* but SparseCore does not sort it, so
    this distinguishes "fast path taken" from "fast path silently skipped"."""
    inp = _inputs(q_lens, seq_lens, seed=13)
    num_tokens = int(sum(q_lens))
    expect = _fast_path_closed_form(inp, num_tokens)
    np.testing.assert_array_equal(_run_incumbent(inp, True), expect)
    for hoist in HOISTS:
        np.testing.assert_array_equal(_run_dl(inp, hoist, True), expect,
                                      err_msg=f"hoist={hoist}")
