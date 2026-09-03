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
"""Correctness gate for Option D.

Option D must not change a single value -- only where it is computed. So the
assertions here are all zero-tolerance:

1. `test_hoisted_metadata_equals_per_layer` -- the assertion
   `measurement_and_design.md` §3.3 specifies under "Fallback gate". Every
   quantity `precompute()` hoists is compared, element for element, against
   what the incumbent builds inside the `cond`: `_effective_row_lengths` (the
   incumbent's own function, imported, not reimplemented), the SparseCore
   `lengths` pad, the three `start_end_seq_idx` prefetch operands, `sem_ids`,
   `bo_sz` and the three grid extents. Run over a spread of batch shapes
   including padded buckets and mixed prefill/decode.
2. `test_option_d_matches_incumbent` -- end to end, D against the incumbent at
   `enable_early_exit` both True and False, on real TPU. Compared as sorted
   index sets per row, which is the contract `sparsecore_topk` documents
   ("unsorted, -1 suffix-padded") and the contract
   `tests/kernels/test_dsv4_streamindex_topk.py` asserts against.
3. `test_option_d_matches_incumbent_exact` -- the stronger claim, that the raw
   rows are identical and not merely set-equal. D does not reorder anything, so
   this should hold; it is separated so a tie-break difference would be visible
   as a distinct failure rather than hiding inside (2).

`tests/kernels/test_dsv4_streamindex_topk.py` is run **unmodified** alongside
this file; it covers the incumbent and is the guard that the mapped-in kernels
are the ones under test.
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

import streamidx_hoist as D  # noqa: E402

PAGE_SIZE = 128
PAGES_PER_SEQ = 32
BKV_P = 1
BQ_SZ = 4
H_I = 4
HEAD_DIM = 128
K = 64
DRBS = 4


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


# `q_lens`, `seq_lens`, label.
CASES = [
    ([1] * 8, [700] * 8, "decode_b8_homog"),
    ([1] * 8, [64, 700, 129, 4000, 1, 2500, 33, 900], "decode_b8_ragged"),
    ([1] * 4, [3000, 3000, 3000, 3000], "decode_b4_exact_group"),
    ([1] * 6, [3000] * 6, "decode_b6_ragged_group"),
    ([1] * 8, [30] * 8, "decode_b8_all_short_fastpath_would_fire"),
    ([1, 1, 1, 1, 64, 128], [900, 900, 900, 900, 900, 900], "mixed_b6"),
    ([16, 32], [1000, 2000], "prefill_b2"),
]


def _num_positions(page_indices_len, max_num_seqs):
    return D.num_positions_for(
        page_indices_len=page_indices_len,
        max_num_seqs=max_num_seqs,
        page_size=PAGE_SIZE,
        num_kv_pages_per_blocks=(BKV_P, BKV_P, BKV_P),
        k=K,
    )


@pytest.mark.parametrize("q_lens,seq_lens,label", CASES,
                         ids=[c[2] for c in CASES])
def test_hoisted_metadata_equals_per_layer(q_lens, seq_lens, label):
    """Zero tolerance: what D hoists must equal what the incumbent rebuilds."""
    inp = _inputs(q_lens, seq_lens, seed=7)
    num_tokens = int(sum(q_lens))
    max_num_seqs = len(q_lens)
    num_positions = _num_positions(int(inp["page_indices"].shape[0]),
                                   max_num_seqs)

    meta = jax.jit(
        lambda s, c, d: D.precompute(
            s,
            c,
            d,
            num_tokens=num_tokens,
            num_positions=num_positions,
            compression_ratio=1,
            decode_req_batch_size=DRBS,
        ))(inp["seq_lens"], inp["cu_q_lens"], inp["distribution"])

    # --- what the incumbent computes, from the incumbent's own source ---
    eff_ref = jax.jit(
        lambda s, c, d: _effective_row_lengths(
            seq_lens=s,
            cu_q_lens=c,
            distribution=d,
            num_tokens=num_tokens,
            num_positions=num_positions,
            compression_ratio=1,
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

    expected_grid = [
        decode_batch_end // DRBS,
        dist[1] - decode_batch_end,
        dist[2] - dist[1],
    ]
    for i, exp in enumerate(expected_grid):
        assert int(meta.grid[i]) == int(exp), (i, meta.grid[i], exp)


def _run_incumbent(inp, enable_early_exit):
    return np.asarray(
        streamindex_topk(
            **inp,
            k=K,
            compression_ratio=1,
            num_kv_pages_per_block=BKV_P,
            num_queries_per_block=BQ_SZ,
            enable_early_exit=enable_early_exit,
        ))


def _run_d(inp, hoist):
    meta = None
    if hoist:
        num_tokens = int(inp["q"].shape[0])
        meta = D.precompute(
            inp["seq_lens"],
            inp["cu_q_lens"],
            inp["distribution"],
            num_tokens=num_tokens,
            num_positions=_num_positions(int(inp["page_indices"].shape[0]),
                                         int(inp["seq_lens"].shape[0])),
            compression_ratio=1,
            decode_req_batch_size=DRBS,
        )
    return np.asarray(
        D.streamindex_topk_d(
            inp["q"],
            inp["indexer_weights"],
            inp["cache_kv"],
            inp["seq_lens"],
            inp["page_indices"],
            inp["cu_q_lens"],
            inp["distribution"],
            meta,
            k=K,
            compression_ratio=1,
            num_kv_pages_per_block=BKV_P,
            num_queries_per_block=BQ_SZ,
            decode_req_batch_size=DRBS,
        ))


def _assert_same_sets(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    for token, (ra, rb) in enumerate(zip(a, b)):
        np.testing.assert_array_equal(np.sort(ra[ra >= 0]),
                                      np.sort(rb[rb >= 0]),
                                      err_msg=f"token {token}")


@pytest.mark.parametrize("hoist", [False, True], ids=["d1_cond_only", "d2_hoist"])
@pytest.mark.parametrize("q_lens,seq_lens,label", CASES,
                         ids=[c[2] for c in CASES])
def test_option_d_matches_incumbent(q_lens, seq_lens, label, hoist):
    inp = _inputs(q_lens, seq_lens, seed=13)
    base = _run_incumbent(inp, enable_early_exit=True)
    got = _run_d(inp, hoist)
    _assert_same_sets(got, base)


@pytest.mark.parametrize("q_lens,seq_lens,label", CASES,
                         ids=[c[2] for c in CASES])
def test_option_d_matches_incumbent_exact(q_lens, seq_lens, label):
    """D reorders nothing, so the raw rows should be identical to the
    incumbent's `enable_early_exit=False` path, element for element."""
    inp = _inputs(q_lens, seq_lens, seed=13)
    base = _run_incumbent(inp, enable_early_exit=False)
    got = _run_d(inp, hoist=True)
    np.testing.assert_array_equal(got, base)
