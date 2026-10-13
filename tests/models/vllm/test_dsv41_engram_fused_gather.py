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
"""CPU equivalence tests for the fused DeepSeek-V4.1 Engram host gather.

``engram_hash_and_gather_omp`` fuses ``TpuNgramHashState.cpu_hashes`` (NumPy)
with ``engram_gather_omp`` (C++). Both paths must produce byte-identical packed
``[value | scale]`` rows, including the int64 wrap-around multiply, the floored
modulo, blocked look-back at the start of a prompt and out-of-range row clamps.
"""

import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41 import engram as engram_mod

VAL_DIM = 64
SCL_DIM = 4
ROWS_PER_COL = 2048


def _make_hash_state(rng, max_ngram, n_heads, n_tables):
    vocab, compressed = 4096, 1024
    token_map = rng.integers(0, compressed, size=vocab).astype(np.int64)
    n_hash_cols = (max_ngram - 1) * n_heads
    # Multipliers near the int64 range so ``tok * mult`` wraps like NumPy does.
    mult = rng.integers(1 << 40, 1 << 62, size=(n_tables, max_ngram), dtype=np.int64)
    primes = rng.integers(
        500, ROWS_PER_COL, size=(n_tables, max_ngram - 1, n_heads)
    ).astype(np.int64)
    offsets = (np.arange(n_hash_cols, dtype=np.int64) * ROWS_PER_COL)[None, :]
    offsets = np.repeat(offsets, n_tables, axis=0)
    layout = SimpleNamespace(max_ngram_size=max_ngram, n_heads=n_heads)
    hs = SimpleNamespace(
        layout=layout,
        pad_id=int(token_map[7]),
        offsets=torch.from_numpy(offsets.astype(np.int32)),
        _token_map_np=np.ascontiguousarray(token_map),
        _mult_np=np.ascontiguousarray(mult),
        _primes_np=np.ascontiguousarray(primes),
        _offsets_np=np.ascontiguousarray(offsets),
    )
    hs.cpu_hashes = types.MethodType(engram_mod.TpuNgramHashState.cpu_hashes, hs)
    return hs, n_hash_cols


def _install_tables(monkeypatch, rng, n_tables, n_hash_cols):
    n_rows = n_hash_cols * ROWS_PER_COL
    table_ids = list(range(100, 100 + n_tables))
    tables = {}
    for tid in table_ids:
        w = torch.from_numpy(
            rng.integers(0, 256, size=(n_rows, VAL_DIM)).astype(np.uint8)
        )
        s = torch.from_numpy(
            rng.integers(0, 256, size=(n_rows, SCL_DIM)).astype(np.uint8)
        )
        tables[tid] = (w, s)
    monkeypatch.setattr(engram_mod, "_CPU_ENGRAM_WEIGHTS", tables)
    monkeypatch.setattr(engram_mod, "_ENGRAM_PTRS_CACHE", {})
    return table_ids, tables, n_rows


def _make_history(rng, num_tokens=512, num_reqs=3, max_len=300):
    token_ids = rng.integers(0, 4096, size=(num_reqs, max_len)).astype(np.int32)
    token_ids[:, 250:] = -1  # never-written slots: both paths clamp to id 0
    req_idx = rng.integers(0, num_reqs, size=num_tokens).astype(np.int32)
    pos = rng.integers(0, max_len, size=num_tokens).astype(np.int32)
    pos[:8] = np.arange(8)  # look-back < 0 -> pad_id for the blocked shifts
    return token_ids, req_idx, pos


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "max_ngram,n_heads,n_tables", [(3, 8, 2), (4, 4, 3), (2, 16, 1), (5, 16, 2)]
)
def test_fused_hash_and_gather_matches_reference(
    monkeypatch, max_ngram, n_heads, n_tables
):
    lib = engram_mod._get_engram_gather_lib()
    if lib is None:
        pytest.skip("Engram OpenMP gather library unavailable (no g++)")
    rng = np.random.default_rng(1234)
    hs, n_hash_cols = _make_hash_state(rng, max_ngram, n_heads, n_tables)
    table_ids, tables, n_rows = _install_tables(monkeypatch, rng, n_tables, n_hash_cols)
    token_ids, req_idx, pos = _make_history(rng)
    num_tokens = pos.shape[0]
    packed_dim = VAL_DIM + SCL_DIM

    ref_hashes = hs.cpu_hashes(token_ids, req_idx, pos)
    assert ref_hashes.shape == (num_tokens, n_tables, n_hash_cols)
    ref = torch.zeros(
        num_tokens * n_tables * n_hash_cols * packed_dim, dtype=torch.uint8
    )
    engram_mod._gather_into(
        lib,
        np.ascontiguousarray(ref_hashes, dtype=np.int32),
        table_ids,
        packed_dim,
        ref,
    )

    out = torch.full_like(ref, 0xAB)
    engram_mod._hash_and_gather_into(
        lib, hs, token_ids, req_idx, pos, table_ids, packed_dim, out
    )
    assert torch.equal(out, ref)

    # Spot-check the reference kernel against a pure Python row lookup so the
    # equivalence above is anchored to the table contents, not just to itself.
    rows = ref.view(num_tokens, n_tables, n_hash_cols, packed_dim)
    for t in (0, 5, num_tokens - 1):
        for tbl in range(n_tables):
            w, s = tables[table_ids[tbl]]
            for c in (0, n_hash_cols - 1):
                h = int(np.clip(ref_hashes[t, tbl, c], 0, n_rows - 1))
                assert torch.equal(rows[t, tbl, c, :VAL_DIM], w[h])
                assert torch.equal(rows[t, tbl, c, VAL_DIM:], s[h])


@pytest.mark.cpu_test
def test_fused_hash_and_gather_accepts_non_int32_index_arrays(monkeypatch):
    lib = engram_mod._get_engram_gather_lib()
    if lib is None:
        pytest.skip("Engram OpenMP gather library unavailable (no g++)")
    rng = np.random.default_rng(7)
    hs, n_hash_cols = _make_hash_state(rng, 3, 8, 2)
    table_ids, _, _ = _install_tables(monkeypatch, rng, 2, n_hash_cols)
    token_ids, req_idx, pos = _make_history(rng, num_tokens=64)
    packed_dim = VAL_DIM + SCL_DIM
    nbytes = 64 * 2 * n_hash_cols * packed_dim

    expected = torch.zeros(nbytes, dtype=torch.uint8)
    engram_mod._hash_and_gather_into(
        lib, hs, token_ids, req_idx, pos, table_ids, packed_dim, expected
    )
    # int64 / non-contiguous index arrays are converted, not reinterpreted.
    got = torch.zeros(nbytes, dtype=torch.uint8)
    engram_mod._hash_and_gather_into(
        lib,
        hs,
        token_ids,
        req_idx.astype(np.int64),
        np.ascontiguousarray(pos.astype(np.int64)[::-1])[::-1],
        table_ids,
        packed_dim,
        got,
    )
    assert torch.equal(got, expected)


@pytest.mark.cpu_test
@pytest.mark.parametrize("max_ngram,n_heads", [(18, 2), (3, 40)])
def test_prefetch_prefill_falls_back_for_wide_layouts(monkeypatch, max_ngram, n_heads):
    """tok[16]/h_arr[64] scratch in the C++ kernel bounds the fused path."""
    lib = engram_mod._get_engram_gather_lib()
    if lib is None:
        pytest.skip("Engram OpenMP gather library unavailable (no g++)")
    rng = np.random.default_rng(99)
    hs, n_hash_cols = _make_hash_state(rng, max_ngram, n_heads, 1)
    assert (
        max_ngram > engram_mod._FUSED_MAX_NGRAM
        or n_hash_cols > engram_mod._FUSED_MAX_HASH_COLS
    )
    table_ids, _, _ = _install_tables(monkeypatch, rng, 1, n_hash_cols)
    token_ids, req_idx, pos = _make_history(rng, num_tokens=32)
    monkeypatch.setattr(engram_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(engram_mod, "_PIN_HOST_BUFS", False)

    def _boom(*_a, **_k):
        raise AssertionError("fused kernel must not run for wide layouts")

    monkeypatch.setattr(engram_mod, "_hash_and_gather_into", _boom)
    calls = []
    orig_gather = engram_mod._gather_into
    monkeypatch.setattr(
        engram_mod,
        "_gather_into",
        lambda *a, **k: (calls.append(1), orig_gather(*a, **k))[1],
    )
    engram_mod.engram_prefetch_prefill(
        hs,
        table_ids,
        VAL_DIM + SCL_DIM,
        token_ids,
        req_idx,
        pos,
        np.ones(pos.shape[0], dtype=np.bool_),
        padded_num_tokens=32,
    )
    assert calls == [1]
    assert engram_mod._ENGRAM_PREFETCH is not None
    assert engram_mod._ENGRAM_PREFETCH["all_known"] is True
