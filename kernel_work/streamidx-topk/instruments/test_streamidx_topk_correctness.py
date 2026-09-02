#!/usr/bin/env python3
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
"""Step-1 correctness gate for `streamindex_topk`.

This exists to catch what `bench_streamidx_topk.py` structurally cannot: a
kernel that skips work benchmarks beautifully. Every candidate in step 3 is
judged by this file and that one, unchanged.

Oracle
------
`streamindex_topk_ref` from `tests/kernels/test_dsv4_streamindex_topk.py`
(lines 58-118) -- a pure-NumPy triple loop over (token, position, head). It is
loaded by path rather than copied, so it cannot drift from the in-tree
reference. `tests/` is not an importable package, hence `importlib`.

Because the oracle is O(tokens x positions x heads x head_dim) in Python, the
shapes here are small. They are chosen for *coverage of the code paths*, not
for resemblance to GLM-5.2 -- the benchmark owns the production shapes:

  * all three `pallas_call`s in one batch: 4 batched decodes (seq_batch_size=4)
    + a 5th decode (the remainder call) + a prefill sequence (the mixed call),
  * `compression_ratio == 1` (the GLM/DSv3.2 caller) and
    `compression_ratio > 1` (the DSv4 caller),
  * `enable_early_exit` both False and True, and, separately, a batch short
    enough that the `lax.cond` fast path actually fires -- at both compression
    ratios, since the guard divides by it,
  * `head_dim == 128`, so the packed record width is
    `align_to(128 + 1, 128) == 256`, the same `lkv_dim` production uses.

Tolerance
---------
**Zero.** The comparison is exact integer equality of the row-wise *sorted*
index sets (`np.sort` on both sides, as the in-tree test does at :411-418):
top-k order is deliberately unspecified -- the scoring path emits in
`approx_max_k` order and the early-exit path in increasing KV position -- so
comparing raw rows would assert something the kernel never promised.

There is no score tolerance because scores are never compared. The kernel and
the oracle consume bit-identical dequantized fp8 keys, so their scores differ
only by f32 accumulation order (~1e-6 relative). A pair of scores closer than
that straddling the k-th boundary would flip one index and fail this test
spuriously; the seeds are fixed so that is deterministic rather than flaky,
and with continuous random data the probability is negligible. If a future
shape does trip it, the fix is a different seed, not a tolerance.

Usage (from the vllm-torchtpu checkout root)
--------------------------------------------
`pytest` is in the `[test]` extra, so it is in neither `uv_venv` nor the pod
image. It is needed even for `--oracle-only`, because the in-tree test module
this file loads the oracle out of imports it at module scope.

    # CPU dry run. Proves the wiring, not the kernel: Pallas does not execute
    # on CPU, so every test that calls streamindex_topk will fail there.
    uv pip install --target /tmp/dryrun_pytest -q pytest
    JAX_PLATFORMS=cpu PYTHONPATH=/tmp/dryrun_pytest ./uv_venv/bin/python \\
        -m pytest --collect-only -q -p no:cacheprovider \\
        kernel_work/streamidx-topk/instruments/test_streamidx_topk_correctness.py
    JAX_PLATFORMS=cpu PYTHONPATH=/tmp/dryrun_pytest ./uv_venv/bin/python \\
        kernel_work/streamidx-topk/instruments/test_streamidx_topk_correctness.py \\
        --oracle-only

    # On a TPU, through /run-command-in-cdk. Read the `N passed` summary line:
    # pytest exit code 5 is "collected nothing", not success.
    uv pip install --system -q pytest && \\
        cd /root/cloud-devkit/vllm-torchtpu && \\
        python -m pytest -q -p no:cacheprovider --timeout=900 \\
        kernel_work/streamidx-topk/instruments/test_streamidx_topk_correctness.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk

# --- the in-tree oracle, loaded by path ------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REF_FILE = _REPO_ROOT / "tests" / "kernels" / "test_dsv4_streamindex_topk.py"


def _load_reference_module():
    spec = importlib.util.spec_from_file_location(
        "_dsv4_streamindex_topk_reference", _REF_FILE)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load the oracle from {_REF_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REF = _load_reference_module()
streamindex_topk_ref = _REF.streamindex_topk_ref
quantize_fp8_ue8m0 = _REF.quantize_fp8_ue8m0
_to_byte_lane = _REF._to_byte_lane

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

PAGE_SIZE = 16  # small; `page_size * bkv_p` must be a multiple of 128
BKV_P = 8  # -> bkv_sz = 128, the scores kernel's DMA contract
BQ_SZ = (1, 8, 8)  # (decode, prefill, mixed); decode is forced to 1 anyway
PAGES_PER_SEQ = 8  # -> num_positions = align_to(8, 8) * 16 = 128
HEAD_DIM = 128  # -> record width align_to(128 + 1, 128) = 256, as in prod


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    q_lens: tuple[int, ...]
    seq_lens: tuple[int, ...]
    k: int
    comp_ratio: int
    num_heads: int
    seed: int
    fast_path_expected: bool
    why: str


CASES: tuple[Case, ...] = (
    # 5 decodes + 1 prefill exercises the batched-decode call (4 sequences),
    # the decode-remainder call (the 5th) and the mixed call, in one graph.
    Case(
        name="glm_all_three_kernels",
        q_lens=(1, 1, 1, 1, 1, 6),
        seq_lens=(100, 128, 96, 112, 120, 120),
        k=64,
        comp_ratio=1,
        num_heads=32,
        seed=1,
        fast_path_expected=False,
        why="compression_ratio=1 (GLM), every sequence longer than k",
    ),
    # compression_ratio > 1 is the DSv4 caller. kv_len = seq_len // ratio, and
    # the causal bound is on the *compressed* position, so this is a different
    # masking path, not just a smaller one.
    Case(
        name="dsv4_comp4_scoring",
        q_lens=(1, 4),
        seq_lens=(256, 240),
        k=32,
        comp_ratio=4,
        num_heads=8,
        seed=2,
        fast_path_expected=False,
        why="compression_ratio=4, kv_len 64/60 both above k=32",
    ),
    # max(seq_lens) // comp_ratio <= k, so `enable_early_exit=True` takes the
    # lax.cond shortcut and no scores are computed at all. The oracle still
    # applies: every visible position wins, and the causal bound must hold.
    Case(
        name="glm_fast_path",
        q_lens=(1, 1, 4),
        seq_lens=(48, 60, 64),
        k=64,
        comp_ratio=1,
        num_heads=8,
        seed=3,
        fast_path_expected=True,
        why="compression_ratio=1, max seq_len 64 == k, shortcut fires",
    ),
    # Same shortcut reached through the divisor rather than through short
    # sequences: 256 // 4 == 64 <= k. A guard that forgot to divide would miss
    # this and silently run the scoring path (still correct, but the branch
    # under test would never execute), so `fast_path_expected` is asserted.
    Case(
        name="dsv4_comp4_fast_path",
        q_lens=(1, 4),
        seq_lens=(256, 240),
        k=64,
        comp_ratio=4,
        num_heads=8,
        seed=4,
        fast_path_expected=True,
        why="compression_ratio=4, 256 // 4 = 64 <= k, shortcut fires",
    ),
)


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def pack_cache(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fp8-quantize ``f32[num_pages, page_size, head_dim]`` keys.

    Returns ``(cache_kv, dequantized)``: the uint8 cache in the layout the
    kernel reads, and the exact values the hardware will see, for the oracle.

    Record layout, per `test_dsv4_streamindex_topk.py:316-339`:
    ``[fp8 x D | one e8m0 scale byte | pad]`` at ``[page, s // 4, s % 4, :]``,
    the row rounded up to a 128-lane group.
    """
    num_pages, page_size, head_dim = keys.shape
    assert page_size % 4 == 0, "the cache minor axis packs 4 slots per word"

    fp8, scale = quantize_fp8_ue8m0(jnp.asarray(keys), head_dim)
    # `quantize_fp8_ue8m0` squeezes the block axis, leaving one scale per row.
    scale_f32 = np.asarray(scale.astype(jnp.float32))  # [pages, slots, 1]
    dequantized = np.asarray(fp8.astype(jnp.float32)) * scale_f32

    record = jnp.concatenate([_to_byte_lane(fp8), _to_byte_lane(scale)],
                             axis=-1)
    q_lkv_dim = ((head_dim + 127) // 128) * 128
    width = ((q_lkv_dim + q_lkv_dim // 128 + 127) // 128) * 128
    record = jnp.pad(record,
                     ((0, 0), (0, 0), (0, width - record.shape[-1])))
    cache_kv = np.asarray(record).reshape(num_pages, page_size // 4, 4, width)
    return cache_kv, dequantized


def build_case(case: Case) -> dict:
    """Inputs, the block table and the oracle's dequantized keys."""
    rng = np.random.default_rng(case.seed)
    num_seqs = len(case.q_lens)
    num_tokens = int(sum(case.q_lens))
    num_pages = num_seqs * PAGES_PER_SEQ

    for q_len, seq_len in zip(case.q_lens, case.seq_lens):
        assert seq_len // case.comp_ratio <= PAGES_PER_SEQ * PAGE_SIZE, (
            f"{case.name}: kv_len {seq_len // case.comp_ratio} does not fit "
            f"in {PAGES_PER_SEQ} pages of {PAGE_SIZE}")
        assert q_len <= seq_len, f"{case.name}: q_len exceeds seq_len"

    # Shuffled, disjoint physical pages: a sequential table would let a bug in
    # the page walk hide behind the layout.
    block_table = rng.permutation(num_pages).astype(np.int32).reshape(
        num_seqs, PAGES_PER_SEQ)
    keys = rng.standard_normal((num_pages, PAGE_SIZE, HEAD_DIM),
                               dtype=np.float32)
    cache_kv, dequantized = pack_cache(keys)

    # `distribution` requires the decode-only sequences to lead the batch.
    num_decodes = 0
    while num_decodes < num_seqs and case.q_lens[num_decodes] == 1:
        num_decodes += 1

    return {
        "q":
        rng.standard_normal((num_tokens, case.num_heads, HEAD_DIM),
                            dtype=np.float32),
        "indexer_weights":
        rng.uniform(-1.5, 1.5, (num_tokens, case.num_heads)).astype(np.float32),
        "cache_kv":
        cache_kv,
        "seq_lens":
        np.asarray(case.seq_lens, np.int32),
        "page_indices":
        block_table.reshape(-1),
        "cu_q_lens":
        np.concatenate([[0], np.cumsum(case.q_lens)]).astype(np.int32),
        "distribution":
        np.array([num_decodes, num_decodes, num_seqs], np.int32),
        # not kernel inputs:
        "_block_table":
        block_table,
        # the oracle wants [page, slot, kv_head, dim]; the indexer is MQA.
        "_dequantized_kv":
        dequantized[:, :, None, :],
        # token -> (its sequence's uncompressed length, its absolute position)
        "_token_seq":
        _token_positions(case),
    }


def _token_positions(case: Case) -> list[tuple[int, int]]:
    """``(seq_len, q_abs_pos)`` per token, mirroring the oracle's indexing.

    The oracle computes ``q_abs_pos = (S_total - T_seq) + t_idx`` -- the queries
    are the tail of their sequence -- and masks ``s_idx * comp > q_abs_pos``.
    """
    out: list[tuple[int, int]] = []
    for q_len, seq_len in zip(case.q_lens, case.seq_lens):
        for t_idx in range(q_len):
            out.append((seq_len, seq_len - q_len + t_idx))
    return out


def expected_topk(case: Case, built: dict) -> np.ndarray:
    return streamindex_topk_ref(
        q=built["q"],
        weights=built["indexer_weights"],
        kv=built["_dequantized_kv"],
        block_table=built["_block_table"],
        T_list=list(case.q_lens),
        S_list=list(case.seq_lens),
        cu_q_lens=built["cu_q_lens"],
        k=case.k,
        comp_ratio=case.comp_ratio,
        H_I=case.num_heads,
        H_KV=1,
    )


def run_kernel(case: Case, built: dict, enable_early_exit: bool) -> np.ndarray:
    kernel_inputs = {
        name: jnp.asarray(value)
        for name, value in built.items() if not name.startswith("_")
    }
    return np.asarray(
        streamindex_topk(
            **kernel_inputs,
            k=case.k,
            compression_ratio=case.comp_ratio,
            num_kv_pages_per_block=BKV_P,
            num_queries_per_block=BQ_SZ,
            enable_early_exit=enable_early_exit,
        ))


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def assert_matches_oracle(actual: np.ndarray, expected: np.ndarray,
                          label: str) -> None:
    """Exact equality of the selected index *sets*, row by row."""
    assert actual.shape == expected.shape, (label, actual.shape,
                                            expected.shape)
    assert actual.dtype == np.int32, (label, actual.dtype)
    np.testing.assert_array_equal(
        np.sort(actual, axis=-1),
        np.sort(expected, axis=-1),
        err_msg=f"{label}: selected index set differs from the NumPy oracle",
    )


def assert_well_formed(actual: np.ndarray, case: Case, built: dict,
                       label: str) -> None:
    """Structural claims the oracle comparison alone would not localize.

    Cheap, and they turn "the sets differ" into a named failure: a duplicate
    index, padding that is not a suffix, or a causality violation.
    """
    for token, row in enumerate(actual):
        kept = row[row >= 0]
        assert len(set(kept.tolist())) == len(kept), (
            f"{label}: token {token} reported a duplicate index: {row}")
        assert np.all(row[len(kept):] < 0), (
            f"{label}: token {token} -1 padding is not a suffix: {row}")
        if not len(kept):
            continue
        seq_len, q_abs_pos = built["_token_seq"][token]
        # Causality, in compressed space: the oracle skips s_idx whenever
        # s_idx * comp_ratio > q_abs_pos, and nothing past kv_len exists.
        bound = min((seq_len // case.comp_ratio) - 1,
                    q_abs_pos // case.comp_ratio)
        assert int(kept.max()) <= bound, (
            f"{label}: token {token} selected index {kept.max()}, past its "
            f"causal bound {bound} (seq_len {seq_len}, q_abs_pos {q_abs_pos}, "
            f"comp {case.comp_ratio})")


def fast_path_fires(case: Case) -> bool:
    """The guard at `streamindex_topk.py:860-865`, restated."""
    return max(case.seq_lens) // case.comp_ratio <= case.k


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_IDS = [c.name for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=_IDS)
@pytest.mark.parametrize("enable_early_exit", [False, True],
                         ids=["ee_off", "ee_on"])
def test_matches_numpy_oracle(case: Case, enable_early_exit: bool) -> None:
    """The kernel's top-k index set equals the NumPy oracle's, exactly.

    Run at both settings of `enable_early_exit`, because the two production
    callers differ on it (`mla_attention_op.py:57` sets it True; the DSv4
    default at `streamindex_topk.py:537` is False) and because the flag must
    change cost, never the answer.
    """
    built = build_case(case)
    expected = expected_topk(case, built)
    actual = run_kernel(case, built, enable_early_exit)

    label = f"{case.name}/ee={enable_early_exit}"
    assert_well_formed(actual, case, built, label)
    assert_matches_oracle(actual, expected, label)


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_early_exit_is_inert(case: Case) -> None:
    """Turning the flag on may not change the answer, only the work.

    Asserted as set equality, not bit-for-bit, in both directions. Where the
    shortcut fires the two paths emit in different orders by design (increasing
    KV position vs `approx_max_k` order); where it does not, `lax.cond` still
    reaches the same code but XLA is free to fuse it differently, and a
    different f32 accumulation order could flip a near-tie. Bit-exactness is
    therefore not something the kernel promises, and asserting it here would
    make the yardstick flaky rather than strict.

    This complements the oracle test: it holds even on a batch where the oracle
    is too slow to build, and it names which of the two paths moved.
    """
    built = build_case(case)
    off = run_kernel(case, built, False)
    on = run_kernel(case, built, True)

    assert fast_path_fires(case) == case.fast_path_expected, (
        f"{case.name}: the case no longer exercises the branch it was written "
        f"for ({case.why})")

    np.testing.assert_array_equal(
        np.sort(on, axis=-1),
        np.sort(off, axis=-1),
        err_msg=f"{case.name}: enable_early_exit changed the selected index "
        f"set (shortcut fires: {case.fast_path_expected})")


def test_fast_path_is_still_causal() -> None:
    """The shortcut answers from positions alone, so its mask is its own.

    The whole batch fits in k, so a shortcut that returned `[0, k)` instead of
    `[0, causal_position]` would still look plausible against a shape check.
    This pins the exact per-token count.
    """
    case = next(c for c in CASES if c.name == "glm_fast_path")
    built = build_case(case)
    actual = run_kernel(case, built, True)

    counts = []
    for q_len, seq_len in zip(case.q_lens, case.seq_lens):
        # At compression_ratio 1 the queries are the tail of their sequence, so
        # query i sits at absolute position seq_len - q_len + i and sees every
        # position up to and including it.
        counts.extend(seq_len - q_len + i + 1 for i in range(q_len))
    kept = [int(np.count_nonzero(row >= 0)) for row in actual]
    assert kept == counts, (kept, counts)


# ---------------------------------------------------------------------------


def _oracle_only() -> int:
    """CPU dry run: build every case and run the oracle. No Pallas.

    Proves the case table, the packing and the oracle wiring are not typos.
    It proves nothing about the kernel -- Pallas does not execute on CPU.
    """
    for case in CASES:
        built = build_case(case)
        expected = expected_topk(case, built)
        num_tokens = int(sum(case.q_lens))
        kept = [int(np.count_nonzero(row >= 0)) for row in expected]
        print(f"[oracle] {case.name}: tokens={num_tokens} k={case.k} "
              f"comp={case.comp_ratio} "
              f"cache_kv={built['cache_kv'].shape} "
              f"expected={expected.shape} kept_per_token={kept} "
              f"fast_path={fast_path_fires(case)} "
              f"(declared {case.fast_path_expected})")
        assert expected.shape == (num_tokens, case.k)
        assert fast_path_fires(case) == case.fast_path_expected, case.name
    print(f"[oracle] {len(CASES)} case(s) built and scored on CPU")
    return 0


def main(argv) -> int:
    if "--oracle-only" in argv:
        return _oracle_only()
    return pytest.main([__file__, "-p", "no:cacheprovider", "-q"])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
