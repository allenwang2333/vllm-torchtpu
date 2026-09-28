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
"""Tests for the threshold-gated SparseCore top-k (`TPU_STREAMIDX_SCTOPK_L1`).

`sparsecore_topk_l1` must select what `sparsecore_topk` selects. Two layers:

* **Kernel.** Against the exact score-multiset oracle (the incumbent test's
  bar: exact, no tolerance, indices only up to boundary ties), and against the
  incumbent's own output, at the production chunk shape (b=128, n=264192 for
  `--max-model-len 263168`) plus short, ragged and edge shapes. Every
  distribution that matters to the algorithm is covered: one where no slice
  overflows its threshold, one where the hot recent slice does (the per-slice
  radix fallback runs), heavy ties (T lands inside a tie), and rows too short
  for any threshold.
* **Call site.** `streamindex_topk(sctopk_l1=True)` against `sctopk_l1=False`
  at the production shape, with long, short, decode and padded rows. There T
  comes from the StreamIdxTC scoring epilogue (no T kernel), and it must equal
  the T kernel's `row_threshold_keys` on every live row.

A passing oracle proves something only if it can fail, so the positive controls
are asserted red: a threshold one key above the true k-th score (kernel and
call site), stage 1' with the per-slice fallback removed, and an epilogue whose
group maxima are inflated.

`python test_dsv4_sparsecore_topk_l1.py` runs the flag on/off microbench
instead of the tests.
"""

import difflib
import importlib
import re
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    KEY_NEG_INF,
    _align_to,
    _pick_partition,
    sparsecore_topk,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import sctopk_l1 as l1

sim = importlib.import_module(
    "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SEED = 20260928
K = 2048
B = 128  # TPU_STREAMIDX_CHUNK_TOKENS=128 rows per top-k call.
# --max-model-len 263168 -> 257 pages, aligned to bkv_p=2 -> 258 * 1024.
N_PROD = 264192
N_128K = 133120
N_SHORT = 10240


# --------------------------------------------------------------------------- #
# Oracle.
# --------------------------------------------------------------------------- #


def _keys(scores: np.ndarray) -> np.ndarray:
    bits = scores.view(np.int32)
    return bits ^ ((bits >> 31) & np.int32(0x7FFFFFFF))


def _reference_scores(scores, k, row_lengths):
    out = np.full((scores.shape[0], k), -np.inf, np.float32)
    for i in range(scores.shape[0]):
        row = scores[i, : int(row_lengths[i])]
        vals = np.sort(row[row > -np.inf])[::-1][:k]
        out[i, : vals.size] = vals
    return out


def _mismatch(scores, idx, k, row_lengths) -> str | None:
    """`None` if `idx` is an exact top-k (as a score multiset), else why not."""
    b, n = scores.shape
    ref = _reference_scores(scores, k, row_lengths)
    got = np.full((b, k), -np.inf, np.float32)
    for i in range(b):
        live = idx[i][idx[i] >= 0]
        m = live.size
        if not (np.all(idx[i, m:] < 0) and np.all(idx[i, :m] >= 0)):
            return f"row {i}: -1 padding is not a suffix"
        if m and live.max() >= min(n, int(row_lengths[i])):
            return f"row {i}: index {live.max()} at or beyond {int(row_lengths[i])}"
        if np.unique(live).size != m:
            return f"row {i}: duplicate indices"
        vals = np.sort(scores[i, live])[::-1]
        got[i, :m] = vals
    n_ref = np.isfinite(ref).sum(1)
    n_got = np.isfinite(got).sum(1)
    if (n_ref != n_got).any():
        r = int(np.flatnonzero(n_ref != n_got)[0])
        return f"row {r}: {n_got[r]} valid entries, expected {n_ref[r]}"
    fin = np.isfinite(ref)
    if fin.any() and np.abs(ref[fin] - got[fin]).max() != 0.0:
        return f"max score difference {np.abs(ref[fin] - got[fin]).max()!r}"
    return None


def _set_diff_beyond_ties(scores, a, b) -> str | None:
    """`None` if index sets `a` and `b` differ only in scores tied at v_k."""
    for i in range(scores.shape[0]):
        sa = set(a[i][a[i] >= 0].tolist())
        sb = set(b[i][b[i] >= 0].tolist())
        diff = sa ^ sb
        if not diff:
            continue
        vk = scores[i, list(sa)].min()
        off = [c for c in diff if scores[i, c] != vk]
        if off:
            return f"row {i}: {len(off)} differing indices not tied at v_k={vk}"
    return None


def _t_reference(scores, k, row_lengths) -> np.ndarray:
    b, n = scores.shape
    key = np.where(
        np.arange(n)[None, :] < row_lengths[:, None], _keys(scores), KEY_NEG_INF
    ).astype(np.int32)
    n_pad = _align_to(n, k)
    key = np.pad(key, ((0, 0), (0, n_pad - n)), constant_values=KEY_NEG_INF)
    return key.reshape(b, n_pad // k, k).max(1).min(1)


def _overflowing_slices(scores, k, row_lengths, t) -> int:
    """How many stage-1' slices keep more than k_p survivors (take the radix)."""
    b, n = scores.shape
    p = _pick_partition(b, n, k)
    n_p = n // p
    k_p = min(k, n_p)
    col = np.arange(n)[None, :]
    surv = (_keys(scores) >= t[:, None]) & (col < row_lengths[:, None])
    surv &= scores > -np.inf
    return int((surv.reshape(b, p, n_p).sum(2) > k_p).sum())


# --------------------------------------------------------------------------- #
# Inputs.
# --------------------------------------------------------------------------- #


def _scores(kind, b, n, rng, row_lengths=None):
    if kind == "iid":
        return rng.standard_normal((b, n), dtype=np.float32)
    if kind == "hot_recent":
        # A prefill chunk's recent context scores high: the last ~12K visible
        # positions of every row sit in the same slice, which overflows.
        x = rng.standard_normal((b, n), dtype=np.float32)
        lens = np.full(b, n) if row_lengths is None else row_lengths
        col = np.arange(n)[None, :]
        hot = (col >= lens[:, None] - 12000) & (col < lens[:, None])
        return np.where(hot, x + 4.0, x).astype(np.float32)
    if kind == "heavy_ties":
        return rng.integers(0, 17, size=(b, n), dtype=np.int8).astype(np.float32)
    if kind == "strict_desc":
        return np.repeat(-np.arange(n, dtype=np.float32)[None, :], b, axis=0)
    if kind == "all_neg_inf":
        return np.full((b, n), -np.inf, np.float32)
    if kind == "sparse_neg_inf":
        # Masked positions inside the visible range, as the scorer can emit.
        x = rng.standard_normal((b, n), dtype=np.float32)
        return np.where(rng.random((b, n)) < 0.3, -np.inf, x).astype(np.float32)
    raise ValueError(kind)


def _ragged_lengths(b, n, k):
    p = _pick_partition(b, n, k)
    n_p = n // p
    edges = [0, 1, 16, k - 1, k, k + 1, n_p - 1, n_p, n_p + 1, 2 * n_p + 7,
             n // 2, n - 1, n]
    return np.array([edges[i % len(edges)] for i in range(b)], np.int32)


def _as_bits(scores):
    return jnp.asarray(scores.view(np.int32))


def _run_both(scores, k, row_lengths):
    s, lens = _as_bits(scores), jnp.asarray(row_lengths)
    on = np.asarray(l1.sparsecore_topk_l1(s, k, lens))
    off = np.asarray(sparsecore_topk(s, k, lens))
    return on, off


# --------------------------------------------------------------------------- #
# Kernel tests.
# --------------------------------------------------------------------------- #


def test_support_gate():
    """The production shape runs the threshold path; the rest is explicit."""
    assert l1.l1_supported(B, N_PROD, K)  # P=16, 16512-word slices
    assert l1.l1_supported(B, N_128K, K)  # P=8
    assert l1.l1_supported(16, N_SHORT, K)  # P=2, b < 32
    assert l1.l1_supported(4, N_PROD, K)  # P=16 by the b*P <= 32 cap
    assert l1.l1_supported(1024, N_PROD, K)  # unchunked prefill
    assert not l1.l1_supported(B, N_SHORT, K)  # P=1: single stage
    assert not l1.l1_supported(B, 32768, K)  # P=1
    assert not l1.l1_supported(B, 1048576, K)  # P*k_p=65536: recursive stage 2
    assert not l1.l1_supported(1, N_PROD, K)  # P=32 -> P*k_p=65536
    assert not l1.l1_supported(B, N_PROD, 2000)  # k % 128
    assert l1.l1_supported(B, 16 * l1.L1_MAX_SLICE_WORDS, K)
    assert not l1.l1_supported(B, 16 * (l1.L1_MAX_SLICE_WORDS + 16), K)


SHAPES = [
    pytest.param(B, N_PROD, K, id="prod-b128-n264192"),
    pytest.param(B, N_128K, K, id="b128-n133120"),
    pytest.param(16, N_SHORT, K, id="p2-b16-n10240"),
    # b=32/40 with i32 scores fail XLA memory-space assignment without the
    # optimization_barrier in _sparsecore_topk_l1_impl; keep them as regressions.
    pytest.param(40, N_PROD, 512, id="b40-n264192-k512"),
    pytest.param(32, N_PROD, K, id="b32-n264192"),
]
KINDS = ["iid", "hot_recent", "heavy_ties", "strict_desc", "all_neg_inf",
         "sparse_neg_inf"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("b,n,k", SHAPES)
def test_matches_oracle_and_incumbent(b, n, k, kind):
    assert l1.l1_supported(b, n, k)
    rng = np.random.default_rng(SEED)
    lens = np.full(b, n, np.int32)
    scores = _scores(kind, b, n, rng, lens)
    on, off = _run_both(scores, k, lens)
    assert (err := _mismatch(scores, on, k, lens)) is None, err
    assert (err := _mismatch(scores, off, k, lens)) is None, err
    assert (err := _set_diff_beyond_ties(scores, on, off)) is None, err


@pytest.mark.parametrize("kind", ["iid", "hot_recent", "heavy_ties"])
@pytest.mark.parametrize("b,n,k", SHAPES)
def test_ragged_row_lengths(b, n, k, kind):
    """Empty rows, rows shorter than k, and lengths on slice edges."""
    rng = np.random.default_rng(SEED + 1)
    lens = _ragged_lengths(b, n, k)
    scores = _scores(kind, b, n, rng, lens)
    on, off = _run_both(scores, k, lens)
    assert (err := _mismatch(scores, on, k, lens)) is None, err
    assert (err := _set_diff_beyond_ties(scores, on, off)) is None, err


def test_slice_width_limit_compiles_and_is_exact():
    """The widest slice the gate admits fits SparseCore VMEM."""
    b, n = B, 16 * l1.L1_MAX_SLICE_WORDS
    assert _pick_partition(b, n, K) == 16
    rng = np.random.default_rng(SEED + 2)
    lens = np.full(b, n, np.int32)
    scores = _scores("hot_recent", b, n, rng, lens)
    on, _ = _run_both(scores, K, lens)
    assert (err := _mismatch(scores, on, K, lens)) is None, err


@pytest.mark.parametrize("kind", ["iid", "hot_recent", "heavy_ties"])
def test_threshold_kernel_is_bit_exact(kind):
    rng = np.random.default_rng(SEED + 3)
    lens = _ragged_lengths(B, N_PROD, K)
    scores = _scores(kind, B, N_PROD, rng, lens)
    got = np.asarray(l1.row_threshold_keys(_as_bits(scores), K, jnp.asarray(lens)))
    np.testing.assert_array_equal(got, _t_reference(scores, K, lens))


def test_distributions_exercise_both_branches():
    """`iid` stays under k_p everywhere; `hot_recent` makes slices overflow.

    Without this, a pass on `hot_recent` might never have run the fallback.
    """
    rng = np.random.default_rng(SEED)
    lens = np.full(B, N_PROD, np.int32)
    for kind, want in (("iid", False), ("hot_recent", True)):
        scores = _scores(kind, B, N_PROD, rng, lens)
        t = _t_reference(scores, K, lens)
        n_over = _overflowing_slices(scores, K, lens, t)
        assert (n_over > 0) == want, (kind, n_over)


# --------------------------------------------------------------------------- #
# Positive controls: the oracle above must go red on a wrong stage 1'.
# --------------------------------------------------------------------------- #


def _impl(scores, lens, **kw):
    return np.asarray(
        l1._sparsecore_topk_l1_impl(
            _as_bits(scores),
            K,
            jnp.asarray(lens),
            write_empty_rows=True,
            scheduling_group_id=None,
            stage2_scheduling_group_id=None,
            threshold_scheduling_group_id=None,
            **kw,
        )[0]
    )


def test_control_threshold_above_vk_goes_red():
    """T = v_k + 1 drops the k-th score: every full row must fail."""
    rng = np.random.default_rng(SEED + 4)
    lens = np.full(B, N_PROD, np.int32)
    scores = _scores("iid", B, N_PROD, rng, lens)
    vk = np.sort(_keys(scores), axis=1)[:, -K]
    ok_t = _impl(scores, lens, t_keys=jnp.asarray(_t_reference(scores, K, lens)))
    assert _mismatch(scores, ok_t, K, lens) is None
    bad = _impl(scores, lens, t_keys=jnp.asarray(vk + 1))
    red_rows = sum(_mismatch(scores[i : i + 1], bad[i : i + 1], K, lens[:1])
                   is not None for i in range(B))
    assert red_rows == B, f"only {red_rows}/{B} rows red with T = v_k + 1"


def test_control_no_fallback_goes_red():
    """Copying the first k_p survivors of an overflowing slice is wrong."""
    rng = np.random.default_rng(SEED + 5)
    lens = np.full(B, N_PROD, np.int32)
    scores = _scores("hot_recent", B, N_PROD, rng, lens)
    bad = _impl(scores, lens, fallback=False)
    assert _mismatch(scores, bad, K, lens) is not None


def test_comparison_detects_a_wrong_index():
    rng = np.random.default_rng(SEED + 6)
    lens = np.full(B, N_PROD, np.int32)
    scores = _scores("iid", B, N_PROD, rng, lens)
    on, off = _run_both(scores, K, lens)
    bad = on.copy()
    top = set(bad[3].tolist())
    bad[3, 0] = next(c for c in range(N_PROD) if c not in top)
    assert _mismatch(scores, bad, K, lens) is not None
    assert _set_diff_beyond_ties(scores, bad, off) is not None


# --------------------------------------------------------------------------- #
# Call site: streamindex_topk with the flag on and off.
# --------------------------------------------------------------------------- #

PAGE = 1024
IH, D = 32, 128  # GLM-5.2 indexer heads and head dim
NUM_TOKENS = 1024  # max_num_tokens bucket; 8 chunks of 128


def _build_cache(pages_per_seq, recent_ctx, hot, seed):
    """fp8 + UE8M0 HEAD_ALONG_SUBLANE cache, pages shuffled.

    Logical position j of every sequence reads physical page perm[j // PAGE].
    With `hot`, the keys at the most recent positions (just below
    `recent_ctx`) are scaled up, so the recent slice dominates the scores the
    way a prefill chunk's does.
    """
    num_pages = _align_to(pages_per_seq, 2)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_pages).astype(np.int32)
    x = jax.random.normal(jax.random.PRNGKey(seed), (num_pages * PAGE, D))
    if hot:
        pos = jnp.arange(num_pages * PAGE)
        near = (pos >= recent_ctx - 12000) & (pos < recent_ctx)
        x = jnp.where(near[:, None], x * 4.0, x)
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.clip(jnp.max(jnp.abs(x), -1, keepdims=True), 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    quant = (x / scale).astype(jnp.float8_e4m3fn)
    rec = jnp.concatenate(
        [
            jax.lax.bitcast_convert_type(quant, jnp.uint8),
            jax.lax.bitcast_convert_type(scale.astype(jnp.float8_e8m0fnu), jnp.uint8),
        ],
        -1,
    )
    width = _align_to(D + D // 128, 128)
    rec = jnp.pad(rec, ((0, 0), (0, width - rec.shape[-1])))
    logical = rec.reshape(num_pages, PAGE // 4, 4, width)
    cache = jnp.zeros_like(logical).at[jnp.asarray(perm)].set(logical)
    return cache, perm[:pages_per_seq]


def _call_kwargs(
    T_list, S_list, pages_per_seq=257, hot=False, seed=0, num_tokens=NUM_TOKENS,
    chunk_tokens=128,
):
    cache, table = _build_cache(pages_per_seq, max(S_list), hot, seed)
    rng = np.random.default_rng(seed + 1)
    num_decodes = 0
    while num_decodes < len(T_list) and T_list[num_decodes] == 1:
        num_decodes += 1
    q = rng.standard_normal((num_tokens, IH, D), dtype=np.float32)
    w = rng.uniform(-1.5, 1.5, size=(num_tokens, IH)).astype(np.float32)
    return dict(
        q=jnp.asarray(q).astype(jnp.float8_e4m3fn),
        indexer_weights=jnp.asarray(w),
        cache_kv=cache,
        seq_lens=jnp.asarray(np.array(S_list, np.int32)),
        page_indices=jnp.asarray(np.tile(table, len(S_list))),
        cu_q_lens=jnp.asarray(np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32)),
        distribution=jnp.asarray(
            np.array([num_decodes, num_decodes, len(T_list)], np.int32)
        ),
        k=K,
        compression_ratio=1,
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 128, 128),
        chunk_tokens=chunk_tokens,
    )


CALL_CASES = [
    # One long prefill, the production case.
    pytest.param((1024,), (230000,), False, {}, id="prefill-230k"),
    pytest.param((1024,), (230000,), True, {}, id="prefill-230k-hot"),
    # Short context at the production n: most of every row is past its length.
    pytest.param((1024,), (4096,), False, {}, id="prefill-4k"),
    # Decodes, long and short prefills, and 120 padded (invalid) rows.
    pytest.param(
        (1, 1, 1, 1, 600, 300),
        (250000, 1500, 64, 200000, 230000, 3000),
        True,
        {},
        id="mixed-padded",
    ),
    # TPU_STREAMIDX_CHUNK_TOKENS unset: one b=1024 top-k call.
    pytest.param((1024,), (230000,), True, dict(chunk_tokens=None), id="unchunked"),
    # A small token bucket (16 does not divide by 128, so unchunked b=16).
    pytest.param((16,), (230000,), True, dict(num_tokens=16), id="bucket16"),
    # The contexts the L1v2 microbench sweeps.
    pytest.param((1024,), (29000,), True, {}, id="prefill-29k-hot"),
    pytest.param((1024,), (100000,), False, {}, id="prefill-100k"),
    pytest.param((1024,), (185000,), True, {}, id="prefill-185k-hot"),
    # Every row shorter than k (T = -inf), and rows straddling k.
    pytest.param((1024,), (2048,), False, {}, id="prefill-2048"),
    pytest.param((1024,), (2600,), False, {}, id="prefill-2600"),
    # Two sequences inside one 128-row chunk (rows 640..767 split at 700), a
    # context ending mid KV block, and a decode block ahead of them.
    pytest.param(
        (1, 1, 698, 324),
        (70001, 99, 150001, 3333),
        True,
        {},
        id="two-seqs-in-chunk",
    ),
]


def _red_rows(kw, T_list, **on_kw):
    """Rows whose flag-on selection differs from flag-off beyond v_k ties."""
    on_i, on_s = streamindex_topk(**kw, sctopk_l1=True, return_scores=True, **on_kw)
    off_i, off_s = streamindex_topk(**kw, sctopk_l1=False, return_scores=True)
    on_i, off_i = np.asarray(on_i), np.asarray(off_i)
    on_s = np.asarray(on_s).view(np.float32)
    off_s = np.asarray(off_s).view(np.float32)
    red = []
    for i in range(on_i.shape[0]):
        # The selected scores must agree exactly as multisets ...
        if not np.array_equal(np.sort(on_s[i]), np.sort(off_s[i])):
            red.append(i)
            continue
        # ... and the index sets except among scores tied at v_k.
        a = dict(zip(on_i[i].tolist(), on_s[i].tolist()))
        b = dict(zip(off_i[i].tolist(), off_s[i].tolist()))
        a.pop(-1, None), b.pop(-1, None)
        if a.keys() == b.keys():
            continue
        vk = min(b.values())
        if any({**a, **b}[c] != vk for c in a.keys() ^ b.keys()):
            red.append(i)
    # Rows past cu_q_lens[-1] are padding: empty either way.
    pad0 = sum(T_list)
    assert (off_i[pad0:] == -1).all()
    if (on_i[pad0:] != -1).any():
        red.extend(range(pad0, on_i.shape[0]))
    return sorted(set(red))


@pytest.fixture
def patched_sim(monkeypatch):
    """Monkeypatch `streamindex_topk`'s module globals with fresh jit caches."""
    jax.clear_caches()
    yield lambda name, value: monkeypatch.setattr(sim, name, value)
    monkeypatch.undo()
    jax.clear_caches()


@pytest.mark.parametrize("T_list,S_list,hot,extra", CALL_CASES)
def test_call_site_flag_on_matches_off(T_list, S_list, hot, extra):
    kw = _call_kwargs(T_list, S_list, hot=hot, **extra)
    hlo = streamindex_topk.lower(**kw, sctopk_l1=True).as_text()
    assert "sc_topk_l1" in hlo, "flag on did not reach L1"
    # T comes from the scoring epilogue: no T kernel, no relayout for it.
    assert "tc_rowT" not in hlo, "flag on still runs the T kernel"
    red = _red_rows(kw, T_list)
    assert not red, f"{len(red)} rows differ, first {red[:8]}"


def _capture_t(store):
    """A `sparsecore_topk_l1` that records (epilogue T, T kernel T, lengths)."""

    def wrapped(scores, k, row_lengths=None, *, t_rows=None, **kw):
        assert t_rows is not None, "the call site passed no epilogue T"
        ref = l1.row_threshold_keys(scores, k, row_lengths)
        jax.debug.callback(
            lambda a, b, c: store.append(
                (np.asarray(a), np.asarray(b), np.asarray(c))
            ),
            t_rows[:, 0],
            ref,
            row_lengths,
        )
        return l1.sparsecore_topk_l1(scores, k, row_lengths, t_rows=t_rows, **kw)

    return wrapped


@pytest.mark.parametrize(
    "T_list,S_list,hot,extra",
    [c for c in CALL_CASES if c.id in (
        "prefill-230k-hot", "prefill-4k", "mixed-padded", "unchunked",
        "prefill-2600", "two-seqs-in-chunk")],
)
def test_epilogue_t_equals_threshold_kernel(patched_sim, T_list, S_list, hot, extra):
    """The scoring epilogue's T is the T kernel's T on every live row."""
    store = []
    patched_sim("sparsecore_topk_l1", _capture_t(store))
    kw = _call_kwargs(T_list, S_list, hot=hot, **extra)
    jax.block_until_ready(streamindex_topk(**kw, sctopk_l1=True))
    jax.effects_barrier()
    assert store, "no chunk reached sparsecore_topk_l1"
    live = 0
    for chunk, (t_epi, t_ref, lens) in enumerate(store):
        m = lens > 0
        live += int(m.sum())
        bad = np.flatnonzero(m & (t_epi != t_ref))
        assert not bad.size, (
            f"chunk {chunk}: {bad.size} rows differ, row {bad[0]} "
            f"epi={t_epi[bad[0]]} ref={t_ref[bad[0]]} len={lens[bad[0]]}"
        )
    assert live == sum(T_list), (live, sum(T_list))


def _t_above_vk(scores, k, row_lengths=None, *, t_rows=None, **kw):
    """Positive control: T one key above v_k, the row's true k-th score."""
    b, n = scores.shape
    bits = (
        scores
        if scores.dtype == jnp.int32
        else jax.lax.bitcast_convert_type(scores, jnp.int32)
    )
    key = l1._monotone_key(bits)
    col = jax.lax.broadcasted_iota(jnp.int32, (b, n), 1)
    key = jnp.where(col < row_lengths[:, None], key, jnp.int32(KEY_NEG_INF))
    vk = jax.lax.top_k(key, k)[0][:, -1]
    bad = jnp.broadcast_to((vk + 1)[:, None], (b, 128))
    return l1.sparsecore_topk_l1(scores, k, row_lengths, t_rows=bad, **kw)


def test_call_site_control_t_above_vk_goes_red(patched_sim):
    """Every row with >= k live scores must fail when T = v_k + 1."""
    patched_sim("sparsecore_topk_l1", _t_above_vk)
    T_list, S_list = (1024,), (230000,)
    kw = _call_kwargs(T_list, S_list)
    red = _red_rows(kw, T_list)
    assert len(red) == 1024, f"only {len(red)}/1024 rows red with T = v_k + 1"


def test_call_site_control_inflated_epilogue_max_goes_red(patched_sim):
    """Group maxima inflated by one binade in the epilogue put T above v_k.

    Only rows whose min group max sits within one binade (x2) of v_k go red:
    812/1024 at this shape [measured: j-061c0789]. The bar is half the rows.
    """
    patched_sim("_TEST_EPILOGUE_T_FAULT", "inflate")
    T_list, S_list = (1024,), (230000,)
    kw = _call_kwargs(T_list, S_list)
    red = _red_rows(kw, T_list)
    print(f"inflate control: {len(red)}/1024 rows red")
    assert len(red) >= 512, f"only {len(red)}/1024 rows red with inflated maxima"


def test_call_site_unsupported_shape_falls_back():
    """n=10240 (P=1) with the flag on is the incumbent, bit for bit."""
    kw = _call_kwargs((1024,), (9000,), pages_per_seq=9)
    hlo = streamindex_topk.lower(**kw, sctopk_l1=True).as_text()
    assert "sc_topk_l1" not in hlo and "tc_rowT" not in hlo
    assert (
        streamindex_topk.lower(**kw, sctopk_l1=False).as_text().count("pallas")
        == hlo.count("pallas")
    ), "an unsupported shape still grew a scoring output"
    on = np.asarray(streamindex_topk(**kw, sctopk_l1=True))
    off = np.asarray(streamindex_topk(**kw, sctopk_l1=False))
    np.testing.assert_array_equal(on, off)


def _normalized(hlo: str) -> str:
    # Each trace reserves fresh scheduling group ids; renumber in order.
    ids = {}
    return re.sub(
        r"_scheduling_group_id(\W+)(\d+)",
        lambda m: f"_scheduling_group_id{m.group(1)}"
        f"{ids.setdefault(m.group(2), len(ids))}",
        hlo,
    )


def test_flag_off_lowers_to_the_default_program():
    kw = _call_kwargs((1024,), (230000,))
    default = streamindex_topk.lower(**kw).as_text()
    off = streamindex_topk.lower(**kw, sctopk_l1=False).as_text()
    assert "sc_topk_l1" not in off
    a, b = _normalized(default), _normalized(off)
    diff = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""))
    assert not diff, "\n".join(diff[:60])


# --------------------------------------------------------------------------- #
# Microbench: `python test_dsv4_sparsecore_topk_l1.py`.
# --------------------------------------------------------------------------- #


def _chain_slope_us(fn, scores, lens, ms=(1, 2, 4, 8), reps=30):
    """Per-call device time as the slope of an M-call dependent chain."""

    def chained(m):
        @jax.jit
        def f(s, l):
            acc = jnp.int32(0)
            for _ in range(m):
                s, acc = jax.lax.optimization_barrier((s, acc))
                out = fn(s, l)
                acc = acc + out[0, 0]
            return acc

        return f

    walls = []
    for m in ms:
        f = chained(m)
        jax.block_until_ready(f(scores, lens))
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(f(scores, lens))
            t.append(time.perf_counter() - t0)
        walls.append(statistics.median(t) * 1e6)
    slope = float(np.polyfit(np.array(ms, float), np.array(walls), 1)[0])
    return slope, walls


def _wall_us(fn, iters=50):
    jax.block_until_ready(fn())
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        t.append(time.perf_counter() - t0)
    return statistics.median(t) * 1e6


def main():
    print(f"devices: {jax.devices()}")
    rng = np.random.default_rng(SEED)
    # (a) The top-k call alone at the production chunk shape.
    for kind in ("iid", "hot_recent"):
        lens_np = np.full(B, N_PROD, np.int32)
        if kind == "hot_recent":
            lens_np = (230000 - 128 + np.arange(B) + 1).astype(np.int32)
        scores_np = _scores(kind, B, N_PROD, rng, lens_np)
        scores, lens = _as_bits(scores_np), jnp.asarray(lens_np)
        t = _t_reference(scores_np, K, lens_np)
        n_over = _overflowing_slices(scores_np, K, lens_np, t)
        on = _chain_slope_us(
            lambda s, l: l1.sparsecore_topk_l1(s, K, l), scores, lens
        )
        off = _chain_slope_us(lambda s, l: sparsecore_topk(s, K, l), scores, lens)
        print(
            f"BENCH topk b={B} n={N_PROD} k={K} {kind} overflow_slices={n_over}: "
            f"on={on[0]:.1f} us off={off[0]:.1f} us saving={off[0] - on[0]:.1f} us "
            f"(walls on={[round(w) for w in on[1]]} off={[round(w) for w in off[1]]})"
        )
    # (b) The whole streamindex_topk call, as the adapter makes it.
    for name, T_list, S_list, hot in (
        ("prefill-230k", (1024,), (230000,), False),
        ("prefill-230k-hot", (1024,), (230000,), True),
        ("prefill-4k", (1024,), (4096,), False),
    ):
        kw = _call_kwargs(T_list, S_list, hot=hot)
        kw["enable_early_exit"] = True
        # streamindex_topk is jitted; the arrays stay arguments, not constants.
        def f_on(kw=kw):
            return streamindex_topk(**kw, sctopk_l1=True)

        def f_off(kw=kw):
            return streamindex_topk(**kw, sctopk_l1=False)

        res = {}
        for _ in range(3):  # interleave to spread drift over both arms
            res.setdefault("on", []).append(_wall_us(f_on))
            res.setdefault("off", []).append(_wall_us(f_off))
        on, off = min(res["on"]), min(res["off"])
        print(
            f"BENCH streamindex_topk {name} tokens={NUM_TOKENS} chunk=128: "
            f"on={on:.1f} us off={off:.1f} us saving={off - on:.1f} us "
            f"(rounds on={[round(x) for x in res['on']]} "
            f"off={[round(x) for x in res['off']]})"
        )


if __name__ == "__main__":
    main()
