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
"""Correctness for `streamindex_topk` at GLM-5.2's real shapes.

THIS FILE IS READ-ONLY AFTER STEP 1.

Why this exists when `tests/kernels/test_dsv4_streamindex_topk.py` already
passes: that file's executed reference cases top out at `seq_len=384`,
`num_heads=8`, `head_dim=128`, `k=1024`, because its oracle
(`streamindex_topk_ref`, L58) scores in a triple Python loop. GLM-5.2 runs
`seq_len` up to 65536, `num_heads=32`, `k=2048`. The oracle here is the same
algorithm vectorized and evaluated in float64, so it reaches the production
shape in seconds.

The criterion is on RANKS, not on scores
----------------------------------------
A top-k returns a SET, so "close" is not meaningful -- an index is in or out.
The first version of this file therefore compared scores against the k-th
largest with a small absolute epsilon, and it reported 64 of 64 rows bad on the
shipped kernel: 4-8 indices per row sat 0.11-0.34 below a cut whose magnitude
was 11-49. An absolute epsilon on a quantity whose scale is data-dependent is
not a criterion, and it cannot distinguish the two things that matter here.

What does distinguish them is the RANK of the offending index in the oracle
ordering. An index at oracle rank 2051 of 8192 is a rank-boundary disagreement
between two ways of computing nearly-equal scores. An index at rank 5000 is a
bug. Rank is scale-free, so it needs no tolerance tuned to the data. Two
numbers per row:

    recall      |returned set  intersect  oracle top-k| / expected_n
    worst_rank  the worst oracle rank (1-indexed) among the returned indices

A row passes when `worst_rank <= expected_n + RANK_SLACK` and
`recall >= MIN_RECALL`. Both aggregates are printed on PASS as well as on FAIL,
so the margin is visible rather than assumed -- a check whose slack you cannot
see is a check you cannot judge.

`--negative-control` deliberately breaks the kernel and asserts the check goes
red. A correctness check that has never failed is not evidence. See
`references/hardware-facts.md` section "Instruments that lie".
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

from . import inputs as I
from . import shapes as S

# A returned index may sit at most this many ranks past the true cut. 64 is
# 3.1% of k=2048. Chosen to be far below the rank error a real defect produces
# (the negative control lands in the thousands) and far above float64-vs-MXU
# boundary churn; the printed `worst_rank` says how much of it is being used.
RANK_SLACK = 64
MIN_RECALL = 0.99
TOKEN_CHUNK = 64  # oracle is evaluated in token chunks to bound memory


def oracle_scores(d, regime: S.Regime) -> np.ndarray:
    """float64 scores[num_tokens, kv_len], -inf where masked.

    Vectorized restatement of tests/kernels/test_dsv4_streamindex_topk.py:100-106:
        score[t, s] = sum_h relu(q[t,h] . kv[s]) * weights[t,h]
    GLM-5.2's indexer cache holds ONE key per compressed position
    (index_head_dim=128, no KV-head axis), so the reference's `h_kv` GQA map is
    the identity here and is dropped.
    """
    q = d["_q_f32"].astype(np.float64)          # [T, H, D]
    w = d["_weights"].astype(np.float64)        # [T, H]
    dequant = d["_dequant"].astype(np.float64)  # [num_pages, PAGE_SIZE, D]
    block_table = d["_block_table"]
    q_lens = d["_q_lens"]
    cu = np.asarray(d["cu_q_lens"])
    seq_lens = np.asarray(d["seq_lens"])

    T = q.shape[0]
    kv_len = regime.kv_len
    out = np.full((T, kv_len), -np.inf, dtype=np.float64)

    for b in range(regime.max_num_seqs):
        if q_lens[b] == 0:
            continue
        pages = block_table[b]
        seq_kv = dequant[pages].reshape(-1, S.INDEX_HEAD_DIM)[:kv_len]
        t0, t1 = int(cu[b]), int(cu[b + 1])
        seq_len = int(seq_lens[b])
        for c0 in range(t0, t1, TOKEN_CHUNK):
            c1 = min(c0 + TOKEN_CHUNK, t1)
            # inner[t, h, s]
            inner = np.einsum("thd,sd->ths", q[c0:c1], seq_kv, optimize=True)
            np.maximum(inner, 0.0, out=inner)
            inner *= w[c0:c1, :, None]
            sc = inner.sum(axis=1)  # [chunk, kv_len]
            # causal: token at absolute position p sees positions 0..p
            local = np.arange(c0, c1) - t0
            q_abs_pos = (seq_len - int(q_lens[b])) + local
            vis = np.arange(kv_len)[None, :] <= (
                q_abs_pos[:, None] // S.COMPRESSION_RATIO)
            out[c0:c1] = np.where(vis, sc, -np.inf)
    return out


def check(topk_idxs: np.ndarray, scores: np.ndarray, k: int):
    """Return (n_rows_checked, failures, stats).

    `stats` carries the aggregate margin -- worst rank seen anywhere and the
    minimum per-row recall -- so a PASS can be read for how much slack it used.
    """
    failures = []
    worst_rank_seen = 0
    min_recall = 1.0
    T = scores.shape[0]
    n_checked = 0
    for t in range(T):
        row = scores[t]
        finite = np.isfinite(row)
        n_visible = int(finite.sum())
        if n_visible == 0:
            continue
        n_checked += 1
        expected_n = min(k, n_visible)
        got = topk_idxs[t]
        got_valid = got[got >= 0]
        if len(got_valid) != expected_n:
            failures.append(
                f"row {t}: returned {len(got_valid)} indices, expected "
                f"{expected_n} (n_visible={n_visible}, k={k})")
            continue
        if len(np.unique(got_valid)) != len(got_valid):
            failures.append(f"row {t}: duplicate indices returned")
            continue
        if got_valid.max() >= len(row) or got_valid.min() < 0:
            failures.append(f"row {t}: index out of range")
            continue
        # rank_of[i] = 0-based position of index i in the oracle's descending
        # score order. -inf positions sort last, so a masked index that leaks
        # into the output lands at a rank >= n_visible and fails loudly.
        order = np.argsort(-row, kind="stable")
        rank_of = np.empty(len(row), dtype=np.int64)
        rank_of[order] = np.arange(len(row))
        ranks = rank_of[got_valid]
        worst_rank = int(ranks.max()) + 1  # 1-indexed for reading
        recall = float((ranks < expected_n).mean())
        worst_rank_seen = max(worst_rank_seen, worst_rank)
        min_recall = min(min_recall, recall)
        if worst_rank > expected_n + RANK_SLACK or recall < MIN_RECALL:
            n_out = int((ranks >= expected_n).sum())
            failures.append(
                f"row {t}: worst_rank={worst_rank} (allowed "
                f"{expected_n + RANK_SLACK}), recall={recall:.5f} (min "
                f"{MIN_RECALL}), {n_out} of {expected_n} indices outside the "
                f"true top-{expected_n}")
    stats = dict(worst_rank_seen=worst_rank_seen, min_recall=min_recall)
    return n_checked, failures, stats


def run(impl, regime: S.Regime, *, negative_control=False, seed=0):
    d = I.build(regime, seed=seed, with_dequant=True)
    k = S.INDEX_TOPK
    q = d["q"]
    if negative_control:
        # Break the scores in a way a top-k check must notice: zero the query
        # rows for the second half of the heads, which changes the ranking
        # without changing any shape, dtype or code path.
        # np.asarray of a jax array is read-only; copy before mutating.
        qm = np.array(q.astype(jnp.float32), copy=True)
        qm[:, S.INDEX_N_HEADS // 2:, :] = 0.0
        q = jnp.asarray(qm).astype(jnp.float8_e4m3fn)

    idxs = impl.streamindex_topk(
        q,
        d["indexer_weights"],
        d["cache_kv"],
        d["seq_lens"],
        d["page_indices"],
        d["cu_q_lens"],
        d["distribution"],
        k=k,
        compression_ratio=S.COMPRESSION_RATIO,
        num_kv_pages_per_block=S.NUM_KV_PAGES_PER_BLOCK,
        num_queries_per_block=S.NUM_QUERIES_PER_BLOCK,
        decode_req_batch_size=S.DECODE_REQ_BATCH_SIZE,
        enable_early_exit=False,
    )
    idxs = np.asarray(jax.block_until_ready(idxs))
    sc = oracle_scores(d, regime)  # oracle always uses the UNMUTATED q
    return check(idxs, sc, k)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl",
                    default="vllm_torchtpu.kernels.deepseek_v4.streamindex_topk")
    ap.add_argument("--regimes", default="decode_prod")
    ap.add_argument("--negative-control", action="store_true",
                    help="mutate the kernel input and require the check to FAIL")
    a = ap.parse_args(argv)

    import importlib
    impl = importlib.import_module(a.impl)

    rc = 0
    for name in [n for n in a.regimes.split(",") if n]:
        regime = S.REGIMES_BY_NAME[name]
        n, failures, st = run(impl, regime,
                              negative_control=a.negative_control)
        margin = (f"worst_rank={st['worst_rank_seen']} of allowed "
                  f"{S.INDEX_TOPK + RANK_SLACK}, min_recall="
                  f"{st['min_recall']:.5f} of required {MIN_RECALL}")
        if a.negative_control:
            if failures:
                print(f"[{name}] NEGATIVE CONTROL OK -- check fired on a "
                      f"mutated kernel ({len(failures)} of {n} rows bad). "
                      f"{margin}. first: {failures[0]}")
            else:
                print(f"[{name}] NEGATIVE CONTROL DID NOT FIRE -- the mutated "
                      f"kernel PASSED over {n} rows ({margin}). The check is "
                      f"vacuous and must not be quoted as evidence.")
                rc = 1
        else:
            if failures:
                print(f"[{name}] FAIL -- {len(failures)} of {n} rows bad "
                      f"({margin})")
                for f in failures[:5]:
                    print("   ", f)
                rc = 1
            else:
                print(f"[{name}] PASS -- all {n} rows are a valid top-"
                      f"{S.INDEX_TOPK} against the float64 oracle. {margin}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
