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
"""On-device time and numerics of the megakernel attention vs the stock kernels.

Usage (TPU): python bench_attn_device.py hca [--ctx 32768,131072] [--req 1,8]
    [--heads 8,128]

Each timed call is chained ``n`` times inside one jit (output fed back as the
next query) and the per-call time is the slope between two chain lengths, so
host dispatch drops out.  Prints one ``RESULT`` line per configuration.
"""

import argparse
import functools
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_cases as C  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout  # noqa: E402
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla import (  # noqa: E402
    mla_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (  # noqa: E402
    mla_sliding_window_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (  # noqa: E402
    sparse_ragged_paged_attention,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (  # noqa: E402
    streamindex_topk,
)
from vllm_torchtpu.kernels.deepseek_v4_megakernel import (  # noqa: E402
    attn_csa,
    attn_ref,
    attn_swa_hca,
)

HBM_BPS = 3.10e12
CSA_KW = {}
BF16_FLOPS = 734.6e12


def chain_time(step, x0, args, ns=(4, 24), reps=5):
    """Median/min/max seconds per `step` call (slope over chain lengths)."""
    def chained(n, x, *a):
        return lax.fori_loop(0, n, lambda i, x: step(x, *a), x)

    fns = {n: jax.jit(functools.partial(chained, n)) for n in ns}
    for f in fns.values():
        jax.block_until_ready(f(x0, *args))
    per = []
    for _ in range(reps):
        t = {}
        for n, f in fns.items():
            s = time.perf_counter()
            jax.block_until_ready(f(x0, *args))
            t[n] = time.perf_counter() - s
        per.append((t[ns[1]] - t[ns[0]]) / (ns[1] - ns[0]))
    return float(np.median(per)), float(np.min(per)), float(np.max(per))


def chain_time_pure(fn, args, **kw):
    """`chain_time` for a call with no state: each call is tied to the previous
    one's output through an optimization barrier, so none can be hoisted."""
    def step(prev, *a):
        a, _ = lax.optimization_barrier((a, prev))
        return fn(*a)

    return chain_time(step, jax.jit(fn)(*args), args, **kw)


def rel_err(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / np.abs(b).max()), float(np.abs(a - b).mean())


# ------------------------------------------------------------------ stock path


def stock_hca(q2, sw_cache, new_kv2, hca_u8, swa_table, hca_table, seq_lens, sinks,
              *, sm_scale):
    """`_attention_hca` (deepseek_v4_attention_op.py) with verify_q_len=7."""
    R = seq_lens.shape[0]
    cu = jnp.arange(R + 1, dtype=jnp.int32) * C.Q_LEN
    dist = jnp.array([0, R, R], jnp.int32)
    pos = (seq_lens[:, None] - C.Q_LEN + jnp.arange(C.Q_LEN)[None, :]).reshape(-1)
    swa_out, new_cache, swa_l, swa_m = mla_sliding_window_ragged_paged_attention(
        q=q2, new_kv=new_kv2, cache_kv=sw_cache, kv_lens=seq_lens,
        page_indices=swa_table.reshape(-1), cu_q_lens=cu, distribution=dist,
        attention_sinks=sinks, sm_scale=sm_scale, sliding_window=C.WINDOW,
        logical_page_size=C.SWA_PAGE, num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 32, 32), q_compute_block_size=4,
        unnormalized_output=True, chunk_prefill_size=C.Q_LEN,
    )
    out = mla_ragged_paged_attention(
        q=q2, cache_kv=hca_u8, kv_lens=seq_lens // C.HCA_RATIO,
        kv_lens_to_attend=((pos + 1) // C.HCA_RATIO).astype(jnp.int32),
        page_indices=hca_table.reshape(-1), cu_q_lens=cu, distribution=dist,
        attention_sinks=sinks, swa_accumution=swa_out, swa_l=swa_l, swa_m=swa_m,
        sm_scale=sm_scale, num_kv_pages_per_block=(32, 64, 64),
        num_queries_per_block=(1, 32, 32), chunk_prefill_size=C.Q_LEN,
    )
    return out, new_cache


# ------------------------------------------------------------------ HCA bench


def hca_floor(R, H, ctx):
    n_c = ctx // C.HCA_RATIO
    kv = R * ((C.WINDOW - 1) + n_c) * 1024
    qo = 2 * R * C.Q_LEN * H * 512 * 2
    flops = R * C.Q_LEN * H * (C.WINDOW - 1 + C.Q_LEN + n_c) * 512 * 4
    return (kv + qo) / HBM_BPS, flops / BF16_FLOPS, kv, qo


def bench_hca(R, H, ctx, check, rng):
    seq = np.full(R, ctx, np.int32) - rng.integers(0, 64, R).astype(np.int32)
    case = C.swa_hca_case(rng, seq, H, max_len=ctx)
    args = (case.new_kv, case.swa_words, case.swa_table, case.hca_words, case.hca_table,
            case.seq_lens, case.sinks)

    def mine(q, *a):
        return attn_swa_hca.swa_hca_attention(q, *a, sm_scale=case.sm_scale)

    hca_u8 = csa_cache_layout.words_to_slabs(case.hca_words)  # [N, 16, 4, 128]
    q2 = case.q.reshape(R * C.Q_LEN, H, 512)
    nkv2 = case.new_kv.reshape(R * C.Q_LEN, 512)

    def stock(state, nkv2, hca_u8, swa_table, hca_table, seq_lens, sinks):
        q2, cache = state
        return stock_hca(q2, cache, nkv2, hca_u8, swa_table, hca_table, seq_lens, sinks,
                         sm_scale=case.sm_scale)

    sargs = (nkv2, hca_u8, case.swa_table, case.hca_table, case.seq_lens, case.sinks)
    fl_hbm, fl_mxu, kvb, qob = hca_floor(R, H, ctx)
    t_m = chain_time(mine, case.q, args)
    t_s = chain_time(stock, (q2, case.swa_words), sargs)

    def mine_w(state, nkv, swt, hw, ht, sl, sk):
        q, cache = state
        return attn_swa_hca.swa_hca_attention(q, nkv, cache, swt, hw, ht, sl, sk,
                                              sm_scale=case.sm_scale, write_swa=True)

    t_w = chain_time(mine_w, (case.q, case.swa_words), (case.new_kv,) + args[2:])
    extra = ""
    if check:
        out_m = jax.jit(mine)(case.q, *args)
        out_s, _ = jax.jit(stock_hca, static_argnames="sm_scale")(
            q2, case.swa_words, nkv2, hca_u8, case.swa_table, case.hca_table,
            case.seq_lens, case.sinks, sm_scale=case.sm_scale)
        ref = attn_ref.swa_hca_reference(
            case.q, case.new_kv, case.swa_words, case.swa_table, case.hca_words,
            case.hca_table, case.seq_lens, case.sinks, sm_scale=case.sm_scale)
        em, es = rel_err(out_m, ref), rel_err(out_s.reshape(ref.shape), ref)
        ems = rel_err(out_m, out_s.reshape(ref.shape))
        _, cache_s = jax.jit(stock_hca, static_argnames="sm_scale")(
            q2, case.swa_words, nkv2, hca_u8, case.swa_table, case.hca_table,
            case.seq_lens, case.sinks, sm_scale=case.sm_scale)
        _, cache_m = jax.jit(functools.partial(
            attn_swa_hca.swa_hca_attention, sm_scale=case.sm_scale, write_swa=True))(
            case.q, *args)
        same = bool(np.array_equal(np.asarray(cache_m), np.asarray(cache_s)))
        changed = int(np.sum(np.any(np.asarray(cache_m) != np.asarray(case.swa_words), -1)))
        extra = (f" err_mine_ref={em[0]:.2e}/{em[1]:.2e} err_stock_ref={es[0]:.2e}/{es[1]:.2e}"
                 f" err_mine_stock={ems[0]:.2e}/{ems[1]:.2e}"
                 f" swa_write_bitexact_vs_stock={same} rows_changed={changed}")
    print(f"RESULT kind=swa_hca R={R} H={H} ctx={ctx} "
          f"mine_us={t_m[0]*1e6:.1f} [{t_m[1]*1e6:.1f},{t_m[2]*1e6:.1f}] "
          f"mine_with_swa_write_us={t_w[0]*1e6:.1f} "
          f"stock_us={t_s[0]*1e6:.1f} [{t_s[1]*1e6:.1f},{t_s[2]*1e6:.1f}] "
          f"floor_hbm_us={fl_hbm*1e6:.2f} floor_mxu_us={fl_mxu*1e6:.2f} "
          f"kv_MB={kvb/1e6:.2f} qo_MB={qob/1e6:.2f}{extra}", flush=True)


# ------------------------------------------------------------------ CSA bench


def stock_indexer(q2, w2, idx_u8, seq_lens, table, *, return_scores=False, strict_causal=False):
    """`_indexer_jax` after rope_quant (deepseek_v4_indexer.py), SAL layout."""
    R = seq_lens.shape[0]
    cu = jnp.arange(R + 1, dtype=jnp.int32) * C.Q_LEN
    return streamindex_topk(
        q=q2, indexer_weights=w2, cache_kv=idx_u8, seq_lens=seq_lens,
        page_indices=table.reshape(-1), cu_q_lens=cu, distribution=jnp.array([0, R, R], jnp.int32),
        k=1024, compression_ratio=C.CSA_RATIO, num_kv_pages_per_block=(32, 32, 16),
        num_queries_per_block=(1, 128, 128), decode_req_batch_size=1, uniform_q_len=C.Q_LEN,
        kv_layout="seq_along_lane", return_scores=return_scores, strict_causal=strict_causal)


def stock_csa(q2, sw_cache, new_kv2, nope, rope, topk2, swa_table, csa_table, seq_lens, sinks,
              *, sm_scale):
    """`_attention_csa` (deepseek_v4_attention_op.py) with verify_q_len=7."""
    R = seq_lens.shape[0]
    cu = jnp.arange(R + 1, dtype=jnp.int32) * C.Q_LEN
    dist = jnp.array([0, R, R], jnp.int32)
    swa_out, new_cache, swa_l, swa_m = mla_sliding_window_ragged_paged_attention(
        q=q2, new_kv=new_kv2, cache_kv=sw_cache, kv_lens=seq_lens,
        page_indices=swa_table.reshape(-1), cu_q_lens=cu, distribution=dist,
        attention_sinks=sinks, sm_scale=sm_scale, sliding_window=C.WINDOW,
        logical_page_size=C.SWA_PAGE, num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 32, 32), q_compute_block_size=4,
        unnormalized_output=True, chunk_prefill_size=C.Q_LEN,
    )
    out = sparse_ragged_paged_attention(
        q=q2, cache_kv_nope=nope, cache_kv_rope=rope, topk_indices=topk2,
        page_indices=csa_table.reshape(-1), cu_q_lens=cu, distribution=dist,
        attention_sinks=sinks, swa_accumution=swa_out, swa_l=swa_l, swa_m=swa_m,
        sm_scale=sm_scale, gather_and_attention_chunk_size=32 if q2.shape[0] <= 128 else 128,
        attention_kernel_batch_size=8,
    )
    return out, new_cache


def overlapping_topk(rng, seq_lens, share):
    """Synthetic per-row top-1024 lists: each row keeps a request-wide base set
    and swaps a (1 - share) fraction for its own random rows."""
    R = len(seq_lens)
    out = np.full((R, C.Q_LEN, 1024), -1, np.int32)
    for r, s in enumerate(seq_lens):
        n = int(s) // C.CSA_RATIO - C.Q_LEN  # visible to every row
        base = rng.choice(n, 1024, replace=False)
        for t in range(C.Q_LEN):
            sel = base.copy()
            k = int(round((1 - share) * 1024))
            if k:
                swap = rng.choice(1024, k, replace=False)
                pool = np.setdiff1d(np.arange(n), sel)
                sel[swap] = rng.choice(pool, k, replace=False)
            out[r, t] = np.sort(sel)
    return jnp.asarray(out)


def union_of(topk):
    """[R, T, k] lists -> (packed union, lengths) as `topk_union_placeholder`."""
    R, T, k = topk.shape
    n = int(topk.max()) + 1
    bit = (1 << jnp.arange(T, dtype=jnp.int32))[None, :, None]
    rows = jnp.broadcast_to(jnp.arange(R)[:, None, None], topk.shape)
    bm = jnp.zeros((R, n), jnp.int32).at[rows, jnp.maximum(topk, 0)].add(
        jnp.where(topk >= 0, bit, 0))

    def one(b):
        (pos,) = jnp.nonzero(b, size=T * k, fill_value=0)
        return (pos << attn_csa.UNION_BITS) | b[pos], jnp.sum(b != 0)

    u, l = jax.vmap(one)(bm)
    return u.astype(jnp.int32), l.astype(jnp.int32)


def bench_csa(R, H, ctx, check, rng, share, gathers=("paged",), do_indexer=True):
    seq = np.full(R, ctx, np.int32) - rng.integers(0, 64, R).astype(np.int32)
    case = C.csa_case(rng, seq, H, max_len=ctx)
    w = case.win
    T = R * C.Q_LEN
    n = ctx // C.CSA_RATIO
    idx_u8 = csa_cache_layout.words_to_slabs(case.idx_words)  # [N, 64, 4, 256]
    q2 = case.q_idx.reshape(T, C.IDX_HEADS, C.IDX_DIM)
    w2 = case.weights.reshape(T, C.IDX_HEADS)
    tag = f"R={R} H={H} ctx={ctx}"

    # Indexer (scores; top-k for stock).
    def mine_idx(q, wt, words, table, sl):
        return attn_csa.indexer_scores(q, wt, words, table, sl, max_len=ctx)

    def stock_idx(q, wt, u8, sl, table):
        return stock_indexer(q, wt, u8, sl, table)

    if H == 8 and do_indexer:  # the indexer does not depend on H
        t_mi = chain_time_pure(mine_idx, (case.q_idx, case.weights, case.idx_words,
                                          case.csa_table, w.seq_lens))
        t_si = chain_time_pure(stock_idx, (q2, w2, idx_u8, w.seq_lens, case.csa_table))
        sc, rl = jax.jit(mine_idx)(case.q_idx, case.weights, case.idx_words, case.csa_table,
                                   w.seq_lens)
        t_tk = chain_time_pure(lambda s, l: attn_csa.topk_union_placeholder(s, l, 1024)[:2],
                               (sc, rl))
        extra = ""
        if check:
            s_idx, s_sc = jax.jit(functools.partial(stock_indexer, return_scores=True))(
                q2, w2, idx_u8, w.seq_lens, case.csa_table)
            m_idx = attn_csa.topk_union_placeholder(sc, rl, 1024)[2].reshape(T, 1024)
            s_idx, s_sc, m_idx = map(np.asarray, (s_idx, s_sc, m_idx))
            if s_sc.dtype == np.int32:  # f32 bits
                s_sc = s_sc.view(np.float32)
            same = float(np.mean(np.all(s_idx == m_idx, axis=1)))
            sel = s_idx >= 0
            mine_at = np.take_along_axis(np.asarray(sc).reshape(T, -1), np.maximum(s_idx, 0), 1)
            bitexact = float(np.mean(mine_at[sel] == s_sc[sel]))
            relm = float(np.abs(mine_at[sel] - s_sc[sel]).max() / np.abs(s_sc[sel]).max())
            extra = (f" rows_topk_identical={same:.3f} score_bitexact_frac={bitexact:.4f}"
                     f" score_max_rel={relm:.2e}")
        keys_b = R * n * 132
        print(f"RESULT kind=indexer {tag} mine_us={t_mi[0]*1e6:.1f} [{t_mi[1]*1e6:.1f},"
              f"{t_mi[2]*1e6:.1f}] stock_us(scores+SC topk)={t_si[0]*1e6:.1f} [{t_si[1]*1e6:.1f},"
              f"{t_si[2]*1e6:.1f}] xla_topk_union_us={t_tk[0]*1e6:.1f}"
              f" floor_hbm_us={keys_b/HBM_BPS*1e6:.2f}"
              f" floor_mxu_us={R*7*64*n*128*2/1.295e15*1e6:.2f}{extra}", flush=True)

    # Sparse attention on the same lists for both: the stock indexer's top-k
    # (random data: little overlap) or a synthetic list with row overlap `share`.
    if share is None:
        topk = jax.jit(stock_idx)(q2, w2, idx_u8, w.seq_lens, case.csa_table).reshape(R, 7, 1024)
    else:
        topk = overlapping_topk(rng, seq, share)
    union, ulen = union_of(topk)
    nkv2 = w.new_kv.reshape(T, 512)
    qh2 = w.q.reshape(T, H, 512)

    margs = (w.new_kv, w.swa_words, w.swa_table, case.nope_words, case.rope_words,
             case.csa_table, union, ulen, w.seq_lens, w.sinks)

    def stock(state, nkv2, nope, rope, topk2, swa_table, csa_table, sl, sinks):
        q, cache = state
        return stock_csa(q, cache, nkv2, nope, rope, topk2, swa_table, csa_table, sl, sinks,
                         sm_scale=w.sm_scale)

    sargs = (nkv2, case.nope_words, case.rope_words, topk.reshape(T, 1024), w.swa_table,
             case.csa_table, w.seq_lens, w.sinks)
    t_s = chain_time(stock, (qh2, w.swa_words), sargs)
    U = np.asarray(ulen)
    for gather in gathers:
        bench_csa_mine(case, topk, margs, t_s, U, tag, share, gather, check, R, H, T, qh2, nkv2)


def bench_csa_mine(case, topk, margs, t_s, U, tag, share, gather, check, R, H, T, qh2, nkv2):
    w = case.win

    kw = dict(CSA_KW)
    g = gather
    if gather.startswith("hybrid:"):
        g, kw["k_dense"] = "hybrid", int(gather.split(":")[1])

    def mine_attn(q, *a):
        return attn_csa.csa_attention(q, *a, sm_scale=w.sm_scale, gather=g, **kw)

    t_m = chain_time(mine_attn, w.q, margs)
    extra = ""
    if check:
        out_m = jax.jit(mine_attn)(w.q, *margs)
        out_s, _ = jax.jit(stock_csa, static_argnames="sm_scale")(
            qh2, w.swa_words, nkv2, case.nope_words, case.rope_words, topk.reshape(T, 1024),
            w.swa_table, case.csa_table, w.seq_lens, w.sinks, sm_scale=w.sm_scale)
        r0 = slice(0, 1)  # reference on request 0 (per-row loops)
        ref = attn_ref.csa_reference(
            w.q[r0], w.new_kv[r0], w.swa_words, w.swa_table[r0], case.nope_words,
            case.rope_words, case.csa_table[r0], topk[r0], w.seq_lens[r0], w.sinks,
            sm_scale=w.sm_scale)
        em = rel_err(out_m[r0], ref)
        es = rel_err(out_s.reshape(w.q.shape)[r0], ref)
        ems = rel_err(out_m, out_s.reshape(w.q.shape))
        extra = (f" err_mine_ref={em[0]:.2e}/{em[1]:.2e} err_stock_ref={es[0]:.2e}/{es[1]:.2e}"
                 f" err_mine_stock={ems[0]:.2e}/{ems[1]:.2e}")
    gather_b = int(U.sum()) * 640
    qo = 2 * R * 7 * H * 512 * 2
    swa_b = R * 127 * 1024
    flops = R * 7 * H * (135 + 1024) * 512 * 4
    print(f"RESULT kind=csa_attn {tag} gather={gather} kw={CSA_KW} share={share} union_mean={U.mean():.0f} "
          f"mine_us={t_m[0]*1e6:.1f} [{t_m[1]*1e6:.1f},{t_m[2]*1e6:.1f}] "
          f"stock_us={t_s[0]*1e6:.1f} [{t_s[1]*1e6:.1f},{t_s[2]*1e6:.1f}] "
          f"floor_hbm_us={(gather_b + qo + swa_b)/HBM_BPS*1e6:.2f} "
          f"floor_mxu_us(per-row)={flops/BF16_FLOPS*1e6:.2f} "
          f"descriptors={2*int(U.sum())} dense_MB={R*int(case.csa_table.shape[1])*256*640/1e6:.1f}"
          f"{extra}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["hca", "csa"])
    ap.add_argument("--share", default="none")
    ap.add_argument("--gather", default="paged")
    ap.add_argument("--indexer", default="1")
    ap.add_argument("--granule", type=int, default=128)
    ap.add_argument("--blk", type=int, default=8)
    ap.add_argument("--kdense", default="4")
    ap.add_argument("--ctx", default="32768,131072")
    ap.add_argument("--req", default="1,8")
    ap.add_argument("--heads", default="8,128")
    ap.add_argument("--check", default="1")
    a = ap.parse_args()
    CSA_KW.update(granule=a.granule, blk_pages=a.blk)
    print(jax.devices()[0], flush=True)
    rng = np.random.default_rng(0)
    for ctx in map(int, a.ctx.split(",")):
        for R in map(int, a.req.split(",")):
            for H in map(int, a.heads.split(",")):
                if a.kind == "hca":
                    bench_hca(R, H, ctx, a.check == "1", rng)
                    continue
                for sh in a.share.split(","):
                    bench_csa(R, H, ctx, a.check == "1", rng, None if sh == "none" else float(sh),
                              gathers=[g2 for g in a.gather.split(",") for g2 in (
                                  [f"hybrid:{k}" for k in a.kdense.split(",")]
                                  if g == "hybrid" else [g])],
                              do_indexer=a.indexer == "1")


if __name__ == "__main__":
    main()
