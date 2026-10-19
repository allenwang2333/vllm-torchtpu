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
"""Streaming decode kernel for the DCP CACHE_ONLY attention pass (TPU_RPA_DECODE_V2).

Scope: decode steps whose sequences all have q_len <= decode_query_size, a
SEQ_ALONG_LANE ("nhs") KV cache with one KV head per device, per-tensor KV
scales, CACHE_ONLY scope with context parallelism (interleave = page size), and
an LSE output. That is the DCP decode leg's context pass; every other case keeps
the batched RPA kernel.

Why a separate kernel: the batched RPA decode kernel is compute-bound, not
bandwidth-bound. On v7x at 64 sequences x 80K context its DMA-only floor is
~206 us but its compute-only floor is ~478 us per call, about twice what the
attention math itself costs: every 4096-token block pays for causal and range
masks, the new-token stitch and a per-step output write, and its eight lanes
chain one online-softmax state, so their matmuls and softmaxes serialize.

Here the work is a flat list of (sequence, KV block) items, built in the kernel
prologue:

- one item = the decode_query_size * heads query rows of one sequence (64 for
  Qwen3.5 on the decode leg) against `bkv` cached tokens of its KV head;
- K/V stream HBM -> VMEM into per-item slots, one DMA per run of consecutive
  pages (vLLM's 640-token blocks are 5 consecutive 128-token kernel pages), from
  a run-length table built outside the kernel with elementwise ops;
- each loop iteration computes `group` items as independent softmax partials
  (local max, sum, P V) in one basic block, so the scheduler overlaps one item's
  matmuls with another's softmax; the partials are then merged in order into a
  running state that is reset at each sequence's first block, and the output
  and LSE are written at its last block;
- a partial last block has its K/V tail zeroed in VMEM (it may hold any bytes,
  fp8 NaN included) and its scores masked with a -inf bias row.

VMEM layout of a slot: [page, K/V, Dp, pack, page_size] fp8, i.e. pages stay
contiguous so a run of pages is one DMA; K and V are reassembled as [D, bkv] in
vregs by a lane-aligned concatenation, which moves no data.

Numerics: S = Q K^T in f32 on the MXU (bf16 Q, fp8 K widened to bf16), softmax
in f32, P cast to bf16 for P V with f32 accumulation, f32 LSE. The batched RPA
kernel rounds S to bf16 before the softmax and keeps m / l / acc and the LSE in
bf16, so this kernel is not bit-identical to it; it is closer to a float64
reference (scripts/attn/test_decode_v2.py).
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32
BF16 = jnp.bfloat16
_NT = (((1,), (1,)), ((), ()))
_LANES = 128
_P_SCALE = 256.0
_LN_P_SCALE = math.log(_P_SCALE)
_SPLIT = 16.0

_INTERPRET = False


def _interpret_kwargs():
    return {"interpret": pltpu.InterpretParams()} if _INTERPRET else {}


def _cdiv(a, b):
    return (a + b - 1) // b


def _local_cache_len(global_cache_len, cp_size, cp_rank, page):
    """Cache tokens of a sequence that this CP rank holds (interleave = page)."""
    super_page = cp_size * page
    full = (global_cache_len // super_page) * page
    rem = jnp.clip((global_cache_len % super_page) - cp_rank * page, 0, page)
    return full + rem


def build_run_lengths(page_indices: jax.Array, num_seqs: int, ppb: int) -> jax.Array:
    """run_len[s * pps + p]: length of the run of consecutive physical pages
    starting at page p of sequence s (0 where p continues a run). Runs never
    cross a `ppb`-page block boundary."""
    pps = page_indices.shape[0] // num_seqs
    pg = page_indices.reshape(num_seqs, pps)
    p = jnp.arange(pps, dtype=jnp.int32)
    cont = (pg[:, 1:] == pg[:, :-1] + 1) & ((p[1:] % ppb) != 0)[None, :]
    start = jnp.concatenate(
        [jnp.ones((num_seqs, 1), jnp.bool_), jnp.logical_not(cont)], axis=1
    )
    idx = jnp.where(start, p[None, :], pps)
    after = jnp.concatenate(
        [idx[:, 1:], jnp.full((num_seqs, 1), pps, jnp.int32)], axis=1
    )
    nxt = lax.cummin(after, axis=1, reverse=True)
    return jnp.where(start, nxt - p[None, :], 0).reshape(-1).astype(jnp.int32)


RUNS_PER_BLOCK = 8  # vLLM 640-token blocks give at most 8 runs per 32-page block


def build_block_runs(page_indices: jax.Array, num_seqs: int, ppb: int):
    """Per (sequence, block): the first RUNS_PER_BLOCK runs of consecutive pages.

    Returns (src [N*nb*RUNS], dst_len [N*nb*RUNS] = dst_page << 8 | len,
    covered [N*nb] = pages covered by those runs (ppb when every run fits)).
    Entries past the sequence end are garbage; the kernel clips runs to it.
    """
    pps = page_indices.shape[0] // num_seqs
    nb = pps // ppb
    pg = page_indices.reshape(num_seqs, nb, ppb)
    p = jnp.arange(ppb, dtype=jnp.int32)
    start = jnp.concatenate(
        [jnp.ones((num_seqs, nb, 1), jnp.bool_), pg[..., 1:] != pg[..., :-1] + 1],
        axis=2,
    )
    rid = jnp.cumsum(start.astype(jnp.int32), axis=2) - 1  # [N, nb, ppb]
    r = jnp.arange(RUNS_PER_BLOCK, dtype=jnp.int32)
    in_run = rid[..., None, :] == r[:, None]  # [N, nb, R, ppb]
    first = in_run & start[..., None, :]
    src = jnp.sum(jnp.where(first, pg[..., None, :], 0), axis=-1)
    dst = jnp.sum(jnp.where(first, p, 0), axis=-1)
    ln = jnp.sum(in_run.astype(jnp.int32), axis=-1)
    nruns = rid[..., -1] + 1
    covered = jnp.where(nruns > RUNS_PER_BLOCK, dst[..., -1] + ln[..., -1], ppb)
    dst_len = (dst << 8) | ln
    return (
        src.reshape(-1).astype(jnp.int32),
        dst_len.reshape(-1).astype(jnp.int32),
        covered.reshape(-1).astype(jnp.int32),
    )


def _kernel(
    # scalar prefetch (SMEM)
    lens_ref,  # [N] rank-local cache length (0: no work)
    pages_ref,  # [N * pps] rank-local page table
    runlen_ref,  # [N * pps] run lengths ("loop" issue mode)
    rsrc_ref,  # [N * nb * RUNS] run source pages ("runs" issue mode)
    rdl_ref,  # [N * nb * RUNS] run dst << 8 | len
    rcov_ref,  # [N * nb] pages covered by the runs
    # inputs
    q_ref,  # VMEM [N, R, D] bf16, pre-scaled
    kv_hbm,  # HBM [P, 2, Dp, pack, page] fp8
    # outputs
    o_ref,  # VMEM [N, R, D]
    lse_ref,  # VMEM [N, R, 128] f32
    # scratch
    kv_buf,  # VMEM [S, ppb, 2, Dp, pack, page]
    sem,  # DMA sems [S]
    item_seq_sc,  # SMEM [max_items]
    item_blk_sc,  # SMEM [max_items]
    m_sc,  # [R, 128] f32
    l_sc,  # [R, 128] f32
    acc_sc,  # [R, D] f32
    *maybe_dbg_kv,  # measurement only ("fp8native"): [S, 2, D, bkv] fp8
    num_seqs: int,
    bkv: int,
    page: int,
    pages_per_seq: int,
    group: int,
    depth: int,
    issue_mode: str,
    pv_mode: str,
    chunk: int,
    v_scale: float | None,
    out_dtype,
    dbg: str,
):
    flags = set(dbg.split(",")) if dbg else set()
    dbg_kv = maybe_dbg_kv[0] if maybe_dbg_kv else None
    ppb = bkv // page  # pages per block
    nb = pages_per_seq // ppb  # blocks per sequence
    num_slots = group * depth
    pack_dim = kv_hbm.shape[2]  # head_dim // packing (u32 rows of K or V)
    pack = kv_hbm.shape[3]
    rows_per_page = 2 * pack_dim * page // _LANES  # u32 [., 128] rows of one page
    kv_dtype = kv_hbm.dtype
    dma = "nodma" not in flags

    # Prologue: the flat (sequence, block) work list.

    def add_seq(s, cnt):
        nbs = _cdiv(jnp.maximum(lens_ref[s], 0), bkv)

        def add_blk(j, c):
            item_seq_sc[c] = s
            item_blk_sc[c] = j
            return c + 1

        return lax.fori_loop(0, nbs, add_blk, cnt)

    n_items = lax.fori_loop(0, num_seqs, add_seq, jnp.int32(0))

    def geom(it):
        """(seq, block, valid tokens, live) of item `it` (clamped when past the end)."""
        live = it < n_items
        itc = jnp.minimum(it, jnp.maximum(n_items - 1, 0))
        s = item_seq_sc[itc]
        j = item_blk_sc[itc]
        valid = jnp.where(live, jnp.minimum(lens_ref[s] - j * bkv, bkv), 0)
        return s, j, valid, live

    def copy_pages(src, dst, n, slot):
        pltpu.make_async_copy(
            kv_hbm.at[pl.ds(src, n)], kv_buf.at[slot, pl.ds(dst, n)], sem.at[slot]
        ).start()

    def issue_loop(it, slot):
        """Top-of-iteration issue: one DMA per run (while loop over the runs)."""
        s, j, valid, live = geom(it)
        npg = _cdiv(valid, page)
        base = s * pages_per_seq + j * ppb

        def body(p):
            ln = jnp.maximum(jnp.minimum(runlen_ref[base + p], npg - p), 1)
            copy_pages(pages_ref[base + p], p, ln, slot)
            return p + ln

        lax.while_loop(lambda p: p < npg, body, jnp.int32(0))

    def issue_inblock(it, slot):
        """Branch-free issue (sizes 0 when not live), for the compute block."""
        s, j, valid, live = geom(it)
        npg = _cdiv(valid, page)
        if issue_mode == "pages":
            base = s * pages_per_seq + j * ppb
            for p in range(ppb):
                n = jnp.where(p < npg, 1, 0)
                copy_pages(
                    pages_ref[base + jnp.minimum(p, jnp.maximum(npg - 1, 0))],
                    p,
                    n,
                    slot,
                )
        else:  # "runs"
            rb = (s * nb + j) * RUNS_PER_BLOCK
            for r in range(RUNS_PER_BLOCK):
                dl = rdl_ref[rb + r]
                d = lax.shift_right_logical(dl, 8)
                n = jnp.clip(jnp.minimum(dl & 0xFF, npg - d), 0, ppb)
                copy_pages(rsrc_ref[rb + r], jnp.minimum(d, ppb - 1), n, slot)

    def issue_fallback(it, slot):
        """Pages past the first RUNS_PER_BLOCK runs of a fragmented block."""
        s, j, valid, live = geom(it)
        npg = _cdiv(valid, page)
        cov = rcov_ref[s * nb + j]
        base = s * pages_per_seq + j * ppb

        @pl.when(jnp.logical_and(live, cov < npg))
        def _():
            @pl.loop(cov, npg)
            def _(p):
                copy_pages(pages_ref[base + p], p, 1, slot)

    def wait(it, slot):
        _, _, valid, live = geom(it)
        npg = _cdiv(valid, page)
        if _INTERPRET:
            view = kv_buf.at[slot, pl.ds(0, npg)]
            pltpu.make_async_copy(view, view, sem.at[slot]).wait()
            return
        rows = npg * rows_per_page
        flat = kv_buf.at[slot].bitcast(jnp.uint32).reshape(-1, _LANES)
        pltpu.make_async_copy(
            flat.at[pl.ds(0, rows)], flat.at[pl.ds(0, rows)], sem.at[slot]
        ).wait()

    def page_ref(slot, p, h):
        """u32 [Dp, page] view of K (h=0) or V (h=1) of page p in a slot."""
        return kv_buf.at[slot, p, h].bitcast(jnp.uint32).reshape(pack_dim, page)

    def zero_unfetched(slot, valid):
        """Zero K and V of the pages past the last fetched one (they hold stale or
        uninitialized bytes, fp8 NaN included)."""
        npg = _cdiv(valid, page)

        @pl.loop(npg, ppb)
        def _(p):
            for h in range(2):
                if _INTERPRET:
                    kv_buf[slot, p, h] = jnp.zeros(kv_buf.shape[3:], kv_dtype)
                else:
                    page_ref(slot, p, h)[...] = jnp.zeros((pack_dim, page), jnp.uint32)

    def zero_page_tail(slot, valid):
        """Zero K and V of the last fetched page at tokens >= valid."""
        p_last = _cdiv(valid, page) - 1
        rem = valid - p_last * page
        for h in range(2):
            if _INTERPRET:
                x = kv_buf[slot, p_last, h]
                li = lax.broadcasted_iota(jnp.int32, x.shape, 2)
                kv_buf[slot, p_last, h] = jnp.where(li < rem, x, jnp.zeros_like(x))
            else:
                lane = lax.broadcasted_iota(jnp.int32, (pack_dim, page), 1)
                r = page_ref(slot, p_last, h)
                r[...] = jnp.where(
                    lane < rem, r[...], jnp.zeros((pack_dim, page), jnp.uint32)
                )

    def load_kv(slot):
        """K and V of a slot as fp8 [D, bkv] (dim = dp * pack + j)."""
        if "fp8native" in flags:  # measurement only: KV already as fp8 [D, bkv]
            assert dbg_kv is not None
            return dbg_kv[slot, 0], dbg_kv[slot, 1]
        if _INTERPRET:
            kk = jnp.concatenate(
                [kv_buf[slot, p, 0].reshape(pack_dim * pack, page) for p in range(ppb)],
                axis=1,
            )
            vv = jnp.concatenate(
                [kv_buf[slot, p, 1].reshape(pack_dim * pack, page) for p in range(ppb)],
                axis=1,
            )
            return kk, vv
        k_u32 = jnp.concatenate([page_ref(slot, p, 0)[...] for p in range(ppb)], axis=1)
        v_u32 = jnp.concatenate([page_ref(slot, p, 1)[...] for p in range(ppb)], axis=1)
        return pltpu.bitcast(k_u32, kv_dtype), pltpu.bitcast(v_u32, kv_dtype)

    def partial(it, slot):
        if chunk >= bkv:
            return partial_block(it, slot)
        # Online softmax over `chunk`-token pieces of the block (smaller score tiles).
        s, _, valid, _ = geom(it)
        q = q_ref[s]
        m = lsum = acc = None
        cp = chunk // page
        for c in range(bkv // chunk):
            k_u32 = jnp.concatenate(
                [page_ref(slot, c * cp + p, 0)[...] for p in range(cp)], axis=1
            )
            v_u32 = jnp.concatenate(
                [page_ref(slot, c * cp + p, 1)[...] for p in range(cp)], axis=1
            )
            k = pltpu.bitcast(k_u32, kv_dtype)
            v = pltpu.bitcast(v_u32, kv_dtype)
            sc = lax.dot(q, k, preferred_element_type=F32)
            lane = lax.broadcasted_iota(jnp.int32, (1, chunk), 1) + c * chunk
            sc = sc + jnp.where(lane < valid, 0.0, -jnp.inf).astype(F32)
            m_c = jnp.max(sc, axis=-1, keepdims=True)
            m_new = m_c if m is None else jnp.maximum(m, m_c)
            m_safe = jnp.where(m_new == -jnp.inf, 0.0, m_new)
            p_ = jnp.exp(sc - m_safe)
            l_c = jnp.sum(p_, axis=-1, keepdims=True)
            pv = lax.dot_general(p_.astype(BF16), v, _NT, preferred_element_type=F32)
            if m is None:
                lsum, acc = l_c, pv
            else:
                a = jnp.exp(jnp.where(m == -jnp.inf, 0.0, m) - m_safe)
                a = jnp.where(m == -jnp.inf, 0.0, a)
                lsum = a * lsum + l_c
                acc = a * acc + pv
            m = m_new
        return m, lsum, acc

    def partial_block(it, slot):
        """Independent softmax partial of one item: (m [R,1], l [R,1], acc [R,D])."""
        s, _, valid, _ = geom(it)
        q = q_ref[s]
        k, v = load_kv(slot)
        sc = lax.dot(q, k, preferred_element_type=F32)  # [R, bkv]
        if "nobias" not in flags:
            lane = lax.broadcasted_iota(jnp.int32, (1, bkv), 1)
            bias = jnp.where(lane < valid, 0.0, -jnp.inf).astype(F32)
            sc = sc + bias
        m = jnp.max(sc, axis=-1, keepdims=True)
        m_safe = jnp.where(m == -jnp.inf, 0.0, m)
        if pv_mode == "fp8x2":
            # P in [0, 1] scaled by 2^8 straight out of the exp, split into fp8
            # hi + lo / 16 (about bf16 precision); V stays fp8 on the MXU.
            pf = jnp.exp(sc - (m_safe - _LN_P_SCALE))
            lsum = jnp.sum(pf, axis=-1, keepdims=True) * (1.0 / _P_SCALE)
            ph = pf.astype(kv_dtype)
            plo = ((pf - ph.astype(F32)) * _SPLIT).astype(kv_dtype)
            pv2 = lax.dot_general(
                jnp.concatenate([ph, plo], axis=0), v, _NT, preferred_element_type=F32
            )
            rows = q.shape[0]
            acc = (pv2[:rows] + pv2[rows:] * (1.0 / _SPLIT)) * (1.0 / _P_SCALE)
            return m, lsum, acc
        p = jnp.exp(sc - m_safe)
        lsum = jnp.sum(p, axis=-1, keepdims=True)
        if "qkonly" in flags:
            return m, lsum, p[:, : acc_sc.shape[1]]
        acc = lax.dot_general(p.astype(BF16), v, _NT, preferred_element_type=F32)
        return m, lsum, acc

    def merge(it, part):
        s, j, valid, live = geom(it)
        m_b, l_b, acc_b = part
        if "nomerge" in flags:
            acc_sc[...] += acc_b
            l_sc[...] += l_b
            return
        is_first = j == 0
        is_last = (j + 1) * bkv >= lens_ref[s]

        @pl.when(live)
        def _():
            m_st = jnp.where(is_first, -jnp.inf, m_sc[...])  # [R, 128]
            l_st = jnp.where(is_first, 0.0, l_sc[...])
            acc_st = jnp.where(is_first, 0.0, acc_sc[...])
            m_new = jnp.maximum(m_st, m_b)
            a_st = jnp.exp(m_st - m_new)
            a_b = jnp.exp(m_b - m_new)
            l_new = a_st * l_st + a_b * l_b
            acc_new = a_st[:, :1] * acc_st + a_b[:, :1] * acc_b
            m_sc[...] = m_new
            l_sc[...] = l_new
            acc_sc[...] = acc_new

            @pl.when(is_last)
            def _():
                inv = 1.0 / l_new[:, :1]
                if v_scale is not None and v_scale != 1.0:
                    inv = inv * v_scale
                o_ref[s] = (acc_new * inv).astype(out_dtype)
                lse_ref[s] = m_new + jnp.log(l_new)

    def issue_top(it, slot):
        issue_loop(it, slot) if issue_mode == "loop" else issue_inblock(it, slot)
        if issue_mode == "runs":
            issue_fallback(it, slot)

    @pl.when(n_items > 0)
    def _run():
        if dma:
            for it in range(group * (depth - 1)):

                @pl.when(it < n_items)
                def _(it=it):
                    issue_top(it, it)

        @pl.loop(0, _cdiv(n_items, group))
        def _iter(k):
            base = k * group
            nxt = [base + group * (depth - 1) + g for g in range(group)]
            nslot = [lax.rem(x, num_slots) for x in nxt]
            if dma:
                for g in range(group):
                    wait(base + g, lax.rem(base + g, num_slots))
                if issue_mode == "loop":
                    for g in range(group):

                        @pl.when(nxt[g] < n_items)
                        def _(g=g):
                            issue_loop(nxt[g], nslot[g])

                elif issue_mode == "runs":
                    for g in range(group):
                        issue_fallback(nxt[g], nslot[g])

            for g in range(group):
                _, _, valid, live = geom(base + g)

                @pl.when(jnp.logical_and(live, valid % page != 0))
                def _(g=g, valid=valid):
                    zero_page_tail(lax.rem(base + g, num_slots), valid)

                @pl.when(jnp.logical_and(live, valid < bkv))
                def _(g=g, valid=valid):
                    zero_unfetched(lax.rem(base + g, num_slots), valid)

            # One basic block: the next items' DMA issue (branch-free, so the
            # scheduler can spread it over the compute) and this group's partials.
            if dma and issue_mode != "loop":
                for g in range(group):
                    issue_inblock(nxt[g], nslot[g])
            if "nocompute" in flags:
                return
            parts = [
                partial(base + g, lax.rem(base + g, num_slots)) for g in range(group)
            ]
            for g in range(group):
                merge(base + g, parts[g])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DecodeMeta:
    """Per-step metadata of decode_cache_pass. It depends only on the step's
    lengths and page tables, so every layer of a forward computes the same
    one and XLA CSE keeps a single copy (build it outside any lax.cond)."""

    local: jax.Array  # [N] rank-local cache tokens (0: no work)
    page_indices: jax.Array  # [N * pps] (pps padded to whole blocks)
    rsrc: jax.Array  # [N * nb * RUNS]
    rdl: jax.Array  # [N * nb * RUNS]
    rcov: jax.Array  # [N * nb]
    q_idx: jax.Array  # [N * Tq] token row of (sequence, query row)
    q_ok: jax.Array  # [N * Tq] that row exists (t < q_len)
    row: jax.Array  # [T] (sequence, query row) of each token row
    row_ok: jax.Array  # [T] token row is real and its sequence has cache


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_tokens",
        "cp_group_size",
        "decode_query_size",
        "page_size",
        "bkv",
    ),
)
def prepare(
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cp_rank: jax.Array,
    num_decode_seqs: jax.Array | None = None,
    *,
    num_tokens: int,
    cp_group_size: int,
    decode_query_size: int,
    page_size: int,
    bkv: int = 4096,
) -> DecodeMeta:
    N = kv_lens.shape[0]
    T, Tq, page = num_tokens, decode_query_size, page_size
    ppb = bkv // page
    page_indices = page_indices.reshape(-1)
    pps = page_indices.shape[0] // N
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    local = _local_cache_len(kv_lens - q_lens, cp_group_size, cp_rank[0], page).astype(
        jnp.int32
    )
    if num_decode_seqs is not None:
        # Sequences past the decode range (padding) may carry stale lengths and
        # page tables: give them no work.
        local = jnp.where(jnp.arange(N) < num_decode_seqs, local, 0)
    if pps % ppb:
        # Whole blocks of page-table entries (the padding is never fetched).
        pad = ppb - pps % ppb
        page_indices = jnp.pad(
            page_indices.reshape(N, pps), ((0, 0), (0, pad))
        ).reshape(-1)
        pps += pad
    local = jnp.minimum(local, pps * page)
    rsrc, rdl, rcov = build_block_runs(page_indices, N, ppb)
    t_idx = jnp.arange(Tq, dtype=jnp.int32)
    tok = cu_q_lens[:N, None] + t_idx[None, :]
    q_idx = jnp.clip(tok, 0, T - 1).reshape(-1).astype(jnp.int32)
    q_ok = (t_idx[None, :] < q_lens[:, None]).reshape(-1)
    r_idx = jnp.arange(T, dtype=jnp.int32)
    seq_of_tok = jnp.sum(
        (cu_q_lens[None, 1:] <= r_idx[:, None]).astype(jnp.int32), axis=1
    )
    seq_of_tok = jnp.minimum(seq_of_tok, N - 1)
    t_of_tok = jnp.clip(r_idx - cu_q_lens[seq_of_tok], 0, Tq - 1)
    row = (seq_of_tok * Tq + t_of_tok).astype(jnp.int32)
    row_ok = (r_idx < cu_q_lens[N]) & (local[seq_of_tok] > 0)
    return DecodeMeta(local, page_indices, rsrc, rdl, rcov, q_idx, q_ok, row, row_ok)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "k_scale",
        "v_scale",
        "decode_query_size",
        "bkv",
        "group",
        "depth",
        "pv_mode",
        "chunk",
        "dbg",
        "vmem_limit_bytes",
    ),
)
def run(
    queries: jax.Array,  # [T, H, D]
    kv_cache: jax.Array,  # [P, 2*KVH, Dp, pack, page] fp8 (SEQ_ALONG_LANE)
    meta: DecodeMeta,
    *,
    sm_scale: float,
    k_scale: float | None,
    v_scale: float | None,
    decode_query_size: int,
    bkv: int = 4096,
    group: int = 8,
    depth: int = 2,
    pv_mode: str = "bf16",
    chunk: int = 4096,
    dbg: str = "",
    vmem_limit_bytes: int = 60 << 20,
) -> tuple[jax.Array, jax.Array]:
    """The per-layer part of decode_cache_pass (see prepare)."""
    T, H, D = queries.shape
    N = meta.local.shape[0]
    P, kvh2, dp, pack, page = kv_cache.shape
    assert kvh2 == 2, "decode_v2 supports one KV head per device"
    assert bkv % page == 0
    ppb = bkv // page
    pages_per_seq = meta.page_indices.shape[0] // N
    Tq = decode_query_size
    R = Tq * H
    max_items = N * (pages_per_seq // ppb)
    # VMEM: K/V slots plus the resident q / out / lse blocks; halve the group
    # until it fits (large batch buckets).
    io_bytes = N * R * D * 2 * jnp.dtype(queries.dtype).itemsize + N * R * _LANES * 4
    while (
        group > 1
        and group * depth * bkv * dp * pack * 2 + io_bytes > (vmem_limit_bytes * 3) // 4
    ):
        group //= 2

    # Queries per sequence: row t * H + h of sequence s is token cu[s] + t, head h.
    scale = float(sm_scale) * (1.0 if k_scale is None else float(k_scale))
    q_seq = jnp.take(queries, meta.q_idx, axis=0)  # [N * Tq, H, D]
    q_seq = jnp.where(meta.q_ok[:, None, None], q_seq.astype(F32) * scale, 0.0)
    q_seq = q_seq.astype(queries.dtype).reshape(N, R, D)

    kernel = functools.partial(
        _kernel,
        num_seqs=N,
        bkv=bkv,
        page=page,
        pages_per_seq=pages_per_seq,
        group=group,
        depth=depth,
        issue_mode="runs",
        pv_mode=pv_mode,
        chunk=min(chunk, bkv),
        v_scale=v_scale,
        out_dtype=queries.dtype,
        dbg=dbg,
    )
    vmem = pltpu.VMEM
    num_slots = group * depth
    one = jnp.zeros((1,), jnp.int32)
    o_seq, lse_seq = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((N, R, D), queries.dtype),
            jax.ShapeDtypeStruct((N, R, _LANES), F32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=[
                pl.BlockSpec(memory_space=vmem),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=(
                pl.BlockSpec(memory_space=vmem),
                pl.BlockSpec(memory_space=vmem),
            ),
            scratch_shapes=[
                vmem((num_slots, ppb, 2, dp, pack, page), kv_cache.dtype),
                pltpu.SemaphoreType.DMA((num_slots,)),
                pltpu.SMEM((max_items,), jnp.int32),
                pltpu.SMEM((max_items,), jnp.int32),
                vmem((R, _LANES), F32),
                vmem((R, _LANES), F32),
                vmem((R, D), F32),
            ]
            + (
                [vmem((num_slots, 2, dp * pack, bkv), kv_cache.dtype)]
                if "fp8native" in dbg
                else []
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=f"RPAd2-{pv_mode}-p{page}-k{bkv}-g{group}-d{depth}",
        **_interpret_kwargs(),
    )(
        meta.local,
        meta.page_indices,
        one,
        meta.rsrc,
        meta.rdl,
        meta.rcov,
        q_seq,
        kv_cache,
    )

    # Back to token rows; rows of padding tokens or of sequences without cache
    # get out 0 and lse -inf.
    o = jnp.take(o_seq.reshape(N * Tq, H, D), meta.row, axis=0)
    lse = jnp.take(lse_seq[:, :, 0].reshape(N * Tq, H), meta.row, axis=0)
    o = jnp.where(meta.row_ok[:, None, None], o, jnp.zeros_like(o))
    lse = jnp.where(meta.row_ok[:, None], lse, -jnp.inf)
    return o, lse


def decode_cache_pass(
    queries: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cp_rank: jax.Array,
    num_decode_seqs: jax.Array | None = None,
    *,
    cp_group_size: int,
    decode_query_size: int,
    bkv: int = 4096,
    **run_kwargs,
) -> tuple[jax.Array, jax.Array]:
    """DCP CACHE_ONLY pass over this rank's KV shard: prepare() + run().

    Every sequence below `num_decode_seqs` must have q_len <= decode_query_size
    (callers check); sequences at or past it are skipped.
    Returns (out [T, H, D] in the query dtype, lse [T, H] f32); rows of
    sequences with no local cache get out 0 and lse -inf.
    """
    meta = prepare(
        kv_lens,
        page_indices,
        cu_q_lens,
        cp_rank,
        num_decode_seqs,
        num_tokens=queries.shape[0],
        cp_group_size=cp_group_size,
        decode_query_size=decode_query_size,
        page_size=kv_cache.shape[-1],
        bkv=bkv,
    )
    run_kwargs.pop(
        "issue_mode", None
    )  # always "runs" (the others were measurement variants)
    return run(
        queries,
        kv_cache,
        meta,
        decode_query_size=decode_query_size,
        bkv=bkv,
        **run_kwargs,
    )


def supported(
    *,
    kv_cache_shape,
    kv_dtype,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    sliding_window,
    soft_cap,
    kv_layout_is_seq_along_lane: bool,
    bkv: int = 4096,
) -> bool:
    """Whether decode_cache_pass covers this layer's configuration."""
    return (
        kv_layout_is_seq_along_lane
        and num_kv_heads == 1
        and sliding_window is None
        and soft_cap is None
        and jnp.dtype(kv_dtype).itemsize == 1
        and head_dim % _LANES == 0
        and bkv % page_size == 0
        and len(kv_cache_shape) == 5
        and math.prod(kv_cache_shape[2:4]) == head_dim
    )
