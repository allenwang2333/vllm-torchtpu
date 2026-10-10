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
"""DeepSeek V4 CSA attention with the sliding window folded in.

One kernel attends each query token to its gathered top-k compressed KV rows
and to its own `sliding_window` raw KV rows under a single softmax. It replaces
the SWA kernel plus the SparseMLA kernel pair, whose unnormalized SWA partials
made an HBM round trip between them. A small kernel writes the step's new raw
KV rows to the SWA cache first, so the attention kernels only read the cache
and every chunk's SparseCore gather can start before the first attention
kernel, as it does after the SWA kernel.

Window rows before a sequence's first new token come from the SWA cache, the
rest straight from `new_kv`. A batch of tokens that are consecutive positions
of one sequence shares one window span; any other batch fetches a window per
token.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import mla_swa, sparse_mla

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
# Query tokens per grid step.
ATTENTION_KERNEL_BATCH_SIZE = 16
# Query tokens per SparseCore gather call.
GATHER_AND_ATTENTION_CHUNK_SIZE = 128
# New KV rows staged in VMEM per grid step of the cache-write kernel.
WRITE_BLOCK_TOKENS = 256

cdiv = sparse_mla.cdiv


def _sw_rows(sw_cache_ref, token_rows: int):
    """The SWA cache as one `[token_rows, lanes]` uint8 row per token slot."""
    slabs = csa_cache_layout.as_u8_slabs(sw_cache_ref)
    num_pages, page_rows, kv_packing, lanes = slabs.shape
    tokens_per_page = page_rows // (token_rows // kv_packing)
    rows = slabs.reshape(num_pages * tokens_per_page, token_rows, lanes)
    return rows, tokens_per_page


def _is_span(win_pos_ref, win_base_ref, t0, n: int):
    """Whether tokens [t0, t0 + n) are consecutive positions of one sequence."""
    t1 = jnp.minimum(t0 + n - 1, win_pos_ref.shape[0] - 1)
    p0 = win_pos_ref[t0]
    return (
        (p0 >= 0)
        & (win_pos_ref[t1] - p0 == n - 1)
        & (win_base_ref[t1] == win_base_ref[t0])
    )


def _write_new_rows_kernel(
    # Prefetch
    num_valid_ref,  # [1] valid tokens
    win_pos_ref,  # [max_num_tokens] position in the SWA page table; -1 = none
    win_base_ref,  # [max_num_tokens] the sequence's first SWA page-table entry
    win_pages_ref,  # [max_num_seqs * window_pages] SWA page table
    # Input
    new_kv_hbm_ref,  # uint8[max_num_tokens, token_rows, 128]
    sw_cache_hbm_ref,  # SWA cache, int32 or uint8 slabs
    # Output
    updated_sw_cache_hbm_ref,  # aliased with sw_cache_hbm_ref
    # Scratch
    buf_ref,  # uint8[block, token_rows, 128]
    sems,  # DMA [2] (fetch, write)
    *,
    logical_page_size: int,
    batch_size: int,
):
    """Writes each valid token's new KV row to its SWA cache slot."""
    del sw_cache_hbm_ref
    block, token_rows, _ = buf_ref.shape
    sw_rows, tokens_per_page = _sw_rows(updated_sw_cache_hbm_ref, token_rows)
    num_win_pages = win_pages_ref.shape[0]
    page = logical_page_size
    own_span_pages = cdiv(batch_size, page) + 1
    t_blk = pl.program_id(0) * block
    t_end = t_blk + jnp.clip(num_valid_ref[0] - t_blk, 0, block)
    staged = buf_ref.at[pl.ds(0, t_end - t_blk)]

    def write_rows(t, lo, n, num_pages):
        """Writes the staged rows of tokens [t, t + n) to positions [lo, lo + n)."""
        first_page = lo // page
        for i in range(num_pages):
            p_start = (first_page + i) * page
            seg_lo = jnp.clip(lo, p_start, p_start + page)
            seg_hi = jnp.clip(lo + n, p_start, p_start + page)
            sz = jnp.maximum(seg_hi - seg_lo, 0)
            idx = jnp.minimum(win_base_ref[t] + first_page + i, num_win_pages - 1)
            dst = win_pages_ref[idx] * tokens_per_page
            # Zero-size copies get in-range offsets: they still issue.
            dst = jnp.where(sz > 0, dst + seg_lo - p_start, 0)
            src = jnp.where(sz > 0, t - t_blk + seg_lo - lo, 0)
            pltpu.make_async_copy(
                buf_ref.at[pl.ds(src, sz)], sw_rows.at[pl.ds(dst, sz)], sems.at[1]
            ).start()

    def write_batch(b, carry):
        t0 = t_blk + b * batch_size

        # Tokens past t_end have position -1, so a span lies inside the block.
        @pl.when(_is_span(win_pos_ref, win_base_ref, t0, batch_size))
        def _():
            write_rows(t0, win_pos_ref[t0], batch_size, own_span_pages)

        @pl.when(jnp.logical_not(_is_span(win_pos_ref, win_base_ref, t0, batch_size)))
        def _():
            def token(t, c):
                write_rows(t, win_pos_ref[t], 1, 1)
                return c

            lax.fori_loop(t0, jnp.minimum(t0 + batch_size, t_end), token, 0)

        return carry

    @pl.when(t_end > t_blk)
    def _():
        fetch = pltpu.make_async_copy(
            new_kv_hbm_ref.at[pl.ds(t_blk, t_end - t_blk)], staged, sems.at[0]
        )
        fetch.start()
        fetch.wait()
        lax.fori_loop(0, cdiv(t_end - t_blk, batch_size), write_batch, 0)
        # Every staged row is written exactly once.
        pltpu.make_async_copy(staged, staged, sems.at[1]).wait()


def write_new_rows(
    new_kv: jax.Array,  # uint8[max_num_tokens, token_rows, 128]
    sw_cache: jax.Array,
    win_pos: jax.Array,
    win_base: jax.Array,
    win_pages: jax.Array,
    num_valid_tokens: jax.Array,
    *,
    logical_page_size: int,
    batch_size: int,
) -> jax.Array:
    """Writes the step's new KV rows to the SWA cache; returns the cache."""
    num_tokens, token_rows, lanes = new_kv.shape
    block = min(WRITE_BLOCK_TOKENS, num_tokens)
    scalar_prefetches = (
        jnp.reshape(num_valid_tokens, (1,)).astype(jnp.int32),
        win_pos,
        win_base,
        win_pages,
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    n = len(scalar_prefetches)
    return pl.pallas_call(
        functools.partial(
            _write_new_rows_kernel,
            logical_page_size=logical_page_size,
            batch_size=batch_size,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=n,
            in_specs=[hbm, hbm],
            out_specs=hbm,
            grid=(cdiv(num_tokens, block),),
            scratch_shapes=[
                pltpu.VMEM((block, token_rows, lanes), jnp.uint8),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        out_shape=jax.ShapeDtypeStruct(sw_cache.shape, sw_cache.dtype),
        input_output_aliases={n + 1: 0},
        name="SWA-write-new-rows",
    )(*scalar_prefetches, new_kv, sw_cache)


def _fused_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_tokens] valid top-k entries per token
    start_end_seq_idx_ref,  # [3] (start_token, end_token, num_valid_tokens)
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [2] token start of the in-flight bo DMA per bo sem
    win_pos_ref,  # [max_num_tokens] position in the SWA page table; -1 = none
    win_start_ref,  # [max_num_tokens] position of the sequence's first new token
    win_base_ref,  # [max_num_tokens] the sequence's first SWA page-table entry
    win_pages_ref,  # [max_num_seqs * window_pages] SWA page table
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # int32[chunk_size, topk, 128]
    cache_kv_rope_hbm_ref,  # int32[chunk_size, topk // 4, 128]
    new_kv_hbm_ref,  # uint8[max_num_tokens, token_rows, 128]
    sw_cache_hbm_ref,  # SWA cache, int32 or uint8 slabs; read only
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_nope_x2_ref,  # [2, batch_size, topk * 4, 128]
    bkv_rope_x2_ref,  # [2, batch_size, topk // 2, rope_dim * 2]
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bwin_x2_ref,  # uint8[2, batch_size * sliding_window, token_rows, 128]
    win_state_ref,  # SMEM i32[2, 2] (span flag, rows fetched)
    sems,  # DMA [5, 2] (nope, q, o, window fetch, rope)
    *,
    sm_scale: float,
    batch_size: int,
    sliding_window: int,
    logical_page_size: int,
):
    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    cache_kv_nope_hbm_ref = cache_kv_nope_hbm_ref.bitcast(jnp.uint8)
    cache_kv_rope_hbm_ref = cache_kv_rope_hbm_ref.bitcast(jnp.bfloat16)
    bkv_sz = cache_kv_rope_hbm_ref.shape[1] * 2
    q_dtype = q_hbm_ref.dtype
    assert head_dim % 128 == 0

    _, token_rows, _ = new_kv_hbm_ref.shape
    sw_rows, phys_tokens_per_page = _sw_rows(sw_cache_hbm_ref, token_rows)
    num_win_pages = win_pages_ref.shape[0]
    window = sliding_window
    page = logical_page_size
    # Pages a span of n rows starting anywhere in a page can touch.
    span_rows = window - 1 + batch_size
    span_pages = cdiv(span_rows, page) + 1
    token_pages = cdiv(window, page) + 1

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    num_valid_tokens = start_end_seq_idx_ref[2]
    batch_start = start_seq_idx + pl.program_id(0) * batch_size
    batch_last = batch_start + batch_size - 1

    def _async_copy(src, dst, sem, wait=False):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _page_id(t, logical_page):
        idx = jnp.minimum(win_base_ref[t] + logical_page, num_win_pages - 1)
        return win_pages_ref[idx]

    def _fetch_window_rows(t, lo, hi, buf, dst_row, sem, num_pages):
        """Copies positions [lo, hi) of token t's sequence to buf[dst_row:].

        Zero-size copies get in-range offsets: they still issue.
        """
        p = win_pos_ref[t]
        start = win_start_ref[t]
        cache_hi = jnp.minimum(hi, start)
        first_page = lo // page
        for i in range(num_pages):
            p_start = (first_page + i) * page
            seg_lo = jnp.clip(lo, p_start, p_start + page)
            seg_hi = jnp.clip(cache_hi, p_start, p_start + page)
            sz = jnp.maximum(seg_hi - seg_lo, 0)
            src = _page_id(t, first_page + i) * phys_tokens_per_page
            src = jnp.where(sz > 0, src + seg_lo - p_start, 0)
            dst = jnp.where(sz > 0, dst_row + seg_lo - lo, 0)
            _async_copy(sw_rows.at[pl.ds(src, sz)], buf.at[pl.ds(dst, sz)], sem)
        new_lo = jnp.maximum(lo, start)
        new_sz = jnp.maximum(hi - new_lo, 0)
        # Position x of the sequence is new_kv row t + x - p.
        src_tok = jnp.where(new_sz > 0, t + new_lo - p, 0)
        dst_tok = jnp.where(new_sz > 0, dst_row + new_lo - lo, 0)
        _async_copy(
            new_kv_hbm_ref.at[pl.ds(src_tok, new_sz)],
            buf.at[pl.ds(dst_tok, new_sz)],
            sem,
        )
        return jnp.maximum(hi - lo, 0)

    def start_fetch_window(t0, sem_idx):
        sem = sems.at[3, sem_idx]
        buf = bwin_x2_ref.at[sem_idx]
        is_span = _is_span(win_pos_ref, win_base_ref, t0, batch_size)
        win_state_ref[0, sem_idx] = is_span.astype(jnp.int32)

        # Span row r holds position p0 - window + 1 + r, so token j's window is
        # rows [j, j + window).
        @pl.when(is_span)
        def _():
            hi = win_pos_ref[t0] + batch_size
            first = hi - span_rows
            lo = jnp.maximum(first, 0)
            win_state_ref[1, sem_idx] = _fetch_window_rows(
                t0, lo, hi, buf, lo - first, sem, span_pages
            )

        # Otherwise token j's window is rows [j * window, (j + 1) * window).
        @pl.when(jnp.logical_not(is_span))
        def _():
            def body(j, rows):
                t = t0 + j
                hi = win_pos_ref[t] + 1
                first = hi - window
                lo = jnp.maximum(first, 0)
                return rows + _fetch_window_rows(
                    t, lo, hi, buf, j * window + lo - first, sem, token_pages
                )

            win_state_ref[1, sem_idx] = lax.fori_loop(0, batch_size, body, jnp.int32(0))

    def wait_fetch_window(sem_idx):
        buf = bwin_x2_ref.at[sem_idx]
        rows = buf.at[pl.ds(0, win_state_ref[1, sem_idx])]
        _async_copy(rows, rows, sems.at[3, sem_idx], wait=True)

    def _fetch_bkv_batch(t0, sem_idx, *, wait=False):
        nope_vmem = bkv_nope_x2_ref.at[sem_idx]
        rope_vmem = bkv_rope_x2_ref.at[sem_idx]
        if wait:
            _async_copy(nope_vmem, nope_vmem, sems.at[0, sem_idx], wait=True)
            _async_copy(rope_vmem, rope_vmem, sems.at[4, sem_idx], wait=True)
        else:
            # The gathered rows are indexed relative to the current chunk.
            idx = t0 - start_seq_idx
            _async_copy(
                cache_kv_nope_hbm_ref.at[pl.ds(idx, batch_size)],
                nope_vmem,
                sems.at[0, sem_idx],
            )
            _async_copy(
                cache_kv_rope_hbm_ref.at[pl.ds(idx, batch_size)],
                rope_vmem,
                sems.at[4, sem_idx],
            )

    def _fetch_bq_batch(t0, sem_idx, *, wait=False):
        bq_vmem = bq_x2_ref.at[sem_idx]
        if wait:
            _async_copy(bq_vmem, bq_vmem, sems.at[1, sem_idx], wait=True)
        else:
            _async_copy(
                q_hbm_ref.at[pl.ds(t0, batch_size)], bq_vmem, sems.at[1, sem_idx]
            )

    def _send_bo_batch(t0, sem_idx, *, wait=False):
        sz = jnp.clip(num_valid_tokens - t0, 1, batch_size)
        vmem = bo_x2_ref.at[sem_idx, pl.ds(0, sz)]
        if wait:
            _async_copy(vmem, vmem, sems.at[2, sem_idx], wait=True)
        else:
            # The wait rebuilds `sz` from this start; a mismatch deadlocks.
            bo_ids_ref[sem_idx] = t0
            _async_copy(vmem, o_hbm_ref.at[pl.ds(t0, sz)], sems.at[2, sem_idx])

    def wait_send_bo_batch(sem_idx):
        old_t0 = bo_ids_ref[sem_idx]

        @pl.when(old_t0 >= 0)
        def _():
            _send_bo_batch(old_t0, sem_idx, wait=True)

    def start_fetch_batch(t0, sem_idx):
        _fetch_bq_batch(t0, sem_idx)
        start_fetch_window(t0, sem_idx)
        _fetch_bkv_batch(t0, sem_idx)

    def load_bkv(sem_idx, j):
        bkv_nope = bkv_nope_x2_ref.at[sem_idx, j][...].reshape(bkv_sz, -1)
        bkv_nope = sparse_mla._dequant_dsv4_fp8(bkv_nope)
        # Entry i of the rope block is in lanes [0, rope_dim) of row i and
        # entry i + bkv_sz // 2 in the upper half (see csa_gather).
        bkv_rope = bkv_rope_x2_ref.at[sem_idx, j][...].reshape(bkv_sz // 2, -1)
        rope_dim = bkv_rope.shape[-1] // 2
        bkv_rope = jnp.concatenate(
            [bkv_rope[:, :rope_dim], bkv_rope[:, rope_dim:]], axis=0
        )
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)
        # Rows past kv_len may hold other caches' bytes that decode to NaN.
        kv_len = kv_lens_ref[batch_start + j]
        k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        return jnp.where(k_span < kv_len, bkv, 0)

    def load_window(sem_idx, j, is_span):
        """Token j's window rows as bf16 and the first row holding a key."""
        row = jnp.where(is_span, j, j * window)
        win_u8 = bwin_x2_ref.at[sem_idx, pl.ds(row, window)][...]
        win = pltpu.bitcast(win_u8, jnp.bfloat16).reshape(window, head_dim)
        # Row r holds position p - window + 1 + r; rows before position 0 were
        # not fetched and may hold NaN bytes. p == -1 masks every row.
        first_row = window - 1 - win_pos_ref[batch_start + j]
        r = lax.broadcasted_iota(jnp.int32, win.shape, 0)
        return jnp.where(r >= first_row, win, 0), first_row

    def step1_qk_softmax(q, kv, kv_len, win, first_row, sinks):
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        k_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(k_span < kv_len, s, jnp.finfo(s.dtype).min)

        sw = jnp.einsum("nd,md->nm", q, win, preferred_element_type=jnp.float32)
        sw *= sm_scale
        w_span = lax.broadcasted_iota(jnp.int32, sw.shape, 1)
        sw = jnp.where(w_span >= first_row, sw, jnp.finfo(sw.dtype).min)

        m = jnp.maximum(
            jnp.max(s, axis=1, keepdims=True), jnp.max(sw, axis=1, keepdims=True)
        )
        p = jnp.exp(s - m)
        pw = jnp.exp(sw - m)
        l_total = (
            jnp.sum(p, axis=1, keepdims=True)
            + jnp.sum(pw, axis=1, keepdims=True)
            + jnp.exp(sinks - m)
        )
        return p, pw, l_total

    def step2_pv(p, kv, pw, win, l_total):
        acc = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
        acc += jnp.einsum("nm,md->nd", pw, win, preferred_element_type=jnp.float32)
        if q_dtype == jnp.float32:
            return lax.div(acc, l_total)
        return (acc * pl.reciprocal(l_total, approx=True)).astype(q_dtype)

    def process():
        bi_sem_idx = sem_ids_ref[0]
        next_t0 = batch_start + batch_size
        next_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)

        @pl.when(next_t0 < end_seq_idx)
        def _():
            sem_ids_ref[0] = next_sem_idx
            start_fetch_batch(next_t0, next_sem_idx)

        bo_sem_idx = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)
        sinks = attention_sinks_ref[...][..., None]

        _fetch_bq_batch(batch_start, bi_sem_idx, wait=True)
        wait_fetch_window(bi_sem_idx)
        _fetch_bkv_batch(batch_start, bi_sem_idx, wait=True)

        @pl.when(pl.program_id(0) >= 2)
        def _():
            wait_send_bo_batch(bo_sem_idx)

        is_span = win_state_ref[0, bi_sem_idx] == 1
        prev = None
        prev_out = None
        for j in range(batch_size):
            bkv = load_bkv(bi_sem_idx, j)
            bq = bq_x2_ref.at[bi_sem_idx, j][...]
            if prev_out is not None:
                # Artificial dependency that limits QK runahead so MXU and VPU
                # work interleave; prev_out is never inf.
                bq = jnp.where(prev_out == jnp.inf, prev_out, bq)
            win, first_row = load_window(bi_sem_idx, j, is_span)
            p, pw, l_total = step1_qk_softmax(
                bq, bkv, kv_lens_ref[batch_start + j], win, first_row, sinks
            )
            if prev is not None:
                out = step2_pv(*prev)
                bo_x2_ref.at[bo_sem_idx, j - 1][...] = out
                prev_out = out
            prev = (p, bkv, pw, win, l_total)

        bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = step2_pv(*prev)
        _send_bo_batch(batch_start, bo_sem_idx)

    @pl.when(batch_start == start_seq_idx)
    def prologue():
        start_fetch_batch(batch_start, 0)

    process()

    @pl.when(batch_last == end_seq_idx - 1)
    def epilogue():
        wait_send_bo_batch(0)

        @pl.when(pl.num_programs(0) >= 2)
        def _():
            wait_send_bo_batch(1)


def _window_metadata(
    num_tokens: int,
    kv_lens: jax.Array,  # i32[max_num_seqs], after narrow_page_indices
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    window_pages: int,
):
    """Per-token window position, sequence start position and page-table base."""
    max_num_seqs = kv_lens.shape[0]
    t = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.searchsorted(cu_q_lens[1:], t, side="right").astype(jnp.int32)
    seq = jnp.minimum(seq, max_num_seqs - 1)
    valid = t < cu_q_lens[distribution[2]]
    q_start = cu_q_lens[seq]
    start = kv_lens[seq] - (cu_q_lens[seq + 1] - q_start)
    pos = jnp.where(valid, start + t - q_start, -1)
    start = jnp.where(valid, start, 0)
    return pos, start, seq * window_pages


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "logical_page_size",
        "nope_is_sw_cache",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
    ),
    donate_argnames=("sw_cache",),
)
def sparse_swa_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    new_kv: jax.Array,  # bf16[max_num_tokens, head_dim]
    sw_cache: jax.Array,  # SWA cache: uint8 slabs or the int32 CSA host array
    swa_kv_lens: jax.Array,  # i32[max_num_seqs]
    swa_page_indices: jax.Array,  # i32[max_num_seqs * swa_pages_per_seq]
    swa_cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    swa_distribution: jax.Array,  # i32[3]
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    nope_is_sw_cache: bool = False,
    attention_kernel_batch_size: int = ATTENTION_KERNEL_BATCH_SIZE,
    gather_and_attention_chunk_size: int = GATHER_AND_ATTENTION_CHUNK_SIZE,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[jax.Array, jax.Array]:
    """CSA attention over top-k compressed KV plus the sliding window.

    Computes what `mla_sliding_window_ragged_paged_attention` (unnormalized)
    followed by `sparse_ragged_paged_attention` computes, and writes `new_kv`
    to `sw_cache` the same way.

    Args:
      nope_is_sw_cache: `cache_kv_nope` is the same buffer as `sw_cache`; the
        gather then reads the cache this call writes, so the two stay one
        buffer.

    Returns:
      The attention output and the updated SWA cache.
    """
    assert cache_kv_nope.dtype == jnp.int32
    assert cache_kv_rope.dtype == jnp.int32
    assert new_kv.dtype == jnp.bfloat16
    _, actual_num_q_heads, actual_head_dim = q.shape
    num_tokens = q.shape[0]

    q = sparse_mla.prepare_q_inputs(q)
    _, num_q_heads, head_dim = q.shape
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, num_q_heads - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )

    _, sw_page_rows, kv_packing, lanes = csa_cache_layout.u8_slab_shape(
        sw_cache.shape, sw_cache.dtype
    )
    token_rows = head_dim * 2 // lanes
    assert token_rows % kv_packing == 0
    assert logical_page_size <= sw_page_rows // (token_rows // kv_packing)
    new_kv = mla_swa.prepare_kv_inputs(new_kv).reshape(num_tokens, token_rows, lanes)

    batch = attention_kernel_batch_size
    swa_kv_lens, swa_page_indices = mla_swa.narrow_page_indices(
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        max_q_len=num_tokens,
        max_kv_pages_per_block=cdiv(sliding_window - 1 + batch, logical_page_size),
    )
    window_pages = swa_page_indices.shape[0] // swa_kv_lens.shape[0]
    win_pos, win_start, win_base = _window_metadata(
        num_tokens, swa_kv_lens, swa_cu_q_lens, swa_distribution, window_pages
    )
    sw_cache = write_new_rows(
        new_kv,
        sw_cache,
        win_pos,
        win_base,
        swa_page_indices,
        swa_cu_q_lens[swa_distribution[2]],
        logical_page_size=logical_page_size,
        batch_size=batch,
    )

    kv_lens, gather_indices = sparse_mla.csa_gather_indices(
        topk_indices,
        page_indices,
        cu_q_lens,
        total_num_pages=cache_kv_nope.shape[0],
        page_size=cache_kv_nope.shape[1],
    )
    topk = topk_indices.shape[-1]
    num_valid_tokens = cu_q_lens[distribution[2]]

    def run_kernel(q, sw_cache, gathered_nope, gathered_rope, chunk):
        b = chunk.kernel_batch_size
        end = jnp.maximum(chunk.start_pos, chunk.batch_end)
        grid = (cdiv(end - chunk.start_pos, b),)
        _, topk_rows, nope_lanes = gathered_nope.shape
        _, rope_rows, rope_lanes = gathered_rope.shape
        scratch_shapes = [
            pltpu.VMEM((2, b, topk_rows * 4, nope_lanes), jnp.uint8),
            pltpu.VMEM((2, b, rope_rows * 2, rope_lanes), jnp.bfloat16),
            pltpu.VMEM((2, b, num_q_heads, head_dim), q.dtype),  # bq
            pltpu.VMEM((2, b, num_q_heads, head_dim), q.dtype),  # bo
            pltpu.VMEM((2, b * sliding_window, token_rows, lanes), jnp.uint8),
            pltpu.SMEM((2, 2), jnp.int32),
            pltpu.SemaphoreType.DMA((5, 2)),
        ]
        scalar_prefetches = (
            kv_lens,
            jnp.array([chunk.start_pos, end, num_valid_tokens], jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.full((2,), -1, jnp.int32),
            win_pos,
            win_start,
            win_base,
            swa_page_indices,
        )
        hbm = pl.BlockSpec(memory_space=pltpu.HBM)
        n = len(scalar_prefetches)
        scope_name = f"SparseMLA-SWA-p_{topk_rows}-bz_{b}-gcz_{chunk.chunk_size}"
        kernel = pl.pallas_call(
            functools.partial(
                _fused_attention_kernel,
                sm_scale=sm_scale,
                batch_size=b,
                sliding_window=sliding_window,
                logical_page_size=logical_page_size,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=n,
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 5,
                out_specs=hbm,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
            input_output_aliases={n + 1: 0},
            name=scope_name,
        )
        return jax.named_scope(scope_name)(kernel)(
            *scalar_prefetches,
            attention_sinks,
            q,
            gathered_nope,
            gathered_rope,
            new_kv,
            sw_cache,
        )

    for chunk in sparse_mla.csa_chunks(
        gather_indices,
        cu_q_lens,
        distribution,
        topk=topk,
        num_tokens=num_tokens,
        chunk_size=gather_and_attention_chunk_size,
        max_batch_size=batch,
    ):
        nope = sw_cache if nope_is_sw_cache else cache_kv_nope
        gathered_nope, gathered_rope = chunk.gather(nope, cache_kv_rope)
        q = run_kernel(q, sw_cache, gathered_nope, gathered_rope, chunk)

    out = sparse_mla.prepare_outputs(q, actual_num_q_heads, actual_head_dim)
    return out, sw_cache
