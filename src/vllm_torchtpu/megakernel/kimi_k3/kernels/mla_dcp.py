# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode context parallelism (DCP) for Kimi-K3 MLA.

Without DCP every TP32 rank holds its 3 heads' expanded K|V for every token.
Under DCP the ``G`` ranks of a group (consecutive ranks, ``member = rank %
G``) each hold the compressed latent (576 of 640 lanes) of a disjoint 1/G of
the tokens and attend *all* the group's heads over them in latent space
(``q_abs = [q_nope @ W_UK^T | q_rope | 0]``, score scale folded in):

1. **q all-gather.** Each rank writes its ``[B, heads, 640]`` absorbed
   queries into slot ``member`` of every group peer's gather buffer.
2. **Local attention.** For each row, over the rank's own pages (global page
   ``j G + member`` is local page ``j``; tokens ``< position``), ``span``
   pages per step: ``Q @ C^T`` with each member's ``slots`` head slots
   (``packet_layout``) on sublanes, online softmax in FP32,
   ``O += P @ C[:, :512]``.
3. **Combine.** Each rank normalizes its partial (``o = O / l``, ``lse = m +
   log l``) and sends member ``d`` a BF16 ``[slots B, 640]`` packet of ``d``'s
   heads: row ``slots b + h``, ``o`` in lanes 0:512, ``lse`` as a high and a
   low BF16 part in lanes 512 and 513.
4. **Merge** on the owner: the log-sum-exp merge of the G partials plus the
   new token (replicated, so in no rank's pages yet), giving the latent-space
   output ``[B, heads, 512]`` (the caller applies W_UV).

Under attention DP (``heads`` = 6 for TP16 x DP2) the DCP group is the DP
group of ``32 / dp`` ranks, over that group's rows. Global page ``g`` lives on
member ``g % G``. Block tables are the same on every rank, in one of two
conventions (``KernelConfig.dcp_tables``):

* **local** (the default; vLLM's ``cp_kv_cache_interleave_size`` of one page):
  entry ``j`` covers G pages and every rank keeps its page in its block of that
  id; pools shrink by G, blocks are shared per G pages.
* **global**: entry ``g`` is page ``g``'s block, exactly vLLM's non-DCP table;
  only its owner fills that block, so every rank's pool has all the block ids.

The cache is vLLM's unified pools ``[blocks, 128, packing, 640]`` (packing 2 in
BF16, 4 in FP8); ``paged.commit_latent_rows`` commits new rows on their owners.
In-kernel pieces: :class:`DcpCache` (pool and table), :class:`DcpWorkspace`
(:func:`persistent_scratch`, which peers write into, so it must not alias other
scoped VMEM, and :func:`scoped_scratch`) and :class:`DcpAttention` (``prime``
the first page DMAs early, ``attend`` later).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.config import active as _config

PAGE_TOKENS = 256
LANES = 640  # latent 512 | rope 64 | zero 64
LATENT = 512
HEADS_PER_RANK = 3
HEAD_SLOTS = 4  # most rows per rank and batch row in a packet (see packet_layout)
NEG = -1e30  # finite "no tokens" log-sum-exp
MAX_ROWS = 8  # batch rows of an unchunked call; a larger batch runs in chunks (see _attend_chunked)


def chunk_layout(batch: int, heads: int = HEADS_PER_RANK) -> tuple[int, int]:
    """``(chunk rows, ring slots)`` of a chunked call (``batch > MAX_ROWS``):
    8-row chunks at B16/B32/B48/B64 (amortizing exchanges), else 4; ring slots
    4 at B32 with three heads per rank, 3 for 8-row chunks, else 2
    (``persistent_scratch`` uses 4 for FP8 query packets)."""
    wide = batch <= 16 or batch in (32, 48, 64)
    rows = 8 if wide else 4
    ring = 4 if batch == 32 and heads == HEADS_PER_RANK else 3 if wide else 2
    if batch % rows or rows > MAX_ROWS or ring not in (2, 3, 4):
        raise ValueError(f"DCP chunks: {rows} rows must divide the batch {batch} (<= {MAX_ROWS}), ring 2, 3 or 4")
    return rows, ring

GROUP = _config().dcp_group
GLOBAL_PAGES = _config().dcp_tables == "global"
SPAN = 4  # pages per step
BUFFERS = 4  # steps in flight


def page_plan(batch: int, heads: int = HEADS_PER_RANK, sequence_rows: int = 1) -> dict:
    """The decoder's page buffers for ``batch`` rows of ``heads`` heads per
    rank (``sequence_rows`` per sequence): ``span`` pages per step and
    ``buffers`` steps in flight, sized to fit VMEM. Up to four rows and plain
    decode: ``SPAN`` x ``BUFFERS`` (3 buffers chunked). Verifies above four rows:
    half the span, twice the buffers (4 at B64 and attention DP's B32), 3/4 chunked."""
    if batch <= 4:
        return {"span": SPAN, "buffers": BUFFERS}
    if sequence_rows == 1:
        return {"span": SPAN, "buffers": 3 if chunked(batch) else BUFFERS}
    span = max(1, SPAN // 2)
    buffers = 4 if batch == 64 or (batch == 32 and heads != HEADS_PER_RANK) else BUFFERS * SPAN // span
    return {"span": span, "buffers": buffers if not chunked(batch) else max(2, buffers * 3 // 4)}


MAX_GROUPS = 2  # row groups of a call (see packet_layout)


def packet_layout(batch: int, heads: int = HEADS_PER_RANK) -> tuple[int, int]:
    """``(head slots, row groups)`` of a call with ``batch`` rows of ``heads``
    heads per rank. Head slots are the packet rows per rank and batch row.
    B2 runs two groups whose exchanges overlap the other group's attention
    (used at B2 only; above it the DMA count doubles); a group's packet slice
    must be a multiple of 8 rows unless it has 4-row slots, so B2 pads its
    three heads with a zero slot. Otherwise one group of a slot per head."""
    return (4, 2) if batch == 2 and heads == HEADS_PER_RANK else (heads, 1)

def owner_index(table):
    """This rank's page-owner index from a static per-mesh-position table
    (``table[p]`` for mesh position ``p``, e.g. its vLLM TP rank); None: the
    mesh position. Works in a kernel or at the XLA level."""
    rank = jax.lax.axis_index("tp")
    if table is None:
        return rank
    owner = 0
    for position, value in enumerate(table):
        owner = jnp.where(rank == position, value, owner)
    return owner


@dataclasses.dataclass(frozen=True)
class DcpCache:
    """This rank's latent pages in HBM: ``pool_ref`` the per-MLA-layer tuple
    of vLLM's unified pools (``[blocks, 128, packing, 640]``, token ``t`` of a
    block at ``[t // packing, t % packing]``), picked by the traced layer index
    ``layer``. ``table_ref`` is the flat ``[rows * pages]`` int32 block table
    in SMEM (``pages`` entries per row), ``group`` the DCP group size,
    ``global_pages`` the table convention (see the module docstring).

    vLLM's pages hold stale bytes past a request's tokens and in lanes
    576:640 (freed blocks, KDA state that reads as NaN in bf16); the
    attention zeroes both before they reach the MXU."""

    pool_ref: Any
    table_ref: Any
    pages: int
    layer: Any = None
    group: int = GROUP
    global_pages: bool = GLOBAL_PAGES
    # Which pages this rank holds: global page g is on the rank whose owner
    # index is g % group. Default: its member index (mesh order); a P/D
    # decoder uses its vLLM TP rank (see owner_index), the order the KV
    # transfer's destination units follow.
    owner: Any = None
    # The table row of attention row ``b`` (default ``b``): attention DP
    # attends its group's rows of the batch-wide table.
    table_row: Callable[[Any], Any] | None = None

    @property
    def dtype(self):
        return self.pool_ref[0].dtype

    @property
    def page_tokens(self):
        ref = self.pool_ref[0]
        return ref.shape[-3] * ref.shape[-2]

    def start_page_copy(self, row, local, member, destination, semaphore):
        """Start the DMA of ``row``'s local page ``local`` on ``member``."""
        entry = local * self.group + member if self.global_pages else local
        if self.table_row is not None:
            row = self.table_row(row)
        block = self.table_ref[row * self.pages + entry]
        for index, ref in enumerate(self.pool_ref):
            @pl.when(self.layer == index)
            def start(ref=ref):
                tpu.make_async_copy(ref.at[block], destination, semaphore).start()


@dataclasses.dataclass(frozen=True)
class DcpWorkspace:
    # Persistent: peers' remote copies land here.
    gathered: Any  # BF16 queries, or FP8 query/scale packets as uint8
    send: Any  # [G, slots B, 640] bf16
    receive: Any  # [2, G, slots B, 640] bf16
    gather_semaphores: Any  # DMA [2 (send, receive), 2 (parity), MAX_GROUPS (row group)]
    combine_semaphores: Any  # DMA [2, 2, MAX_GROUPS]
    # Scoped to one MLA invocation.
    queries: Any  # [B, head_rows, 640], cache dtype; head slots padded to 16
    pages: Any  # [buffers, span * page_rows, packing, 640], cache dtype
    page_semaphores: Any  # DMA [buffers]
    maximum: Any  # [B, head_rows, 1] f32 (per head slot)
    denominator: Any  # [B, head_rows, 1] f32
    accumulator: Any  # [B, head_rows, 512] f32
    own_rows: Any  # [slots B, 640] f32 (chunked calls: chunk_rows_shape)
    new_rows: Any  # [slots B, 640] f32 (chunked calls: chunk_rows_shape)
    merged: Any  # chunked calls: chunk_rows_shape(512) merged outputs (else a placeholder)
    lookup_bounds: Any
    lookup_counts: Any
    query_scale: Any  # [held, heads, 1] FP32 dequantization scale for FP8 Q


def chunked(batch) -> bool:
    """Whether a call of ``batch`` rows runs as chunks of ``MAX_ROWS``."""
    return batch > MAX_ROWS


def _grouped_spec(batch, group, heads, sequence_rows):
    """Verifier shapes supported by grouped FP8 QK/PV and query transport."""
    return (group == 32 and heads == HEADS_PER_RANK
            and (batch, sequence_rows) in ((4, 4), (8, 4), (8, 8), (16, 8), (32, 8), (32, 4), (64, 8), (64, 4)))


def persistent_scratch(batch, group=GROUP, heads=HEADS_PER_RANK, *,
                       cache_dtype=jnp.bfloat16, sequence_rows=1):
    # Send the receiver's identical FP8 operand and lossless FP32 scale.
    # Plain attention DP2 holds six heads for B/2 rows. Speculative attention
    # retains full-precision owner queries for its causal new-token fold.
    plain = sequence_rows == 1 and (
        (batch in (32, 64) and heads == HEADS_PER_RANK)
        or (group == 16 and batch in (16, 32, 48) and heads == 6))
    query_dtype = (jnp.uint8 if cache_dtype == jnp.float8_e4m3fn
                   and (plain or _grouped_spec(batch, group, heads, sequence_rows))
                   else jnp.bfloat16)
    if chunked(batch):
        chunk, ring = chunk_layout(batch, heads)
        rows = packet_layout(batch, heads)[0] * chunk
        if query_dtype == jnp.uint8:
            # The smaller query buffers make two-chunk lookahead fit at B64.
            ring = 4
        return (
            tpu.VMEM((ring, group, rows, LANES), query_dtype),
            tpu.VMEM((group, rows, LANES), jnp.bfloat16),
            tpu.VMEM((ring, group, rows, LANES), jnp.bfloat16),
            tpu.SemaphoreType.DMA((2, ring)),
            tpu.SemaphoreType.DMA((2, ring)),
        )
    rows = packet_layout(batch, heads)[0] * batch
    return (
        tpu.VMEM((2, group, rows, LANES), query_dtype),
        tpu.VMEM((group, rows, LANES), jnp.bfloat16),
        tpu.VMEM((2, group, rows, LANES), jnp.bfloat16),
        tpu.SemaphoreType.DMA((2, 2, MAX_GROUPS)),
        tpu.SemaphoreType.DMA((2, 2, MAX_GROUPS)),
    )


PERSISTENT_NAMES = ("dcp gathered queries", "dcp send packets", "dcp received packets",
                    "dcp gather semaphores", "dcp combine semaphores")


def scoped_scratch(batch, span=SPAN, buffers=BUFFERS, heads=HEADS_PER_RANK, group=GROUP,
                   cache_dtype=jnp.bfloat16, page_tokens=PAGE_TOKENS, *, sequence_rows=1):
    rows = packet_layout(batch, heads)[0] * batch
    tile = head_rows(batch, group, heads)
    packing = 4 // jnp.dtype(cache_dtype).itemsize
    # FP8 doubles tokens per physical page. The opt-in mla_decode_ktile=2048
    # folds 2048 tokens per step for plain DP2 B64 (32 local rows).
    wide = _config().mla_decode_ktile == 2048
    if wide and (group != 16 or heads != 6 or cache_dtype != jnp.float8_e4m3fn
                 or page_tokens != 512):
        raise ValueError(
            "mla_decode_ktile=2048 requires DCP16, six heads per rank and "
            "FP8 e4m3fn KV with 512-token pages")
    if cache_dtype == jnp.float8_e4m3fn and batch in (8, 16, 32, 64):
        span = 4 if wide and batch == 32 and sequence_rows == 1 else min(span, SPAN * PAGE_TOKENS // page_tokens)
    else:
        span = span * PAGE_TOKENS // page_tokens
    span = max(1, span)
    pages = (buffers, span * page_tokens // packing, packing, LANES)
    held = chunk_layout(batch, heads)[0] if chunked(batch) else batch  # rows whose attention state is live at once
    return (
        tpu.VMEM((held, tile, LANES), cache_dtype),
        tpu.VMEM(pages, cache_dtype),
        tpu.SemaphoreType.DMA((buffers, )),
        tpu.VMEM((held, tile, 1), jnp.float32),
        tpu.VMEM((held, tile, 1), jnp.float32),
        tpu.VMEM((held, tile, LATENT), jnp.float32),
        *((tpu.VMEM(chunk_rows_shape(batch, LANES, heads), jnp.float32),
           tpu.VMEM(chunk_rows_shape(batch, LANES, heads), jnp.float32),
           tpu.VMEM(chunk_rows_shape(batch, LATENT, heads), jnp.float32)) if chunked(batch) else
          (tpu.VMEM((rows, LANES), jnp.float32),
           tpu.VMEM((rows, LANES), jnp.float32),
           tpu.VMEM((8, 128), jnp.float32))),
        tpu.SMEM((batch if batch in (32, 48, 64) else 1,), jnp.int32),
        tpu.SMEM((batch if batch in (32, 48, 64) else 1,), jnp.int32),
        tpu.VMEM((held, tile, 1), jnp.float32),
    )


def head_rows(batch, group=GROUP, heads=HEADS_PER_RANK):
    """Rows of a step's score tile: the group's head slots, rounded up to 16."""
    slots = packet_layout(batch, heads)[0] * group
    return -(-slots // 16) * 16


def chunk_rows_shape(batch, lanes, heads=HEADS_PER_RANK):
    """Chunked calls: per-chunk row buffers ``[chunks, rows padded to 8,
    lanes]`` (a chunk's packet rows, ``slots x chunk``, need not be a
    multiple of the f32 tile; chunks are indexed on the leading dim)."""
    chunk = chunk_layout(batch, heads)[0]
    rows = packet_layout(batch, heads)[0] * chunk
    return (batch // chunk, -(-rows // 8) * 8, lanes)


SCOPED_NAMES = ("dcp query columns", "dcp latent pages", "dcp page semaphores", "dcp maximum",
                "dcp denominator", "dcp accumulator", "dcp own rows", "dcp new rows", "dcp merged",
                "dcp lookup bounds", "dcp lookup counts", "dcp query scales")


class DcpAttention:
    """One DCP attention invocation for ``batch`` rows at ``position_ref``."""

    def __init__(self, cache: DcpCache, workspace: DcpWorkspace, position_ref, batch, *,
                 probability_hilo=False, skip=(), sequence_rows=1, heads=HEADS_PER_RANK):
        if batch > MAX_ROWS:
            chunk = chunk_layout(batch, heads)[0]  # validates
            if chunk % sequence_rows:
                raise ValueError(
                    f"DCP chunks of {chunk} rows must hold whole {sequence_rows}-row sequences")
        if sequence_rows < 1 or batch % sequence_rows:
            raise ValueError("sequence_rows must divide the batch")
        if cache.group not in (8, 16, 32):
            raise ValueError("DCP groups are 8, 16 or 32 consecutive ranks")
        self.cache, self.ws, self.position_ref, self.batch = cache, workspace, position_ref, batch
        # sequence_rows > 1 (speculative verify): each group of rows is one
        # sequence's consecutive tokens. All rows attend the pool below the
        # group's first position; its new tokens fold in causally after the merge.
        self.sequence_rows = sequence_rows
        self.heads = heads
        self.slots, self.groups = packet_layout(batch, heads)
        self.group = cache.group
        self.tile = workspace.queries.shape[1]  # score-tile rows (head_rows)
        if self.group * self.slots > self.tile:
            raise ValueError(f"{self.group} ranks x {self.slots} head slots exceed {self.tile} rows")
        self.buffers = workspace.pages.shape[0]
        self.page_tokens = cache.page_tokens
        self.fp8 = cache.dtype == jnp.float8_e4m3fn
        self.packing = 4 // jnp.dtype(cache.dtype).itemsize
        self.page_rows = self.page_tokens // self.packing
        self.span = workspace.pages.shape[1] // self.page_rows
        self.probability_hilo, self.skip = probability_hilo, set(skip)
        rank = jax.lax.axis_index("tp")
        self.member = rank % self.group
        self.base = rank - self.member
        self.owner = self.member if cache.owner is None else cache.owner
        # Local pages per row: global pages below ceil(limit / page_tokens)
        # congruent to ``owner``, ``span`` per step; the rows' steps form one
        # flat sequence. Shared (whole sequences per row group): a sequence's
        # first row owns its steps and each step folds into all its rows.
        self.shared = sequence_rows > 1 and all(
            first % sequence_rows == 0 and end % sequence_rows == 0 for first, end in self.row_groups())
        self.counts, starts = [], [0]
        for b in range(batch):
            needed = (self._limit(b) + self.page_tokens - 1) // self.page_tokens
            count = jnp.maximum(needed - self.owner + self.group - 1, 0) // self.group
            self.counts.append(count if not self.shared or b % sequence_rows == 0 else 0)
            starts.append(starts[-1] + (self.counts[-1] + self.span - 1) // self.span)
        self.starts, self.total = starts, starts[-1]
        if batch in (32, 48, 64):
            for b in range(batch):
                workspace.lookup_bounds[b] = starts[b]
                workspace.lookup_counts[b] = self.counts[b]

    def _limit(self, row):
        """Pool tokens ``row`` may see: those below its position, or below
        its sequence's first position (``sequence_rows > 1``); ``row`` may be
        traced."""
        if self.sequence_rows == 1:
            return self.position_ref[row]
        return self.position_ref[row // self.sequence_rows * self.sequence_rows]

    def _locate(self, step):
        if self.batch in (32, 48, 64):
            row = jnp.int32(0)
            for bit in range((self.batch - 1).bit_length() - 1, -1, -1):
                candidate = jnp.minimum(row + (1 << bit), self.batch - 1)
                row = jnp.where(step >= self.ws.lookup_bounds[candidate], candidate, row)
            return row, step - self.ws.lookup_bounds[row], self.ws.lookup_counts[row]
        row, first, count = 0, 0, self.counts[0]
        for b in range(1, self.batch):
            later = step >= self.starts[b]
            row = jnp.where(later, b, row)
            first = jnp.where(later, self.starts[b], first)
            count = jnp.where(later, self.counts[b], count)
        return row, step - first, count

    def _start_step(self, step):
        """The step's ``span`` page DMAs; pages past the row's last re-read its
        last page (masked by position)."""
        row, local_step, count = self._locate(step)
        slot = step % self.buffers
        for index in range(self.span):
            local = jnp.minimum(local_step * self.span + index, count - 1)
            self.cache.start_page_copy(
                row, local, self.owner,
                self.ws.pages.at[slot, pl.ds(index * self.page_rows, self.page_rows)],
                self.ws.page_semaphores.at[slot])

    def _wait_step(self, step):
        slot = step % self.buffers
        tpu.make_async_copy(self.ws.pages.at[slot], self.ws.pages.at[slot],
                            self.ws.page_semaphores.at[slot]).wait()

    def _attend_step(self, step, base=0):
        """One step of the local attention: wait for its pages, prefetch
        ahead, and fold the pages' scores into its row's running softmax
        (held at row - ``base``: the chunk's first row)."""
        self._wait_step(step)

        @pl.when(step + self.buffers - 1 < self.total)
        def prefetch():
            self._start_step(step + self.buffers - 1)

        if "pagecompute" in self.skip:
            return
        row, visible, load = self._step_pages(step)
        if self.shared and self.fp8 and _grouped_spec(self.batch, self.group, self.heads, self.sequence_rows):
            # Four independent queries fill the matrix row dimension together.
            # Preserve each query's softmax reduction and FP8 conversion; only
            # QK/PV dispatch is grouped. One/two groups cover four/eight rows.
            def fold_group(g, score):
                upcoming = self._group_scores(load, visible, row - base + (g + 1) * 4, 4)
                self._group_fold(score, load, row - base + g * 4, 4)
                return upcoming
            final = jax.lax.fori_loop(0, self.sequence_rows // 4 - 1, fold_group,
                                     self._group_scores(load, visible, row - base, 4))
            self._group_fold(final, load, row - base + self.sequence_rows - 4, 4)
        elif self.shared and (self.fp8 or self.batch in (16, 32)):
            # Each query has independent softmax state. Issue the next QK
            # before folding the current scores so its MXU work can overlap
            # softmax/PV. Keep this page buffer live until the final query's
            # PV completes; the caller refills it only in a later page step.
            # Peeling the last fold avoids an unused extra QK per page tile.
            def fold_row(r, score):
                upcoming = self._scores(load, visible, row - base + r + 1)
                self._fold_scores(score, load, row - base + r)
                return upcoming

            final = jax.lax.fori_loop(0, self.sequence_rows - 1, fold_row,
                                     self._scores(load, visible, row - base))
            self._fold_scores(final, load, row - base + self.sequence_rows - 1)
        elif self.shared:
            # BF16 verifier: fold rows one at a time (no QK lookahead).
            @pl.loop(0, self.sequence_rows)
            def fold_row(r):
                self._fold_pages(load, visible, row - base + r)
        else:
            self._fold_pages(load, visible, row - base)

    def _step_pages(self, step):
        """Step ``step``'s row, the mask of its visible tokens and a reader of
        its pages (their stale bytes zeroed first)."""
        row, local_step, _ = self._locate(step)
        offset = jax.lax.broadcasted_iota(jnp.int32, (1, self.span * self.page_tokens), 1)  # tokens on lanes
        global_page = (local_step * self.span + offset // self.page_tokens) * self.group + self.owner
        token = global_page * self.page_tokens + offset % self.page_tokens
        visible = token < self._limit(row)
        slot = step % self.buffers
        self._clean_step(slot, row, local_step)
        return row, visible, self._pages(slot)

    def _pages(self, slot):
        """``load(lanes)``: buffer ``slot``'s pages in the cache dtype."""
        words = self.ws.pages.bitcast(jnp.uint32).at[slot].reshape(self.span * self.page_rows, LANES)
        return lambda lanes: pltpu.bitcast(words[:, lanes], self.cache.dtype)

    def _attend_range(self, first, end, base):
        # FP8-query plain decode at B32/B64 (and DP2's 32 local rows): rows with
        # a partial last step fold its 1 .. span - 1 pages with static tail shapes.
        if (self.sequence_rows != 1 or (self.heads != HEADS_PER_RANK and not (self.group == 16 and self.heads == 6 and self.batch == 32)) or self.batch not in (32, 64) or self.ws.gathered.dtype != jnp.uint8
                or (self.span != 2 and not (_config().mla_decode_ktile == 2048 and self.span == 4))
                or "attend" in self.skip or "pagecompute" in self.skip):
            return self._attend_pipeline(first, end, base)
        # Branch once per chunk so QK/PV shapes in the page loop stay static.
        chunk = chunk_layout(self.batch, self.heads)[0]
        remainder = jnp.int32(0)
        for j in range(chunk):
            remainder = remainder | (self.ws.lookup_counts[base + j] % self.span)

        @pl.when(remainder == 0)
        def whole():
            self._attend_pipeline(first, end, base)

        @pl.when(remainder != 0)
        def split():
            @pl.loop(0, chunk)
            def row_loop(j):
                row = base + j
                begin = self.ws.lookup_bounds[row]
                count = self.ws.lookup_counts[row]
                full_end = begin + count // self.span
                self._attend_pipeline(begin, full_end, base, peel=True)

                for tail_pages in range(1, self.span):
                    @pl.when(count % self.span == tail_pages)
                    def tail(tail_pages=tail_pages):
                        self._wait_step(full_end)

                        @pl.when(full_end + self.buffers - 1 < self.total)
                        def prefetch():
                            self._start_step(full_end + self.buffers - 1)

                        _, visible, load = self._step_pages(full_end)
                        score = self._scores(load, visible, row - base, tail_tokens=tail_pages * self.page_tokens)
                        self._fold_scores(score, load, row - base, tail_tokens=tail_pages * self.page_tokens)

    def _attend_pipeline(self, first, end, base, *, peel=False):
        """Steps ``first .. end - 1`` of a chunk whose state rows start at row
        ``base``. Plain decode pipelines them: step s + 1's scores are issued
        before step s's softmax and P V, so the next QK overlaps them, and a
        step's buffer is refilled only after its P V. Normally the last step
        recomputes its own scores (unused), avoiding a branch before QK.
        ``peel`` is static: the split-tail path finishes PV outside the loop
        instead, before issuing the separate half-width QK/PV."""
        if self.sequence_rows > 1 or "pagecompute" in self.skip:
            pl.loop(first, end)(lambda step: self._attend_step(step, base=base))
            return

        def scores(step):
            row, visible, load = self._step_pages(step)
            return self._scores(load, visible, row - base), row

        @pl.when(first < end)
        def attend():
            self._wait_step(first)

            @pl.when(first + self.buffers - 1 < self.total)
            def prefetch_first():
                self._start_step(first + self.buffers - 1)

            def fold(step, carry):
                score, row = carry
                more = step + 1 < end
                if peel:
                    self._wait_step(step + 1)
                    upcoming = scores(step + 1)
                else:
                    @pl.when(more)
                    def wait_next():
                        self._wait_step(step + 1)

                    upcoming = scores(jnp.minimum(step + 1, end - 1))
                self._fold_scores(score, self._pages(step % self.buffers), row - base)

                @pl.when((True if peel else more) & (step + self.buffers < self.total))
                def prefetch():
                    self._start_step(step + self.buffers)

                return upcoming

            carry = jax.lax.fori_loop(first, end - int(peel), fold, scores(first))
            if peel:
                score, row = carry
                self._fold_scores(score, self._pages((end - 1) % self.buffers), row - base)

    def _store_query(self, held, query):
        """Quantize once per query, outside the streamed page loop.

        Scale each absorbed head independently; its latent/RoPE columns share
        one scale, so the QK dot needs only one FP32 multiply per score row.
        Padding heads are zero and get scale one. KV uses scale one.
        """
        if self.fp8:
            maximum = jnp.max(jnp.abs(query), axis=1, keepdims=True)
            scale = jnp.where(maximum > 0, maximum / 448.0, 1.0)
            self.ws.queries[held] = (query / scale).astype(jnp.float8_e4m3fn)
            self.ws.query_scale[held] = scale
        else:
            self.ws.queries[held] = query.astype(jnp.bfloat16)

    def _group_scores(self, load, visible, held, count):
        """QK for adjacent queries, with independent head rows and scales."""
        query = jnp.concatenate([self.ws.queries[held + i] for i in range(count)], axis=0)
        scales = jnp.concatenate([self.ws.query_scale[held + i] for i in range(count)], axis=0)
        latent = load(pl.ds(0, LANES))
        score = jax.lax.dot_general(query, latent, (((1,), (1,)), ((), ())),
                                    preferred_element_type=jnp.float32)
        return jnp.where(visible, score * scales, -jnp.inf)

    def _group_fold(self, score, load, held, count):
        """Fold grouped PV back into each query's own online state."""
        ws = self.ws
        previous = jnp.concatenate([ws.maximum[held + i] for i in range(count)], axis=0)
        new_maximum = jnp.maximum(previous, jnp.max(score, axis=1, keepdims=True))
        scale = jnp.exp(previous - new_maximum)
        probabilities = jnp.exp(score - new_maximum)
        totals = jnp.sum(probabilities, axis=1, keepdims=True)
        high = (probabilities * 256).astype(jnp.float8_e4m3fn)
        values = load(pl.ds(0, LATENT))
        weighted = jax.lax.dot_general(high, values, (((1,), (0,)), ((), ())),
                                      preferred_element_type=jnp.float32) * (1.0 / 256)
        for i in range(count):
            rows = slice(i * self.tile, (i + 1) * self.tile)
            ws.denominator[held + i] = ws.denominator[held + i] * scale[rows] + totals[rows]
            ws.maximum[held + i] = new_maximum[rows]
            ws.accumulator[held + i] = ws.accumulator[held + i] * scale[rows] + weighted[rows]

    def _fold_pages(self, load, visible, held):
        """Fold a step's pages (``load(lanes)`` reads them) into the running
        softmax of state row ``held``."""
        self._fold_scores(self._scores(load, visible, held), load, held)

    def _scores(self, load, visible, held, *, tail_tokens=0):
        """State row ``held``'s masked scores of a step's pages."""
        # Heads-major (flash-attention layout): scores [head_rows head slots,
        # span * page_tokens tokens] = Q C^T, so the probabilities feed P V as is.
        latent = load(pl.ds(0, LANES))
        if tail_tokens:
            latent = latent[:tail_tokens]
        score = jax.lax.dot_general(self.ws.queries[held], latent, (((1, ), (1, )), ((), ())),
                                    preferred_element_type=jnp.float32)
        if self.fp8:
            score = score * self.ws.query_scale[held]
        if tail_tokens:
            score = jnp.pad(score, ((0, 0), (0, self.span * self.page_tokens - tail_tokens)), constant_values=-jnp.inf)
        return jnp.where(visible, score, -jnp.inf)

    def _fold_scores(self, score, load, held, *, tail_tokens=0):
        """Fold a step's ``score`` into the running softmax of state row
        ``held``; the values are loaded after the softmax (a load, not a
        live 320-vreg value across it)."""
        ws = self.ws
        previous = ws.maximum[held]  # [head_rows, 1]
        new_maximum = jnp.maximum(previous, jnp.max(score, axis=1, keepdims=True))
        scale = jnp.exp(previous - new_maximum)
        probabilities = jnp.exp(score - new_maximum)  # [head_rows, span * page_tokens]
        ws.denominator[held] = ws.denominator[held] * scale + jnp.sum(probabilities, axis=1, keepdims=True)
        ws.maximum[held] = new_maximum
        # Keep FP32 probabilities for the denominator; use scaled E4M3 for PV.
        high = ((probabilities * 256).astype(jnp.float8_e4m3fn) if self.fp8
                else probabilities.astype(jnp.bfloat16))
        values = load(pl.ds(0, LATENT))
        if tail_tokens:
            values = values[:tail_tokens]
            high = high[:, :tail_tokens]
        weighted = jax.lax.dot_general(high, values, (((1, ), (0, )), ((), ())),
                                       preferred_element_type=jnp.float32)
        if self.fp8:
            weighted = weighted * (1.0 / 256)
        elif self.probability_hilo:
            low = (probabilities - high.astype(jnp.float32)).astype(jnp.bfloat16)
            weighted += jax.lax.dot_general(low, values, (((1, ), (0, )), ((), ())),
                                            preferred_element_type=jnp.float32)
        ws.accumulator[held] = ws.accumulator[held] * scale + weighted

    def _clean_step(self, slot, row, local_step):
        """A step's pages as 32-bit rows ``[span * page_rows, 640]`` (a word
        packs ``packing`` tokens), with vLLM's stale bytes zeroed in place:
        lanes 576:640 of every token and every token at or past the row's
        limit (a step's visible offsets are a prefix). Stores into the buffer,
        not a masked copy of the whole tile as a value (which spills)."""
        ws, rows = self.ws, self.span * self.page_tokens
        words = ws.pages.bitcast(jnp.uint32).at[slot].reshape(self.span * self.page_rows, LANES)
        if (self.batch in (16, 32, 48, 64) or self.heads != HEADS_PER_RANK
                or (self.batch, self.sequence_rows) == (8, 1)):
            pad_shape = (self.span * self.page_rows, LANES - LATENT)
            pad_lane = jax.lax.broadcasted_iota(jnp.int32, pad_shape, 1)
            tpu.store(words.at[:, pl.ds(LATENT, LANES - LATENT)],
                      jnp.zeros(pad_shape, jnp.uint32), mask=pad_lane >= 64)
        else:
            pad = words[:, pl.ds(LATENT, LANES - LATENT)]  # a word's two tokens share its lane
            pad_lane = jax.lax.broadcasted_iota(jnp.int32, pad.shape, 1)
            words[:, pl.ds(LATENT, LANES - LATENT)] = jnp.where(pad_lane < 64, pad, jnp.zeros((), pad.dtype))
        limit = self._limit(row)
        cut = 0  # visible offsets
        for k in range(self.span):
            first_token = ((local_step * self.span + k) * self.group + self.owner) * self.page_tokens
            cut = cut + jnp.clip(limit - first_token, 0, self.page_tokens)
        tile_tokens = 8 * self.packing
        tile = cut // tile_tokens  # one native eight-word-row tile

        @pl.when(tile < rows // tile_tokens)
        def boundary():  # the tile holding the cut
            at = pl.ds(pl.multiple_of(tile * 8, 8), 8)
            block = pltpu.bitcast(words[at, :], self.cache.dtype)
            offset = tile * tile_tokens + jax.lax.broadcasted_iota(jnp.int32, block.shape, 0)
            words[at, :] = pltpu.bitcast(jnp.where(offset < cut, block, jnp.zeros((), block.dtype)), jnp.uint32)

        @pl.loop(tile + 1, rows // tile_tokens)
        def invisible(t):
            words[pl.ds(pl.multiple_of(t * 8, 8), 8), :] = jnp.zeros((8, LANES), jnp.uint32)

        return words

    def _pack_partials(self, first, end, base=0):
        """Rows ``first..end``: normalized outputs and log-sum-exps into each
        peer's send packet (BF16; the lse as a high and a low part); state
        and packet rows are relative to ``base``."""
        ws, group = self.ws, self.group
        for b in range(first, end):
            held = b - base
            total_weight = ws.denominator[held]  # [head_rows, 1]
            row_maximum = ws.maximum[held]
            has = total_weight > 0
            normalized = jnp.where(has, ws.accumulator[held] / jnp.where(has, total_weight, 1.0), 0.0)
            lse = jnp.where(has, row_maximum + jnp.log(jnp.where(has, total_weight, 1.0)), NEG)
            target = pl.ds(self.slots * held, self.slots)
            lse_high = lse.astype(jnp.bfloat16).astype(jnp.float32)
            lane = jax.lax.broadcasted_iota(jnp.int32, (lse.shape[0], LANES - LATENT), 1)
            lse_lanes = jnp.where(lane == 0, lse_high, jnp.where(lane == 1, lse - lse_high, 0.0))
            for peer in range(group):
                heads_of_peer = slice(self.slots * peer, self.slots * (peer + 1))
                ws.send[peer, target, :] = jnp.concatenate(
                    (normalized[heads_of_peer], lse_lanes[heads_of_peer]), axis=1).astype(jnp.bfloat16)

    def prime(self):
        """Issue the first page DMAs (call as early as the positions allow)."""
        for step in range(self.buffers - 1):
            @pl.when(step < self.total)
            def prime(step=step):
                self._start_step(step)

    def row_groups(self):
        """``[(first row, end row)]``: the batch in ``packet_layout``'s groups."""
        groups = self.groups
        bounds = [self.batch * g // groups for g in range(groups + 1)]
        return [(bounds[g], bounds[g + 1]) for g in range(groups)]

    def attend(self, queries, new_latents, parity, between: Callable[[], None] | None = None):
        """``queries`` [B, heads, 640] FP32 absorbed (scale folded in),
        ``new_latents`` [B, 640] FP32, ``parity`` alternating per invocation
        (MLA layer). ``between`` runs while the query gather is in flight.
        Returns the merged latent-space output [B, heads, 512] FP32.

        The rows run in groups (``row_groups``): every group's queries go out
        first; then per group, in order, its gather is awaited, its pages
        attended and its partials sent -- so a group's local attention runs
        while the next group's queries and the previous group's partials are
        in flight -- and the merge follows the last combine."""
        if chunked(self.batch):
            return self._attend_chunked(queries, new_latents, between)
        ws, group, member, base, batch = self.ws, self.group, self.member, self.base, self.batch
        heads = group * self.slots
        groups = self.row_groups()
        ws.own_rows[...] = jnp.zeros(ws.own_rows.shape, jnp.float32)
        ws.new_rows[...] = jnp.zeros(ws.new_rows.shape, jnp.float32)
        for b in range(batch):
            ws.own_rows[pl.ds(self.slots * b, self.heads), :] = queries[b]
            ws.new_rows[pl.ds(self.slots * b, self.heads), :] = jnp.broadcast_to(
                new_latents[b][None, :], (self.heads, LANES))

        def rows_of(g):
            first, end = groups[g]
            return pl.ds(self.slots * first, self.slots * (end - first))

        # 1. q all-gather into slot `member` of every peer, group by group.
        ws.gathered[parity, member] = (_pack_fp8_query(ws.own_rows[...]) if ws.gathered.dtype == jnp.uint8
                                       else ws.own_rows[...].astype(jnp.bfloat16))
        for g in range(len(groups)):
            for peer in range(0 if "gather" in self.skip else group):
                @pl.when(peer != member)
                def send_query(peer=peer, g=g):
                    tpu.make_async_remote_copy(
                        ws.gathered.at[parity, member, rows_of(g)], ws.gathered.at[parity, member, rows_of(g)],
                        ws.gather_semaphores.at[0, parity, g], ws.gather_semaphores.at[1, parity, g],
                        device_id=(base + peer, ), device_id_type=pl.DeviceIdType.MESH,
                    ).start()
        if between is not None:
            between()
        ws.maximum[...] = jnp.full(ws.maximum.shape, -jnp.inf, jnp.float32)
        ws.denominator[...] = jnp.zeros(ws.denominator.shape, jnp.float32)
        ws.accumulator[...] = jnp.zeros(ws.accumulator.shape, jnp.float32)

        for g, (first, end) in enumerate(groups):
            others = ws.gathered.at[parity, pl.ds(0, group - 1), rows_of(g)]
            if "gather" not in self.skip:
                tpu.make_async_copy(others, others, ws.gather_semaphores.at[0, parity, g]).wait()
                tpu.make_async_copy(others, others, ws.gather_semaphores.at[1, parity, g]).wait()
            for b in range(first, end):
                # [G, slots, 640] -> [slots G, 640] (slot `slots` d + h is member
                # d's head h), padded to the tile's rows.
                gathered = ws.gathered[parity, :, pl.ds(self.slots * b, self.slots), :].reshape(heads, LANES)
                padding = ((0, self.tile - heads), (0, 0))
                if ws.gathered.dtype == jnp.uint8:
                    query, scale = _unpack_fp8_query(gathered)
                    ws.queries[b] = jnp.pad(query, padding)
                    ws.query_scale[b] = jnp.pad(scale, padding, constant_values=1)
                else:
                    self._store_query(b, jnp.pad(gathered.astype(jnp.float32), padding))

            # 2. Local attention over this rank's pages (the group's steps).
            # Attention DP's group rows pipeline the fold as chunked calls do.
            if self.heads != HEADS_PER_RANK:
                self._attend_pipeline(self.starts[first], self.starts[first] if "attend" in self.skip
                                      else self.starts[end], 0)
            else:
                @pl.loop(self.starts[first], self.starts[first] if "attend" in self.skip else self.starts[end])
                def attend(step):
                    self._attend_step(step)

            # 3. Normalize and send each owner its heads' rows.
            self._pack_partials(first, end)
            for peer in range(0 if "combine" in self.skip else group):
                @pl.when(peer != member)
                def send_partial(peer=peer, g=g):
                    tpu.make_async_remote_copy(
                        ws.send.at[peer, rows_of(g)], ws.receive.at[parity, member, rows_of(g)],
                        ws.combine_semaphores.at[0, parity, g], ws.combine_semaphores.at[1, parity, g],
                        device_id=(base + peer, ), device_id_type=pl.DeviceIdType.MESH,
                    ).start()
            ws.receive[parity, member, rows_of(g)] = ws.send[member, rows_of(g)]
        for g in range(len(groups)):
            others = ws.receive.at[parity, pl.ds(0, group - 1), rows_of(g)]
            if "combine" not in self.skip:
                tpu.make_async_copy(others, others, ws.combine_semaphores.at[0, parity, g]).wait()
                tpu.make_async_copy(others, others, ws.combine_semaphores.at[1, parity, g]).wait()

        # 4. Log-sum-exp merge of the G partials and the new token.
        partials = ws.receive[parity].astype(jnp.float32)  # [G, rows, 640]
        lse = partials[:, :, LATENT:LATENT + 1] + partials[:, :, LATENT + 1:LATENT + 2]
        if self.sequence_rows == 1:
            new_score = jnp.sum(ws.own_rows[...] * ws.new_rows[...], axis=1, keepdims=True)  # [rows, 1]
            top = jnp.maximum(jnp.max(lse, axis=0), new_score)
            weights = jnp.exp(lse - top[None])
            new_weight = jnp.exp(new_score - top)
            merged = (jnp.sum(weights * partials[:, :, :LATENT], axis=0)
                      + new_weight * ws.new_rows[:, :LATENT]) / (jnp.sum(weights, axis=0) + new_weight)
        else:
            # Row b's head slots also attend the new tokens of rows j <= b of
            # its sequence (every rank holds all rows' new latents).
            latents = new_latents  # [B, 640]
            scores = jax.lax.dot_general(ws.own_rows[...], latents, (((1, ), (1, )), ((), ())),
                                         precision=jax.lax.Precision.HIGHEST,
                                         preferred_element_type=jnp.float32)  # [rows, B]
            slot_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0) // self.slots
            key_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
            causal = ((key_row // self.sequence_rows == slot_row // self.sequence_rows)
                      & (key_row <= slot_row))
            scores = jnp.where(causal, scores, -jnp.inf)
            top = jnp.maximum(jnp.max(lse, axis=0), jnp.max(scores, axis=1, keepdims=True))
            weights = jnp.exp(lse - top[None])
            new_weights = jnp.exp(scores - top)  # [rows, B]
            new_values = jax.lax.dot_general(new_weights, latents[:, :LATENT], (((1, ), (0, )), ((), ())),
                                             precision=jax.lax.Precision.HIGHEST,
                                             preferred_element_type=jnp.float32)
            merged = ((jnp.sum(weights * partials[:, :, :LATENT], axis=0) + new_values)
                      / (jnp.sum(weights, axis=0) + jnp.sum(new_weights, axis=1, keepdims=True)))
        return jnp.stack([merged[self.slots * b:self.slots * b + self.heads] for b in range(batch)])


def _merge_received(received, own_rows, new_rows, group):
    """Log-sum-exp merge of the ``group`` peers' partials in ``received``
    [G, rows, 640] (a ref: normalized outputs, lse hi/lo in lanes 512:514)
    with the new token (``own_rows`` the rows' queries, ``new_rows`` its
    latent per head slot): [rows, 512]. Online over the peers in a loop, so
    one peer's [rows, 640] is live at a time (bounds register pressure)."""
    top = jnp.sum(own_rows * new_rows, axis=1, keepdims=True)  # [rows, 1]: the new token's score
    accumulator = new_rows[:, :LATENT]
    denominator = jnp.ones_like(top)

    def fold(peer, carry):
        top, accumulator, denominator = carry
        partial = received[peer].astype(jnp.float32)  # [rows, 640]
        lse = partial[:, LATENT:LATENT + 1] + partial[:, LATENT + 1:LATENT + 2]
        new_top = jnp.maximum(top, lse)
        keep, weight = jnp.exp(top - new_top), jnp.exp(lse - new_top)
        return (new_top, accumulator * keep + weight * partial[:, :LATENT], denominator * keep + weight)

    _, accumulator, denominator = jax.lax.fori_loop(0, group, fold, (top, accumulator, denominator))
    return accumulator / denominator


def _merge_received_sequences(received, own_rows, latents, group, slots, sequence_rows, *, unroll=1):
    """``_merge_received`` for a chunk of whole multi-row sequences: each
    row's head slots also fold in, causally, the new tokens of its
    sequence's earlier rows in the chunk (``latents`` [chunk, 640]: every
    row's new latent; every rank holds them all), then the peers' partials
    online as there."""
    scores = jax.lax.dot_general(own_rows, latents, (((1, ), (1, )), ((), ())),
                                 precision=jax.lax.Precision.HIGHEST,
                                 preferred_element_type=jnp.float32)  # [rows, chunk]
    slot_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 0) // slots
    key_row = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1)
    causal = (key_row // sequence_rows == slot_row // sequence_rows) & (key_row <= slot_row)
    scores = jnp.where(causal, scores, -jnp.inf)
    top = jnp.max(scores, axis=1, keepdims=True)  # finite: a row sees its own token
    weights = jnp.exp(scores - top)
    accumulator = jax.lax.dot_general(weights, latents[:, :LATENT], (((1, ), (0, )), ((), ())),
                                      precision=jax.lax.Precision.HIGHEST,
                                      preferred_element_type=jnp.float32)
    denominator = jnp.sum(weights, axis=1, keepdims=True)

    def fold(peer, carry):
        top, accumulator, denominator = carry
        partial = received[peer].astype(jnp.float32)  # [rows, 640]
        lse = partial[:, LATENT:LATENT + 1] + partial[:, LATENT + 1:LATENT + 2]
        new_top = jnp.maximum(top, lse)
        keep, weight = jnp.exp(top - new_top), jnp.exp(lse - new_top)
        return (new_top, accumulator * keep + weight * partial[:, :LATENT], denominator * keep + weight)

    # Bounded unrolling exposes the next peer load without changing the
    # online merge order or materializing all 32 peers in FP32.
    def fold_group(peer, carry):
        for i in range(unroll):
            carry = fold(peer * unroll + i, carry)
        return carry
    _, accumulator, denominator = jax.lax.fori_loop(0, group // unroll, fold_group, (top, accumulator, denominator))
    return accumulator / denominator


def _pack_fp8_query(query):
    """FP8 query bytes plus a lossless FP32 scale in padding lanes 576:580.

    Round to BF16 first, matching the BF16 query path. Integer transport
    preserves every scale bit, including bytes that would encode NaNs if
    treated as FP8 values. The owner's FP32 query stays live separately for
    the new-token score.
    """
    query = query.astype(jnp.bfloat16).astype(jnp.float32)
    maximum = jnp.max(jnp.abs(query), axis=1, keepdims=True)
    scale = jnp.where(maximum > 0, maximum / 448.0, 1.0)
    # Perform masks/broadcasts in 32-bit lanes; packed-byte sublane broadcasts
    # are unsupported by Mosaic. Only the stored wire stays byte packed.
    payload = jax.lax.bitcast_convert_type((query / scale).astype(jnp.float8_e4m3fn), jnp.uint8).astype(jnp.uint32)
    bits = jax.lax.bitcast_convert_type(scale, jnp.uint32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, query.shape, 1)
    scale_bytes = (bits >> (((lanes - 576) % 4) * 8)) & 255
    return jnp.where(lanes < 576, payload, scale_bytes).astype(jnp.uint8)


def _unpack_fp8_query(packet):
    """Recover the exact query/scale and clear metadata before the QK dot."""
    packet = packet.astype(jnp.uint32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, packet.shape, 1)
    # Reassemble the scale with a lane reduction; narrow byte slices lose bits on TPU.
    words = packet << (((lanes - 576) % 4) * 8)
    bits = jnp.sum(jnp.where((lanes >= 576) & (lanes < 580), words, jnp.uint32(0)),
                   axis=1, keepdims=True)
    scale = jax.lax.bitcast_convert_type(bits, jnp.float32)
    payload = jnp.where(lanes < 576, packet, jnp.uint32(0)).astype(jnp.uint8)
    return jax.lax.bitcast_convert_type(payload, jnp.float8_e4m3fn), scale


def _attend_chunked(self, queries, new_latents, between):
    """``attend`` for a batch of ``C`` chunks (``chunk_layout``), pipelined:
    queries run two chunks ahead with four ring slots, else one; earlier
    chunks' combines overlap the current chunk's page attention. Packets
    rotate through ``R`` slots across layers (chunk c of MLA layer L: slot
    ``(L C + c) % R``). A query for chunk q leaves only after chunk q - R has
    merged: every peer has then sent that chunk's partials, so it has read the
    slot's queries. The chunk loop is rolled, so code and spills stay O(1) in
    the batch."""
    ws, group, member, base, batch = self.ws, self.group, self.member, self.base, self.batch
    chunk, slots = chunk_layout(batch, self.heads)[0], self.slots
    ring_slots = ws.gathered.shape[0]
    count = batch // chunk
    lookahead = 2 if ring_slots == 4 else 1
    rows = slots * chunk  # packet rows per chunk
    heads = group * slots
    layer = self.cache.layer if self.cache.layer is not None else 0
    ws.own_rows[...] = jnp.zeros(ws.own_rows.shape, jnp.float32)
    ws.new_rows[...] = jnp.zeros(ws.new_rows.shape, jnp.float32)
    for b in range(batch):
        c, j = divmod(b, chunk)
        ws.own_rows[c, pl.ds(slots * j, self.heads), :] = queries[b]
        ws.new_rows[c, pl.ds(slots * j, self.heads), :] = jnp.broadcast_to(
            new_latents[b][None, :], (self.heads, LANES))
    # Page steps before each row (scalars), for a chunk's step range.
    steps = [(self.counts[b] + self.span - 1) // self.span for b in range(batch)]

    def first_step(first):
        total = jnp.int32(0)
        for b in range(batch):
            total = total + jnp.where(b < first, steps[b], 0)
        return total

    def ring(c):
        return (layer * count + c) % ring_slots

    def send_queries(c):
        slot = ring(c)
        own = ws.own_rows[c, :rows, :]
        ws.gathered[slot, member] = (_pack_fp8_query(own) if ws.gathered.dtype == jnp.uint8
                                     else own.astype(jnp.bfloat16))
        for peer in range(0 if "gather" in self.skip else group):
            @pl.when(peer != member)
            def send(peer=peer):
                tpu.make_async_remote_copy(
                    ws.gathered.at[slot, member], ws.gathered.at[slot, member],
                    ws.gather_semaphores.at[0, slot], ws.gather_semaphores.at[1, slot],
                    device_id=(base + peer, ), device_id_type=pl.DeviceIdType.MESH,
                ).start()

    def wait_gather(c):
        if "gather" in self.skip:
            return
        slot = ring(c)
        others = ws.gathered.at[slot, pl.ds(0, group - 1)]
        tpu.make_async_copy(others, others, ws.gather_semaphores.at[0, slot]).wait()
        tpu.make_async_copy(others, others, ws.gather_semaphores.at[1, slot]).wait()

    def send_partials(c):
        slot = ring(c)
        for peer in range(0 if "combine" in self.skip else group):
            @pl.when(peer != member)
            def send(peer=peer):
                tpu.make_async_remote_copy(
                    ws.send.at[peer], ws.receive.at[slot, member],
                    ws.combine_semaphores.at[0, slot], ws.combine_semaphores.at[1, slot],
                    device_id=(base + peer, ), device_id_type=pl.DeviceIdType.MESH,
                ).start()
        ws.receive[slot, member] = ws.send[member]

    def wait_sent(c):  # this rank's partials of chunk c have left the send buffer
        if "combine" in self.skip:
            return
        others = ws.receive.at[ring(c), pl.ds(0, group - 1)]
        tpu.make_async_copy(others, others, ws.combine_semaphores.at[0, ring(c)]).wait()

    def wait_received(c):
        if "combine" in self.skip:
            return
        others = ws.receive.at[ring(c), pl.ds(0, group - 1)]
        tpu.make_async_copy(others, others, ws.combine_semaphores.at[1, ring(c)]).wait()

    def merge(c):
        if "merge" in self.skip:
            return
        if self.sequence_rows == 1:
            ws.merged[c, :rows, :] = _merge_received(
                ws.receive.at[ring(c)], ws.own_rows[c, :rows, :], ws.new_rows[c, :rows, :], group)
            return
        # Whole sequences per chunk: fold each row's earlier in-step tokens too.
        latents = jnp.concatenate(
            [ws.new_rows[c, pl.ds(slots * j, 1), :] for j in range(chunk)], axis=0)  # [chunk, 640]
        ws.merged[c, :rows, :] = _merge_received_sequences(
            ws.receive.at[ring(c)], ws.own_rows[c, :rows, :], latents, group, slots,
            self.sequence_rows,
            unroll=4 if self.fp8 and _grouped_spec(batch, group, self.heads, self.sequence_rows) else 1)

    for initial in range(min(lookahead, count)):
        send_queries(initial)
    if between is not None:
        between()

    @pl.loop(0, count)
    def chunk_step(c):
        done = c + lookahead - ring_slots  # merge before issuing the query that reuses this slot

        @pl.when(done >= 0)
        def merge_done():
            wait_received(done)
            merge(done)

        @pl.when(c + lookahead < count)
        def send_next():
            send_queries(c + lookahead)

        wait_gather(c)
        for j in range(0 if "build" in self.skip else chunk):
            # [G, slots, 640] -> [slots G, 640] (slot `slots` d + h is member d's head h),
            # padded to the tile's rows.
            gathered = ws.gathered[ring(c), :, pl.ds(slots * j, slots), :].reshape(heads, LANES)
            padding = ((0, self.tile - heads), (0, 0))
            if ws.gathered.dtype == jnp.uint8:
                query, scale = _unpack_fp8_query(gathered)
                ws.queries[j] = jnp.pad(query, padding)
                ws.query_scale[j] = jnp.pad(scale, padding, constant_values=1)
            else:
                self._store_query(j, jnp.pad(gathered.astype(jnp.float32), padding))
        ws.maximum[...] = jnp.full(ws.maximum.shape, -jnp.inf, jnp.float32)
        ws.denominator[...] = jnp.zeros(ws.denominator.shape, jnp.float32)
        ws.accumulator[...] = jnp.zeros(ws.accumulator.shape, jnp.float32)
        first = chunk * c

        self._attend_range(first_step(first),
                           first_step(first) if "attend" in self.skip else first_step(first + chunk), first)

        @pl.when(c >= 1)
        def free_send_buffer():
            wait_sent(c - 1)

        self._pack_partials(0, 0 if "pack" in self.skip else chunk)  # the chunk's rows: state rows 0..chunk
        send_partials(c)

    if batch != 32:
        wait_sent(count - 1)
    for c in range(max(count - ring_slots + lookahead, 0), count):
        wait_received(c)
        merge(c)
    if batch == 32:
        # Merges read receive/own/new buffers, leaving the final send free
        # to drain in parallel. Finish it before the next layer can pack.
        wait_sent(count - 1)
    return jnp.stack([ws.merged[b // chunk, pl.ds(slots * (b % chunk), self.heads), :]
                      for b in range(batch)])


DcpAttention._attend_chunked = _attend_chunked
