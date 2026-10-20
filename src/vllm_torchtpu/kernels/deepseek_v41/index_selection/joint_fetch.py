# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode index scores with one key-tile loop shared by a request group.

Same arithmetic and output as the request-major 2D path of
`decode.grouped_decode_scores` (bitwise). What differs is the page fetch:

* The ``group`` requests of a grid step share one key-tile loop, so their
  page copies for the next tile are in flight together (``group`` x 1 MiB at
  the serving shape instead of 1 MiB).
* The copies are issued unconditionally (no contiguity test; past the end the
  last tile is fetched again and drained), into statically indexed buffer
  slots, with one DMA semaphore per slot.
* Runtime DMA bounds checks are disabled for this kernel. Every copy address
  is clamped into the bank it reads (see ``copy``), and every VMEM destination
  is a static slice of the scratch, so no copy can leave its buffer.

Tiles past a shorter member's length are scored and masked to -inf, exactly as
the per-request loop leaves them.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_word_scores import (
    DIM,
    ROW_WORDS,
    SLOTS,
    unpack_word_rows,
)

# Key tiles prefetched ahead plus the one being scored. Deeper rings measured
# no faster at the serving shape (the shared loop is no longer DMA-latency
# bound) and cost another group x tile of VMEM per slot.
DEPTH = 2


def _kernel(
    pages,
    limits,
    row_lengths,
    q,
    weights,
    pool,
    out,
    words,
    *scratch,
    group,
    page_records,
    tile,
    halfplane,
):
    unpacked, sems = (None, scratch[0]) if halfplane else scratch
    queries = q.shape[0] // group
    heads = q.shape[1]
    page_count = pages.shape[0] // limits.shape[0]
    page_rows = page_records // SLOTS
    tile_pages = tile // page_records
    window_start = pl.program_id(1) * out.shape[1]
    first_tile = window_start // tile
    banks = pool if isinstance(pool, tuple) else (pool,)
    total_rows = sum(bank.shape[0] for bank in banks)
    requests = [pl.program_id(0) * group + r for r in range(group)]
    steps = functools.reduce(
        jnp.maximum,
        [
            pl.cdiv(jnp.clip(limits[request] - window_start, 0, out.shape[1]), tile)
            for request in requests
        ],
    )
    last = steps - 1
    out[...] = jnp.full(out.shape, -jnp.inf, jnp.float32)
    member_rows = [pl.ds(m * queries, queries) for m in range(group)]
    member_q = [q[rows].reshape(queries * heads, DIM) for rows in member_rows]
    member_w = [weights[rows] for rows in member_rows]

    def copy(address, dest, sem):
        """Copy ``page_rows`` rows at pool row ``address`` (already in range)."""
        if len(banks) == 1:
            pltpu.make_async_copy(
                banks[0].at[pl.ds(address, page_rows), :], dest, sem
            ).start()
            return
        lo, hi = banks

        # A page id that is not page-aligned could straddle the bank boundary;
        # clamp it into the low bank rather than read past it.
        @pl.when(address < lo.shape[0])
        def _():
            start = jnp.minimum(address, lo.shape[0] - page_rows)
            pltpu.make_async_copy(lo.at[pl.ds(start, page_rows), :], dest, sem).start()

        # address <= total_rows - page_rows, so this start is <= hi rows - page_rows.
        @pl.when(address >= lo.shape[0])
        def _():
            pltpu.make_async_copy(
                hi.at[pl.ds(address - lo.shape[0], page_rows), :], dest, sem
            ).start()

    def fetch(block, slot):
        base = (first_tile + block) * tile_pages
        for m, request in enumerate(requests):
            for p in range(tile_pages):
                page = pages[request * page_count + base + p]
                # Floor division by SLOTS == 2 (also for -1 pages), clamped so
                # the copy stays inside the pool.
                address = jnp.clip(
                    jax.lax.shift_right_arithmetic(page, 1), 0, total_rows - page_rows
                )
                copy(
                    address,
                    words.at[slot, m, pl.ds(p * page_rows, page_rows), :],
                    sems.at[slot],
                )

    def wait(slot):
        pltpu.make_async_copy(words.at[slot], words.at[slot], sems.at[slot]).wait()

    def score(block, slot):
        for m, request in enumerate(requests):
            keys = (
                pltpu.bitcast(words[slot, m], jnp.bfloat16)
                if halfplane
                else unpack_word_rows(words[slot, m], unpacked)
            )
            logits = jax.lax.dot_general(
                member_q[m],
                keys,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).reshape(queries, heads, tile)
            values = (jnp.maximum(logits, 0) * member_w[m]).sum(1)
            position = window_start + block * tile + jnp.arange(tile)
            causal = (
                position[None, :]
                < jnp.stack(
                    [row_lengths[request * queries + i] for i in range(queries)]
                )[:, None]
            )
            base = request * page_count + (first_tile + block) * tile_pages
            page_valid = jnp.concatenate(
                [
                    jnp.broadcast_to(
                        (pages[base + p] >= 0).astype(jnp.int32), (1, page_records)
                    )
                    for p in range(tile_pages)
                ],
                axis=1,
            )
            values = jnp.where(causal & (page_valid != 0), values, -jnp.inf)
            out[member_rows[m], pl.ds(block * tile, tile)] = values

    @pl.when(steps > 0)
    def _():
        for d in range(DEPTH - 1):
            fetch(jnp.minimum(d, last), d)

    def visit(block, slot):
        wait(slot)
        fetch(jnp.minimum(block + DEPTH - 1, last), (slot + DEPTH - 1) % DEPTH)
        score(block, slot)

    def ring(i, _):
        visit(DEPTH * i, 0)
        for j in range(1, DEPTH):
            block = DEPTH * i + j
            pl.when(block < steps)(functools.partial(visit, block, j))

    jax.lax.fori_loop(0, pl.cdiv(steps, DEPTH), ring, None)

    # Drain the DEPTH - 1 refetches still in flight: virtual tiles
    # steps .. steps + DEPTH - 2, which went to slots v % DEPTH.
    @pl.when(steps > 0)
    def _():
        for t in range(DEPTH - 1):
            wait((steps + t) % DEPTH)


def joint_decode_scores(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    pages: jax.Array,
    lengths: jax.Array,
    *,
    page_records: int,
    key_tile: int,
    window: int,
    request_group: int,
    halfplane: bool,
    interpret: bool,
) -> jax.Array:
    """Masked [requests * queries, width] scores; callers validate shapes.

    ``window`` columns per grid step, ``request_group`` requests per grid step
    (``request_group * queries`` a multiple of 8), as in `_flat_decode_scores`.
    """
    requests, queries, heads, _ = q.shape
    width = pages.shape[1] * page_records
    rows = request_group * queries
    if requests % request_group or rows % 8 or width % window or window % key_tile:
        raise ValueError("Joint fetch needs whole 8-row request groups and tiles")
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            group=request_group,
            page_records=page_records,
            tile=key_tile,
            halfplane=halfplane,
        ),
        out_shape=jax.ShapeDtypeStruct((requests * queries, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(requests // request_group, width // window),
            in_specs=(
                pl.BlockSpec((rows, heads, DIM), lambda i, j, *_: (i, 0, 0)),
                pl.BlockSpec((rows, heads, 1), lambda i, j, *_: (i, 0, 0)),
                (hbm, hbm) if isinstance(pool, tuple) else hbm,
            ),
            out_specs=pl.BlockSpec((rows, window), lambda i, j, *_: (i, j)),
            scratch_shapes=(
                pltpu.VMEM(
                    (DEPTH, request_group, key_tile // SLOTS, ROW_WORDS), jnp.int32
                ),
                *(
                    (pltpu.VMEM((key_tile, ROW_WORDS), jnp.int32),)
                    if not halfplane
                    else ()
                ),
                pltpu.SemaphoreType.DMA((DEPTH,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            # Two (24, 131072) f32 out blocks plus 8 MiB of key slots.
            vmem_limit_bytes=(
                None
                if interpret
                else int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes)
            ),
            # Every DMA address is clamped into its buffer above.
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name="decode_joint_paged_index_scores_2d",
    )(
        pages.reshape(-1),
        jnp.max(lengths, axis=1),
        lengths.reshape(-1),
        q.reshape(-1, heads, DIM),
        weights.reshape(-1, heads, 1),
        pool,
    )
