# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prototype: decode index scoring with the page fetch overlapped with compute.

Same arithmetic as ``index_selection.decode._flat_decode_scores`` (2D output,
halfplane keys, fused masks), so scores are expected bitwise equal. What changes
is how the next key tile's page DMAs are issued. The incumbent issues them in
predicated regions (a contiguity test, then 32 per-page copies, each with a
per-bank branch), which the LLO scheduler cannot overlap with the dot and the
head-sum epilogue. Here the key-tile loop is unrolled by two with static
buffer slots, the next tile's page copies are issued unconditionally in the same
basic block as the compute, and page addresses use a shift instead of a floor
division.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DIM = 128
ROW_WORDS = 128
# Records per int32[128] word row, by key storage format. Only "bf16" is the
# serving format; the others exist to price narrower keys (scores differ).
RECORDS_PER_ROW = {"bf16": 2, "fp8_widen": 4, "fp8_mxu": 4, "fp4_widen": 8}


def _fp4_e2m1_to_bf16(words):
    """Widen eight e2m1 nibbles per int32 into eight [rows, 128] BF16 planes.

    Per-32 E8M0 scales are not applied, so this is a lower bound on the cost of
    the reference format's widening.
    """
    planes = []
    for i in range(8):
        n = jax.lax.shift_right_logical(words, 4 * i) & 0xF
        sign = (n & 8) << 28
        exponent = jax.lax.shift_right_logical(n, 1) & 3
        mantissa = n & 1
        normal = ((exponent + 126) << 23) | (mantissa << 22)
        bits = jnp.where(exponent == 0, mantissa * 0x3F000000, normal) | sign
        planes.append(jax.lax.bitcast_convert_type(bits, jnp.float32))
    return jnp.concatenate(planes, axis=0).astype(jnp.bfloat16)


def _score_requests(
    members,
    row_lengths,
    pool,
    out,
    words,
    sems,
    *,
    page_records,
    tile,
    key_format,
):
    """Score requests that share one key-tile loop.

    ``members`` holds (pages, limit, q, weights, row_offset, out_row_offset)
    per request; ``words`` is [depth slots, len(members), tile rows, 128], with
    one DMA semaphore per slot in ``sems``. The loop
    runs to the longest member; tiles past a shorter member's length are
    computed and fully masked, so outputs equal the per-request loops.
    """
    per_row = RECORDS_PER_ROW[key_format]
    shift = per_row.bit_length() - 1
    page_rows = page_records // per_row
    tile_pages = tile // page_records
    window_start = pl.program_id(1) * out.shape[1]
    banks = pool if isinstance(pool, tuple) else (pool,)
    total_rows = sum(b.shape[0] for b in banks)
    steps = functools.reduce(
        jnp.maximum,
        [
            pl.cdiv(jnp.clip(m[1][0] - window_start, 0, out.shape[1]), tile)
            for m in members
        ],
    )
    queries = []
    for _, _, q_ref, w_ref, _, _ in members:
        rows, heads, _ = q_ref.shape
        query = q_ref[...].reshape(rows * heads, DIM)
        if key_format == "fp8_mxu":
            query = query.astype(jnp.float8_e4m3fn)
        queries.append((query, w_ref[...], rows, heads))

    def copy(address, count, dest, sem):
        if len(banks) == 1:
            pltpu.make_async_copy(
                banks[0].at[pl.ds(address, count), :], dest, sem
            ).start()
            return
        lo, hi = banks

        # Clamped so that a misaligned page id cannot read past a bank: with
        # bounds checks disabled, every address must be in range by construction.
        @pl.when(address < lo.shape[0])
        def _():
            start = jnp.minimum(address, lo.shape[0] - count)
            pltpu.make_async_copy(lo.at[pl.ds(start, count), :], dest, sem).start()

        @pl.when(address >= lo.shape[0])
        def _():
            pltpu.make_async_copy(
                hi.at[pl.ds(address - lo.shape[0], count), :], dest, sem
            ).start()

    def fetch(block, slot):
        base = (window_start // tile + block) * tile_pages
        for m, (pages, *_rest) in enumerate(members):
            for p in range(tile_pages):
                # Floor division by a power of two is an arithmetic shift.
                address = jnp.clip(
                    jax.lax.shift_right_arithmetic(pages[base + p], shift),
                    0,
                    total_rows - page_rows,
                )
                copy(
                    address,
                    page_rows,
                    words.at[slot, m, pl.ds(p * page_rows, page_rows), :],
                    sems.at[slot],
                )

    def wait(slot):
        pltpu.make_async_copy(words.at[slot], words.at[slot], sems.at[slot]).wait()

    def compute(block, slot):
        for m, (pages, _, _, _, row_offset, out_row_offset) in enumerate(members):
            query, weights, rows, heads = queries[m]
            if key_format == "bf16":
                keys = pltpu.bitcast(words[slot, m], jnp.bfloat16)
            elif key_format == "fp4_widen":
                keys = _fp4_e2m1_to_bf16(words[slot, m])
            else:
                keys = pltpu.bitcast(words[slot, m], jnp.float8_e4m3fn)
                if key_format == "fp8_widen":
                    keys = keys.astype(jnp.bfloat16)
            logits = jax.lax.dot_general(
                query,
                keys,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).reshape(rows, heads, tile)
            values = (jnp.maximum(logits, 0) * weights).sum(1)
            position = window_start + block * tile + jnp.arange(tile)
            causal = (
                position[None, :]
                < jnp.stack([row_lengths[row_offset + i] for i in range(rows)])[:, None]
            )
            page_valid = jnp.concatenate(
                [
                    jnp.broadcast_to(
                        (
                            pages[(window_start // tile + block) * tile_pages + p] >= 0
                        ).astype(jnp.int32),
                        (1, page_records),
                    )
                    for p in range(tile_pages)
                ],
                axis=1,
            )
            values = jnp.where(causal & (page_valid != 0), values, -jnp.inf)
            out[pl.ds(out_row_offset, rows), pl.ds(block * tile, tile)] = values

    last = steps - 1
    depth = words.shape[0]

    # Prefetch depth - 1 tiles ahead; past the end, refetch the last tile.
    @pl.when(steps > 0)
    def _():
        for d in range(depth - 1):
            fetch(jnp.minimum(d, last), d)

    def visit(block, slot):
        wait(slot)
        # No branch around the copies: the next fetch is unconditional.
        fetch(jnp.minimum(block + depth - 1, last), (slot + depth - 1) % depth)
        compute(block, slot)

    def ring(i, _):
        visit(depth * i, 0)
        for j in range(1, depth):
            block = depth * i + j
            pl.when(block < steps)(functools.partial(visit, block, j))

    jax.lax.fori_loop(0, pl.cdiv(steps, depth), ring, None)

    # Drain the depth - 1 refetches still in flight (virtual tiles
    # steps .. steps + depth - 2, which went to slots v % depth).
    @pl.when(steps > 0)
    def _():
        for t in range(depth - 1):
            wait((steps + t) % depth)


def _flat_kernel(
    pages,
    limits,
    query_limits,
    q,
    weights,
    pool,
    out,
    words,
    sems,
    *,
    group,
    joint,
    page_records,
    tile,
    key_format,
):
    queries = q.shape[0] // group
    page_count = pages.shape[0] // limits.shape[0]
    out[...] = jnp.full(out.shape, -jnp.inf, jnp.float32)
    members = []
    for r in range(group):
        request = pl.program_id(0) * group + r
        members.append(
            (
                pages.at[pl.ds(request * page_count, page_count)],
                limits.at[pl.ds(request, 1)],
                q.at[pl.ds(r * queries, queries)],
                weights.at[pl.ds(r * queries, queries)],
                request * queries,
                r * queries,
            )
        )
    kw = dict(page_records=page_records, tile=tile, key_format=key_format)
    if joint:
        _score_requests(members, query_limits, pool, out, words, sems, **kw)
    else:
        for member in members:
            _score_requests([member], query_limits, pool, out, words, sems, **kw)


@functools.partial(
    jax.jit,
    static_argnames=(
        "page_records",
        "key_tile",
        "request_group",
        "interpret",
        "disable_bounds_checks",
        "key_format",
        "joint",
        "depth",
    ),
)
def overlapped_decode_scores(
    q,
    weights,
    pool,
    pages,
    lengths,
    *,
    page_records,
    key_tile,
    request_group,
    interpret=False,
    disable_bounds_checks=True,
    key_format="bf16",
    joint=False,
    depth=2,
):
    """``grouped_decode_scores(..., fuse_mask=True, halfplane=True, request_group=g)``.

    ``pool`` holds ``RECORDS_PER_ROW[key_format]`` records per int32[128] row.
    """
    requests, queries, heads, dim = q.shape
    width = pages.shape[1] * page_records
    rows = request_group * queries
    if (
        dim != DIM
        or requests % request_group
        or rows % 8
        or width % key_tile
        or key_tile % page_records
    ):
        raise ValueError("unsupported shape")
    window = width
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _flat_kernel,
            group=request_group,
            joint=joint,
            page_records=page_records,
            tile=key_tile,
            key_format=key_format,
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
                    (
                        depth,
                        request_group if joint else 1,
                        key_tile // RECORDS_PER_ROW[key_format],
                        ROW_WORDS,
                    ),
                    jnp.int32,
                ),
                pltpu.SemaphoreType.DMA((depth,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=None
            if interpret
            else int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes),
            # Every DMA address above is clamped into its bank.
            disable_bounds_checks=disable_bounds_checks,
        ),
        interpret=interpret,
        name="proto_overlapped_index_scores_2d",
    )(
        pages.reshape(-1),
        jnp.max(lengths, axis=1),
        lengths.reshape(-1),
        q.reshape(-1, heads, DIM),
        weights.reshape(-1, heads, 1),
        pool,
    )
