# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request prefill scoring, with resident Q and pipelined packed K.

Uses a two-bank packed word cache and preserves FP32 multiply/reduce semantics.
"""

import contextlib
import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .. import index_key_layout
from ..index_word_scores import DIM, ROW_WORDS, SLOTS
from . import tuned_block_sizes


def supports_chunked_prefill(
    q: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    page_records: int | None,
) -> bool:
    """Static envelope for the measured chunk-2K packed-cache route."""
    banks = pool if isinstance(pool, tuple) else (pool,)
    return (
        q.shape == (2048, 32, DIM)
        and q.dtype == jnp.bfloat16
        and page_records is not None
        and 0 < page_records <= tuned_block_sizes.KEY_TILE
        and page_records % 16 == 0
        and tuned_block_sizes.KEY_TILE % page_records == 0
        and all(
            b.dtype == jnp.int32
            and b.ndim == 2
            and b.shape[1] == ROW_WORDS
            and b.shape[0] % (page_records // SLOTS) == 0
            for b in banks
        )
    )


def context_buckets(capacity: int) -> tuple[int, ...]:
    """Power-of-two score widths, capped at the page-aligned capacity."""
    tile = tuned_block_sizes.KEY_TILE
    capacity = pl.cdiv(capacity, tile) * tile
    widths = []
    width = tile
    while width < capacity:
        widths.append(width)
        width *= 2
    return (*widths, capacity)


def _kernel(
    pages,
    limit,
    q_ref,
    w_ref,
    pool,
    out,
    words,
    unpacked,
    sem,
    *,
    page_records,
    tile,
    coalesce,
    prefetch,
    profile_stages,
    row_lengths=None,
    row_offset=0,
    depth=2,
    native=False,
):
    """``depth`` key tiles are buffered: ``depth - 1`` fetches stay in flight.

    Depth 2 shares one DMA semaphore (one fetch in flight at every wait).
    Deeper pipelines need ``words`` [depth, ...] and ``sem`` DMA[depth].
    ``native`` pools (`index_key_layout`) bitcast each tile straight to BF16
    keys; ``unpacked`` is then unused.
    """
    scope = jax.named_scope if profile_stages else contextlib.nullcontext
    with scope("prefill_kernel_body"):
        rows, heads, _ = q_ref.shape
        query = q_ref[...].reshape(rows * heads, DIM)
        weights = w_ref[...]
        page_rows = page_records // SLOTS
        tile_pages = tile // page_records
        window_start = pl.program_id(1) * out.shape[1]
        steps = pl.cdiv(jnp.clip(limit[0] - window_start, 0, out.shape[1]), tile)
        out[...] = jnp.full(out.shape, -jnp.inf, jnp.float32)

        def slot_sem(bank):
            return sem.at[bank] if depth > 2 else sem

        def copy(address, count, dest, sem):
            if isinstance(pool, tuple):
                lo, hi = pool

                @pl.when(address < lo.shape[0])
                def low():
                    pltpu.make_async_copy(
                        lo.at[pl.ds(address, count), :], dest, sem
                    ).start()

                @pl.when(address >= lo.shape[0])
                def high():
                    pltpu.make_async_copy(
                        hi.at[pl.ds(address - lo.shape[0], count), :], dest, sem
                    ).start()
            else:
                pltpu.make_async_copy(
                    pool.at[pl.ds(address, count), :], dest, sem
                ).start()

        def fetch(block, bank):
            block += window_start // tile
            first = pages[block * tile_pages] // SLOTS
            contiguous = first >= 0
            for p in range(tile_pages):
                contiguous &= (
                    pages[block * tile_pages + p] // SLOTS == first + p * page_rows
                )
            total_rows = (
                sum(x.shape[0] for x in pool)
                if isinstance(pool, tuple)
                else pool.shape[0]
            )
            contiguous &= first + tile // SLOTS <= total_rows
            if isinstance(pool, tuple):
                contiguous &= (first >= pool[0].shape[0]) | (
                    first + tile // SLOTS <= pool[0].shape[0]
                )

            def scattered():
                for p in range(tile_pages):
                    address = jnp.clip(
                        pages[block * tile_pages + p] // SLOTS,
                        0,
                        total_rows - page_rows,
                    )
                    copy(
                        address,
                        page_rows,
                        words.at[bank, pl.ds(p * page_rows, page_rows), :],
                        slot_sem(bank),
                    )

            if coalesce:

                @pl.when(contiguous)
                def contiguous_copy():
                    copy(first, tile // SLOTS, words.at[bank], slot_sem(bank))

                @pl.when(~contiguous)
                def scattered_copy():
                    scattered()
            else:
                scattered()

        if prefetch:
            for ahead in range(depth - 1):

                @pl.when(ahead < steps)
                def first_fetch():
                    with scope("prefill_k_prologue_dma"):
                        fetch(jnp.int32(ahead), jnp.int32(ahead))

        def body(block, _):
            bank = block % depth
            if not prefetch:
                with scope("prefill_k_serial_dma"):
                    fetch(block, bank)
            with scope("prefill_k_dma_wait"):
                pltpu.make_async_copy(
                    words.at[bank], words.at[bank], slot_sem(bank)
                ).wait()

            if prefetch:
                # The slot of block - 1, whose compute has finished.
                @pl.when(block + depth - 1 < steps)
                def next_fetch():
                    with scope("prefill_k_prefetch_dma"):
                        fetch(block + depth - 1, (block + depth - 1) % depth)

            # Keep the fused vector/MXU region in one scope: splitting dot and
            # reduction introduces materialization/spills in the Mosaic compiler.
            with scope("prefill_unpack_dot_reduce_store"):
                keys = index_key_layout.rows_to_keys(words[bank], unpacked, native)
                logits = jax.lax.dot_general(
                    query,
                    keys,
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                ).reshape(rows, heads, tile)
                values = (jnp.maximum(logits, 0) * weights).sum(1)
                if row_lengths is not None:
                    position = window_start + block * tile + jnp.arange(tile)
                    causal = (
                        position[None, :]
                        < jnp.stack([row_lengths[row_offset + i] for i in range(rows)])[
                            :, None
                        ]
                    )
                    page_valid = jnp.concatenate(
                        [
                            jnp.broadcast_to(
                                (
                                    pages[
                                        (window_start // tile + block) * tile_pages + p
                                    ]
                                    >= 0
                                ).astype(jnp.int32),
                                (1, page_records),
                            )
                            for p in range(tile_pages)
                        ],
                        axis=1,
                    )
                    valid = causal & (page_valid != 0)
                    values = jnp.where(valid, values, -jnp.inf)
                out[:, pl.ds(block * tile, tile)] = values

        jax.lax.fori_loop(0, steps, body, None)


@functools.partial(
    jax.jit,
    static_argnames=(
        "page_records",
        "query_tile",
        "key_tile",
        "coalesce",
        "interpret",
        "prefetch",
        "profile_stages",
        "native_keys",
    ),
)
def prefill_index_scores(
    q: jax.Array,
    weights: jax.Array,
    pool: jax.Array | tuple[jax.Array, jax.Array],
    pages: jax.Array,
    row_limit: jax.Array,
    *,
    page_records: int,
    query_tile: int,
    key_tile: int,
    coalesce: bool = True,
    interpret: bool = False,
    prefetch: bool = True,
    profile_stages: bool = False,
    native_keys: bool = index_key_layout.NATIVE_KEYS,
) -> jax.Array:
    """Score one shared request; caller applies per-query causal/record masks.

    Q is split-order BF16 [T,H,128]; weights are FP32 [T,H,1]. Pages are
    first-record addresses [1,N]. The limit [1] bounds the shared scan.
    ``prefetch=False`` is a diagnostic serial-DMA ablation; ``profile_stages``
    enables standalone JAX trace scopes. Serving defaults remain uninstrumented.
    """
    if min(page_records, query_tile, key_tile) <= 0:
        raise ValueError("Tile and page sizes must be positive")
    tokens, heads, dim = q.shape
    banks = pool if isinstance(pool, tuple) else (pool,)
    width = pages.shape[1] * page_records
    if (
        tokens < 1
        or heads < 1
        or width < 1
        or pages.dtype != jnp.int32
        or row_limit.dtype != jnp.int32
        or dim != DIM
        or q.dtype != jnp.bfloat16
        or weights.dtype != jnp.float32
        or weights.shape != (tokens, heads, 1)
        or pages.shape[0] != 1
        or row_limit.shape != (1,)
        or tokens % query_tile
        or page_records % 16
        or key_tile % page_records
        or width % key_tile
        or any(
            b.dtype != jnp.int32
            or b.shape[1] != ROW_WORDS
            or b.shape[0] % (page_records // SLOTS)
            for b in banks
        )
    ):
        raise ValueError("Unsupported single-request prefill score shape or dtype")
    # Bound VMEM independently of context. Each window is stored in its
    # logical HBM columns before the full-row SparseCore selector runs.
    window = (
        width
        if width <= 2 * tuned_block_sizes.SCORE_WINDOW
        else math.gcd(width, max(key_tile, tuned_block_sizes.SCORE_WINDOW))
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            page_records=page_records,
            tile=key_tile,
            coalesce=coalesce,
            prefetch=prefetch,
            profile_stages=profile_stages,
            native=native_keys,
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(tokens // query_tile, width // window),
            in_specs=(
                pl.BlockSpec((query_tile, heads, DIM), lambda i, j, *_: (i, 0, 0)),
                pl.BlockSpec((query_tile, heads, 1), lambda i, j, *_: (i, 0, 0)),
                (hbm, hbm) if isinstance(pool, tuple) else hbm,
            ),
            out_specs=pl.BlockSpec((query_tile, window), lambda i, j, *_: (i, j)),
            scratch_shapes=(
                pltpu.VMEM((2, key_tile // SLOTS, ROW_WORDS), jnp.int32),
                pltpu.VMEM((8 if native_keys else key_tile, ROW_WORDS), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=(
                None
                if interpret
                else int(0.9 * pltpu.get_tpu_info().vmem_capacity_bytes)
            ),
        ),
        interpret=interpret,
        name="prefill_paged_bf16_index_scores",
    )(pages.reshape(-1), row_limit, q, weights, pool)
