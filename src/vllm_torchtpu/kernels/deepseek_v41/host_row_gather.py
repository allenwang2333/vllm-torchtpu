# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather fixed-width table rows straight from TPU ``pinned_host`` memory.

Host-DRAM Engram tables (``TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD``) otherwise
need the sampled tokens on the host every step: hash, gather and H2D all sit
between two decode steps. Here the table lives in the TPU's pinned host memory
space and a Pallas kernel DMAs only the hashed rows into HBM, so the lookup
runs inside the compiled graph like a device table.

TPU host DMAs move whole ``(8, 128)`` int32 tiles (4 KiB): a one-row slice of
a host operand fails ``multiplier_in_bytes % 4096`` in LLO. Rows are therefore
packed ``ROWS_PER_TILE`` to a tile (``pack_rows``), the kernel fetches one tile
per requested row, and the row's words are picked out of the tile in HBM.
Host-to-VMEM DMA is unimplemented, so the kernel output is pinned to HBM.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

SUBLANES = 8
LANES = 128
TILE_WORDS = SUBLANES * LANES  # one 4 KiB host DMA
TILES_PER_MAJOR = 8  # [majors, 8, 1024]: 8 tiles side by side per major index
MAJOR_LANES = TILES_PER_MAJOR * LANES  # [majors, 8, MAJOR_LANES] host layout


def row_words(row_bytes: int) -> int:
    assert row_bytes % 4 == 0 and LANES % (row_bytes // 4) == 0, row_bytes
    return row_bytes // 4


def rows_per_tile(row_bytes: int) -> int:
    return TILE_WORDS // row_words(row_bytes)


def num_majors(num_rows: int, row_bytes: int) -> int:
    per_major = TILES_PER_MAJOR * rows_per_tile(row_bytes)
    return -(-num_rows // per_major)


def pack_rows(rows: np.ndarray, majors: int | None = None) -> np.ndarray:
    """``uint8[num_rows, row_bytes]`` -> ``int32[majors, 8, 1024]`` host tiles.

    Row ``r`` lands in tile ``t = r // rows_per_tile`` (major ``t // 8``,
    lanes ``[128 * (t % 8), 128 * (t % 8 + 1))``), sublane ``s // per_sub``
    and lanes ``w * (s % per_sub)`` onward within the tile, where
    ``s = r % rows_per_tile``, ``w`` is the row's word count and
    ``per_sub = 128 // w``. Missing tail rows are zero.
    """
    num_rows, row_bytes = rows.shape
    w = row_words(row_bytes)
    per_sub = LANES // w
    per_major = TILES_PER_MAJOR * rows_per_tile(row_bytes)
    majors = num_majors(num_rows, row_bytes) if majors is None else majors
    assert majors * per_major >= num_rows, (majors, per_major, num_rows)
    words = np.zeros((majors * per_major, w), dtype=np.int32)
    words[:num_rows] = np.ascontiguousarray(rows).view(np.int32)
    return np.ascontiguousarray(
        words.reshape(majors, TILES_PER_MAJOR, SUBLANES, per_sub, w)
        .transpose(0, 2, 1, 3, 4)
        .reshape(majors, SUBLANES, MAJOR_LANES)
    )


def _gather_kernel(majors_per_chunk, tile_ref, *refs):
    *chunk_refs, out_ref, sem = refs
    n = out_ref.shape[0]

    def copy(i, chunk, major, lanes):
        return pltpu.make_async_copy(
            chunk_refs[chunk].at[pl.ds(major, 1), :, pl.ds(lanes, LANES)],
            out_ref.at[pl.ds(i, 1)],
            sem,
        )

    def start(i, carry):
        tile = tile_ref[i]
        major = tile // TILES_PER_MAJOR
        lanes = pl.multiple_of((tile % TILES_PER_MAJOR) * LANES, LANES)
        if len(chunk_refs) == 1:
            copy(i, 0, major, lanes).start()
        else:
            chunk = major // majors_per_chunk
            local = major - chunk * majors_per_chunk
            for c in range(len(chunk_refs)):

                @pl.when(chunk == c)
                def _(c=c):
                    copy(i, c, local, lanes).start()

        return carry

    jax.lax.fori_loop(0, n, start, 0)

    def wait(i, carry):
        # Every copy moves one tile on the same semaphore.
        copy(0, 0, 0, 0).wait()
        return carry

    jax.lax.fori_loop(0, n, wait, 0)


def select_rows(tiles: jax.Array, slot: jax.Array, row_bytes: int) -> jax.Array:
    """Fetched ``int32[n, 8, 128]`` tiles -> each one's row ``slot[i]``.

    A tile's row ``s`` sits at sublane ``s // per_sub``, lanes
    ``w * (s % per_sub)`` onward (``pack_rows``), i.e. sub-row ``s`` of the
    tile viewed as ``[8 * per_sub, w]``.
    """
    w = row_words(row_bytes)
    sub_rows = tiles.reshape(tiles.shape[0], SUBLANES * (LANES // w), w)
    return jnp.take_along_axis(sub_rows, slot[:, None, None], axis=1)[:, 0]


def gather_rows_jax(
    rows: jax.Array, *chunks: jax.Array, row_bytes: int, interpret: bool = False
) -> jax.Array:
    """``int32[n]`` row ids -> ``int32[n, row_bytes // 4]`` from host chunks.

    ``chunks`` are consecutive ``pack_rows`` slabs, all but the last holding
    the same number of majors. Ids must be in range: the caller clamps them.
    ``interpret`` runs the Pallas interpreter (CPU tests).
    """
    per_tile = rows_per_tile(row_bytes)
    per_chunk = chunks[0].shape[0]
    assert all(c.shape[0] == per_chunk for c in chunks[:-1]), [c.shape for c in chunks]
    assert chunks[-1].shape[0] <= per_chunk, [c.shape for c in chunks]
    n = rows.shape[0]
    rows = rows.astype(jnp.int32)
    gathered = pl.pallas_call(
        functools.partial(_gather_kernel, per_chunk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pl.HOST)] * len(chunks),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA(())],
        ),
        out_shape=pltpu.HBM((n, SUBLANES, LANES), jnp.int32),
        name="host_row_gather",
        interpret=interpret,
    )(rows // per_tile, *chunks)
    return select_rows(gathered, rows % per_tile, row_bytes)
