# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merge token columns in VMEM and DMA only the updated FP8 pages."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .tuned_block_sizes import PACKED_FP8_WRITE_BATCH


def write_packed_pages(
    bank: jax.Array,
    bases: jax.Array,
    columns: jax.Array,
    values: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Update unique token slots; negative bases leave the bank unchanged."""
    count, page_rows = values.shape
    if not count:
        return bank
    page_tokens = bank.shape[1]
    batch = min(count, PACKED_FP8_WRITE_BATCH)
    valid = (bases >= 0) & (bases + page_rows <= bank.shape[0])
    bases, order = jax.lax.sort_key_val(
        jnp.where(valid, bases, bank.shape[0]), jnp.arange(count, dtype=jnp.int32)
    )
    columns, values = columns[order], values[order]
    last = jnp.concatenate((bases[:-1] != bases[1:], jnp.ones((1,), jnp.bool_)))
    ends = jax.lax.associative_scan(
        jnp.minimum,
        jnp.where(last, jnp.arange(count, dtype=jnp.int32) + 1, count),
        reverse=True,
    )
    first = jnp.concatenate((jnp.ones((1,), jnp.bool_), last[:-1]))
    bases = jnp.where(first & (bases < bank.shape[0]), bases, -1)
    bases, columns, ends = (
        jnp.pad(x, (0, -count % batch), constant_values=-1)
        for x in (bases, columns, ends)
    )
    values = jnp.pad(values.T, ((0, 0), (0, -count % page_tokens)))

    def kernel(
        base_ref, column_ref, end_ref, value_ref, _bank_ref, out_ref, pages, sem
    ):
        def copies(*, write, wait):
            def body(slot, carry):
                record = pl.program_id(0) * batch + slot
                base = base_ref[record]

                @pl.when(base >= 0)
                def _():
                    hbm = out_ref.at[pl.ds(base, page_rows), :]
                    vmem = pages.at[slot]
                    src, dst = (vmem, hbm) if write else (hbm, vmem)
                    copy = pltpu.make_async_copy(src, dst, sem)
                    if wait:
                        copy.wait()
                    else:
                        copy.start()

                return carry

            jax.lax.fori_loop(0, batch, body, 0)

        copies(write=False, wait=False)
        copies(write=False, wait=True)

        def merge(slot, carry):
            start = pl.program_id(0) * batch + slot

            @pl.when(base_ref[start] >= 0)
            def _():
                def update(record, carry):
                    word = jnp.take_along_axis(
                        value_ref[
                            :, pl.ds(record // page_tokens * page_tokens, page_tokens)
                        ],
                        jnp.full((page_rows, page_tokens), record % page_tokens),
                        axis=1,
                        mode="fill",
                    )
                    pages[slot] = jnp.where(
                        jnp.arange(page_tokens)[None, :] == column_ref[record],
                        word,
                        pages[slot],
                    )
                    return carry

                jax.lax.fori_loop(start, end_ref[start], update, 0)

            return carry

        jax.lax.fori_loop(0, batch, merge, 0)
        copies(write=True, wait=False)
        copies(write=True, wait=True)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(bank.shape, bank.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(bases.size // batch,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((batch, page_rows, page_tokens), jnp.int32),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        input_output_aliases={4: 0},
        interpret=pltpu.InterpretParams() if interpret else False,
        name="write_packed_fp8_pages",
    )(bases, columns, ends, values, bank)
