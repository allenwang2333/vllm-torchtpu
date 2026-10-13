# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct writes to records in a shared, TensorCore-tiled word pool."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@functools.partial(jax.jit, static_argnames=("interpret",))
def scatter_word_records(
    cache: jax.Array,
    records: jax.Array,
    words: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Write unique records without reading their old contents or padding.

    Negative and out-of-range records are ignored. Records can occupy a
    contiguous part of a word row or several whole rows. The owner retains
    its layout; only the small update batch is flattened for DMA addressing.
    """
    row_words = cache.shape[1]
    count, width = words.shape
    if cache.dtype != jnp.int32 or words.dtype != cache.dtype:
        raise ValueError("Cache and updates must contain int32 words")
    if width % row_words and row_words % width:
        raise ValueError("Records must divide a cache row or span whole rows")
    if width * words.dtype.itemsize % 16:
        raise ValueError("Record writes require 16-byte aligned lengths")
    if records.shape != (count,):
        raise ValueError("Supply one destination record per update")
    if not count:
        return cache
    # Match the existing full-row writer's in-flight DMA batch.
    block = 64
    records = jnp.pad(records, (0, -count % block), constant_values=-1)
    chunks = max(1, width // row_words)
    segment = min(width, row_words)
    per_row = max(1, row_words // width)

    def kernel(record_ref, values_ref, _input_ref, output_ref, sem):
        start = pl.program_id(0) * block

        def copy(token, chunk):
            record = record_ref[token]
            return pltpu.make_async_copy(
                values_ref.at[pl.ds(token * width + chunk * segment, segment)],
                output_ref.at[
                    record // per_row * chunks + chunk,
                    pl.ds(record % per_row * segment, segment),
                ],
                sem,
            )

        def run(wait):
            def body(index, _):
                token = start + index
                record = record_ref[token]

                @pl.when((record >= 0) & (record < cache.size // width))
                def _():
                    for chunk in range(chunks):
                        dma = copy(token, chunk)
                        if wait:
                            dma.wait()
                        else:
                            dma.start()

            jax.lax.fori_loop(0, block, body, None)

        run(False)
        run(True)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(records.size // block,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={2: 0},
        interpret=interpret,
        name="scatter_word_records",
    )(records, words.reshape(-1), cache)
