# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct writes to records in a shared, TensorCore-tiled word pool."""

import jax
import jax.numpy as jnp


@jax.jit
def scatter_word_records(
    cache: jax.Array,
    records: jax.Array,
    words: jax.Array,
) -> jax.Array:
    """Write unique records without reading their old contents or padding.

    Negative and out-of-range records are ignored. Records can occupy a
    contiguous part of a word row or several whole rows. The owner retains
    its layout; only the small update batch is flattened for DMA addressing.
    """
    if cache.ndim != 2 or cache.shape[1] != 128:
        raise ValueError("Expected a word cache with shape [rows, 128]")
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
    from vllm_torchtpu.kernels.mla.scatter_utils import scatter_rows

    # Keep the owner's dtype and row boundaries intact so donation can alias
    # it through the SparseCore write without copying the entire shared pool.
    return scatter_rows(
        cache,
        words,
        records,
        name="scatter_word_records",
    )
