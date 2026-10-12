# SPDX-License-Identifier: Apache-2.0
"""Measured QSA attention and cache I/O tile choices."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Query block -> scoped VMEM MiB. Q256/KV256 compiles to 61.13 MiB on v7;
# Q128 is retained for environments where the full graph cannot spare 62 MiB.
PAIRED_PAGE_VMEM_MIB = {128: 61, 256: 62}

# (chip, tokens, query heads, KV heads, head dim, compression ratio,
#  physical cache page shape, query dtype, cache dtype) -> (query block, KV tile).
# T8192, contexts 0/8192, uniform/recent selections: Q256/KV256 reduced isolated
# latency 22-37% relative to Q256/KV128 with FP32 attention accumulation.
# The unified serving pool uses 1664-token pages (13 mask segments);
# dedicated attention caches use 128. Both require segment-wise addressing.
SPARSE_PREFILL_TILES = {
    (
        pltpu.ChipVersion.TPU_7X,
        8192,
        24,
        2,
        256,
        4,
        (page_size, 1, 4, 256),
        "bfloat16",
        "float8_e4m3fn",
    ): (256, 256)
    for page_size in (128, 1664)
}

# (chip, physical page shape, carrier dtype, ring elements, tag elements).
# B8/B128 exact-byte and native XProf checks validate eight outstanding DMAs
# per tile for these BF16 ring/int32 tag regions. B128 state I/O is 27 us
# on v7, versus 236 us for the original four serial region kernels.
RING_TAG_DMA_BATCHES = {
    (
        pltpu.ChipVersion.TPU_7X,
        (1664, 1, 4, 256),
        jnp.dtype(jnp.float8_e4m3fn),
        1024,
        16,
    ): 8,
    (pltpu.ChipVersion.TPU_7X, (896, 1, 2, 256), jnp.dtype(jnp.bfloat16), 1024, 16): 8,
}

# (chip, physical page shape, carrier dtype, packed row shape) -> DMA batch.
# Eight concurrent decode-row DMAs preserve ordered writes across packets.
# Native FP8 B128 writer latency falls from 72.39 us to 10.48 us; contiguous
# prefill tiles retain one bulk DMA. Duplicate destinations use serial DMAs.
MAIN_KV_DMA_BATCHES = {
    (
        pltpu.ChipVersion.TPU_7X,
        (1664, 1, 4, 256),
        jnp.dtype(jnp.float8_e4m3fn),
        (1, 4, 256),
    ): 8,
    (
        pltpu.ChipVersion.TPU_7X,
        (896, 1, 2, 256),
        jnp.dtype(jnp.bfloat16),
        (1, 2, 256),
    ): 8,
}


def get_main_kv_dma_batch(pool: jax.Array, rows: jax.Array) -> int:
    """Select measured row concurrency, preserving serial writes on misses."""
    key = (
        pltpu.get_tpu_info().chip_version,
        pool.shape[1:],
        jnp.dtype(pool.dtype),
        rows.shape[1:],
    )
    if key not in MAIN_KV_DMA_BATCHES:
        logger.warning_once("QSA uses serial main KV DMA tiles for key: %s", key)
        return 1
    return MAIN_KV_DMA_BATCHES[key] if rows.dtype == pool.dtype else 1


def get_ring_tag_dma_batch(
    pool: jax.Array, requests: int, *, ring_elements: int, tag_elements: int
) -> int:
    """Use a measured DMA batch that divides the request count."""
    key = (
        pltpu.get_tpu_info().chip_version,
        pool.shape[1:],
        jnp.dtype(pool.dtype),
        ring_elements,
        tag_elements,
    )
    if key not in RING_TAG_DMA_BATCHES:
        logger.warning_once("QSA uses single-request state DMA tiles for key: %s", key)
        return 1
    batch = RING_TAG_DMA_BATCHES[key]
    return batch if requests % batch == 0 else 1


def get_sparse_prefill_tiles(
    query: jax.Array, cache: jax.Array, *, kv_heads: int, ratio: int
) -> tuple[int, int]:
    """Select measured tiles, otherwise use one 32-group selection word."""
    tokens, query_heads, head_dim = query.shape
    key = (
        pltpu.get_tpu_info().chip_version,
        tokens,
        query_heads,
        kv_heads,
        head_dim,
        ratio,
        cache.shape[1:],
        str(query.dtype),
        str(cache.dtype),
    )
    if key not in SPARSE_PREFILL_TILES:
        logger.warning_once("QSA uses default prefill tiles for key: %s", key)
        # Keep query/state windows within the measured Q24/D256 footprint
        # when fewer TP shards give this rank more attention heads.
        rows = max(8, 256 * 24 * 256 // (query_heads * cache.shape[-1]))
        query_block = min(256, 1 << (rows.bit_length() - 1))
        return query_block, ratio * 32
    return SPARSE_PREFILL_TILES[key]
