# SPDX-License-Identifier: Apache-2.0
"""Initial offload fusion candidates, pending v7x measurements."""

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# (input width, stream count, hidden width) -> (token tile, contracting tile).
# Keep the full output width for the gate reduction. At the released shape,
# these tiles need a 3.2 MiB accumulator and a 6.25 MiB FP8 weight tile.
BLOCK_SIZES = {(6144, 4, 5120): (32, 256)}


def get_block_sizes(tokens: int, width: int, streams: int, dim: int) -> tuple[int, int]:
    key = (width, streams, dim)
    if key not in BLOCK_SIZES:
        logger.warning_once("Engram fusion has no measured tile sizes for %s", key)
    tile_m, tile_k = BLOCK_SIZES.get(key, (32, 256))
    return min(tile_m, tokens), min(tile_k, width)
