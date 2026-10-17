# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass

from vllm_torchtpu.logger import init_logger

# At Bq=128, a 32K-key FP32 output window occupies 16 MiB. Mosaic double
# buffers grid-window outputs, so long contexts need 32 MiB plus scorer
# scratch. A single <=64K window retains the original single-buffer layout.
SCORE_WINDOW = 32768
QUERY_TILE = 128
KEY_TILE = 2048


# Decode tables are calibrated with decode_indexer's default score/top-k options.
# The dimensions describe one logical device, not global DP concurrency.
logger = init_logger(__name__)
DEFAULT_DECODE_REQUEST_TILE = 16


@dataclass(frozen=True)
class DecodeTuningKey:
    device_kind: str
    requests: int
    queries: int
    heads: int
    score_capacity: int
    page_records: int
    key_tile: int
    k: int
    banks: int


# Calibrated on TPU7x, JAX 0.10.2 / libtpu 0.0.47; DSpark K5.
# Seeds 517/519, effective lengths 49408/98816; all other options default.
DECODE_REQUEST_TILES: dict[DecodeTuningKey, int] = {
    DecodeTuningKey("TPU7x", 48, 6, 32, 65536, 128, 4096, 512, 2): 12,
    DecodeTuningKey("TPU7x", 48, 6, 32, 131072, 128, 4096, 512, 2): 16,
}


def get_decode_request_tile(key: DecodeTuningKey) -> int:
    """Resolve a static request group; unknown shapes keep a bounded divisor."""
    if key.requests < 1:
        raise ValueError("decode requests must be positive")
    tile = DECODE_REQUEST_TILES.get(key)
    if tile is None:
        tile = next(
            n
            for n in range(min(DEFAULT_DECODE_REQUEST_TILE, key.requests), 0, -1)
            if key.requests % n == 0
        )
        logger.warning_once(
            "[decode_indexer] No tuned request_tile for %s; using %d", key, tile
        )
    if type(tile) is not int or tile < 1 or key.requests % tile:
        raise ValueError(f"Invalid decode request_tile={tile} for {key}")
    return tile


@dataclass(frozen=True)
class CandidateTuningKey:
    device_kind: str
    requests: int
    queries: int
    heads: int
    capacity: int
    key_tile: int | None
    k: int
    packed: bool = False


@dataclass(frozen=True)
class CandidateBlockSizes:
    key_tile: int
    request_tile: int
    topk_block_size: int = 8192
    reconstruct_threshold: int = 8192
    compute_tile: int | None = None
    head_stripe: int | None = None
    fused_pipeline: bool = False


_TPU7X_CACHED_DECODE = CandidateBlockSizes(
    key_tile=49152,
    request_tile=16,
    topk_block_size=16384,
    reconstruct_threshold=1024,
    compute_tile=12288,
    head_stripe=8,
    fused_pipeline=True,
)


# Exact configurations take precedence over the bounded upstream fallback.
CANDIDATE_BLOCK_SIZES: dict[CandidateTuningKey, CandidateBlockSizes] = {
    CandidateTuningKey("TPU7x", 48, 6, 32, 40960, None, 512): CandidateBlockSizes(
        4096, 48
    ),
    CandidateTuningKey("TPU7x", 48, 6, 32, 40960, 4096, 512): CandidateBlockSizes(
        4096, 48
    ),
    CandidateTuningKey("TPU7x", 48, 6, 32, 49152, 4096, 512): CandidateBlockSizes(
        4096, 48
    ),
    CandidateTuningKey("TPU7x", 48, 6, 32, 49152, 8192, 512): CandidateBlockSizes(
        8192, 48
    ),
    CandidateTuningKey("TPU7x", 48, 6, 32, 49152, 16384, 512): CandidateBlockSizes(
        16384, 24, 16384, 1024
    ),
    CandidateTuningKey("TPU7x", 48, 6, 32, 49152, 24576, 512): CandidateBlockSizes(
        24576, 24, 16384, 1024, compute_tile=4096
    ),
    CandidateTuningKey("TPU7x", 48, 6, 32, 49152, 49152, 512): _TPU7X_CACHED_DECODE,
    CandidateTuningKey("TPU7x", 48, 6, 32, 49152, None, 512): _TPU7X_CACHED_DECODE,
}
# 40K overflowed in 12/48 profiled rank-steps. The 48K bucket covers the
# captured maximum of 5138 blocks with headroom, at 576 MiB per 48 requests.
CANDIDATE_KEY_CAPACITY = 49152
# Generalize the measured 8192/48 configuration without increasing its score
# buffer or per-request dot-product footprint. These defaults are heuristics;
# only exact table entries have measured performance evidence.
DEFAULT_CANDIDATE_KEY_TILE = 8192
DEFAULT_CANDIDATE_REQUEST_TILE = 48
CANDIDATE_SCORE_ELEMENTS = DEFAULT_CANDIDATE_REQUEST_TILE * 6 * CANDIDATE_KEY_CAPACITY


def get_candidate_block_sizes(key: CandidateTuningKey) -> CandidateBlockSizes:
    """Resolve scorer and selector tiles together for the complete pipeline."""
    if key.requests < 1:
        raise ValueError("candidate requests must be positive")
    if key.queries < 1 or key.capacity < 1:
        raise ValueError("candidate queries and capacity must be positive")
    config = CANDIDATE_BLOCK_SIZES.get(key)
    if config is None:
        related = (
            key.device_kind == "TPU7x"
            and not key.packed
            and 1 <= key.queries <= 6
            and key.heads == 32
            and key.k == 512
        )
        key_tile = key.key_tile
        if key_tile is None:
            key_tile = math.gcd(
                DEFAULT_CANDIDATE_KEY_TILE if related else 4096, key.capacity
            )
        limit = (
            min(
                DEFAULT_CANDIDATE_REQUEST_TILE,
                max(1, CANDIDATE_SCORE_ELEMENTS // (key.queries * key.capacity)),
            )
            if related and key_tile <= DEFAULT_CANDIDATE_KEY_TILE
            else DEFAULT_DECODE_REQUEST_TILE
        )
        request_tile = next(
            n for n in range(min(limit, key.requests), 0, -1) if key.requests % n == 0
        )
        config = CandidateBlockSizes(key_tile, request_tile)
        logger.warning_once(
            "[candidate_indexer] No tuned block sizes for %s; using %s", key, config
        )
    if config.request_tile < 1 or key.requests % config.request_tile:
        raise ValueError(
            f"Invalid candidate request_tile={config.request_tile} for {key}"
        )
    return config
