# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Full-context paged indexer: (key columns, query rows). Two 32K SparseCore
# slices cover up to 128K uncompressed tokens at compression ratio 2. On v7x,
# 128 and 256 query rows have comparable latency; 128 uses half the score HBM
# (32 MiB). Longer contexts retain the bounded 32-row pipeline.
PAGED_INDEXER_PREFILL_BUCKET = (65536, 128)

# v7x TP1, H64: three chunks per decode bucket limit launch/join overhead.
# At 288 queries, 96-row chunks also retain whole six-query MTP groups.
CSA2_DECODE_GATHER_CHUNKS = {288: 96, 384: 128}


def csa2_gather_chunk_size(num_tokens: int) -> int:
    # On v7x TP1 with 64 heads, 128-row chunks overlap SC gather with TC
    # attention while retaining full gathered KV for subsequent layers.
    # Other small buckets retain one launch to avoid extra staging costs.
    if num_tokens <= 512:
        return CSA2_DECODE_GATHER_CHUNKS.get(num_tokens, num_tokens)
    return 128
