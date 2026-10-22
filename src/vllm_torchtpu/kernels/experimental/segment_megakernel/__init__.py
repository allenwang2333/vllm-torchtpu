# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Decode segment megakernel: MoE layer N's tail and skip layer N+1's head in
one Pallas call (kernel.py)."""

from vllm_torchtpu.kernels.experimental.segment_megakernel.kernel import (
    KV_TOKEN_ROUND,
    MAX_TOKENS,
    TOKEN_BUCKETS,
    SegCfg,
    fstack_row,
    kernel_name,
    kv_dst_block,
    operand_shapes,
    pack_act,
    pack_meta,
    rope_lane_tables,
    segment_call,
    supported,
)

__all__ = [
    "KV_TOKEN_ROUND",
    "MAX_TOKENS",
    "TOKEN_BUCKETS",
    "SegCfg",
    "fstack_row",
    "kernel_name",
    "kv_dst_block",
    "operand_shapes",
    "pack_act",
    "pack_meta",
    "rope_lane_tables",
    "segment_call",
    "supported",
]
