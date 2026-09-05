# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-XLA pieces of the GLM-5.3-Flash k-pool DSA indexer.

Everything here is plain ``jnp`` on purpose. GLM's pooling is a 4-way
per-channel softmax plus one 256-byte record per four tokens -- roughly 1 MFLOP
and 512 KB of scatters at an 8192-token chunk, against ~33 GFLOP of scoring in
the same layer. A Pallas kernel would be under 0.1% of layer time and would
cost the CPU test rung, which is where every silent-if-wrong detail in this
design lives.

The expensive stages are *not* here: pool-space top-k is
``kernels/deepseek_v4/streamindex_topk.py`` at
``compression_ratio=index_kpool``
and ``causal_row_offset=index_kpool - 1``, and the attention itself is
``sparse_mla_attention``.
"""

from vllm_torchtpu.kernels.glm5_kpool.expand import (align_to,
                                                     derive_out_width,
                                                     expand_pools_with_tail)
from vllm_torchtpu.kernels.glm5_kpool.pool_math import (
    FP8_E4M3_MAX, POOLED_CACHE_KV_PACKING, pack_fp8_ue8m0, pool_window,
    pooled_record_bytes, quantize_fp8_ue8m0_rows, unpack_fp8_ue8m0)
from vllm_torchtpu.kernels.glm5_kpool.pool_write import (
    DIAG_NAMES, DIAG_POSITION_MISMATCHES, DIAG_SLOT_DUPLICATES,
    DIAG_SLOT_OUT_OF_RANGE, DIAG_TAG_VIOLATIONS, KPOOL_WRITE_DIAG_SIZE,
    PENDING_TAG_EMPTY, PendingStateGeometry, assert_pending_depth_sufficient,
    clear_pending_slots, derive_pending_state_geometry, describe_diagnostics,
    empty_pending_state, kpool_pooled_rows, kpool_write, read_pooled_records)

__all__ = [
    "DIAG_NAMES",
    "DIAG_POSITION_MISMATCHES",
    "DIAG_SLOT_DUPLICATES",
    "DIAG_SLOT_OUT_OF_RANGE",
    "DIAG_TAG_VIOLATIONS",
    "FP8_E4M3_MAX",
    "KPOOL_WRITE_DIAG_SIZE",
    "PENDING_TAG_EMPTY",
    "POOLED_CACHE_KV_PACKING",
    "PendingStateGeometry",
    "align_to",
    "assert_pending_depth_sufficient",
    "clear_pending_slots",
    "derive_out_width",
    "derive_pending_state_geometry",
    "describe_diagnostics",
    "empty_pending_state",
    "expand_pools_with_tail",
    "kpool_pooled_rows",
    "kpool_write",
    "pack_fp8_ue8m0",
    "pool_window",
    "pooled_record_bytes",
    "read_pooled_records",
    # The raw (values, scale_bits) pair, not just the concatenated record:
    # ``pool_write`` scatters the fp8 bytes and the scale byte into
    # separate offsets of a 256-byte padded row, and without this it would
    # either re-slice the packed record or re-derive the ue8m0 rounding. A
    # second derivation of that rounding is precisely what the bit-identity
    # test against ``quantize_tensor`` exists to prevent.
    "quantize_fp8_ue8m0_rows",
    "unpack_fp8_ue8m0",
]
