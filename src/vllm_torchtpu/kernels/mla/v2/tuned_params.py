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

from dataclasses import dataclass

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TuningKey:
    case: str  # A string identifier for the case, support only: "batched_decode", "decode_only", "mixed"
    max_num_tokens: int  # Maximum number of tokens in the batch
    actual_num_q_heads: int  # Actual number of Q heads, <= num_q_heads in the model config, fixed at 128 for now
    actual_lkv_dim: int  # Actual NOPE head dimension, <= lkv_dim in the model config, fixed at 512 for now
    actual_r_dim: int  # Actual ROPE head dimension, <= r_dim in the model config, fixed at 64 for now
    kv_dtype: str = "float8_e4m3fn"  # KV cache and KV input data type, fixed at fp8 for now
    q_dtype: str = "float8_e4m3fn"  # Q activation dtype, fixed at fp8 for now
    page_size_per_kv_packing: int = 256  # Page size per KV packing, should be aligned with the kernel configuration
    kv_packing: int = 4  # Packing factor for KV, determined by the data type (e.g., 4 for fp8)
    max_num_seqs: int = 160  # Maximum number of sequences in the batch, should be large enough to cover all sequences in the batch
    pages_per_seq: int = 9  # Number of pages per sequence, determined by the maximum KV length and page size. Should be large enough to cover the longest sequence in the batch.

    s_dtype: str = "bfloat16"  # Post QK einsum data type feeding into softmax, fixed at bf16 for now
    soft_cap: float | None = None  # Optional softmax cap, if None, no capping is applied. If set, should be a positive value.
    # sm_scale: float = 0.1352337788608801 # Scaling factor applied to the softmax input
    # mask_value: float | None = -3.38953e+38 # Optional mask value for masked positions

    chunk_prefill_size: int | None = None  # Chunk size for prefill in the decode case, range from 1 to max_num_tokens with steps of powers of two
    sliding_window: int | None = None  # Sliding window size, [None, 5, 128]
    p_same_dtype_as_v: bool = True  # Whether the softmax input should have the same data type as V, fixed at True for now


@dataclass(frozen=True)
class MLATuningKey:
    """Topology-local key for MLA kernel launch parameters."""

    device_name: str
    case: str
    max_num_tokens: int
    actual_num_q_heads: int
    actual_lkv_dim: int
    actual_r_dim: int
    q_dtype: str
    kv_dtype: str
    page_size: int


@dataclass
class TunableParams:
    num_kv_pages_per_block: int  # Number of KV pages to process per block. Range from 1 to as high as possible before OOM,
    # with steps of powers of two.
    num_queries_per_block: int  # for batched_decode, this is always 1
    vmem_limit_bytes: int  # 16MiB(?) to 64MiB, increments of 8MiB.
    # Select lowest value that gives the highest performance
    decode_batch_size: int = 1  # range from 1 to as high as possible before OOM with steps powers of two
    # Constraint: batch size % decode_batch_size = 0
    q_split: int = 1  # number of query split for running parallel.


@dataclass(frozen=True)
class MLAKernelConfig:
    """Complete MLA launch configuration for decode and mixed programs."""

    num_kv_pages_per_block: tuple[int, int, int]
    num_queries_per_block: tuple[int, int, int]
    vmem_limit_bytes: int
    decode_batch_size: int
    mixed_q_split: int


tuned_params_mapping: dict[TuningKey, TunableParams] = {
    # Mistral-Small batched decode. (TPU v7-8)
    TuningKey(
        case="batched_decode",
        max_num_tokens=4,
        actual_num_q_heads=32,
        actual_lkv_dim=256,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=16,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=8,
        actual_num_q_heads=32,
        actual_lkv_dim=256,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=16,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=16,
        actual_num_q_heads=32,
        actual_lkv_dim=256,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=16,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=256,
        actual_num_q_heads=32,
        actual_lkv_dim=256,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=16,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    # DeepSeekV3 batched decode. (TPU v7-8)
    TuningKey(
        case="batched_decode",
        max_num_tokens=4,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=8,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=16,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=32,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=64,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=128,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=160,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=256,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=512,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    # Tuned parameters for Mistral-Large-3 on TPU v7x-16 (page_size=1024, kv_packing=32, max_num_seqs=8):
    TuningKey(
        case="batched_decode",
        max_num_tokens=4,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=8,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=16,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=2,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=32,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=64,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=128,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=160,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=256,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=1,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=512,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=1024,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=2,
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=2048,
        actual_num_q_heads=128,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=8,
        pages_per_seq=3,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    # Kimi 2.6 batched_decode. (TPU v7-8)
    TuningKey(
        case="batched_decode",
        max_num_tokens=4,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=8,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=16,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=16,
        num_kv_pages_per_block=2,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=32,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=64,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=77,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=128,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=160,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=256,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=512,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=616,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=1024,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=8,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="batched_decode",
        max_num_tokens=2048,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    ),
    # Kimi 2.6 prefill/mixed. (TPU v7-8)
    TuningKey(
        case="mixed",
        max_num_tokens=4,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=8,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=16,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=32,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=64,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=77,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=128,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=160,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=256,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=512,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=1024,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        page_size_per_kv_packing=32,
        kv_packing=32,
        max_num_seqs=64,
        pages_per_seq=9,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    # Kimi 2.6 prefill/mixed. (TPU v7-8)
    TuningKey(
        case="mixed",
        max_num_tokens=4,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=8,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=16,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=32,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=64,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=77,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=128,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=160,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=256,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=512,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
    TuningKey(
        case="mixed",
        max_num_tokens=1024,
        actual_num_q_heads=64,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="float8_e4m3fn",
        page_size_per_kv_packing=576,
        kv_packing=1024,
        max_num_seqs=77,
        pages_per_seq=9,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=16,
        num_kv_pages_per_block=1,
        num_queries_per_block=64,
        vmem_limit_bytes=62914560,
    ),
}

# Kimi-K3 MLA mixed extend, TP=8 on TPU v7x-8. This is the post-shard-map
# shape (96 / 8 = 12 local Q heads) for cold 4K prefill with a BF16 cache.
# Falcon exp-ex2edkk2t9 selected page_size=256, bkv=4, bq=512 from the
# vllm-torchtpu kernel; cache-hit/long-KV serving remains to be validated.
mla_tuned_params_mapping: dict[MLATuningKey, TunableParams] = {
    MLATuningKey(
        device_name="TPU v7",
        case="mixed",
        max_num_tokens=4096,
        actual_num_q_heads=12,
        actual_lkv_dim=128,
        actual_r_dim=64,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        page_size=256,
    ):
    TunableParams(
        num_kv_pages_per_block=4,
        num_queries_per_block=512,
        vmem_limit_bytes=62914560,
        q_split=1,
    ),
}


def get_tuned_params(
        tuning_key: TuningKey,
        fallback_keys: tuple[TuningKey, ...] = (),
) -> TunableParams:
    """Return tuned parameters, trying compatibility keys before defaults."""
    for candidate in (tuning_key, *fallback_keys):
        if candidate in tuned_params_mapping:
            return tuned_params_mapping[candidate]

    logger.warning(
        "No tuned parameters found for the given tuning key: %s, using "
        "default parameters", tuning_key)
    if tuning_key.case == "mixed":
        return TunableParams(
            num_kv_pages_per_block=1,
            num_queries_per_block=16,
            vmem_limit_bytes=62914560,
        )

    # decode, batched_decode
    return TunableParams(
        decode_batch_size=4,
        num_kv_pages_per_block=3,
        num_queries_per_block=1,
        vmem_limit_bytes=62914560,
    )


def get_mla_kernel_config(
    *,
    device_name: str,
    max_num_tokens: int,
    local_num_q_heads: int,
    global_num_q_heads: int,
    actual_lkv_dim: int,
    actual_r_dim: int,
    q_dtype: str,
    kv_dtype: str,
    page_size_per_kv_packing: int,
    kv_packing: int,
    max_num_seqs: int,
    pages_per_seq: int,
) -> MLAKernelConfig:
    """Resolve separate decode and mixed configs for one compiled MLA shape.

    New entries are keyed by the post-sharding shape and logical page size.
    The legacy table remains a fallback so existing global-head-count entries
    keep working while MLA configurations migrate to topology-local keys.
    """
    common = dict(
        max_num_tokens=max_num_tokens,
        actual_num_q_heads=local_num_q_heads,
        actual_lkv_dim=actual_lkv_dim,
        actual_r_dim=actual_r_dim,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        page_size_per_kv_packing=page_size_per_kv_packing,
        kv_packing=kv_packing,
        max_num_seqs=max_num_seqs,
        pages_per_seq=pages_per_seq,
    )
    global_common = common | {"actual_num_q_heads": global_num_q_heads}
    legacy_common = dict(
        max_num_tokens=max_num_tokens,
        actual_num_q_heads=global_num_q_heads,
        actual_lkv_dim=actual_lkv_dim,
        actual_r_dim=actual_r_dim,
        kv_dtype=kv_dtype,
    )

    def _lookup(case: str) -> TunableParams:
        topology_local_key = MLATuningKey(
            device_name=device_name,
            case=case,
            max_num_tokens=max_num_tokens,
            actual_num_q_heads=local_num_q_heads,
            actual_lkv_dim=actual_lkv_dim,
            actual_r_dim=actual_r_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            page_size=page_size_per_kv_packing * kv_packing,
        )
        if tuned := mla_tuned_params_mapping.get(topology_local_key):
            return tuned
        return get_tuned_params(
            TuningKey(case=case, **common),
            (
                TuningKey(case=case, **global_common),
                TuningKey(case=case, **legacy_common),
            ),
        )

    decode = _lookup("batched_decode")
    mixed = _lookup("mixed")

    return MLAKernelConfig(
        num_kv_pages_per_block=(decode.num_kv_pages_per_block, 1,
                                mixed.num_kv_pages_per_block),
        # Keep the historical prefill slot while selecting mixed separately.
        num_queries_per_block=(decode.num_queries_per_block, 16,
                               mixed.num_queries_per_block),
        vmem_limit_bytes=min(decode.vmem_limit_bytes, mixed.vmem_limit_bytes),
        decode_batch_size=decode.decode_batch_size,
        mixed_q_split=mixed.q_split,
    )
