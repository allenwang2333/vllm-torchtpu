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


_MIXED_VMEM_LIMIT_BYTES = 62914560  # 60 MiB, as every hand-tuned entry uses.


def mixed_vmem_limit_bytes() -> int:
    """VMEM budget the mixed kernel declares.

    60 MiB is what every hand-tuned entry asks for, and it is far more than
    the kernel needs. At 32 pages per block and Kimi's 20 KB pages the KV
    window wants about 4 MiB; the budget only starts to bind the block
    schedule below roughly 2 MiB.

    The request is not free, because it competes with the rest of the program
    for the 64 MiB a core has. TP=8 died during warmup at 63.50 MiB of 64,
    half a megabyte short -- so a budget nobody needed was the thing standing
    between this model and twice the tensor parallelism.

    `TPU_MLA_MIXED_VMEM_MB` lowers it; 0 keeps 60 MiB.
    """
    from vllm_torchtpu import envs as tpu_envs
    override_mb = tpu_envs.TPU_MLA_MIXED_VMEM_MB
    if override_mb > 0:
        return override_mb * 1024 * 1024
    return _MIXED_VMEM_LIMIT_BYTES

# Enough KV pages in flight to cover DMA latency. Measured end to end on
# Kimi-Linear-48B-A3B at TP=4, 8k/1k, concurrency 16:
#
#     pages per block :    1       4       16      32
#     output tok/s    :  581.94  829.70  945.33  964.71
#
# and again at kernel level with `scripts/kernels/autotune_mla.py`, which
# sweeps further than a serving benchmark affordably can. 64 beat 32 at every
# query block size tried:
#
#     pages per block :   1      8      16     32     64     (ms, queries=64)
#     kernel time     : 17.961  2.714  1.630  1.302  1.174
#
# Nothing OOMed anywhere in that sweep, so VMEM is not what ends the curve.
_MIXED_KV_PAGES_TARGET = 64

# Query rows resident per block. The untuned fallback pinned this at 16 and
# nobody had swept it; it turned out to matter as much as the KV block size.
# At 32 pages per block, same harness:
#
#     queries : 8      16     32     64     128
#     ms      : 4.888  2.808  1.756  1.256  1.041
#
# 16 -> 128 is 2.70x on the kernel. Still improving at 128, so this target is
# where the measurement stops rather than where the curve does.
_MIXED_QUERIES_TARGET = 128

_DTYPE_BYTES = {
    "float8_e4m3fn": 1,
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


def _prev_power_of_two(value: int) -> int:
    """Largest power of two <= value, at least 1."""
    result = 1
    while result * 2 <= value:
        result *= 2
    return result


def _mixed_page_bytes(tuning_key: TuningKey) -> int:
    """VMEM one KV page occupies, latent and RoPE dims together."""
    r_dim_padded = ((tuning_key.actual_r_dim + 127) // 128) * 128
    width = tuning_key.actual_lkv_dim + r_dim_padded
    slots = tuning_key.page_size_per_kv_packing * tuning_key.kv_packing
    return slots * width * _DTYPE_BYTES.get(tuning_key.kv_dtype, 2)


def mixed_queries_per_block(tuning_key: TuningKey, kv_pages: int) -> int:
    """Query rows per block for the mixed kernel, from what VMEM is left.

    Sized after the KV window rather than beside it, because the two share one
    budget and the KV side is chosen first. A query row costs its own latent
    and RoPE slice plus the scores it produces against the whole KV window, so
    a wider KV block makes each query row more expensive -- the two knobs
    trade against each other and cannot be picked independently.

    `TPU_MLA_MIXED_QUERIES_PER_BLOCK` overrides the result; 0 computes it.
    """
    from vllm_torchtpu import envs as tpu_envs
    override = tpu_envs.TPU_MLA_MIXED_QUERIES_PER_BLOCK
    if override > 0:
        return override

    r_dim_padded = ((tuning_key.actual_r_dim + 127) // 128) * 128
    width = tuning_key.actual_lkv_dim + r_dim_padded
    dtype_bytes = _DTYPE_BYTES.get(tuning_key.q_dtype, 2)
    row_bytes = tuning_key.actual_num_q_heads * width * dtype_bytes

    kv_positions = kv_pages * (tuning_key.page_size_per_kv_packing *
                               tuning_key.kv_packing)
    score_bytes = kv_positions * 4  # scores accumulate in float32
    per_query = max(row_bytes + score_bytes, 1)

    kv_window = kv_pages * _mixed_page_bytes(tuning_key) * 2  # double buffered
    remaining = max(mixed_vmem_limit_bytes() - kv_window, 0)
    fits = remaining // (per_query * 2)

    queries = min(fits, _MIXED_QUERIES_TARGET, tuning_key.max_num_tokens)
    return max(_prev_power_of_two(queries), 1)


def mixed_kv_pages_per_block(tuning_key: TuningKey) -> int:
    """KV pages per block for the mixed kernel when no tuned entry matches.

    The previous behaviour was to return 1 -- the minimum legal value -- which
    walks the paged KV cache one page per block. Each block is then a tiny DMA
    with nothing to overlap against, so the kernel waits on memory latency
    instead of reading or computing. HLO op profiling on Kimi-Linear-48B put
    that path at 80.3% of device time with the chip at 2.76% FLOPS and 5.82%
    HBM utilization, averaging 27.54 ms per call.

    A hand-written table entry is the wrong repair. `TuningKey` includes
    `max_num_tokens`, `max_num_seqs` and `pages_per_seq`, which are deployment
    settings rather than model properties, so any row matches only the exact
    configuration it was measured on and misses the moment someone changes
    `--max-num-seqs` or `--max-model-len`. That over-specified key is why the
    lookup misses in the first place. Computing the value on the miss path
    cannot fail to match.

    `TPU_MLA_MIXED_KV_PAGES_PER_BLOCK` overrides the result; 0 computes it.
    """
    from vllm_torchtpu import envs as tpu_envs
    override = tpu_envs.TPU_MLA_MIXED_KV_PAGES_PER_BLOCK
    if override > 0:
        return override

    # One page holds `page_size_per_kv_packing * kv_packing` slots of the
    # latent and RoPE dims together, the latter padded to a 128 lane multiple.
    page_bytes = _mixed_page_bytes(tuning_key)

    # Half the budget for the KV window, and it is double buffered.
    fits = (mixed_vmem_limit_bytes() // 2) // max(page_bytes * 2, 1)
    pages = min(fits, _MIXED_KV_PAGES_TARGET, tuning_key.pages_per_seq)
    return max(_prev_power_of_two(pages), 1)


def get_tuned_params(tuning_key: TuningKey) -> TunableParams:
    if tuning_key in tuned_params_mapping:
        return tuned_params_mapping[tuning_key]
    else:
        logger.warning(
            f"No tuned parameters found for the given tuning key: {tuning_key}, using default parameters"
        )
        if tuning_key.case == "mixed":
            # Order matters: the query block is sized from whatever VMEM the
            # KV window leaves.
            kv_pages = mixed_kv_pages_per_block(tuning_key)
            return TunableParams(
                num_kv_pages_per_block=kv_pages,
                num_queries_per_block=mixed_queries_per_block(
                    tuning_key, kv_pages),
                vmem_limit_bytes=mixed_vmem_limit_bytes(),
            )

        # decode, batched_decode
        return TunableParams(
            decode_batch_size=4,
            num_kv_pages_per_block=3,
            num_queries_per_block=1,
            vmem_limit_bytes=62914560,
        )
