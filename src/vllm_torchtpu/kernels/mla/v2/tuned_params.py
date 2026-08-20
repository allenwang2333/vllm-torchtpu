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

# Enough KV pages in flight to cover DMA latency. Measured on
# Kimi-Linear-48B-A3B at TP=4, 8k/1k, concurrency 16, end to end:
#
#     pages per block :    1       4       16      32
#     output tok/s    :  581.94  829.70  945.33  964.71
#
# The curve is latency-bound and flattens: 1->4 is +42.6%, 4->16 is +13.9%,
# 16->32 is +2.1%. Nothing OOMed, so VMEM was never the binding constraint --
# past the knee, extra pages buy nothing and only cost residency. 32 sits just
# past the knee, which is where we want to land when the shape is unknown.
_MIXED_KV_PAGES_TARGET = 32

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
    r_dim_padded = ((tuning_key.actual_r_dim + 127) // 128) * 128
    width = tuning_key.actual_lkv_dim + r_dim_padded
    slots = tuning_key.page_size_per_kv_packing * tuning_key.kv_packing
    page_bytes = slots * width * _DTYPE_BYTES.get(tuning_key.kv_dtype, 2)

    # Half the budget for the KV window, and it is double buffered.
    fits = (_MIXED_VMEM_LIMIT_BYTES // 2) // max(page_bytes * 2, 1)
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
            return TunableParams(
                num_kv_pages_per_block=mixed_kv_pages_per_block(tuning_key),
                num_queries_per_block=16,
                vmem_limit_bytes=_MIXED_VMEM_LIMIT_BYTES,
            )

        # decode, batched_decode
        return TunableParams(
            decode_batch_size=4,
            num_kv_pages_per_block=3,
            num_queries_per_block=1,
            vmem_limit_bytes=62914560,
        )
