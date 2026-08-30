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

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class TuningKey:
    """Exact-equality lookup key: every field must match the row.

    Each one is read off the arrays the kernel receives -- see
    `layers/common/attention_interface.py`. Five of them
    (`max_num_tokens`, `page_size_per_kv_packing`, `kv_packing`,
    `max_num_seqs`, `pages_per_seq`) are serving-configuration values, so a
    table tuned at one serving shape is inert at every other. The defaults
    below are DeepSeekV3-on-v7-8's shape, not neutral values; prefer passing
    every field. `report_missed_tuning_keys` is what makes a miss visible.
    """

    case: str  # A string identifier for the case, support only: "batched_decode", "decode_only", "mixed"
    max_num_tokens: int  # Maximum number of tokens in the batch
    actual_num_q_heads: int  # Q heads per shard, i.e. `ql_nope.shape[0]` inside the kernel
    actual_lkv_dim: int  # Compressed latent dim (`ql_nope.shape[-1]` == kv_lora_rank; NOT qk_nope_head_dim)
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
    # Kimi-K3 mixed prefill on TPU v7x-8 with TP8, BF16, and 4K tokens.
    TuningKey(
        case="mixed",
        max_num_tokens=4096,
        actual_num_q_heads=12,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="bfloat16",
        q_dtype="bfloat16",
        page_size_per_kv_packing=128,
        kv_packing=2,
        max_num_seqs=8,
        pages_per_seq=128,
    ):
    TunableParams(
        q_split=1,
        num_kv_pages_per_block=4,
        num_queries_per_block=256,
        vmem_limit_bytes=62914560,
    ),
    # Kimi-K3 mixed prefill on TPU v7x with TP32, FP8 KV, 8K tokens,
    # and 16-token KV cache pages.
    TuningKey(
        case="mixed",
        max_num_tokens=8192,
        actual_num_q_heads=3,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="bfloat16",
        page_size_per_kv_packing=4,
        kv_packing=4,
        max_num_seqs=8,
        pages_per_seq=520,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=4,
        num_kv_pages_per_block=256,
        num_queries_per_block=512,
        vmem_limit_bytes=58720256,
    ),
    # Same workload with 256-token pages, as required when a cache page must
    # also accommodate the KDA recurrent state for prefix caching.
    TuningKey(
        case="mixed",
        max_num_tokens=8192,
        actual_num_q_heads=3,
        actual_lkv_dim=512,
        actual_r_dim=64,
        kv_dtype="float8_e4m3fn",
        q_dtype="bfloat16",
        page_size_per_kv_packing=64,
        kv_packing=4,
        max_num_seqs=8,
        pages_per_seq=40,
        s_dtype="bfloat16",
        soft_cap=None,
        chunk_prefill_size=None,
        sliding_window=None,
        p_same_dtype_as_v=True,
    ):
    TunableParams(
        q_split=8,
        num_kv_pages_per_block=16,
        num_queries_per_block=512,
        vmem_limit_bytes=54525952,
    ),
}


class TuningKeyMissError(KeyError):
    """A lookup missed the table and strict mode forbids the fallback."""


# Distinct keys looked up, in first-seen order, with their lookup counts. The
# lookup runs under jit tracing, once per layer per bucket, so the same key
# recurs hundreds of times; only the end-of-warm-up summary is worth reading.
_hit_keys: dict[TuningKey, int] = {}
_missed_keys: dict[TuningKey, int] = {}


def _mixed_fallback_kv_pages(tuning_key: TuningKey) -> int:
    """KV pages per block for an untuned chunked-prefill call.

    One page is the shipped answer and it is page-blind: the kernel walks
    `cdiv(kv_len, bkv_p * page)` KV tiles once per query tile with no causal
    skip, at a near-constant cost per tile, so at `--block-size 16` a full 8192
    prefill pays 16384 tiles per layer instead of 1024. `bkv_sz = bkv_p * page`
    also gates the kernel's own fast paths, and 16 misses both.

    TPU_MLA_MIXED_FALLBACK_BKV_TOKENS sizes the block in tokens instead, so a
    small page does not imply a small tile: "at least this many tokens per KV
    tile, when the page size allows". At `page >= tokens` one page already
    clears the bar and this returns 1 -- a correct no-op, not a failure.
    0 keeps the shipped behaviour.
    """
    tokens = envs.TPU_MLA_MIXED_FALLBACK_BKV_TOKENS
    if tokens <= 0:
        return 1
    page = tuning_key.page_size_per_kv_packing * tuning_key.kv_packing
    return max(1, tokens // page)


def get_tuned_params(tuning_key: TuningKey) -> TunableParams:
    if tuning_key in tuned_params_mapping:
        _hit_keys[tuning_key] = _hit_keys.get(tuning_key, 0) + 1
        return tuned_params_mapping[tuning_key]
    seen = _missed_keys.get(tuning_key, 0)
    _missed_keys[tuning_key] = seen + 1
    if envs.TPU_MLA_TUNING_STRICT:
        raise TuningKeyMissError(
            f"No tuned parameters for {tuning_key} and "
            "TPU_MLA_TUNING_STRICT is set. Tune this shape (see "
            "kernels/mla/v2/tuned_params.py) or unset the variable to fall "
            "back to untuned defaults.")
    if seen == 0:
        logger.warning(
            "No tuned parameters found for the given tuning key: %s, using "
            "default parameters", tuning_key)

    if tuning_key.case == "mixed":
        return TunableParams(
            num_kv_pages_per_block=_mixed_fallback_kv_pages(tuning_key),
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


def missed_tuning_keys() -> dict[TuningKey, int]:
    """Keys that fell back to untuned defaults, and how often each was asked for."""
    return dict(_missed_keys)


def report_missed_tuning_keys() -> int:
    """Log one summary of MLA v2 tuning-table coverage; return the miss count.

    Called once per worker after warm-up: by then every serving shape has been
    traced, so a miss here means this recipe runs that shape untuned for the
    life of the process. A per-lookup warning fires once per layer per bucket
    and is read by nobody; this line names the distinct keys instead.
    """
    if not _missed_keys:
        if _hit_keys:
            logger.info("MLA v2 tuned params: all %d distinct keys hit.",
                        len(_hit_keys))
        return 0
    logger.warning(
        "MLA v2 tuned params: %d distinct keys MISSED, %d hit. The missed "
        "shapes ran on default block sizes for this whole process:\n%s",
        len(_missed_keys), len(_hit_keys),
        "\n".join(f"  x{count:<5d} {key}"
                  for key, count in _missed_keys.items()))
    return len(_missed_keys)
