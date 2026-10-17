#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Fixed C64 CI workload; keep it separate from the C4/C8/C16 baseline.
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_NUM_SEQS=64
CONCURRENCY_OPTIONS=64
COMPILE_SIZES='[64,384,8192]'
KV_CACHE_MEMORY_BYTES=21474836480
DSPARK_K=5
export TPU_DEEPSEEK_V41_IMPL="${TPU_DEEPSEEK_V41_IMPL:-v2}"
export TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-1}"
export TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD="${TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD:-1}"
# Encoder-only sequence parallelism (CED decoder layers never use SP): shards
# the T=8192 mixed-step elementwise work 8-way. Measured on tpu7x-8:
# mixed step 320 -> 308 ms, E2E +1.4%, pure-decode step +0.8 ms.
export TPU_DEEPSEEK_V41_SP="${TPU_DEEPSEEK_V41_SP:-1}"
export TPU_DEEPSEEK_V41_LOCAL_Q="${TPU_DEEPSEEK_V41_LOCAL_Q:-1}"
# Chunk the sequence-parallel MoE all-gather/gmm/reduce-scatter (4 chunks per
# 8192-token step) so each chunk's reduce-scatter overlaps the next gmm.
export TPU_MOE_COLLECTION_CHUNK_SIZE="${TPU_MOE_COLLECTION_CHUNK_SIZE:-2048}"
export USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE:-1}"
export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-0}"
export MOE_FUSED_EP_ENABLE_W4A8="${MOE_FUSED_EP_ENABLE_W4A8:-0}"
export MOE_FUSED_EP_V2_SHARDED_PLAN="${MOE_FUSED_EP_V2_SHARDED_PLAN:-0}"
# shellcheck source=/dev/null
source "$CONFIG_DIR/deepseek-v41-flash-tp8-ep-100k-decode-dspark.sh"
MAX_MODEL_LEN=103424
MAX_NUM_BATCHED_TOKENS=8192
KV_CACHE_DTYPE=fp8
# Match the measured C64 DSpark/offload workload.
ISL_OSL_CONFIGS="98000:5000"
