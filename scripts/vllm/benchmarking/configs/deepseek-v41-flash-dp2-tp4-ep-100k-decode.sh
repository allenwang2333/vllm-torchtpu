#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DeepSeek V4.1 Flash 100k long-context decode with DP=2 x TP=4 + EP=8
# (EXT-1B KV-Sharded Attention-DP + MoE-EP on tpu7x-8), for run_decode_benchmark.sh.
# Each DP replica (TP=4, 4 dies = 2 dual-die v7x chips) manages its own
# independent KV cache pool, eliminating 2x MQA KV cache duplication across
# the two TP=4 groups while keeping all 8 ranks in one EP=8 MoE group.
MODEL="${DEEPSEEK_V41_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
TENSOR_PARALLELISM=4
DATA_PARALLELISM=2
ENABLE_EP=true
MAX_MODEL_LEN=103424
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
ASYNC_SCHEDULING=true
GPU_MEMORY_UTILIZATION=0.85
# fp8 selects v2's FP8 main KV record and the SparseMLA kernel. TPU V4.1 does
# not use the fp8_ds_mla layout, so auto would mean BF16 KV and the slower
# generic paged attention.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ISL_OSL_CONFIGS="100000:2048"
CONCURRENCY_OPTIONS="${CONCURRENCY_OPTIONS:-8 16 32}"
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
# Required by run_benchmarks.sh for every config; no eval runs here.
EVAL_TOLERANCE="0.03"
# MXFP4 expert processing alone takes ~40 min on v7x-8.
SERVER_READY_WAIT_MIN=120
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export TPU_DEEPSEEK_V41_FUSED_SWA="${TPU_DEEPSEEK_V41_FUSED_SWA:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
# V4.1 impl v2 (models/vllm/deepseek_v41/v2).
export TPU_DEEPSEEK_V41_IMPL="${TPU_DEEPSEEK_V41_IMPL:-v2}"
export TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-0}"
# 20 GiB per rank (1-bank); with DP=2 x TP=4 each TP=4 replica holds its own
# 20 GiB pool (~83 x 100k requests/replica = ~166 total across DP=2).
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-21474836480}"
COMPILE_SIZES="${COMPILE_SIZES:-[4,8,16,2048]}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--no-enable-prefix-caching --enable-ep-weight-filter --safetensors-load-strategy=prefetch --safetensors-prefetch-num-threads=8 --block-size=16 --kv-cache-memory-bytes=${KV_CACHE_MEMORY_BYTES} --compilation-config={\"compile_sizes\":${COMPILE_SIZES}}"

# Keep the v7x Engram gather on the runtime default path.
if [[ "${LIBTPU_INIT_ARGS:-}" != *"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
fi
