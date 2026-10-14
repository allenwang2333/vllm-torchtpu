#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DeepSeek V4.1 Flash 100k prefill-only throughput at DP=8, TP=1, EP=8.
MODEL="${DEEPSEEK_V41_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=8
ENABLE_EP=true
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
MAX_MODEL_LEN=103424
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=1
ASYNC_SCHEDULING=true
GPU_MEMORY_UTILIZATION=0.85
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
NUM_PROMPTS=8
ISL_OSL_CONFIGS="100000:1"
CONCURRENCY_OPTIONS="8"
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
BENCHMARK_WARMUP_RUNS=1
EVAL_TOLERANCE="0.03"
SERVER_READY_WAIT_MIN=120
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
export TPU_DEEPSEEK_V41_IMPL="${TPU_DEEPSEEK_V41_IMPL:-v2}"
export TPU_DEEPSEEK_V41_CED="${TPU_DEEPSEEK_V41_CED:-1}"
export TPU_DEEPSEEK_V41_SP="${TPU_DEEPSEEK_V41_SP:-1}"
export TPU_DEEPSEEK_V41_LOCAL_Q="${TPU_DEEPSEEK_V41_LOCAL_Q:-1}"
export TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD="${TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD:-1}"
export USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE:-0}"
export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
export MOE_FUSED_EP_ENABLE_W4A8="${MOE_FUSED_EP_ENABLE_W4A8:-1}"
export MOE_FUSED_EP_V2_SHARDED_PLAN="${MOE_FUSED_EP_V2_SHARDED_PLAN:-1}"
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--language-model-only --enable-ep-weight-filter --safetensors-load-strategy=prefetch --safetensors-prefetch-num-threads=8 --block-size=16 --engram-config={\"dp_shared_memory\":false} --attention-config={\"indexer_kv_dtype\":\"fp8\"} --kv-cache-memory-bytes=1342177280 --compilation-config={\"compile_sizes\":[16,128,2048]}"

if [[ "${LIBTPU_INIT_ARGS:-}" != *"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
fi
