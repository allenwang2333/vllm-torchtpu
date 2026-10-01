#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# vLLM v0.30.0 / TorchTPU compiled text baseline. The shared eval flow uses
# GSM8K's first 100 test examples; preserve lm_eval's task and answer filters.
MODEL="${DEEPSEEK_V41_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
MAX_MODEL_LEN=4096
MAX_NUM_BATCHED_TOKENS=128
MAX_NUM_SEQS=8
ASYNC_SCHEDULING=false
# Bound the cache while whole-pool functionalization remains an optimization gap.
GPU_MEMORY_UTILIZATION=0.85
NUM_PROMPTS=2
ISL_OSL_CONFIGS="128:16"
CONCURRENCY_OPTIONS="2"
RANDOM_RANGE_RATIO="0.0"
BENCHMARK_TEMPERATURE=0
LM_EVAL_TASKS="gsm8k"
EXTRA_LM_EVAL_MODEL_ARGS="num_concurrent=8,max_retries=1,timeout=3600"
LM_EVAL_GEN_KWARGS='{"chat_template_kwargs":{"thinking":false},"max_gen_toks":1024,"temperature":0}'
SERVER_READY_WAIT_MIN=60
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_AOT_COMPILE=0
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--no-async-scheduling --no-enable-prefix-caching --enable-ep-weight-filter --safetensors-load-strategy=prefetch --safetensors-prefetch-num-threads=8 --block-size=16 --kv-cache-memory-bytes=268435456 --compilation-config={\"compile_sizes\":[8,128]}"

# Keep the v7x Engram gather on the runtime default path.
if [[ "${LIBTPU_INIT_ARGS:-}" != *"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes"* ]]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=0"
fi
