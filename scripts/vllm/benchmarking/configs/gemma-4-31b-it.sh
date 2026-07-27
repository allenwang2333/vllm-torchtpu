#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: google/gemma-4-31B-it (Dense, TP=8 on v7x-8).
MODEL="google/gemma-4-31B-it"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"
# Reserve 25% HBM headroom (GPU_MEMORY_UTILIZATION=0.75) to accommodate large multimodal vision XLA compilation
# scratch buffers at max-model-len 4096 without triggering PjRt TPU HBM OOM crashes.
export VLLM_WORKER_MULTIPROC_METHOD=fork
GPU_MEMORY_UTILIZATION=0.75
EVAL_TOLERANCE="0.02"
IS_MULTIMODAL=true
MAX_MODEL_LEN=4096
EXTRA_SERVE_ARGS="--enforce-eager --trust-remote-code --generation-config vllm"
MMMU_LIMIT="${MMMU_LIMIT:-100}"
RUN_EVALPLUS=0
