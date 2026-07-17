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
# Reserve 20% HBM headroom (GPU_MEMORY_UTILIZATION=0.8) to accommodate static pre-compiled
# XLA graph executables during _dummy_run() when running in compiled mode without triggering HBM OOM.
GPU_MEMORY_UTILIZATION=0.8
EVAL_TOLERANCE="0.02"
EXTRA_SERVE_ARGS="--enforce-eager --trust-remote-code --max-model-len 2048"
