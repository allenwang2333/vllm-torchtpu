#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Structured decoding performance benchmark for Qwen3-0.6B with TP=1.
MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
ISL_OSL_CONFIGS="512:512"
CONCURRENCY_OPTIONS="1 8 32"
RANDOM_RANGE_RATIO="0.0"
GPU_MEMORY_UTILIZATION=0.9
ATTENTION_BACKEND="CUSTOM"
PERF_TOLERANCE="0.10"
EVAL_TOLERANCE="0.02"

DATASET_PATH="benchmarks/datasets/structured/ast_schema_prompts.jsonl"
