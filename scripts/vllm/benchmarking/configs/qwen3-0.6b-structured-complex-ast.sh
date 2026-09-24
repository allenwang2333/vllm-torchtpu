#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly structured-decoding gate: Qwen3-0.6B TP=1, every request constrained
# by the recursive expression-tree grammar ($ref / anyOf) in
# structured_schemas/complex_ast.json. Same shape as
# qwen3-0.6b-structured-flat-json.sh so the two baselines compare directly.
MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
ISL_OSL_CONFIGS="128:128"
CONCURRENCY_OPTIONS="1 8 32"
RANDOM_RANGE_RATIO="0.0"
NUM_PROMPTS=128
GPU_MEMORY_UTILIZATION=0.9
ATTENTION_BACKEND="CUSTOM"
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
PERF_TOLERANCE="0.10"

BENCHMARK_EXTRA_BODY="$(cat "$SCRIPT_DIR/structured_schemas/complex_ast.json")"
CAPTURE_PROFILE=1
SD_TRACE_SUMMARY=1
SD_VOCAB_SIZE=151936
