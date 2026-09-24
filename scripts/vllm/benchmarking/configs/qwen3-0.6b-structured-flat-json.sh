#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly structured-decoding gate: Qwen3-0.6B TP=1, every request constrained
# by the flat 5-key JSON grammar in structured_schemas/flat_json.json.
# Sibling of qwen3-0.6b-structured-complex-ast.sh; --extra-body applies to
# every request of a run, so each grammar is its own config and baseline.
MODEL="Qwen/Qwen3-0.6B"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=1
ENABLE_EP=false
# Short prompt, 128 generated tokens: decode-dominated, so the per-step mask
# cost is what moves the numbers.
ISL_OSL_CONFIGS="128:128"
CONCURRENCY_OPTIONS="1 8 32"
RANDOM_RANGE_RATIO="0.0"
NUM_PROMPTS=128
GPU_MEMORY_UTILIZATION=0.9
ATTENTION_BACKEND="CUSTOM"
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
# The SD scopes are sub-millisecond, so they swing more than throughput.
PERF_TOLERANCE="0.10"

BENCHMARK_EXTRA_BODY="$(cat "$SCRIPT_DIR/structured_schemas/flat_json.json")"
CAPTURE_PROFILE=1
SD_TRACE_SUMMARY=1
SD_VOCAB_SIZE=151936
