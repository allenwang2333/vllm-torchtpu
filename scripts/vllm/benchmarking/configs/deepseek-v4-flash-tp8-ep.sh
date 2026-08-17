#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: DeepSeek-V4-Flash (sparse MLA + SWA/CSA/HCA, TP=8 + EP).
#
# Compiled, not eager: compiled serves at 9216 and is ~3x faster on generation
# (6.9 vs 20.4 tok/s median).
#
# gpu-memory-utilization is 0.75, the highest value verified to serve
# (2026-08-16, ctx 9216: KV 485,612 tokens, 52.69x concurrency). 0.92 still
# fails, in compilation rather than KV allocation -- HLO temporaries exceed
# available HBM. See docs/dsv4_tracker_20260816.md.
#
# This config has NOT completed a run. It is the starting point for the first
# DSv4 throughput measurement, not a validated baseline.
MODEL="/home/patemotter_google_com/workspace/hf_cache/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
KV_CACHE_DTYPE="fp8"

MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=512
GPU_MEMORY_UTILIZATION=0.75

ISL_OSL_CONFIGS="1024:1024 8192:1024"
CONCURRENCY_OPTIONS="64 256"
RANDOM_RANGE_RATIO="0.8"

export VLLM_QUANT_WEIGHT_CACHE_DIR="/home/patemotter_google_com/workspace/dsv4_quant_cache"
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# Greedy decoding, for a deterministic perf measurement.
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
