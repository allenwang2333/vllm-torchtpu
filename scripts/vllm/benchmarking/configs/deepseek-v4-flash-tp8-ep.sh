#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Benchmark config: DeepSeek-V4-Flash (sparse MLA + SWA/CSA/HCA, TP=8 + EP).
#
# Compiled, not eager: compiled serves at 9216 and is ~3x faster on generation
# (6.9 vs 20.4 tok/s median).
#
# gpu-memory-utilization is 0.55, NOT the 0.92-0.95 house standard, because
# DSv4's KV init allocates outside vLLM's profiled envelope. Measured
# 2026-08-16: startup fails at 0.92 (needs 2.19G, 1.60G free), 0.85 (short by
# 23M) and 0.83 (short by 558M) -- and the shortfall is non-monotonic, so there
# is no threshold to find by nudging. See G5 in docs/dsv4_tracker_20260816.md.
# Raise this only after G5 is fixed.
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
GPU_MEMORY_UTILIZATION=0.55

ISL_OSL_CONFIGS="1024:1024 8192:1024"
CONCURRENCY_OPTIONS="64 256"
RANDOM_RANGE_RATIO="0.8"

export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm
export VLLM_QUANT_WEIGHT_CACHE_DIR="/home/patemotter_google_com/workspace/dsv4_quant_cache"
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# Greedy decoding, for a deterministic perf measurement.
BENCHMARK_TEMPERATURE=0
ATTENTION_BACKEND="CUSTOM"
EVAL_TOLERANCE="0.02"
