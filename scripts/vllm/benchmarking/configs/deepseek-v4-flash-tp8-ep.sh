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

# 9216, not 10240: at util 0.75 the profiler leaves 51.71 GiB for KV and 10240
# needs 53.11, so startup fails outright (measured 2026-08-17). 9216 is also
# exactly 8192+1024 and 1024+8192, so every combo below fits.
MAX_MODEL_LEN=9216
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=512
GPU_MEMORY_UTILIZATION=0.75

# Decode-heavy (1024:8192) is listed second deliberately: ISL/OSL is the outer
# loop, so a timeout truncates the tail, and decode is where this model is
# weakest -- the gap against tpu-inference widens from 1.45x at batch 8 to
# 3.34x at batch 128 -- and where the recovered KV capacity is spent.
ISL_OSL_CONFIGS="1024:1024 1024:8192 8192:1024"
CONCURRENCY_OPTIONS="64 256"
# 320 (the default) x 8192 output tokens is 2.6M tokens per combo, and this
# model's aggregate decode rate at these concurrencies is unmeasured -- that is
# what this sweep exists to find out. 128 bounds the unknown.
NUM_PROMPTS=128
RANDOM_RANGE_RATIO="0.8"
# "min" gives [rL, L]; the default "symmetric" gives [(1-r)L, (1+r)L], which at
# ISL 8192 samples up to 14746 tokens and exceeds max-model-len. The other large
# configs set this for the same reason.
RANGE_RATIO_STYLE="min"

export VLLM_QUANT_WEIGHT_CACHE_DIR="/home/patemotter_google_com/workspace/dsv4_quant_cache"
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# Greedy decoding, for a deterministic perf measurement.
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
