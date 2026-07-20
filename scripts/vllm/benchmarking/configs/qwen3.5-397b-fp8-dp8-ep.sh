#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DP=8 attention-DP + EP sweep for Qwen3.5-397B-A17B on a single tpu7x-8.
# Mirrors PR #349's working local config, plus a perf ISL/OSL sweep so we can
# compare DP=8/TP=1 against the TP=8/DP=1 baseline (qwen3.5-397b-fp8-tp8-ep).
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
TENSOR_PARALLELISM=1
DATA_PARALLELISM=8
ENABLE_EP=true
QUANTIZATION="fp8"
ISL_OSL_CONFIGS="1024:1024"
CONCURRENCY_OPTIONS="64"
RANDOM_RANGE_RATIO="0.8"
SERVER_READY_WAIT_MIN=120
GPU_MEMORY_UTILIZATION=0.9

MAX_NUM_BATCHED_TOKENS=1024
MAX_NUM_SEQS=64

# Use the fused Conv1D+GDN v3 kernel. The v1 (chunked_kernel_pd) path lowers
# the persistent Mamba-conv state (bf16[65,3,12288], I/O-aliased in the 105-
# entry may-alias set) via a ConcatBitcast chain under libtpu 0.0.42.1, whose
# byte-level assembly progressively corrupts the aliased state buffer under
# sustained load. Symptom: MMLU-pro exact_match collapses from ~0.82 to ~0.39
# after ~800 concurrent requests, with later subjects hitting 0. v3 writes
# the state via direct convert_element_type -> copy_bitcast_fusion, matching
# the main-branch (libtpu 0.0.40) code path. Verified 2026-07-10:
# mmlu_pro=0.8229 with v3 vs 0.3929 with v1 on this exact config.
export RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd
export USE_MOE_SPARSE_CORE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# Extra arguments for vllm serve
EXTRA_SERVE_ARGS="--block-size 256 --limit-mm-per-prompt {\"image\":0,\"video\":0} --default-chat-template-kwargs {\"enable_thinking\":false}"
MMLU_PRO_DISABLE_MULTITURN_ARGS=true
