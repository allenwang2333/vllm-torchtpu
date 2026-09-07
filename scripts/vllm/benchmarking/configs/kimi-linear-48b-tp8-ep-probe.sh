#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Kimi-Linear-48B-A3B (the KDA + MLA hybrid that Kimi-K3 also uses, served by
# the same kimi_k3 model code and the compact-mamba KV block layout) on 8
# chips, for the concurrency probe (issue #845). Same serve flags as the K3
# nightly where they apply (fp8 KV, block 256, CUSTOM attention).
MODEL="moonshotai/Kimi-Linear-48B-A3B-Instruct"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="1024:256"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=2
RANDOM_RANGE_RATIO="0.0"
MAX_MODEL_LEN="${K3_SWEEP_CTX:-8192}"
MAX_NUM_SEQS="${K3_SWEEP_MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS=4096
SERVER_READY_WAIT_MIN=120
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ATTENTION_BACKEND="CUSTOM"
GPU_MEMORY_UTILIZATION=0.85
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --block-size 256 --compilation-config {\"compile_sizes\":[1,8,16,32,64,128,256,1024,2048,4096]}${K3_EXTRA_SERVE_ARGS:+ $K3_EXTRA_SERVE_ARGS}"
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.03"
PERF_TOLERANCE="0.05"
