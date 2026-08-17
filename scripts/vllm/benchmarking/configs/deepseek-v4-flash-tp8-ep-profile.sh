#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Diagnostic sibling of deepseek-v4-flash-tp8-ep: one cheap combo with the
# torch profiler on, to capture a trace when the throughput sweep comes in
# below its floor. Not a baseline -- profiling costs roughly +11% TTFT and
# -4% throughput, so nothing here is comparable to the sweep's numbers.
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

# One decode-heavy combo, few prompts: the trace only needs enough steady-state
# decode to be representative, and a long run makes the trace harder to read.
ISL_OSL_CONFIGS="1024:2048"
CONCURRENCY_OPTIONS="64"
NUM_PROMPTS=32
RANDOM_RANGE_RATIO="0.8"

export VLLM_QUANT_WEIGHT_CACHE_DIR="/home/patemotter_google_com/workspace/dsv4_quant_cache"
export VLLM_ENGINE_READY_TIMEOUT_S=1800

BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
