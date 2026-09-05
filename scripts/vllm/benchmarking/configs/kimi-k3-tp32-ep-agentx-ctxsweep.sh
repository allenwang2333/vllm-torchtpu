#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# [branch-local, not for merge] Kimi-K3 on the v7x-32 pod: context-length
# sweep with the AgentX server geometry (kimi-k3-tp32-ep-agentx.sh), bs=1,
# exact lengths. Separates long-context prefill cost (TTFT) from decode cost
# at long context (TPOT, no co-scheduled prefill at conc 1) — the AgentX
# replay (build 6) only shows their sum.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:256 32768:256 65536:256 114688:256"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=2
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"
ENABLE_PREFIX_CACHING=false

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --compilation-config {\"compile_sizes\":[16,1024,2048,4096,8192]}"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
