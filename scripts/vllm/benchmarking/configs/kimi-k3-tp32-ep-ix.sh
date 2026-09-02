#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# [branch-local, not for merge] Kimi-K3 on the v7x-32 pod, server geometry
# for the InferenceX concurrency sweep (8..128 at 8k/1k). Derived from
# kimi-k3-tp32-ep.sh; differences:
#   - throughput regime: max-num-seqs 128, max-num-batched-tokens 2048,
#     async scheduling on (harness default) instead of the bs=1 latency probe
#   - the built-in harness bench is a tiny warmup (conc 1, 4 prompts); the
#     real sweep is driven by k3_ix_sweep.sh against the kept-alive server
#   - compile_sizes covers the decode buckets up to 128 seqs and the 2048
#     prefill chunk; anything missed compiles lazily on first hit
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=4
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN=9216
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=128

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --compilation-config {\"compile_sizes\":[16,32,64,128,256,512,1024,2048]}"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
