#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Concurrency scaling for Kimi-K3 on the v7x-32 pod.
#
# kimi-k3-tp32-ep is a bs=1 latency probe -- the only K3 serving number that
# exists. This config sweeps concurrency 1/4/16 at the same 8k/1k shape so
# there is a throughput curve to reason about, and so the ceiling recorded in
# kimi-k3-12l-proxy-tp8-ep.sh ("above 4 concurrent sequences a stress test
# produced repeated-token degeneration at high KV occupancy, root cause
# unknown") is exercised on the real model rather than the 12-layer proxy.
#
# No baseline JSON on purpose: nothing should gate on a curve whose top end is
# under suspicion.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1 4 16"
# run_benchmarks.sh uses one NUM_PROMPTS for every concurrency cell, so it is
# sized by the slowest one: 16 sequential 8k/1k requests is ~11 min at c1.
NUM_PROMPTS=16
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN=9216
# 512 makes an 8192-token prompt 16 chunk steps; at concurrency 16 that is the
# difference between a queue and a curve. 2048 with compile_sizes to match.
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=16

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --compilation-config {\"compile_sizes\":[16,64,2048]}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
