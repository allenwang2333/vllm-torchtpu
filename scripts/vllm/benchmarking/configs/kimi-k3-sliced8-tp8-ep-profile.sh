#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Single-host rehearsal of kimi-k3-tp32-ep-profile.
#
# Same profiler wiring, same capture protocol, same retrieval path
# (k3_profile_collect.sh -> xprof_reduce.py -> one artifact), on the 8-layer
# surrogate and one v7x-8 host. Its job is to prove the *mechanism* before the
# single v7x-32 agent is spent on the real model; its numbers describe the
# surrogate and gate nothing.
MODEL="moonshotai/Kimi-K3-sliced8"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3-sliced}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
# Two prompts: warmup run (unprofiled -- this is the ITL denominator) and the
# profiled run. The trace is the product here, not the throughput number.
NUM_PROMPTS=2
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN=9216
MAX_NUM_BATCHED_TOKENS=512
MAX_NUM_SEQS=1

SERVER_READY_WAIT_MIN=90
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

# phased_profiler_phases=decode_only: a prefill chunk step is ~20x a decode step
# and the prefill window is the bulk of the capture's bytes. Both keys are in
# HASH_IGNORED_ADDITIONAL_CONFIG_KEYS, so neither busts the compile cache.
PHASED_PROFILER_ARGS='--additional-config={"phased_profiler_phases":"decode_only","phased_profiler_decode_only_kv_len_threshold":8200} --profiler-config.torch_profiler_dir /perf_eval_results/kimi-k3-sliced8-tp8-ep-profile/profile --profiler-config.max_iterations 8'
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --compilation-config {\"compile_sizes\":[16,512]} ${PHASED_PROFILER_ARGS}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-1}"
