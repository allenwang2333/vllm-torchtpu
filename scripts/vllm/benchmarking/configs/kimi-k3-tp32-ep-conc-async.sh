#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Concurrency scaling for Kimi-K3 on the v7x-32 pod, WITH --async-scheduling.
#
# The A arm of an A/B against kimi-k3-tp32-ep-conc, which is byte-identical
# except for this one flag. Every K3 recipe in the tree passes
# --no-async-scheduling and none of them says why; upstream defaults it on and
# nothing in the runner touches hybrid state on that path. On the 8-layer
# surrogate turning it on is worth -35.5% TPOT at c1, -39.3% at c4 and -28.5%
# at c16 (dev build 298). The surrogate exaggerates it -- async hides host-side
# per-step work, which is ~2.8 ms of the surrogate\'s 5.7 ms step but only ~9%
# of the real model\'s 32 ms -- so the real model is the measurement that
# decides, and it has only ever been tried at concurrency 1.
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
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --compilation-config {\"compile_sizes\":[16,64,2048]}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
