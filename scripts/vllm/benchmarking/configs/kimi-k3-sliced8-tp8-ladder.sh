#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Attribution ladder for the surrogate's prefill cost. Same model, same
# workload, same client as kimi-k3-sliced8-tp8-ep; every serving knob that
# moves TTFT is an env override so one job can walk the ladder rung by rung
# with the arms interleaved (see k3_knob_matrix.sh --arms).
#
# The rungs that matter, and why they are not independent: `--block-size` and
# `--max-num-batched-tokens` each change TTFT on their own AND each move a
# field of the MLA v2 tuning key, so the last rung -- where the key finally
# matches `tuned_params.py:1122` -- is only reachable with both set.
MODEL="moonshotai/Kimi-K3-sliced8"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3-sliced}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=5
RANDOM_RANGE_RATIO="0.0"

# Defaults reproduce kimi-k3-sliced8-tp8-ep exactly.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
K3_BLOCK_SIZE="${K3_BLOCK_SIZE:-16}"
K3_COMPILE_SIZES="${K3_COMPILE_SIZES:-[16,512]}"

SERVER_READY_WAIT_MIN=90
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size $K3_BLOCK_SIZE --compilation-config {\"compile_sizes\":$K3_COMPILE_SIZES}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
