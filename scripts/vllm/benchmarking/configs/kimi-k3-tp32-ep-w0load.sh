#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Round-3 W0-b: the per-(layer, expert) router load distribution, for lane W3.
#
# Serve flags are byte-identical to kimi-k3-tp32-ep-w0profile except that the
# profiler is not armed. It runs in its own SERVER LIFETIME, never inside the
# profiled one: the accumulation lives inside the torch.compile'd forward (a
# host read of topk_ids would be one device sync per layer per step), so
# arming it adds HLO, and a capture whose op count differs from #316's is not
# comparable to #316. Separate lifetime, same job, profile first.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
# 2 x 1024 decode steps = 2048, which is exactly what SLOTS=128 x BUCKET=16
# covers at full resolution. No warmup run: it would burn half the slots.
NUM_PROMPTS=2
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=8

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":[16,8192]}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
BENCHMARK_WARMUP_RUNS="${BENCHMARK_WARMUP_RUNS:-0}"
