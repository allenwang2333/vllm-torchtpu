#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Round 3's "fixed config": the best serving shape measured on the real
# Kimi-K3 on the pod (TTFT 4488.3 -> 1432.9 ms across BK #311/#317), and the
# shape a decision would actually be taken at. Identical to the default arm of
# `kimi-k3-tp32-ep-fixed`'s ancestor `kimi-k3-tp32-ep-tunehit`; renamed because
# nothing here is about the tuning table any more.
#
# Load-bearing pieces, none of them free to change:
#   --block-size 256        BK #311, TTFT -42.6% on the real model
#   --max-model-len 10240   pages_per_seq = 40, which selects MLA row :1169
#                           (10496 gives 41 and misses; BK #317, TTFT -30.4%)
#   mnbt 8192 + compile 8192   the bucket that row is keyed on
#   --kv-cache-dtype fp8    kv_packing 4, so page_size_per_kv_packing = 64
#   --enable-ep-weight-filter  K3 cannot load without it: unfiltered streaming
#                           accumulated >1 TB of host memory and drew OOM kills
#
# Async scheduling has never been combined with the above; left off here so
# this config reproduces the measured shape, and overridable for the lane that
# wants to add it.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

ISL_OSL_CONFIGS="8192:1024"
# c1 is where every round-2 pod number was taken and where the MoE
# load-imbalance bubble is largest (topk=16 balls into 32 rank-bins). c8 is
# --max-num-seqs, i.e. a full batch, where the bubble should be much flatter.
# Both run inside ONE server lifetime, so an A/B over server restarts gets the
# concurrency slope for free.
# Private override name: run_benchmarks.sh pre-assigns a non-empty
# CONCURRENCY_OPTIONS before sourcing, so `${CONCURRENCY_OPTIONS:-...}` here
# would never reach its default.
CONCURRENCY_OPTIONS="${K3_CONCURRENCY:-1 8}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
# 5 at c1 matches BK #305/#317, where 5 prompts x 3 reps already resolved 1%;
# 8 at c8 is exactly one full --max-num-seqs batch. Fewer than 8 at c8 would
# quietly measure a smaller batch.
NUM_PROMPTS_BY_CONCURRENCY="${NUM_PROMPTS_BY_CONCURRENCY:-1:5 8:8}"
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=8

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# tpu_runner.py:632 makes the SMALLEST compiled size the decode batch, so this
# is the decode token bucket and not only a compile hint. 16 everywhere on
# record; [8,8192] is the shippable floor at --max-num-seqs 8.
K3_COMPILE_SIZES="${K3_COMPILE_SIZES:-[16,8192]}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --block-size 256 --compilation-config {\"compile_sizes\":$K3_COMPILE_SIZES}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
