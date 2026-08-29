#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Single-host surrogate for kimi-k3-tp32-ep.
#
# gs://tpu-commons-ci/moonshootai/kimi/k3-sliced is the same Kimi-K3
# checkpoint truncated to the first 8 decoder layers (126 GB instead of
# 1561 GB) with the hybrid layer pattern preserved: full_attn_layers
# [4, 8] (MLA) and kda_layers [1, 2, 3, 5, 6, 7] (Kimi Delta Attention),
# 896 experts, mxfp4. So it exercises both attention paths, the EP MoE
# kernel and the hybrid KV-cache sizing on one v7x-8 host, in minutes
# instead of the pod's tens of minutes of streaming.
#
# Numbers from this config measure the surrogate, not Kimi-K3. Nothing
# gates on them; there is deliberately no baseline JSON.
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

MAX_MODEL_LEN=9216
MAX_NUM_BATCHED_TOKENS=512
MAX_NUM_SEQS=1

SERVER_READY_WAIT_MIN=90
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --compilation-config {\"compile_sizes\":[16,512]}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
