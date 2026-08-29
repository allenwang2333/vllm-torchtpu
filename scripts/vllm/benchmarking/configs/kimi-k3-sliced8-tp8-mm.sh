#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Kimi-K3 MULTIMODAL smoke on one v7x-8 host.
#
# gs://.../k3-sliced truncates only the language model: its index still carries
# the complete MoonViT tower (165 tensors) and mm_projector (3), unmodified.
# So dropping --language-model-only here exercises the real vision path --
# NaViT resize, patch embed, 27 encoder blocks, patchmergerv2 projector, the
# bucketed vision scatter and the encoder cache -- at 1/12 the language cost.
# Generated text is meaningless (8 of 93 layers); the encoder path working is
# the measurement.
#
# Every other K3 recipe in the tree serves with --language-model-only, so no
# CI job covers this at all.
MODEL="moonshotai/Kimi-K3-sliced8"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3-sliced}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""

# Text rows only; the image request is driven separately by k3_mm_smoke.sh.
ISL_OSL_CONFIGS="1024:64"
CONCURRENCY_OPTIONS="4"
NUM_PROMPTS=8
RANDOM_RANGE_RATIO="0.0"

# A multimodal K3 declares a large per-request mm token budget, and vLLM
# refuses a max-num-batched-tokens below it.
MAX_MODEL_LEN=16384
MAX_NUM_BATCHED_TOKENS=16384
MAX_NUM_SEQS=8

SERVER_READY_WAIT_MIN=90
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-false}"

# No --language-model-only, and images are allowed.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --limit-mm-per-prompt {\"image\":2,\"video\":0}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:+$EXTRA_BENCH_ARGS }--trust-remote-code"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
