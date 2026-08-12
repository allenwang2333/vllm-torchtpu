#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# Nightly perf sweep for GLM-5.2-FP8 with TP=16, EP on TPU v7x-16.
# Sized for decode batch 512 (64 / device across 8 devices per host).
# Throughput/chip = throughput / 16 (TPxDP recorded in config.json).
MODEL="zai-org/GLM-5.2-FP8"
# v7x-16 hosts have no attached disk, so weights stream from GCS via
# --load-format runai_streamer.
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/glm/GLM-5.2-FP8}"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=2
ENABLE_EP=true
# No --quantization: the FP8 checkpoint self-describes.
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
# Single cell at concurrency 64 with 128 prompts while serving perf sits
# at the ~900ms/step kernel floor (~66 tok/s aggregate): a full
# 3-cell x 512-prompt sweep takes ~14h, and the higher-concurrency cells
# only measure queue drain at that speed. Restore "64 256 512" and
# NUM_PROMPTS=512 once kernel perf lands.
CONCURRENCY_OPTIONS="64"
NUM_PROMPTS=128

# Disable random length jitter: benchmark requests use exact nominal 8192:1024 lengths.
RANDOM_RANGE_RATIO="0.0"

# Server configuration:
# - max-model-len=9216: exact nominal length needed for 8192 input + 1024 output tokens.
# - max-num-batched-tokens=4096 and gpu-memory-utilization=0.92 are the
#   config ceiling (dev builds 70/87/95/101-105): the sparse-MLA compile
#   scratch scales with the KV pool (util) and saturates ~96-100G for any
#   8k+ token bucket vs 94.74G physical HBM. 8k needs kernel-side
#   rematerialization; sweep matrix in PR #246.
# - max-num-seqs=64: limits active sequences per step batch, keeping activation working
#   set and XLA compile sizes manageable.
MAX_MODEL_LEN=10240
MAX_NUM_BATCHED_TOKENS=4096
MAX_NUM_SEQS=512

SERVER_READY_WAIT_MIN=180
GPU_MEMORY_UTILIZATION=0.92
KV_CACHE_DTYPE="fp8"

# GLM 5.2 is an MLA model: tpu_platform.get_attn_backend_cls forces
# FLASH_ATTN_MLA whenever use_mla is set.

# Parser flags for tool call / reasoning compatibility.
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--api-server-count=1 --safetensors-load-strategy=prefetch --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45"

# MoE headroom analysis note:
# FORCE_MOE_RANDOM_ROUTING=1 can be set in the environment for headroom analysis
# to force balanced token allocation across MoE experts. Never use for serving or accuracy eval.

# Greedy decoding for bench requests.
BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
