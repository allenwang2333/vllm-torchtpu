#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# DIAGNOSTIC config, not for CI: Qwen3-VL-2B-Instruct at TP=8 with
# cudagraph_mm_encoder + mm-encoder-tp-mode=data, mirroring
# kimi-k3-pruned75-tp8-multimodal.sh's exact multimodal flags. Qwen3-VL
# natively implements SupportsEncoderCudaGraph upstream, so this isolates
# whether a cross-rank deadlock seen with Kimi-K3-pruned75 (workers stuck,
# "No available shared memory broadcast block found") is a generic
# MMEncoderManager/mm-encoder-tp-mode=data issue or Kimi-specific. Small
# (2B) model chosen for fast iteration, not perf relevance.
MODEL="Qwen/Qwen3-VL-2B-Instruct"
TENSOR_PARALLELISM=8
DATA_PARALLELISM=1
ENABLE_EP=false
QUANTIZATION=""
ISL_OSL_CONFIGS="128:1"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=1
RANDOM_RANGE_RATIO="0.8"
GPU_MEMORY_UTILIZATION=0.90
ATTENTION_BACKEND="CUSTOM"
MAX_NUM_BATCHED_TOKENS=8192
MAX_MODEL_LEN=24576
MAX_NUM_SEQS=64
BENCHMARK_TEMPERATURE=0
EVAL_TOLERANCE="0.02"
export VLLM_ENGINE_READY_TIMEOUT_S=1800
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --mm-encoder-tp-mode data --compilation-config {\"cudagraph_mm_encoder\":true}"
