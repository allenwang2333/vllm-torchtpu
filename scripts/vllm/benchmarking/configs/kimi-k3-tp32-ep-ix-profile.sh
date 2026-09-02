#!/bin/bash
# shellcheck disable=SC2034  # Variables are sourced by run_benchmarks.sh
# [branch-local, not for merge] Same server geometry as kimi-k3-tp32-ep-ix
# but with the phased profiler armed. The harness's tiny built-in bench
# runs one profiled session on warmup traffic; k3_ix_sweep.sh clears the
# trace dir afterwards and drives the real profiled session with the
# InferenceX client (--profile) at the requested concurrency.
MODEL="moonshotai/Kimi-K3"
MODEL_URI="${MODEL_URI:-gs://tpu-commons-ci/moonshootai/kimi/k3}"
TENSOR_PARALLELISM=32
DATA_PARALLELISM=1
ENABLE_EP=true
QUANTIZATION=""
ISL_OSL_CONFIGS="8192:1024"
CONCURRENCY_OPTIONS="1"
NUM_PROMPTS=4
RANDOM_RANGE_RATIO="0.0"

MAX_MODEL_LEN=9216
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=128

SERVER_READY_WAIT_MIN=240
GPU_MEMORY_UTILIZATION=0.7
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-true}"

# Phased profiler: USE_PHASED_PROFILER must reach the server process env
# (exported here; run_benchmarks starts the server from this shell).
# CAPTURE_PROFILE=1 makes the harness arm --profiler-config.* on the serve
# command (PROFILE_DIR defaults to $RESULTS_DIR/profile).
export CAPTURE_PROFILE=1
export USE_PHASED_PROFILER=1

# kv-len thresholds: capture decode_only at steady-state ~8k context and
# prefill_only mid-way through the 8k prompt instead of on its first chunk.
# Excluded from the TPU compile cache key, so no recompile.
# max_iterations 5 bounds the per-phase trace size (32 ranks on the pod).
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:+$EXTRA_SERVE_ARGS }--trust-remote-code --enable-ep-weight-filter --language-model-only --limit-mm-per-prompt {\"image\":0,\"video\":0} --model-loader-extra-config {\"memory_limit\":17179869184} --compilation-config {\"compile_sizes\":[16,32,64,128,256,512,1024,2048]} --profiler-config.max_iterations 5 --additional-config {\"phased_profiler_decode_only_kv_len_threshold\":8192,\"phased_profiler_prefill_only_kv_len_threshold\":4096}"

# Upload destination recorded in config.json (profile_gcs_dir); the dev yml
# step performs the copy from the agent after the run.
PROFILE_GCS_BASE="gs://cuiq-inference-test/k3/vtt/20260902-k3-8k1k-tp32ep/xprof"

BENCHMARK_TEMPERATURE="0"
EVAL_TOLERANCE="0.025"
