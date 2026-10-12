#!/usr/bin/env bash
# Single-command CDK reproduction script for Qwen3.5-397B-A17B-FP8 (tpu7x-8, TP=8, MTP=3, Pallas megakernel) on infx-qwen35-agentx.
# Usage:
#   ./scripts/vllm/benchmark/inferencex/qwen3.5/run_agentx_cdk.sh               # Launches all 8 concurrencies (c=4, 8, 12, 16, 20, 24, 28, 32)
#   ./scripts/vllm/benchmark/inferencex/qwen3.5/run_agentx_cdk.sh 8 16 32       # Launches specified concurrencies
#   EVAL_ONLY=true ./scripts/vllm/benchmark/inferencex/qwen3.5/run_agentx_cdk.sh 8 16 32  # Launches GSM8K quality eval

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CDK_BIN="${CDK_BIN:-cdk}"
CDK_ARGS="${CDK_ARGS:-}"
SERVER_IMAGE="${SERVER_IMAGE:-us-central1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-rdna/vllm-torchtpu-caojx_google_com@sha256:cf3c63be2f49bfb6cdaeeade7c72256dd113f7a2896d4cf1cbdbd57dc9c02e23}"
NODE_POOL="${NODE_POOL:-tpu7x-singlehost-2026}"
DURATION="${DURATION:-3600}"
EVAL_ONLY="${EVAL_ONLY:-false}"
SYNTHETIC_ACCEPTANCE_LENGTH="${SYNTHETIC_ACCEPTANCE_LENGTH:-3.39}"
TAG_PREFIX="${TAG_PREFIX:-qwen35-exp21-clean-c32kv}"

ENGINE_ARGS=(
  "EXTRA_VLLM_ARGS=--renderer-num-workers 4"
  "MAX_NUM_BATCHED_TOKENS=8192"
  "TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS=0"
  "ENABLE_PAW=true"
  "VLLM_TPU_QWEN35_MEGAKERNEL=1"
  "VLLM_TPU_QWEN35_MTP_MK=1"
  "VLLM_TPU_QWEN35_MK_MODE=ep"
  "MTP=true"
  "MTP_NUM_SPECULATIVE_TOKENS=3"
  "MOE_FUSED_EP_KERNEL_IMPL=v3"
  "USE_MOE_FUSED_EP_KERNEL=1"
  "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS=64"
  "TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER=1"
  "RAGGED_GATHER_REDUCE_VERSION=v3"
  "TPU_MOE_SKIP_PADDED_TOKENS=1"
  "TPU_MOE_SINGLE_SORT=1"
  "TPU_ROPE_CACHE_ROW_MAJOR=1"
  "TPU_ROPE_CACHE_TRUNCATE=1"
  "VLLM_TPU_FAST_TOKEN_SUBSTITUTION=1"
  "TPU_RPA_DECODE_BATCHED=1"
  "TPU_GREEDY_TOP_TOKENS=1"
  "TPU_TOKEN_BUCKET_EXTRA=4,8,24,48,96,384,3072"
  "TPU_MOE_SEQUENCE_PARALLEL=1"
  "SERVER_IMAGE=${SERVER_IMAGE}"
)

CONCS=("$@")
if [ ${#CONCS[@]} -eq 0 ]; then
  CONCS=(4 8 12 16 20 24 28 32)
fi

for c in "${CONCS[@]}"; do
  MAX_SEQS="$c"
  if [ "$c" -le 8 ]; then
    KV_OFFLOAD="false"
    OFFLOAD_ARGS=("GPU_MEMORY_UTILIZATION=0.78")
  elif [ "$c" -le 20 ]; then
    KV_OFFLOAD="true"
    OFFLOAD_ARGS=("KV_OFFLOADING_SIZE=429496729600" "GPU_MEMORY_UTILIZATION=0.92")
  else
    KV_OFFLOAD="true"
    OFFLOAD_ARGS=("KV_OFFLOADING_SIZE=536870912000" "GPU_MEMORY_UTILIZATION=0.92")
  fi

  EXTRA_ARGS=()
  if [ "${TPU_DISABLE_PR989_KERNELS:-0}" = "1" ]; then
    EXTRA_ARGS+=("TPU_DISABLE_PR989_KERNELS=1")
  fi
  if [ "$EVAL_ONLY" = "true" ]; then
    EXTRA_ARGS+=("EVAL_ONLY=true" 'SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":3}')
    JOB_TAG="${TAG_PREFIX}-eval-c${c}"
  else
    EXTRA_ARGS+=("DURATION=${DURATION}" "BENCHMARK_DURATION_SECONDS=${DURATION}" "SYNTHETIC_ACCEPTANCE_LENGTH=${SYNTHETIC_ACCEPTANCE_LENGTH}")
    JOB_TAG="${TAG_PREFIX}-c${c}-${DURATION}s"
  fi

  echo "Submitting infx-qwen35-agentx for c=${c} (MAX_NUM_SEQS=${MAX_SEQS}, KV_OFFLOAD=${KV_OFFLOAD}, EVAL_ONLY=${EVAL_ONLY})..."
  # shellcheck disable=SC2086
  "$CDK_BIN" $CDK_ARGS job create infx-qwen35-agentx \
    "CLIENT_CONCURRENCIES=${c}" \
    "MAX_NUM_SEQS=${MAX_SEQS}" \
    "KV_OFFLOAD=${KV_OFFLOAD}" \
    "${OFFLOAD_ARGS[@]}" \
    "${ENGINE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --pending-timeout-seconds 7200 \
    --map-dir "${REPO_ROOT}:vllm-server:/root/cloud-devkit/vllm-torchtpu" \
    --map-dir-copy-dot-git \
    -t "${JOB_TAG}"
done
