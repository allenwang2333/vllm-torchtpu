#!/usr/bin/env bash
# Single-command CDK reproduction script for Qwen3.5-397B-A17B-FP8 (tpu7x-8, TP=8, MTP=3) on infx-qwen35-agentx.
# Usage:
#   ./scripts/vllm/benchmark/inferencex/qwen3.5/run_agentx_cdk.sh            # Launches all 5 concurrencies (c=8, 12, 16, 24, 32)
#   ./scripts/vllm/benchmark/inferencex/qwen3.5/run_agentx_cdk.sh 8 16 32    # Launches specified concurrencies
#   EVAL_ONLY=true ./scripts/vllm/benchmark/inferencex/qwen3.5/run_agentx_cdk.sh 8 16 32  # Launches GSM8K quality eval

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
CDK_BIN="${CDK_BIN:-cdk}"
SERVER_IMAGE="${SERVER_IMAGE:-us-central1-docker.pkg.dev/cloud-tpu-inference-test/inferencex/vllm@sha256:5c70a4beeecc55cd6a46da341cc8c04842d0b8809fb20530ff6668441a7edb29}"
NODE_POOL="${NODE_POOL:-tpu7x-singlehost-2026}"
DURATION="${DURATION:-3600}"
EVAL_ONLY="${EVAL_ONLY:-false}"
TAG_PREFIX="${TAG_PREFIX:-qwen35-agentx-repro}"

CONCS=("$@")
if [ ${#CONCS[@]} -eq 0 ]; then
  CONCS=(8 12 16 24 32)
fi

for c in "${CONCS[@]}"; do
  case "$c" in
    8|12)
      MAX_SEQS=16
      KV_OFFLOAD="false"
      OFFLOAD_ARGS=()
      ;;
    16)
      MAX_SEQS=16
      KV_OFFLOAD="true"
      OFFLOAD_ARGS=("KV_OFFLOADING_SIZE=429496729600")
      ;;
    24)
      MAX_SEQS=24
      KV_OFFLOAD="true"
      OFFLOAD_ARGS=("KV_OFFLOADING_SIZE=536870912000")
      ;;
    32)
      MAX_SEQS=32
      KV_OFFLOAD="true"
      OFFLOAD_ARGS=("KV_OFFLOADING_SIZE=590558003200")
      ;;
    *)
      echo "Unsupported concurrency: $c (expected 8, 12, 16, 24, or 32)" >&2
      exit 1
      ;;
  esac

  EXTRA_ARGS=()
  if [ "${TPU_DISABLE_PR989_KERNELS:-0}" = "1" ]; then
    EXTRA_ARGS+=("TPU_DISABLE_PR989_KERNELS=1")
  fi
  if [ "$EVAL_ONLY" = "true" ]; then
    EXTRA_ARGS+=("EVAL_ONLY=true")
    JOB_TAG="${TAG_PREFIX}-eval-c${c}"
  else
    EXTRA_ARGS+=("DURATION=${DURATION}" "BENCHMARK_DURATION_SECONDS=${DURATION}")
    JOB_TAG="${TAG_PREFIX}-c${c}-${DURATION}s"
  fi

  echo "Submitting infx-qwen35-agentx for c=${c} (MAX_NUM_SEQS=${MAX_SEQS}, KV_OFFLOAD=${KV_OFFLOAD}, EVAL_ONLY=${EVAL_ONLY})..."
  "$CDK_BIN" job create infx-qwen35-agentx \
    "CLIENT_CONCURRENCIES=${c}" \
    "MAX_NUM_SEQS=${MAX_SEQS}" \
    "KV_OFFLOAD=${KV_OFFLOAD}" \
    "${OFFLOAD_ARGS[@]}" \
    "MAX_NUM_BATCHED_TOKENS=8192" \
    "TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS=0" \
    "ENABLE_PAW=true" \
    "${EXTRA_ARGS[@]}" \
    "NODE_POOL=${NODE_POOL}" \
    "SERVER_IMAGE=${SERVER_IMAGE}" \
    --map-dir "${REPO_ROOT}:vllm-server:/root/cloud-devkit/vllm-torchtpu" \
    --map-dir-copy-dot-git \
    -t "${JOB_TAG}"
done
