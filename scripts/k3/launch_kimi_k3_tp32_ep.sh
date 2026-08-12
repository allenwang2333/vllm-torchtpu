#!/usr/bin/env bash
# Launch Kimi-K3 with TP32 + expert parallelism on an existing Ray cluster.
# The caller is responsible for starting Ray and provisioning the TPU hosts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
REVISION="${REVISION:-9f62e4e9fffbd0a83ddd60e1c209d828994b3569}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
PROF_DIR="${PROF_DIR:-${REPO_DIR}/results/kimi-k3-profile}"

export PATH="${REPO_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export TPU_MULTIHOST_BACKEND=ray
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,4}"
export TORCH_TPU_TOPOLOGY="${TORCH_TPU_TOPOLOGY:-2,2,4,2}"
export JAX_PLATFORMS=""
export VLLM_TARGET_DEVICE=tpu
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-10800}"
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false

mkdir -p "${PROF_DIR}"

exec "${REPO_DIR}/.venv/bin/vllm" serve "${MODEL}" \
  --revision "${REVISION}" \
  -tp 32 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --quantization compressed-tensors \
  --safetensors-load-strategy lazy \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --block-size 16 \
  --compilation-config '{"compile_sizes":[2,8,16,512]}' \
  --attention-backend CUSTOM \
  --trust-remote-code \
  --language-model-only \
  --no-enable-prefix-caching \
  --async-scheduling \
  --gpu-memory-utilization 0.82 \
  --port "${PORT}" \
  --profiler-config.profiler=torch \
  "--profiler-config.torch_profiler_dir=$PROF_DIR" \
  --profiler-config.delay_iterations=3 \
  --profiler-config.max_iterations=15 \
  --profiler-config.ignore_frontend=true \
  "$@"
