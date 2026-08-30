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
# ASYNC_SCHEDULING=1 turns on --async-scheduling. Default 0 keeps the
# --no-async-scheduling this script has always passed, so existing runs are
# unchanged. Expressed as one flag rather than appending --async-scheduling to
# "$@": passing both leaves the outcome to argparse ordering, which is not
# something a config this expensive to test should depend on.
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
# Comma-separated static compile buckets. Default is what this script has always
# passed, so existing runs are unchanged. Decode shapes are padded UP to the
# next bucket, so a concurrency sweep needs a bucket per batch size it measures:
# with the default [2,16,512], batches 4 and 8 both execute as 16 and the
# scaling curve is an artifact of bucketing rather than a property of the model.
COMPILE_SIZES="${COMPILE_SIZES:-2,16,512}"
# The prefill bucket must exist or the server pads past its largest bucket.
case ",${COMPILE_SIZES}," in
  *",${MAX_NUM_BATCHED_TOKENS},"*) ;;
  *) echo "COMPILE_SIZES=[${COMPILE_SIZES}] has no bucket for --max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS}" >&2
     exit 2 ;;
esac
# Prefix caching. Expressed as one flag for the same reason ASYNC_SCHEDULING is:
# this script has always passed --no-enable-prefix-caching, and appending
# --enable-prefix-caching to "$@" would leave the outcome to argparse ordering.
# On a hybrid model this is only accepted when the unified KV pool is on.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  PREFIX_CACHING_FLAG=--enable-prefix-caching
else
  PREFIX_CACHING_FLAG=--no-enable-prefix-caching
fi
# K3's tokenizer ships no chat template, so /v1/chat/completions 400s without
# one. Supplying it does not touch /v1/completions, which is what the
# benchmarks and GSM8K use, so it is safe to pass in every arm.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${REPO_DIR}/scripts/kimi_k3_text_chat_template.jinja}"
CHAT_TEMPLATE_ARGS=()
[[ -f "${CHAT_TEMPLATE}" ]] && CHAT_TEMPLATE_ARGS=(--chat-template "${CHAT_TEMPLATE}")
if [[ "${ASYNC_SCHEDULING}" == "1" ]]; then
  ASYNC_SCHEDULING_FLAG=--async-scheduling
else
  ASYNC_SCHEDULING_FLAG=--no-async-scheduling
fi

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
  --compilation-config "{\"compile_sizes\":[${COMPILE_SIZES}]}" \
  --attention-backend CUSTOM \
  --trust-remote-code \
  --language-model-only \
  "${PREFIX_CACHING_FLAG}" \
  "${ASYNC_SCHEDULING_FLAG}" \
  "${CHAT_TEMPLATE_ARGS[@]}" \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.82 \
  --port "${PORT}" \
  "$@"
