#!/usr/bin/env bash
# Launch one host's slice of Kimi-K3 at DP=4 x TP=8 + EP=32 (mp multihost,
# no Ray). One `vllm serve` process per host: DP_START_RANK=0 is the API
# head, the rest pass --headless. Attention is DP=4 x TP=8; the MoE stays
# EP=32 across the unified 32-device slice.
#
# Sizing is PER DP RANK: MAX_NUM_SEQS / MAX_NUM_BATCHED_TOKENS apply to each
# of the 4 engines, so they are job 288's TP=32 values divided by 4 (a naive
# copy would give 4x the intended global batch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-moonshotai/Kimi-K3}"
REVISION="${REVISION:-9f62e4e9fffbd0a83ddd60e1c209d828994b3569}"
PORT="${PORT:-8000}"
DP_SIZE="${DP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-8}"
DP_START_RANK="${DP_START_RANK:-0}"
DP_RPC_PORT="${DP_RPC_PORT:-29500}"
# Workers on every host must agree on the torch_tpu DP rendezvous; the
# platform's default is a per-host pick_unused_port(), which disagrees across
# hosts. Pin both (tpu_platform.py only setdefaults them).
DP_MASTER_PORT="${DP_MASTER_PORT:-29511}"
: "${DP_MASTER_ADDR:?Set DP_MASTER_ADDR to the head host IP}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10240}"
# Per-rank values: job 288's global MAX_NUM_SEQS=32 / mnbt=8192 divided by
# DP=4. Decode buckets cover per-rank batches 1..8 (global c=1..32); the
# largest bucket must equal MAX_NUM_BATCHED_TOKENS.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
COMPILE_SIZES="${COMPILE_SIZES:-1,2,4,8,2048}"
# The prefill bucket must exist or the server pads past its largest bucket.
case ",${COMPILE_SIZES}," in
  *",${MAX_NUM_BATCHED_TOKENS},"*) ;;
  *) echo "COMPILE_SIZES=[${COMPILE_SIZES}] has no bucket for --max-num-batched-tokens=${MAX_NUM_BATCHED_TOKENS}" >&2
     exit 2 ;;
esac
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  PREFIX_CACHING_FLAG=--enable-prefix-caching
else
  PREFIX_CACHING_FLAG=--no-enable-prefix-caching
fi
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${REPO_DIR}/scripts/kimi_k3_text_chat_template.jinja}"
CHAT_TEMPLATE_ARGS=()
[[ -f "${CHAT_TEMPLATE}" ]] && CHAT_TEMPLATE_ARGS=(--chat-template "${CHAT_TEMPLATE}")
if [[ "${ASYNC_SCHEDULING}" == "1" ]]; then
  ASYNC_SCHEDULING_FLAG=--async-scheduling
else
  ASYNC_SCHEDULING_FLAG=--no-async-scheduling
fi
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# Only the head serves the API; followers run their engine core headless.
ROLE_ARGS=()
if [[ "${DP_START_RANK}" == "0" ]]; then
  ROLE_ARGS=(--host 0.0.0.0 --port "${PORT}")
else
  ROLE_ARGS=(--headless)
fi

export PATH="${REPO_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
# mp backend: each host runs its own engine over its local 8 cores; the
# EP=32 collectives span hosts through the unified torch_tpu slice whose
# bootstrap prepare_mp_multihost_dp_env rendezvous over TCPStore.
export TPU_MULTIHOST_BACKEND=mp
export JAX_PLATFORMS=""
export VLLM_TARGET_DEVICE=tpu
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-10800}"
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export TORCH_TPU_DP_MASTER_ADDR="${DP_MASTER_ADDR}"
export TORCH_TPU_DP_MASTER_PORT="${DP_MASTER_PORT}"

exec "${REPO_DIR}/.venv/bin/vllm" serve "${MODEL}" \
  --revision "${REVISION}" \
  -tp "${TP_SIZE}" \
  --data-parallel-size "${DP_SIZE}" \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank "${DP_START_RANK}" \
  --data-parallel-address "${DP_MASTER_ADDR}" \
  --data-parallel-rpc-port "${DP_RPC_PORT}" \
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
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --trust-remote-code \
  --language-model-only \
  "${PREFIX_CACHING_FLAG}" \
  "${ASYNC_SCHEDULING_FLAG}" \
  "${CHAT_TEMPLATE_ARGS[@]}" \
  --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.82 \
  "${ROLE_ARGS[@]}" \
  "$@"
