#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DeepSeek-V4 prefill/decode disaggregation over Raiden Stage 3, DP-attn 8 on
# the prefill replica and DP-attn 8 on the decode replica, with EP-MoE 8 on
# both. That is 16 chips, so the two replicas live on separate tpu7x-8 hosts
# and you run this script once per host with ROLE set.
#
# Set PREFILL_DP=4 DECODE_DP=4 ROLE=all to get the half-size configuration that
# fits one host. The sharding is otherwise identical, which is the point: the
# single-host run is the same code path with fewer ranks.
#
# The serving flags and Raiden environment here are copied from the two-host
# jobset in cdk-dsv4-pd-dp8-dp8-raiden.yml, which is the configuration that has
# actually been run. Keep the two in sync.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Launch one leg of the DeepSeek-V4 DP8/DP8 Raiden P/D pair.

On the prefill host:
  PREFILL_HOST=<p-ip> DECODE_HOST=<d-ip> ROLE=prefill \
    bash examples/disagg/launch_dsv4_dp8_dp8.sh

On the decode host:
  PREFILL_HOST=<p-ip> DECODE_HOST=<d-ip> ROLE=decode \
    bash examples/disagg/launch_dsv4_dp8_dp8.sh

Both addresses must be routable from both hosts. Each leg advertises its own
for the Raiden reshard store, and the peer dials it there.

Half-size rehearsal on a single host:
  ROLE=all PREFILL_DP=4 DECODE_DP=4 bash examples/disagg/launch_dsv4_dp8_dp8.sh

Useful knobs: MODEL_PATH, PYTHON, RUN_DIR, PREFILL_PORT, DECODE_PORT,
PROXY_PORT, CACHE_ROOT, LOAD_FORMAT, MAX_MODEL_LEN, COMPILE_SIZES,
GPU_MEMORY_UTILIZATION, DRY_RUN.

DRY_RUN=1 writes the launch commands to RUN_DIR and exits without touching a
TPU, which is how you diff this against the jobset.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

port_pids() {
  local port="$1"
  ss -ltnp 2>/dev/null \
    | awk -v port=":${port}" '$4 ~ port "$" {print}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | sort -u
}

stop_port_listeners() {
  local pids=()
  local port pid
  for port in "$@"; do
    while IFS= read -r pid; do
      [[ -n "${pid}" ]] && pids+=("${pid}")
    done < <(port_pids "${port}")
  done
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi
  mapfile -t pids < <(printf '%s\n' "${pids[@]}" | sort -u)
  echo "Stopping existing listeners: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    local alive=()
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive+=("${pid}")
      fi
    done
    if [[ "${#alive[@]}" -eq 0 ]]; then
      return 0
    fi
    sleep 0.5
  done
  kill -KILL "${pids[@]}" 2>/dev/null || true
}

get_local_ip() {
  if [[ -n "${LOCAL_IP:-}" ]]; then
    echo "${LOCAL_IP}"
    return 0
  fi
  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -n "${ip}" ]]; then
    echo "${ip}"
    return 0
  fi
  echo "127.0.0.1"
}

ROLE="${ROLE:-}"
if [[ "${ROLE}" != "prefill" && "${ROLE}" != "decode" && "${ROLE}" != "all" ]]; then
  echo "ROLE must be prefill, decode or all" >&2
  usage >&2
  exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python3}"
RUN_ROOT="${RUN_ROOT:-${HOME}/pd_disagg_runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/dsv4_dp8_dp8_${ROLE}_${RUN_ID}}"

MODEL_PATH="${MODEL_PATH:-/mnt/disk/${USER}/models/DeepSeek-V4-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4-Flash}"
# runai_streamer is what the jobset uses, because there MODEL_PATH is a gs://
# prefix. A local snapshot loads faster with the default reader.
LOAD_FORMAT="${LOAD_FORMAT:-}"

LOCAL_ROUTABLE_IP="$(get_local_ip)"
PREFILL_HOST="${PREFILL_HOST:-${LOCAL_ROUTABLE_IP}}"
DECODE_HOST="${DECODE_HOST:-${LOCAL_ROUTABLE_IP}}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"

PREFILL_PORT="${PREFILL_PORT:-8400}"
DECODE_PORT="${DECODE_PORT:-9400}"
PROXY_PORT="${PROXY_PORT:-8000}"

# DP-attn means TP 1 and no prefill context parallelism, so each attention rank
# owns whole heads and the transfer is rank-local. EP-MoE rides along on
# --enable-expert-parallel at the same width.
PREFILL_TP="${PREFILL_TP:-1}"
DECODE_TP="${DECODE_TP:-1}"
PREFILL_DP="${PREFILL_DP:-8}"
DECODE_DP="${DECODE_DP:-8}"

# Co-locating both legs means their reshard store and KV transfer ports must
# not collide. Across two hosts the defaults are free either way, so the same
# numbers work for both layouts.
PREFILL_RESHARD_PORT_BASE="${PREFILL_RESHARD_PORT_BASE:-27000}"
PREFILL_STORE_DISPATCH_PORT_BASE="${PREFILL_STORE_DISPATCH_PORT_BASE:-27100}"
DECODE_RESHARD_PORT_BASE="${DECODE_RESHARD_PORT_BASE:-28000}"
DECODE_STORE_DISPATCH_PORT_BASE="${DECODE_STORE_DISPATCH_PORT_BASE:-28100}"
PREFILL_KV_TRANSFER_PORT="${PREFILL_KV_TRANSFER_PORT:-9100}"
DECODE_KV_TRANSFER_PORT="${DECODE_KV_TRANSFER_PORT:-9200}"
TPU_SIDE_CHANNEL_PORT="${TPU_SIDE_CHANNEL_PORT:-9600}"

# Rank offsets only matter when both legs share a host.
PREFILL_RANK_OFFSET="${PREFILL_RANK_OFFSET:-0}"
if [[ "${ROLE}" == "all" ]]; then
  DECODE_RANK_OFFSET="${DECODE_RANK_OFFSET:-${PREFILL_DP}}"
else
  DECODE_RANK_OFFSET="${DECODE_RANK_OFFSET:-0}"
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
# 512, not the 4096 a dense model would take. DP attention all-gathers every
# rank's tokens before the MoE, so the graph is sized by tokens-per-step times
# DP, and only about 95 GiB of HBM is free once the weights are resident.
# Measured at DP8: 114.93 GiB of XLA temporaries for a 4096-token step, 100.96
# for 2048, and 94.75 for 1024, which misses the 94.74 GiB budget by ten
# megabytes. Halving the step buys roughly 6 GiB, so 512 leaves margin.
COMPILE_SIZES="${COMPILE_SIZES:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"

# Everything cacheable goes on the attached NVMe. Eight DSv4 workers build
# several GB of compile artifacts, this VM's root is 145G and shared, and when
# it fills the workers die mid-compile with a truncated log that never names
# the disk.
CACHE_ROOT="${CACHE_ROOT:-/mnt/disk/${USER}/cache}"

mkdir -p "${RUN_DIR}/logs" "${CACHE_ROOT}"/{vllm,inductor,jax,tpu_logs,tmp}

namespace_base="${TPU_KV_TRANSFER_NAMESPACE_BASE:-dsv4_dp8}"
prefill_namespace="${namespace_base}_prefill"
decode_namespace="${namespace_base}_decode"

compilation_config="{\"backend\":\"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor\",\"compile_sizes\":[${COMPILE_SIZES}],\"inductor_compile_config\":{\"enable_auto_functionalized_v2\":false,\"size_asserts\":false,\"alignment_asserts\":false,\"scalar_asserts\":false}}"
p_kv='{"kv_connector":"TPUConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer"}'
d_kv='{"kv_connector":"TPUConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer"}'

common_env=$(cat <<EOF
export PJRT_DEVICE=TPU TPU_BACKEND_TYPE=jax JAX_PLATFORMS=tpu,cpu VLLM_TARGET_DEVICE=tpu;
export VLLM_PLUGINS=torchtpu,torchtpu_layers;
export MODEL_IMPL_TYPE=vllm NEW_MODEL_DESIGN=0;
export SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0;
export USE_MOE_SPARSE_CORE=0 MOE_REQUANTIZE_WEIGHT_DTYPE=fp4 ENABLE_QUANTIZED_MATMUL_KERNEL=0;
export XLA_PYTHON_CLIENT_PREALLOCATE=false VLLM_ALLOW_LONG_MAX_MODEL_LEN=1;
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_latency_hiding_scheduler=false";
export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled;
# MoE fp4 requantization dominates startup and has taken ~16 minutes to healthy.
export VLLM_ENGINE_READY_TIMEOUT_S=2700;
export VLLM_CACHE_ROOT='${CACHE_ROOT}/vllm';
export TORCHINDUCTOR_CACHE_DIR='${CACHE_ROOT}/inductor';
export JAX_COMPILATION_CACHE_DIR='${CACHE_ROOT}/jax';
export TPU_LOG_DIR='${CACHE_ROOT}/tpu_logs';
export TMPDIR='${CACHE_ROOT}/tmp';
export TPU_SIDE_CHANNEL_PORT='${TPU_SIDE_CHANNEL_PORT}';
# The unified block pool is off. It gives the pools an aliased_raw binding and
# build_dsv4_pool_manifest requires private typed tensors, so it is rejected.
export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0;
export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1;
export TPU_RAIDEN_DSV4_ADMISSION=1;
export TPU_KV_RESHARD_TRANSPORT=raiden;
export TPU_RAIDEN_RESHARD_IMPL=store;
# DP attention gives every rank whole requests, so DSv4 admission rejects any
# transfer parallelism other than 1.
export TPU_RAIDEN_TRANSFER_PARALLELISM=1;
export TPU_ENABLE_D2H_TRANSFER=1;
export TPU_KV_SHM_POOL_GB=\${TPU_KV_SHM_POOL_GB:-8};
unset TPU_KV_RESHARD_TRANSFER_TAGS;
EOF
)

serve_args=(
  "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --seed 42
  --max-model-len "${MAX_MODEL_LEN}"
  --enable-expert-parallel
  --disable-custom-all-reduce
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --kv-cache-dtype fp8_e4m3
  --language-model-only
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --no-disable-hybrid-kv-cache-manager
  --attention-backend CUSTOM
  --enable-prompt-tokens-details
  --no-enable-log-requests
  --compilation-config "${compilation_config}"
  --no-enable-prefix-caching
  --no-async-scheduling
  --prefill-context-parallel-size 1
  --host "${BIND_HOST}"
)
if [[ -n "${LOAD_FORMAT}" ]]; then
  serve_args+=(--load-format "${LOAD_FORMAT}")
fi

build_leg_cmd() {
  # $1 role, $2 port, $3 tp, $4 dp, $5 namespace, $6 advertise host,
  # $7 reshard port base, $8 store dispatch port base, $9 kv transfer port,
  # $10 rank offset, $11 kv-transfer-config
  local role="$1" port="$2" tp="$3" dp="$4" ns="$5" advertise="$6"
  local reshard_base="$7" dispatch_base="$8" kv_port="$9" rank_offset="${10}"
  local kv_config="${11}"
  local args=("${serve_args[@]}" --port "${port}" --tensor-parallel-size "${tp}"
    --data-parallel-size "${dp}" --data-parallel-size-local "${dp}"
    --kv-transfer-config "${kv_config}")
  printf '%s\n%s\n' "${common_env}" \
    "export VLLM_HOST_IP='${advertise}'; export TPU_RAIDEN_ADVERTISE_HOST='${advertise}'; export TPU_KV_TRANSFER_NAMESPACE='${ns}'; export TPU_RAIDEN_JOB_NAME=${role}; export TPU_RAIDEN_ENGINE_ID=${role}-engine; export TPU_RAIDEN_RESHARD_PORT_BASE=${reshard_base}; export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE=${dispatch_base}; export TPU_KV_TRANSFER_PORT=${kv_port}; export DEBUG_TPU_LOCAL_RANK_OFFSET='${rank_offset}'; export TPU_LOCAL_RANK_OFFSET='${rank_offset}'; ${PYTHON} -m vllm.entrypoints.openai.api_server $(printf '%q ' "${args[@]}")"
}

prefill_cmd="$(build_leg_cmd prefill "${PREFILL_PORT}" "${PREFILL_TP}" "${PREFILL_DP}" \
  "${prefill_namespace}" "${PREFILL_HOST}" "${PREFILL_RESHARD_PORT_BASE}" \
  "${PREFILL_STORE_DISPATCH_PORT_BASE}" "${PREFILL_KV_TRANSFER_PORT}" \
  "${PREFILL_RANK_OFFSET}" "${p_kv}")"
decode_cmd="$(build_leg_cmd decode "${DECODE_PORT}" "${DECODE_TP}" "${DECODE_DP}" \
  "${decode_namespace}" "${DECODE_HOST}" "${DECODE_RESHARD_PORT_BASE}" \
  "${DECODE_STORE_DISPATCH_PORT_BASE}" "${DECODE_KV_TRANSFER_PORT}" \
  "${DECODE_RANK_OFFSET}" "${d_kv}")"

printf '%s\n' "${prefill_cmd}" >"${RUN_DIR}/prefill_cmd.sh"
printf '%s\n' "${decode_cmd}" >"${RUN_DIR}/decode_cmd.sh"
echo "run dir: ${RUN_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, wrote prefill_cmd.sh and decode_cmd.sh and stopped."
  exit 0
fi

start_leg() {
  local role="$1" cmd="$2"
  setsid bash -lc "${cmd}" >"${RUN_DIR}/logs/${role}.log" 2>&1 &
  echo $! >"${RUN_DIR}/${role}.pid"
  echo "${role} pid $(cat "${RUN_DIR}/${role}.pid")"
}

wait_healthy() {
  # $1 label, $2 url, $3 pidfile
  local label="$1" url="$2" pidfile="$3"
  local code=""
  for ((attempt = 1; attempt <= 200; attempt++)); do
    code=$(curl -fsS -o /dev/null -w "%{http_code}" "${url}" 2>/dev/null || true)
    echo "$(date +%H:%M:%S) ${label}=${code:-NA}" | tee -a "${RUN_DIR}/logs/health_poll.log"
    if [[ "${code}" == "200" ]]; then
      return 0
    fi
    if ! kill -0 "$(cat "${pidfile}" 2>/dev/null)" 2>/dev/null; then
      echo "${label} died before becoming healthy; check ${RUN_DIR}/logs" >&2
      grep -E "Traceback|RuntimeError|ValueError|LLVM ERROR|No space left" \
        "${RUN_DIR}/logs/"*.log >"${RUN_DIR}/logs/launch_err.log" 2>/dev/null || true
      # A truncated log with no traceback is almost always the root disk.
      df -h / | tail -1
      exit 1
    fi
    sleep 15
  done
  echo "${label} never became healthy" >&2
  exit 1
}

case "${ROLE}" in
  prefill)
    stop_port_listeners "${PREFILL_PORT}"
    start_leg prefill "${prefill_cmd}"
    wait_healthy P "http://127.0.0.1:${PREFILL_PORT}/health" "${RUN_DIR}/prefill.pid"
    echo "prefill healthy. Start the decode leg on ${DECODE_HOST}, then run the proxy."
    ;;
  decode)
    stop_port_listeners "${DECODE_PORT}"
    start_leg decode "${decode_cmd}"
    wait_healthy D "http://127.0.0.1:${DECODE_PORT}/health" "${RUN_DIR}/decode.pid"
    echo "decode healthy."
    ;;
  all)
    stop_port_listeners "${PREFILL_PORT}" "${DECODE_PORT}" "${PROXY_PORT}"
    start_leg prefill "${prefill_cmd}"
    start_leg decode "${decode_cmd}"
    wait_healthy P "http://127.0.0.1:${PREFILL_PORT}/health" "${RUN_DIR}/prefill.pid"
    wait_healthy D "http://127.0.0.1:${DECODE_PORT}/health" "${RUN_DIR}/decode.pid"
    ;;
esac

# The proxy belongs on whichever host you drive traffic from. In the two-host
# layout that is the prefill host by convention, matching the jobset.
if [[ "${ROLE}" == "all" || "${ROLE}" == "prefill" ]]; then
  proxy_cmd="${PYTHON} ${script_dir}/toy_proxy_server.py --host 0.0.0.0 --port ${PROXY_PORT} --prefiller-hosts ${PREFILL_HOST} --prefiller-ports ${PREFILL_PORT} --decoder-hosts ${DECODE_HOST} --decoder-ports ${DECODE_PORT}"
  printf '%s\n' "${proxy_cmd}" >"${RUN_DIR}/proxy_cmd.sh"
  if [[ "${ROLE}" == "all" ]]; then
    setsid bash -lc "${proxy_cmd}" >"${RUN_DIR}/logs/proxy.log" 2>&1 &
    echo $! >"${RUN_DIR}/proxy.pid"
    echo "proxy pid $(cat "${RUN_DIR}/proxy.pid") on port ${PROXY_PORT}"
  else
    echo "decode is remote, so start the proxy once it is healthy:"
    echo "  bash ${RUN_DIR}/proxy_cmd.sh"
  fi
fi
