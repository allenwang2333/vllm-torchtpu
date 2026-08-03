#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this script inside a four-node Slurm allocation}"

repo_dir=$(git rev-parse --show-toplevel)
mapfile -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ ${#hosts[@]} -ne 4 ]]; then
  echo "Expected four Slurm nodes; found ${#hosts[@]}: ${hosts[*]}" >&2
  exit 1
fi

head_host=${hosts[0]}
head_ip=$(srun --jobid="${SLURM_JOB_ID}" --overlap -N1 -n1 -w "${head_host}" \
  bash -lc "hostname -I | awk '{print \$1}'")

export PATH="${repo_dir}/.venv/bin:${PATH}"
export PYTHONPATH="${repo_dir}/src:${PYTHONPATH:-}"
export RAY_ADDRESS="${head_ip}:6379"
export TPU_MULTIHOST_BACKEND=ray
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,4}"
export TORCH_TPU_TOPOLOGY="${TORCH_TPU_TOPOLOGY:-2,2,4,2}"
export JAX_PLATFORMS=""
export VLLM_TARGET_DEVICE=tpu
export RAY_USAGE_STATS_ENABLED=0
export TPU_HBM_REPORT_ENABLE=0
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_diagnostic_coordinator=false --tpu_hbm_report_enable=false --enable_runtime_uptime_telemetry=false}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${repo_dir}/.runtime-cache/vllm}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${repo_dir}/.runtime-cache/vllm-config}"
export VLLM_XLA_CACHE_PATH="${VLLM_XLA_CACHE_PATH:-${repo_dir}/.runtime-cache/xla-k3-compiled-multishape}"
export VLLM_XLA_CHECK_RECOMPILATION=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export TF_CPP_MIN_LOG_LEVEL=1

mkdir -p "${VLLM_CACHE_ROOT}" "${VLLM_CONFIG_ROOT}" \
  "${VLLM_XLA_CACHE_PATH}" "${repo_dir}/logs"

export KIMI_REPRO_REPO_DIR="${repo_dir}"
export KIMI_REPRO_HEAD_IP="${head_ip}"
ray_log="${repo_dir}/logs/kimi-k3-ray-${SLURM_JOB_ID}.log"
# The single-quoted program is intentionally expanded by each Slurm task.
# shellcheck disable=SC2016
srun --jobid="${SLURM_JOB_ID}" --overlap -N4 -n4 --ntasks-per-node=1 \
  --unbuffered --export=ALL bash -lc '
    export PATH="${KIMI_REPRO_REPO_DIR}/.venv/bin:${PATH}"
    export PYTHONPATH="${KIMI_REPRO_REPO_DIR}/src:${PYTHONPATH:-}"
    node_ip=$(hostname -I | awk "{print \$1}")
    if [[ ${SLURM_PROCID} -eq 0 ]]; then
      exec ray start --block --head --node-ip-address="${node_ip}" \
        --port=6379 --include-dashboard=false --num-cpus=56
    fi
    until timeout 1 bash -c "</dev/tcp/${KIMI_REPRO_HEAD_IP}/6379" \
        2>/dev/null; do sleep 1; done
    exec ray start --block --address="${KIMI_REPRO_HEAD_IP}:6379" \
      --node-ip-address="${node_ip}" --num-cpus=56
  ' >"${ray_log}" 2>&1 &
ray_step_pid=$!
trap 'kill "${ray_step_pid}" 2>/dev/null || true' EXIT INT TERM

echo "Waiting for Ray at ${RAY_ADDRESS}; logs: ${ray_log}"
until timeout 1 bash -c "</dev/tcp/${head_ip}/6379" 2>/dev/null; do sleep 1; done
sleep 5

server_args=(
  serve moonshotai/Kimi-K3
  --revision=9f62e4e9fffbd0a83ddd60e1c209d828994b3569
  --tensor-parallel-size=32
  --enable-expert-parallel
  --max-model-len=4096
  --max-num-batched-tokens=512
  --max-num-seqs=4
  '--compilation-config={"compile_sizes":[16,64,256,512]}'
  --attention-backend=CUSTOM
  --trust-remote-code
  --language-model-only
  '--limit-mm-per-prompt={"image":0,"video":0}'
  --no-enable-prefix-caching
  --no-async-scheduling
  --kv-cache-memory-bytes=1073741824
  --num-gpu-blocks-override=260
  --chat-template="${repo_dir}/scripts/kimi_k3_text_chat_template.jinja"
  '--default-chat-template-kwargs={"thinking_effort":"low"}'
  --port=8000
)
if [[ ${KIMI_DUMMY_WEIGHTS:-0} == 1 ]]; then
  server_args+=(--load-format=dummy)
fi

echo "Starting Kimi K3 on ${head_host} (${head_ip}); API port 8000"
srun --jobid="${SLURM_JOB_ID}" --overlap -N1 -n1 -w "${head_host}" \
  --unbuffered --export=ALL --chdir="${repo_dir}" \
  "${repo_dir}/.venv/bin/vllm" "${server_args[@]}"
