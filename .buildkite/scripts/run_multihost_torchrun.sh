#!/bin/bash
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

# [branch-local] `torchrun` backend for run_multihost.sh: run one arbitrary
# per-host script (relative to the repo root inside the image) in a
# TPU-privileged container on every host of the slice, simultaneously, with
# the env a raw torch.distributed program needs to form the multi-host ICI
# mesh. Mirrors what ray_distributed_executor_v2.py / tpu_mp_multihost.py
# export for vLLM workers:
#   TORCH_TPU_SLICEBUILDER_ADDRESSES  ip:8070+local_rank per device, ordered
#                                     by (host, local_rank), head first
#   TORCH_TPU_TOPOLOGY                get_tpu_multihost_topology(devices)
#   TPU_NUM_HOSTS / NNODES / NODE_RANK / MASTER_ADDR / NPROC_PER_NODE
# The script itself launches `python -m torch.distributed.run` with
# --node-rank=$NODE_RANK; head is node 0.
#
# Usage:
#   run_multihost.sh --backend torchrun <script-in-repo> [args...]
# shellcheck disable=SC2154

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  exec bash "${SCRIPT_DIR}/run_multihost.sh" --backend torchrun "$@"
fi

run_torchrun_multihost() {
  if [ "$#" -lt 1 ]; then
    echo "ERROR: Usage: run_multihost.sh --backend torchrun <script-in-repo> [args...]"
    return 1
  fi
  local runner="$1"
  shift
  local runner_args
  runner_args=$(printf '%q ' "$@")
  local nproc="${NPROC_PER_NODE:-8}"

  local all_ips=("${HEAD_INTERNAL_IP}" "${WORKER_IPS_ARRAY[@]}")
  local base_port="${TORCH_TPU_BASE_PORT:-8070}"
  local sb=()
  local ip r
  for ip in "${all_ips[@]}"; do
    for r in $(seq 0 $((nproc - 1))); do
      sb+=("${ip}:$((base_port + r))")
    done
  done
  local sb_csv
  sb_csv=$(IFS=,; echo "${sb[*]}")
  local total_devices=$((nproc * NUM_HOSTS))
  # Same lookup the executors use; falls back to the v7x dual-device map
  # entry for 32 devices if the platform module can't be imported here.
  local topology
  local fallback
  case "${total_devices}" in
    16) fallback="2,2,2,2" ;;
    32) fallback="2,2,4,2" ;;
    64) fallback="2,4,4,2" ;;
    *) fallback="2,2,4,2" ;;
  esac
  topology="${TORCH_TPU_TOPOLOGY:-$(python3 -c "from vllm_torchtpu.platforms.tpu_platform import get_tpu_multihost_topology as g; print(g(${total_devices}, 'tpu7x'))" 2>/dev/null || echo "${fallback}")}"
  echo "torchrun backend: hosts=${NUM_HOSTS} nproc=${nproc} devices=${total_devices} topology=${topology}"
  echo "torchrun backend: slicebuilder=${sb_csv}"

  local mesh_env=(
    -e TORCH_TPU_SLICEBUILDER_ADDRESSES="${sb_csv}"
    -e TORCH_TPU_TOPOLOGY="${topology}"
    -e TPU_NUM_HOSTS="${NUM_HOSTS}"
    -e NNODES="${NUM_HOSTS}"
    -e MASTER_ADDR="${HEAD_INTERNAL_IP}"
    -e NPROC_PER_NODE="${nproc}"
  )
  local cmd="bash /root/torchtpu-vllm/${runner} ${runner_args}"

  local worker_idx=0
  for ip in "${WORKER_IPS_ARRAY[@]}"; do
    worker_idx=$((worker_idx + 1))
    echo "--- Starting torchrun node ${worker_idx} on ${ip}"
    ssh_retry "${SSH_USER}@${ip}" "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet >/dev/null 2>&1 || true; docker rm -f node >/dev/null 2>&1 || true; mkdir -p ~/multihost ~/hf_home"
    base64 < "${RUN_CLUSTER_MP}" > /tmp/run_cluster_mp.b64
    ssh_retry "${SSH_USER}@${ip}" "base64 -d > ~/multihost/run_cluster_mp.sh" < /tmp/run_cluster_mp.b64
    # shellcheck disable=SC2029
    (
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" \
        "docker rm -f node >/dev/null 2>&1 || true; bash ~/multihost/run_cluster_mp.sh '${IMAGE_TAG}' \"\$HOME/hf_home\" '${cmd}' ${CONTAINER_ENV_COMMON[*]} ${mesh_env[*]} -e NODE_RANK=${worker_idx}" \
        > "perf_eval_results/torchrun_node${worker_idx}.log" 2>&1
      echo "torchrun node ${worker_idx} exited $?"
    ) &
  done

  echo "--- Starting torchrun node 0 (head) on ${HEAD_INTERNAL_IP}"
  set +e
  bash "${RUN_CLUSTER_MP}" "${IMAGE_TAG}" "${HOST_HF_HOME}" "${cmd}" \
    "${CONTAINER_ENV_HEAD[@]}" "${mesh_env[@]}" -e NODE_RANK=0 2>&1 | tee "${MULTIHOST_RUN_LOG}"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e
  echo "--- head done (exit ${EXIT_CODE}); waiting for worker nodes"
  wait
  cp "${MULTIHOST_RUN_LOG}" perf_eval_results/torchrun_node0.log 2>/dev/null || true
  return "${EXIT_CODE}"
}
