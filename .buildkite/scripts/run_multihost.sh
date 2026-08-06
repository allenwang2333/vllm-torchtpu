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

# Run a multi-host (Ray) vLLM test from Buildkite.
#
# This script runs ON the head node of a multi-host TPU slice (worker 0), which
# is the host the Buildkite agent lives on for the tpu_v7x_16_queue /
# tpu_v7x_32_queue queues. It:
#
#   1. discovers the slice's peer VM IPs from GCE metadata + gcloud,
#   2. starts a Ray head container locally and a Ray worker container on every
#      peer (both via scripts/multihost/run_cluster.sh, same image),
#   3. runs `vllm serve` inside the head container and waits for /health,
#   4. runs the client command (default: a curl completion) against it,
#   5. tears the whole cluster down on exit.
#
# Usage (from a pipeline step, after the image build step):
#   .buildkite/scripts/run_multihost.sh "<vllm serve ...>" ["<client cmd>"]
#
# The image is the one pushed by .buildkite/scripts/setup_docker_env.sh, tagged
# with $BUILDKITE_COMMIT, so the step must `depends_on` that build step.
#
# NOTE: deliberately no `set -x`. HF_TOKEN is passed to run_cluster.sh and over
# ssh; xtrace would echo it into the public build log.

set -euo pipefail

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
DOCKER_IMAGE="${DOCKER_IMAGE:-${IMAGE_REPO}:${COMMIT_HASH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"

CONTAINER_NAME="node"
SERVE_LOG="/root/vllm_serve.log"

# ---------------------------------------------------------------------------
# Secrets / environment
# ---------------------------------------------------------------------------
# Same source of truth as run_in_docker.sh: /etc/environment, backfilled from
# Secret Manager. Keep this section quiet — it handles a token.
if ! grep -q "^HF_TOKEN=" /etc/environment 2>/dev/null; then
  gcloud secrets versions access latest --secret=bm-agent-hf-token --quiet | \
  sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)" || true
fi
if [ -f /etc/environment ]; then
  # shellcheck disable=SC1091
  source /etc/environment || true
fi

# run_cluster.sh bind-mounts this at /root/.cache/huggingface in every
# container, so the whole slice shares one cache. Fall back to /tmp when the
# persistent disk is absent or not writable by this agent.
HOST_HF_HOME="${HOST_HF_HOME:-/mnt/disks/persist/models}"
if ! mkdir -p "${HOST_HF_HOME}" 2>/dev/null; then
  echo "[multihost] WARNING: cannot create ${HOST_HF_HOME}; falling back to /tmp/hf_home"
  HOST_HF_HOME="/tmp/hf_home"
  mkdir -p "${HOST_HF_HOME}"
fi

export SSH_USER="${SSH_USER:-$(whoami)}"

# ---------------------------------------------------------------------------
# Slice discovery
# ---------------------------------------------------------------------------
# The Buildkite agent runs on worker 0 of the slice. gcloud lists the slice's
# network endpoints in worker order, so element 0 is this host (the Ray head)
# and the rest are the peers that need a Ray worker.
if [[ -z "${WORKER_IPS:-}" ]]; then
  echo "--- [multihost] Discovering slice peers via gcloud"
  if ! command -v gcloud &> /dev/null; then
    echo "[multihost] ERROR: gcloud not found and WORKER_IPS not set." >&2
    exit 1
  fi

  ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')}"
  TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"

  if [[ -z "${TPU_NAME}" || -z "${ZONE}" ]]; then
    echo "[multihost] ERROR: could not read TPU_NAME/ZONE from GCE metadata" \
         "(TPU_NAME='${TPU_NAME}', ZONE='${ZONE}'). Set WORKER_IPS manually." >&2
    exit 1
  fi
  echo "[multihost] slice=${TPU_NAME} zone=${ZONE}"

  ALL_IPS=$(gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone "${ZONE}" \
    --format="value(networkEndpoints[].ipAddress)")
  ALL_IPS="${ALL_IPS//;/ }"
  ALL_IPS="${ALL_IPS//,/ }"
  # shellcheck disable=SC2206
  ALL_IPS_ARRAY=(${ALL_IPS})

  if [[ ${#ALL_IPS_ARRAY[@]} -lt 2 ]]; then
    echo "[multihost] ERROR: slice ${TPU_NAME} reports ${#ALL_IPS_ARRAY[@]} endpoint(s);" \
         "a multi-host run needs >= 2. Is this step on a multi-host queue?" >&2
    exit 1
  fi

  HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-${ALL_IPS_ARRAY[0]}}"
  WORKER_IPS_LIST=("${ALL_IPS_ARRAY[@]:1}")
  WORKER_IPS=$(IFS=, ; echo "${WORKER_IPS_LIST[*]}")
  echo "[multihost] head=${HEAD_INTERNAL_IP} workers=${WORKER_IPS}"
fi

HEAD_INTERNAL_IP="${HEAD_INTERNAL_IP:-$(hostname -I | awk '{print $1}')}"
IFS=',' read -r -a WORKER_IPS_ARRAY <<< "${WORKER_IPS}"

# Passwordless ssh to the peers. The agent image may not carry a key.
if [ ! -f ~/.ssh/id_rsa ]; then
  echo "--- [multihost] Generating ssh key for peer access"
  mkdir -p ~/.ssh
  ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes
          -o UserKnownHostsFile=/dev/null -o IPQoS=none -i ~/.ssh/id_rsa)

# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------
cleanup() {
  if [[ "${CLEANUP_DONE:-}" == "true" ]]; then return; fi
  CLEANUP_DONE="true"
  set +e

  echo "--- [multihost] Tearing down cluster"
  for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
    [[ -z "${worker_ip}" ]] && continue
    if [[ -f "/tmp/worker_${worker_ip}.log" ]]; then
      echo "===== Ray worker ${worker_ip} (last 50 lines) ====="
      tail -n 50 "/tmp/worker_${worker_ip}.log"
      rm -f "/tmp/worker_${worker_ip}.log"
    fi
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      "docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true; \
       sudo -n rm -rf /tmp/ray/* >/dev/null 2>&1 || true" || true
  done

  rm -f /tmp/vllm_serve.log
  docker cp "${CONTAINER_NAME}:${SERVE_LOG}" /tmp/vllm_serve.log >/dev/null 2>&1 || true
  if [[ -f /tmp/vllm_serve.log ]]; then
    echo "===== START OF VLLM SERVE LOG ====="
    cat /tmp/vllm_serve.log
    echo "===== END OF VLLM SERVE LOG ====="
  fi
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  sudo -n rm -rf /tmp/ray/* >/dev/null 2>&1 || true

  echo "[multihost] cleanup complete"
  set -e
}
trap cleanup EXIT SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Health wait
# ---------------------------------------------------------------------------
wait_for_server() {
  local port=$1
  local timeout=${2:-3600}

  echo "--- [multihost] Waiting up to ${timeout}s for vllm serve on port ${port}"
  local pid=""
  for ((i = 0; i < 30; i++)); do
    pid=$(docker exec "${CONTAINER_NAME}" pgrep -n -f "vllm serve" || true)
    [[ -n "${pid}" ]] && break
    sleep 2
  done
  if [[ -z "${pid}" ]]; then
    echo "[multihost] ERROR: 'vllm serve' process never appeared inside container" \
         "'${CONTAINER_NAME}'. Dumping ${SERVE_LOG}:" >&2
    docker exec "${CONTAINER_NAME}" cat "${SERVE_LOG}" || true
    return 1
  fi
  echo "[multihost] vllm serve pid=${pid}"

  local end_time=$((SECONDS + timeout))
  local loop_count=0
  while [[ ${SECONDS} -lt ${end_time} ]]; do
    if curl -fs --max-time 10 "localhost:${port}/health" > /dev/null; then
      echo "[multihost] server healthy on port ${port}"
      return 0
    fi
    if ! docker exec "${CONTAINER_NAME}" kill -0 "${pid}" 2>/dev/null; then
      echo "[multihost] ERROR: vllm serve (pid ${pid}) died inside the container" \
           "before becoming healthy. Dumping ${SERVE_LOG}:" >&2
      docker exec "${CONTAINER_NAME}" cat "${SERVE_LOG}" || true
      return 1
    fi
    # Fast-fail on a crashed engine that leaves the API server zombied.
    if [[ $((loop_count % 3)) -eq 0 ]]; then
      if docker exec "${CONTAINER_NAME}" grep -q -E \
          "EngineCore failed to start|ActorDiedError" "${SERVE_LOG}" 2>/dev/null; then
        echo "[multihost] ERROR: fatal vLLM engine crash detected in ${SERVE_LOG}" \
             "(EngineCore failed to start / ActorDiedError). Dumping log:" >&2
        docker exec "${CONTAINER_NAME}" cat "${SERVE_LOG}" || true
        return 1
      fi
    fi
    sleep 5
    loop_count=$((loop_count + 1))
  done

  echo "[multihost] ERROR: server did not become healthy on port ${port} within" \
       "${timeout}s. Dumping ${SERVE_LOG}:" >&2
  docker exec "${CONTAINER_NAME}" cat "${SERVE_LOG}" || true
  return 1
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
VLLM_SERVE_CMD="${1:-}"
if [[ -z "${VLLM_SERVE_CMD}" ]]; then
  echo "[multihost] ERROR: no serve command given." \
       "Usage: $0 \"<vllm serve ...>\" [\"<client cmd>\"]" >&2
  exit 1
fi
shift
CLIENT_CMD="${*:-}"

VLLM_PORT="8000"
if [[ "${VLLM_SERVE_CMD}" =~ --port[=\ ]+([0-9]+) ]]; then
  VLLM_PORT="${BASH_REMATCH[1]}"
fi
echo "[multihost] port=${VLLM_PORT}"

# ---------------------------------------------------------------------------
# Bring up the cluster
# ---------------------------------------------------------------------------
echo "--- [multihost] Clearing any leftover cluster state"
cleanup
CLEANUP_DONE="false"

echo "--- [multihost] Pruning docker on the head node"
docker system prune -a --volumes -f || true

CLUSTER_ENV_ARGS=(
  -e TPU_MULTIHOST_BACKEND=ray
  -e HF_HOME=/root/.cache/huggingface
)
eval "declare -a EXTRA_DOCKER_ARGS_ARRAY=(${EXTRA_DOCKER_ARGS:-})"

echo "--- [multihost] Starting Ray head locally"
bash "${TOP_DIR}/scripts/multihost/run_cluster.sh" \
  "${DOCKER_IMAGE}" \
  "${HEAD_INTERNAL_IP}" \
  --head \
  "${HOST_HF_HOME}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${CLUSTER_ENV_ARGS[@]}" \
  ${EXTRA_DOCKER_ARGS_ARRAY[@]:+"${EXTRA_DOCKER_ARGS_ARRAY[@]}"} \
  > /tmp/ray_head.log 2>&1 &

# Serialize the non-secret env flags once for the ssh heredocs below.
CLUSTER_ENV_STR=$(printf '%q ' "${CLUSTER_ENV_ARGS[@]}")

sleep 60

for worker_ip in "${WORKER_IPS_ARRAY[@]}"; do
  echo "--- [multihost] Starting Ray worker on ${worker_ip}"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "docker system prune -a --volumes -f" || true
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "mkdir -p ~/vllm-torchtpu/scripts/multihost && mkdir -p ${HOST_HF_HOME}" || true
  base64 < "${TOP_DIR}/scripts/multihost/run_cluster.sh" | \
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      "base64 -d > ~/vllm-torchtpu/scripts/multihost/run_cluster.sh"

  # xtrace is off for this whole script, so the heredoc body (which carries
  # HF_TOKEN) is never echoed. Worker output goes to a file so the two node
  # types don't interleave in the Buildkite log; it is dumped on teardown.
  # shellcheck disable=SC2029,SC2087
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" > "/tmp/worker_${worker_ip}.log" 2>&1 << EOF &
bash ~/vllm-torchtpu/scripts/multihost/run_cluster.sh '${DOCKER_IMAGE}' '${HEAD_INTERNAL_IP}' \
  --worker '${HOST_HF_HOME}' \
  -e HF_TOKEN='${HF_TOKEN:-}' \
  ${CLUSTER_ENV_STR}
EOF
done

echo "--- [multihost] Waiting for workers to join the Ray cluster"
sleep 120
docker exec "${CONTAINER_NAME}" ray status || true

# ---------------------------------------------------------------------------
# Serve + client
# ---------------------------------------------------------------------------
echo "--- [multihost] Starting vllm serve on the head node"
echo "[multihost] serve cmd: ${VLLM_SERVE_CMD}"
docker exec -d \
  -e HF_HOME=/root/.cache/huggingface \
  "${CONTAINER_NAME}" bash -c "${VLLM_SERVE_CMD} > ${SERVE_LOG} 2>&1"

wait_for_server "${VLLM_PORT}" "${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

if [[ -n "${CLIENT_CMD}" ]]; then
  echo "--- [multihost] Running client command"
  docker exec \
    -e HF_HOME=/root/.cache/huggingface \
    "${CONTAINER_NAME}" bash -c "cd /root/torchtpu-vllm && ${CLIENT_CMD}"
else
  echo "--- [multihost] Running default completion request"
  docker exec "${CONTAINER_NAME}" bash -c \
    "curl -sS --fail-with-body http://localhost:${VLLM_PORT}/v1/models"
fi

echo "--- [multihost] Test completed successfully"
