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

# A/B experiment: does libtpu on a tpu7x-16 slice accept 8 processes per
# host, as torch_tpu's own multihost launcher (multihost_wrapper.py +
# torchrun) implies?
#
# This bypasses Ray and vLLM entirely: one plain container per host, the
# official launcher env (TPU_WORKER_HOSTNAMES + JOB_COMPLETION_INDEX ->
# multihost_wrapper.py exports), then torchrun --nproc-per-node 8 running a
# minimal tpu_dist all-reduce across all 16 ranks.
#
# Outcome A (all 16 ranks print ALLREDUCE OK): per-core processes are
#   supported; the Ray executor's remaining delta from torchrun is the bug,
#   and the printed PASSING-CONFIG env is the reference to converge on.
# Outcome B (same "Topology HOST_BOUNDS does not match VM metadata"): the
#   official launcher path is equally broken on tpu7x-16 -> torch_tpu/libtpu
#   must fix or bless a config; no CI-side change can.
set -euo pipefail

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
IMAGE_TAG="${VLLM_TORCHTPU_IMAGE_TAG:-${IMAGE_REPO}:${COMMIT_HASH}}"
CONTAINER=abtest

SSH_USER="${SSH_USER:-$(whoami)}"
if [ ! -f ~/.ssh/id_rsa ]; then
  mkdir -p ~/.ssh
  ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa -q
fi
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null -o IPQoS=none -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o ConnectTimeout=15 -i ~/.ssh/id_rsa)

# ssh to a freshly provisioned worker can drop sessions for the first
# minute or two (key propagation / sshd warmup) — retry with backoff.
ssh_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if ssh "${SSH_OPTS[@]}" "$@"; then
      return 0
    fi
    echo "ssh attempt ${attempt}/5 to ${1##*@} failed; retrying in $((attempt * 10))s"
    sleep $((attempt * 10))
  done
  return 1
}

# Discover the worker host IP (same mechanism as run_multihost.sh).
ZONE="${ZONE:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/zone" | awk -F/ '{print $NF}')}"
TPU_NAME="${TPU_NAME:-$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/description" 2>/dev/null || echo "")}"
ALL_IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" --format="value(networkEndpoints[].ipAddress)" | tr ';,' '  ')
# shellcheck disable=SC2206
ALL_IPS_ARRAY=($ALL_IPS)
WORKER_IP="${ALL_IPS_ARRAY[1]}"
echo "TPU_NAME=$TPU_NAME worker=$WORKER_IP"

RESULTS_DIR="perf_eval_results/abtest"
rm -rf perf_eval_results
mkdir -p "$RESULTS_DIR"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" "docker rm -f $CONTAINER >/dev/null 2>&1 || true" || true
}
trap cleanup EXIT INT TERM
cleanup

# Also make sure no leftover Ray containers hold the TPU devices.
docker rm -f node >/dev/null 2>&1 || true
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" "docker rm -f node >/dev/null 2>&1 || true" || true

# The all-reduce payload every rank runs.
PAYLOAD=$(cat <<'PYEOF'
import os

import torch
import torch_tpu  # noqa: F401
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="tpu_dist")
t = torch.ones(4, device="tpu") * (rank + 1)
dist.all_reduce(t)
expected = world * (world + 1) // 2
print(f"ABTEST RANK {rank}/{world} ALLREDUCE={t.cpu().tolist()} "
      f"expected={expected}", flush=True)
if rank in (0, 8):
    env = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("TORCH_TPU", "TPU_", "MASTER", "NNODES",
                         "NODE_RANK", "RANK", "LOCAL_", "WORLD_"))
    }
    print(f"ABTEST PASSING-CONFIG ENV (rank {rank}): {env}", flush=True)
dist.destroy_process_group()
PYEOF
)

# Per-node runner: official launcher env, then torchrun with 8 procs.
RUNNER=$(cat <<'SHEOF'
#!/bin/bash
set -euo pipefail
NODE_RANK=$1
PREFIX=$(hostname | sed 's/-w-[0-9]*$//')
export TPU_WORKER_HOSTNAMES=${PREFIX}-w-0,${PREFIX}-w-1
export JOB_COMPLETION_INDEX=$NODE_RANK
WRAPPER=$(python3 -c "import torch_tpu, os; print(os.path.join(os.path.dirname(torch_tpu.__file__), '_internal', 'distributed', 'launchers', 'multihost_wrapper.py'))")
eval "$(python3 "$WRAPPER")"
echo "=== node $NODE_RANK launcher env ==="
env | grep -E "^(TORCH_TPU|TPU_|NNODES|NODE_RANK|MASTER)" | sort
timeout 600 torchrun --nnodes 2 --node_rank "$NODE_RANK" --nproc-per-node 8 \
  --master_addr "${PREFIX}-w-0" --master_port 29500 /tmp/tpu_dist_test.py
SHEOF
)

start_container() {
  docker run -d --privileged --net=host --shm-size=8g --name "$CONTAINER" \
    --entrypoint /bin/bash -e JAX_PLATFORMS="" "$IMAGE_TAG" -c "sleep infinity"
  docker exec -i "$CONTAINER" bash -c "cat > /tmp/tpu_dist_test.py" <<<"$PAYLOAD"
  docker exec -i "$CONTAINER" bash -c "cat > /tmp/abtest_runner.sh" <<<"$RUNNER"
}

echo "--- Starting containers on both hosts"
start_container
# shellcheck disable=SC2029
ssh_retry "${SSH_USER}@${WORKER_IP}" \
  "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet >/dev/null 2>&1 || true; docker rm -f $CONTAINER >/dev/null 2>&1 || true; docker run -d --privileged --net=host --shm-size=8g --name $CONTAINER --entrypoint /bin/bash -e JAX_PLATFORMS='' '$IMAGE_TAG' -c 'sleep infinity' && echo REMOTE_CONTAINER_OK"
printf '%s' "$PAYLOAD" | base64 > /tmp/abtest_payload.b64
printf '%s' "$RUNNER" | base64 > /tmp/abtest_runner.b64
ssh_retry "${SSH_USER}@${WORKER_IP}" \
  "base64 -d | docker exec -i $CONTAINER bash -c 'cat > /tmp/tpu_dist_test.py' && echo REMOTE_PAYLOAD_OK" \
  < /tmp/abtest_payload.b64
ssh_retry "${SSH_USER}@${WORKER_IP}" \
  "base64 -d | docker exec -i $CONTAINER bash -c 'cat > /tmp/abtest_runner.sh' && echo REMOTE_RUNNER_OK" \
  < /tmp/abtest_runner.b64

echo "--- Launching torchrun on both nodes concurrently"
set +e
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${WORKER_IP}" \
  "docker exec $CONTAINER bash /tmp/abtest_runner.sh 1" \
  > "$RESULTS_DIR/node1.log" 2>&1 &
WORKER_PID=$!
docker exec "$CONTAINER" bash /tmp/abtest_runner.sh 0 \
  > "$RESULTS_DIR/node0.log" 2>&1
HEAD_RC=$?
wait "$WORKER_PID"
WORKER_RC=$?
set -e

echo "--- node0 (head) log ---"
cat "$RESULTS_DIR/node0.log"
echo "--- node1 (worker) log ---"
cat "$RESULTS_DIR/node1.log"

OK_COUNT=$(grep -h "ALLREDUCE=" "$RESULTS_DIR"/node*.log | wc -l | tr -d ' ')
echo "================================================================"
echo "torchrun exit codes: head=$HEAD_RC worker=$WORKER_RC; ranks OK: $OK_COUNT/16"
if [ "$HEAD_RC" = "0" ] && [ "$WORKER_RC" = "0" ] && [ "$OK_COUNT" = "16" ]; then
  echo "VERDICT: OUTCOME A — 8 procs/host WORKS via torchrun."
  echo "The Ray executor's remaining delta from this flow is the bug;"
  echo "see the ABTEST PASSING-CONFIG ENV lines above for the reference env."
else
  echo "VERDICT: OUTCOME B — official torchrun flow fails too:"
  grep -h "HOST_BOUNDS\|pjrt_state\|TPU initialization failed" \
    "$RESULTS_DIR"/node*.log | sort -u | head -5 || true
  echo "Per-core multihost processes need a torch_tpu/libtpu-side fix."
  exit 1
fi
