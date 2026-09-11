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

# What .buildkite/scripts/run_multihost.sh does, for a pod instead of a VM.
#
#   multihost_entry.sh ray <command> [args...]
#   multihost_entry.sh mp  <config-name> [extra run_eval_flow.sh args...]
#
# run_multihost.sh discovers the slice's hosts from the Cloud TPU API, opens an
# ssh session to each one and starts a container on it. None of that exists
# here: every host of the slice already runs this script, as one pod of an
# indexed Job, and JobSet has given each of them a DNS name. So the whole of
# the discovery, key generation, image pulling and container lifecycle is gone,
# and what is left is the part that was inside the containers - which role each
# host plays, and how they find each other.
#
# Environment, from manifests/v7x/multihost.yaml:
#   HEAD_HOST      the DNS name of index 0, which every host addresses.
#   NUM_HOSTS      hosts in the slice, for the mp rank arithmetic.
#   JOB_COMPLETION_INDEX  set by Kubernetes on an indexed Job. 0 is the head.
set -uo pipefail

MODE="${1:?usage: multihost_entry.sh ray|mp ...}"
shift

: "${HEAD_HOST:?HEAD_HOST must name index 0 of the slice}"
: "${NUM_HOSTS:?NUM_HOSTS must be set}"
INDEX="${JOB_COMPLETION_INDEX:-0}"
REPO_DIR="${REPO_DIR:-/workspace/vllm-torchtpu}"
cd "$REPO_DIR"

# Where the benchmark scripts write, and the only place results can be found
# afterwards. On a VM this is a bind mount of the data disk; a pod is deleted
# when the run ends, so the head uploads it before exiting instead.
mkdir -p /perf_eval_results
umask 000

# Core dumps here are gigabytes each and would fill the node's ephemeral
# storage; the backtrace in the log is the useful artifact.
ulimit -c 0

# Is something listening on the head's port?
#
# bash's /dev/tcp rather than curl or a client of whatever is on the port,
# because both roles need this and the two backends put different servers
# there. A refused connection is the signal, and it does not depend on the
# server answering anything.
head_listening() {
  (exec 3<>"/dev/tcp/${HEAD_HOST}/$1") 2>/dev/null && exec 3>&-
}

# Hold a worker for as long as the head is up.
#
# The pods of an indexed Job are independent: a worker that blocks forever
# outlives the head by the whole ten-hour deadline, holding four chips and
# leaving the Buildkite step hanging on a JobSet that will never complete. So
# every worker watches the head's port and exits when it closes - which happens
# when the head finishes, and also when it dies, so a crashed head ends the run
# instead of stranding the slice.
wait_for_head() {
  local port="$1" ready_deadline=$((SECONDS + $2))
  while ! head_listening "$port"; do
    if [ "$SECONDS" -ge "$ready_deadline" ]; then
      echo "worker ${INDEX}: head never opened port ${port}; exiting"
      return 0
    fi
    sleep 10
  done
  echo "worker ${INDEX}: head is up on port ${port}; waiting for it to finish"
  while head_listening "$port"; do
    sleep 15
  done
  echo "worker ${INDEX}: head is gone; exiting"
}

publish_results() {
  mkdir -p perf_eval_results
  cp -r /perf_eval_results/. perf_eval_results/ 2>/dev/null || true
  find perf_eval_results -type l ! -exec test -e {} \; -delete 2>/dev/null || true
  if [ -n "$(ls -A perf_eval_results 2>/dev/null)" ]; then
    buildkite-agent artifact upload "perf_eval_results/**/*" || true
  fi
}

# ---------------------------------------------------------------- ray backend
# One Ray cluster over the slice: index 0 is the head and runs the command,
# every other index joins it and waits.
run_ray() {
  export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
  export TPU_MULTIHOST_BACKEND=ray
  export JAX_PLATFORMS=""

  if [ "$INDEX" != "0" ]; then
    # Wait for the head's GCS before joining: `ray start --address` fails
    # outright against a head that has not come up yet, and the head is doing a
    # cold model load's worth of work before it does.
    if ! head_listening 6379; then
      echo "worker ${INDEX}: waiting for the Ray head at ${HEAD_HOST}:6379"
      until head_listening 6379; do sleep 10; done
    fi
    # Not --block. A blocking worker ends only when something kills it, and
    # nothing here will.
    ray start --address="${HEAD_HOST}:6379"
    wait_for_head 6379 1800
    ray stop --force >/dev/null 2>&1 || true
    return 0
  fi

  ray start --head --port=6379
  echo "--- Waiting for ${NUM_HOSTS} Ray nodes to be alive"
  deadline=$((SECONDS + 1800))
  while :; do
    alive=$(python3 -c "import ray; ray.init(address='auto', logging_level='error'); print(sum(1 for n in ray.nodes() if n['Alive']))" 2>/dev/null || echo 0)
    if [ "${alive:-0}" -ge "$NUM_HOSTS" ]; then
      echo "Ray cluster complete: ${alive} nodes."
      break
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "ERROR: Ray cluster incomplete after 30 minutes (alive=${alive:-0}/${NUM_HOSTS})"
      ray status || true
      return 1
    fi
    sleep 15
  done
  ray status || true

  echo "--- Running command on the head: $*"
  "$@"
  rc=$?

  echo "~~~ Collecting Ray session logs"
  tar czf /perf_eval_results/ray_logs_head.tgz -C /tmp/ray/session_latest/logs . 2>/dev/null || true
  return "$rc"
}

# ----------------------------------------------------------------- mp backend
# No Ray. Every host runs its own `vllm serve` against the same data-parallel
# address, which is vllm-torchtpu's own multihost DP bootstrap
# (src/vllm_torchtpu/distributed/tpu_mp_multihost.py). Index 0 serves the API
# and runs the eval flow against it; the others are --headless engines.
#
# The serve flags are built from the benchmark config exactly as
# run_multihost_mp.sh builds them, because they are the config's, not the
# transport's.
run_mp() {
  local config_name="${1:?usage: multihost_entry.sh mp <config-name> [args...]}"
  shift

  local config_file="${REPO_DIR}/scripts/vllm/benchmarking/configs/${config_name}.sh"
  if [ ! -f "$config_file" ]; then
    echo "ERROR: config file not found: $config_file"
    return 1
  fi

  MODEL_URI="${MODEL_URI:-}"
  DATA_PARALLELISM_LOCAL="${DATA_PARALLELISM_LOCAL:-}"
  TENSOR_PARALLELISM=1
  DATA_PARALLELISM=1
  ENABLE_EP=false
  QUANTIZATION=""
  GPU_MEMORY_UTILIZATION="0.95"
  KV_CACHE_DTYPE="fp8"
  MAX_MODEL_LEN=16384
  MAX_NUM_BATCHED_TOKENS=8192
  MAX_NUM_SEQS=512
  ATTENTION_BACKEND="CUSTOM"
  ENABLE_PREFIX_CACHING=false
  EXTRA_SERVE_ARGS=""
  SERVER_READY_WAIT_MIN=180
  # shellcheck source=/dev/null
  source "$config_file"

  if [ -z "${MODEL:-}" ]; then
    echo "ERROR: config must set MODEL"
    return 1
  fi
  if [ -z "$DATA_PARALLELISM_LOCAL" ]; then
    echo "ERROR: config must set DATA_PARALLELISM_LOCAL (this host's DP share)"
    return 1
  fi
  if [ "$((DATA_PARALLELISM_LOCAL * NUM_HOSTS))" -ne "$DATA_PARALLELISM" ]; then
    echo "ERROR: DATA_PARALLELISM_LOCAL ($DATA_PARALLELISM_LOCAL) x NUM_HOSTS ($NUM_HOSTS) != DATA_PARALLELISM ($DATA_PARALLELISM)"
    return 1
  fi

  export TPU_MULTIHOST_BACKEND=mp
  local port="${PORT:-8000}"
  local dp_rpc_port="${DP_RPC_PORT:-29500}"

  local extra_args=""
  [ "$ENABLE_EP" = "true" ] && extra_args="--enable-expert-parallel"
  [ -n "$QUANTIZATION" ] && extra_args="$extra_args --quantization $QUANTIZATION"
  extra_args="$extra_args --attention-backend $ATTENTION_BACKEND"
  [ -n "$EXTRA_SERVE_ARGS" ] && extra_args="$extra_args $EXTRA_SERVE_ARGS"
  local prefix_caching_flag="--no-enable-prefix-caching"
  [ "$ENABLE_PREFIX_CACHING" = "true" ] && prefix_caching_flag="--enable-prefix-caching"
  local serve_target="$MODEL"
  if [ -n "$MODEL_URI" ]; then
    serve_target="$MODEL_URI"
    extra_args="$extra_args --load-format runai_streamer --served-model-name $MODEL"
  fi

  # shellcheck disable=SC2206
  local -a serve_args=(
    "${serve_target}"
    --tensor-parallel-size="${TENSOR_PARALLELISM}"
    --data-parallel-size="${DATA_PARALLELISM}"
    --data-parallel-size-local="${DATA_PARALLELISM_LOCAL}"
    --data-parallel-address="${HEAD_HOST}"
    --data-parallel-rpc-port="${dp_rpc_port}"
    --max-model-len="${MAX_MODEL_LEN}"
    --max-num-batched-tokens="${MAX_NUM_BATCHED_TOKENS}"
    --max-num-seqs="${MAX_NUM_SEQS}"
    --async-scheduling
    "${prefix_caching_flag}"
    --gpu-memory-utilization="${GPU_MEMORY_UTILIZATION}"
    --kv-cache-dtype="${KV_CACHE_DTYPE}"
    ${extra_args}
  )

  if [ "$INDEX" != "0" ]; then
    local start_rank=$((INDEX * DATA_PARALLELISM_LOCAL))
    echo "--- mp worker ${INDEX} (data-parallel-start-rank=${start_rank})"
    # Backgrounded rather than exec'd: a headless engine serves until it is
    # stopped, so the worker has to decide for itself when the run is over.
    # The head opening its API port means the whole DP group came up; the port
    # closing again means the head is done or dead.
    vllm serve "${serve_args[@]}" --headless --data-parallel-start-rank="${start_rank}" &
    local serve_pid=$!
    wait_for_head "$port" $((SERVER_READY_WAIT_MIN * 60))
    kill "$serve_pid" 2>/dev/null || true
    wait "$serve_pid" 2>/dev/null || true
    return 0
  fi

  echo "--- mp head"
  vllm serve "${serve_args[@]}" --data-parallel-start-rank=0 --host 0.0.0.0 --port "${port}" &
  local serve_pid=$!
  trap 'kill "$serve_pid" 2>/dev/null || true' EXIT

  echo "--- Waiting for head server /health"
  local deadline=$((SECONDS + SERVER_READY_WAIT_MIN * 60))
  local head_up=0
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -s -o /dev/null --connect-timeout 1 "http://localhost:${port}/health"; then
      head_up=1
      break
    fi
    # A server whose engine died leaves nothing listening, and waiting the full
    # three hours for that tells nobody anything the exit already said.
    if ! kill -0 "$serve_pid" 2>/dev/null; then
      echo "ERROR: the head server exited before becoming healthy"
      return 1
    fi
    sleep 10
  done
  if [ "$head_up" != "1" ]; then
    echo "ERROR: head server did not become healthy within ${SERVER_READY_WAIT_MIN} minutes"
    return 1
  fi
  echo "Head server is healthy."

  bash ./scripts/vllm/benchmarking/run_eval_flow.sh \
    --config "$config_name" \
    --host localhost --port "$port" \
    --results-dir "/perf_eval_results/${config_name}" "$@"
}

case "$MODE" in
  ray) run_ray "$@" ;;
  mp)  run_mp "$@" ;;
  *)
    echo "ERROR: unknown mode '${MODE}'; expected ray or mp" >&2
    exit 2
    ;;
esac
rc=$?

# Only the head has anything to publish, and only the head holds the Buildkite
# job's credential in a way worth spending an upload on: every worker's
# /perf_eval_results is empty.
if [ "$INDEX" = "0" ]; then
  publish_results
fi
exit "$rc"
