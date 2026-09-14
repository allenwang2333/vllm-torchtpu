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
#   multihost_entry.sh <command> [args...]
#
# Every host of the slice runs this, with the same arguments. Nothing outside
# the pods orchestrates them: each one reads JOB_COMPLETION_INDEX, which
# Kubernetes sets on a pod of an indexed Job, and decides for itself whether it
# is the Ray head or a worker.
#
# That is the difference from bare metal, where run_multihost.sh runs on one
# host and reaches every other one - it lists the slice through the Cloud TPU
# API, distributes keys, opens an ssh session per host and starts a container
# on each. JobSet does all of that, so what is left here is only what used to
# run inside those containers: start Ray, join it, and agree on when the run is
# over.
#
# Environment, from manifests/v7x/multihost.yaml:
#   HEAD_HOST      the DNS name of index 0, which every host addresses.
#   NUM_HOSTS      hosts in the slice; the head waits for this many Ray nodes.
#   JOB_COMPLETION_INDEX  set by Kubernetes. 0 is the head.
set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

: "${HEAD_HOST:?HEAD_HOST must name index 0 of the slice}"
: "${NUM_HOSTS:?NUM_HOSTS must be set}"
INDEX="${JOB_COMPLETION_INDEX:-0}"
REPO_DIR="${REPO_DIR:-/workspace/vllm-torchtpu}"
cd "$REPO_DIR" || { echo "$0: no such directory: $REPO_DIR" >&2; exit 2; }

# Where the benchmark scripts write, and the only place results can be found
# afterwards. On a VM this is a bind mount of the data disk; a pod is deleted
# when the run ends, so the head uploads it before exiting instead.
mkdir -p /perf_eval_results
umask 000

# Core dumps here are gigabytes each and would fill the node's ephemeral
# storage; the backtrace in the log is the useful artifact.
ulimit -c 0

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export TPU_MULTIHOST_BACKEND=ray
export JAX_PLATFORMS=""
# The chips are addressed through Ray, so a host has no reason to ask the
# metadata server about the slice it is in - and in a pod the answer would
# describe the node rather than the workload.
export TPU_SKIP_MDS_QUERY=1

RAY_PORT=6379

# Is something listening on the head's Ray port?
#
# bash's /dev/tcp rather than `ray status`, because a worker needs to answer
# this both before it has joined anything and after the head has gone. A
# refused connection is the signal, and it costs no Ray client.
head_listening() {
  (exec 3<>"/dev/tcp/${HEAD_HOST}/${RAY_PORT}") 2>/dev/null && exec 3>&-
}

# How many Ray nodes the head can see, or 0 if it could not ask.
#
# The count comes back through a file rather than on stdout because Ray's C++
# logging writes there the moment a driver attaches: a command substitution
# around this captures some thousands of log lines with the digit lost among
# them, and the numeric test on the result then fails as a malformed
# comparison - which reads as "not enough nodes" for as long as the caller is
# willing to wait.
alive_nodes() {
  local report="${TMPDIR:-/tmp}/ray_alive.$$"
  rm -f "$report"
  python3 -c "
import ray
ray.init(address='auto', logging_level='error')
with open('$report', 'w') as f:
    f.write(str(sum(1 for n in ray.nodes() if n['Alive'])))
" >/dev/null 2>&1
  local alive
  alive=$(cat "$report" 2>/dev/null)
  rm -f "$report"
  echo "${alive:-0}"
}

publish_results() {
  mkdir -p perf_eval_results
  cp -r /perf_eval_results/. perf_eval_results/ 2>/dev/null || true
  # A symlink whose target stayed in the container is a broken link here, and
  # the agent fails the whole upload on one of them.
  find perf_eval_results -type l ! -exec test -e {} \; -delete 2>/dev/null || true
  if [ -n "$(ls -A perf_eval_results 2>/dev/null)" ]; then
    buildkite-agent artifact upload "perf_eval_results/**/*" || true
  fi
}

run_worker() {
  # Wait for the head's GCS before joining: `ray start --address` fails
  # outright against a head that has not come up, and the head is doing a cold
  # model load's worth of work before it does.
  local deadline=$((SECONDS + 1800))
  until head_listening; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "ERROR: no Ray head at ${HEAD_HOST}:${RAY_PORT} after 30 minutes"
      return 1
    fi
    sleep 10
  done

  # Not --block, which is how bare metal runs it. A blocking worker ends only
  # when something kills it, and on a VM the ssh session dying is what does.
  # The pods of an indexed Job are independent: a blocking worker would hold
  # its chips until the JobSet's deadline and leave the step waiting on a
  # JobSet that can never complete. Watching the head's port instead ends the
  # worker when the head finishes - and when it dies, so a crashed head ends
  # the run rather than stranding the slice.
  ray start --address="${HEAD_HOST}:${RAY_PORT}" || return 1
  echo "worker ${INDEX} joined ${HEAD_HOST}; waiting for the head to finish"
  while head_listening; do
    sleep 15
  done
  echo "worker ${INDEX}: head is gone; exiting"
  ray stop --force >/dev/null 2>&1 || true
}

run_head() {
  ray start --head --port="${RAY_PORT}" || return 1

  echo "--- Waiting for ${NUM_HOSTS} Ray nodes to be alive"
  local deadline=$((SECONDS + 1800)) alive reported=-1
  while :; do
    alive=$(alive_nodes)
    if [ "$alive" -ge "$NUM_HOSTS" ]; then
      echo "Ray cluster complete: ${alive} nodes."
      break
    fi
    # Otherwise the wait is up to thirty silent minutes, which reads the same
    # as a hang. Only on a change, so a slow host does not bury the log.
    if [ "$alive" != "$reported" ]; then
      echo "  ${alive}/${NUM_HOSTS} nodes alive"
      reported="$alive"
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "ERROR: Ray cluster incomplete after 30 minutes (alive=${alive}/${NUM_HOSTS})"
      ray status || true
      return 1
    fi
    sleep 15
  done
  ray status || true

  echo "--- Running on the head: $*"
  "$@"
  local rc=$?

  echo "~~~ Collecting Ray session logs"
  tar czf /perf_eval_results/ray_logs_head.tgz -C /tmp/ray/session_latest/logs . 2>/dev/null || true
  return "$rc"
}

if [ "$INDEX" = "0" ]; then
  run_head "$@"
  rc=$?
  # Only the head has anything to publish - every worker's /perf_eval_results
  # is empty - and publishing before the workers notice the port has closed
  # costs nothing, because they are waiting on Ray rather than on the upload.
  publish_results
else
  run_worker
  rc=$?
fi
exit "$rc"
