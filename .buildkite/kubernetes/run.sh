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

# The Kubernetes counterpart of .buildkite/scripts/run_in_docker.sh.
#
#   .buildkite/kubernetes/run.sh <machine-type>/<topology> <command> [args...]
#
# run_in_docker.sh picks an image, mounts the caches and forwards a long -e list
# into a container on a long-lived VM. Here the pod is the container, the
# launcher owns the caches, and the shape names the hardware - so what is left
# is resolving the image, deciding what environment crosses into the pod, and
# getting results back out of a pod the agent cannot see.
#
# It lives in this repo rather than in the launcher because everything in it is
# this repo's: the image repository, /perf_eval_results, TPU_MULTIHOST_BACKEND.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <machine-type>/<topology> <command> [args...]" >&2
  exit 2
fi

# One token rather than two flags, because the pair is what names a shape and
# splitting them across the step invites a step that changes one and not the
# other. The launcher's own names, so there is no table here to fall behind the
# fleet's - a shape it does not run is refused before anything is submitted.
shape="$1"
shift
machine_type="${shape%%/*}"
topology="${shape#*/}"
if [[ "$machine_type" == "$shape" || -z "$topology" ]]; then
  echo "$0: expected <machine-type>/<topology>, got '$shape'" >&2
  exit 2
fi

# The image and the environment that crosses into the pod, shared with
# run_multihost.sh: leaves WORKLOAD_IMAGE exported and the launcher's --env
# flags in `env_args`.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# The prologue and epilogue run_in_docker.sh performs around the container,
# moved inside it. /perf_eval_results is a path the benchmark scripts hardcode;
# on a VM it is a bind mount of a directory on the data disk that the agent
# then copies into its checkout for `artifact_paths`. Here nothing outside the
# pod can read it, so the pod uploads its own results with the buildkite-agent
# CLI the image carries.
#
# Uploaded under the same relative path `artifact_paths` produces on bare
# metal, so a run from either lane is read the same way.
IN_POD='
  set -o pipefail
  umask 000
  rm -rf /perf_eval_results
  mkdir -p /perf_eval_results
  "$@"
  rc=$?
  mkdir -p perf_eval_results
  cp -r /perf_eval_results/. perf_eval_results/ 2>/dev/null || true
  # A symlink whose target stayed in the container is a broken link here, and
  # the agent fails the whole upload on one of them.
  find perf_eval_results -type l ! -exec test -e {} \; -delete 2>/dev/null || true
  if [ -n "$(ls -A perf_eval_results 2>/dev/null)" ]; then
    buildkite-agent artifact upload "perf_eval_results/**/*" || true
  fi
  exit $rc
'

# No --manifest: the launcher built-in Job is one pod on one host holding every
# chip on it, which is what every single-host step is. A slice across hosts is
# a JobSet this repo writes and passes with --manifest instead - see
# .buildkite/kubernetes/manifests/v7x/.
exec /opt/launcher/launch \
  --machine-type "$machine_type" \
  --topology "$topology" \
  "${env_args[@]}" \
  -- bash -c "$IN_POD" -- "$@"
