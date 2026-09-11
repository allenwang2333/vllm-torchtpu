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

# run.sh, for a benchmark that spans the hosts of a slice.
#
#   .buildkite/kubernetes/run_multihost.sh <command> [args...]
#
# The arguments are .buildkite/scripts/run_multihost.sh's, so the two pipelines
# name the same benchmark the same way, and like it this runs the command on a
# Ray cluster spanning the slice.
#
# Not run.sh with a different shape: the launcher's built-in Job is one pod
# holding one host's chips, and a slice is several pods that have to be placed
# on the hosts of one physical slice and be able to address each other. That is
# a JobSet, so this passes --manifest and the shape comes from the manifest's
# nodeSelector rather than from a flag.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"

# Passed by name rather than substituted into the manifest: the launcher's
# renderer would put the value straight into the YAML, where a quote or a brace
# in a benchmark flag becomes a parse error. As an environment variable it is
# opaque to YAML and is split into arguments by the pod's own shell.
MULTIHOST_ARGS="$*"
export MULTIHOST_ARGS

exec /opt/launcher/launch \
  --manifest "${HERE}/manifests/v7x/multihost.yaml" \
  "${env_args[@]}"
