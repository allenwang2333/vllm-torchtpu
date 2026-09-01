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

# Run one profiling benchmark and collect its capture *whatever happens to it*.
# Buildkite skips the remaining commands of a step once one fails, and a
# capture is worth more than the run that produced it: a server that died after
# the decode window still leaves a usable trace.
#
# Usage: k3_profile_run.sh <driver script> <config name> [artifact tag]
set -uo pipefail

DRIVER="${1:?usage: k3_profile_run.sh <driver> <config> [tag]}"
CONFIG="${2:?usage: k3_profile_run.sh <driver> <config> [tag]}"
TAG="${3:-k3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/${DRIVER}" bash ./scripts/vllm/benchmarking/run_eval_flow.sh \
    --config "${CONFIG}" --results-dir "/perf_eval_results/${CONFIG}"
rc=$?
echo "--- benchmark driver exited ${rc}; collecting the capture regardless"
bash "${SCRIPT_DIR}/k3_profile_collect.sh" "perf_eval_results/${CONFIG}" "${TAG}" || true
exit "${rc}"
