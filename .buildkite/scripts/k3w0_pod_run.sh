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

# Round-3 W0 on one v7x-32 pod leg: the profiled decode capture at the fixed
# config plus --async-scheduling.
#
# W0_LOAD_LIFETIME=1 adds a SECOND server lifetime that runs the same serve
# flags without the profiler and with the expert-load dump on. It is off by
# default because the question it answers -- where the 15 padded decode rows
# route -- turned out to be answerable on the free v7x-8 surrogate and, in the
# end, by reading the code (`w4an_mxfp4.py:265` hard-codes
# skip_padded_tokens=True, so the gmm never sees them).
#
# If it is ever turned on it must stay a separate lifetime, never a second
# phase of the profiled one: the accumulation lives inside the torch.compile'd
# forward, so arming it is a recompile, and a capture whose op count differs
# from the run it is compared against is not comparable to it.
#
# The capture is collected and uploaded before anything else can run, so a
# wedged second lifetime cannot cost the deliverable.
set -uo pipefail

DRIVER="${DRIVER:-run_multihost.sh}"
PROFILE_CONFIG="${PROFILE_CONFIG:-kimi-k3-tp32-ep-w0profile}"
LOAD_CONFIG="${LOAD_CONFIG:-kimi-k3-tp32-ep-w0load}"
TAG="${TAG:-k3w0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_one() {  # config, artifact tag
  local cfg="$1" tag="$2"
  bash "${SCRIPT_DIR}/${DRIVER}" bash ./scripts/vllm/benchmarking/run_eval_flow.sh \
      --config "${cfg}" --results-dir "/perf_eval_results/${cfg}"
  local rc=$?
  echo "--- [w0] ${cfg} driver exited ${rc}; collecting regardless"
  bash "${SCRIPT_DIR}/k3_profile_collect.sh" "perf_eval_results/${cfg}" "${tag}" || true
  return "$rc"
}

echo "--- [w0] LIFETIME 1: profiled capture, expert-load dump OFF"
# Unset, not empty-string: envs.py treats "" as off, but an unset name is also
# skipped by passthrough_env.sh, so nothing about the dump reaches the server.
unset TPU_MOE_EXPERT_LOAD_DIR
run_one "${PROFILE_CONFIG}" "${TAG}p"
profile_rc=$?
echo "--- [w0] lifetime 1 exited ${profile_rc}; W0-a bundle is uploaded"

if [ "${W0_LOAD_LIFETIME:-0}" != "1" ]; then
  echo "--- [w0] second lifetime not requested (W0_LOAD_LIFETIME != 1); done"
  exit "$profile_rc"
fi

echo "--- [w0] LIFETIME 2/2: expert-load dump ON, no profiler"
export TPU_MOE_EXPERT_LOAD_DIR="/perf_eval_results/${LOAD_CONFIG}/expert_load"
export TPU_MOE_EXPERT_LOAD_SLOTS="${TPU_MOE_EXPERT_LOAD_SLOTS:-128}"
export TPU_MOE_EXPERT_LOAD_BUCKET="${TPU_MOE_EXPERT_LOAD_BUCKET:-16}"
export TPU_MOE_EXPERT_LOAD_RANKS="${TPU_MOE_EXPERT_LOAD_RANKS:-0,1}"
run_one "${LOAD_CONFIG}" "${TAG}l"
load_rc=$?
echo "--- [w0] lifetime 2 exited ${load_rc}"

# The capture is the deliverable; a failed load lifetime must not mask a green
# one or fail the leg after the bundle is already safe.
exit "$profile_rc"
