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

# Bootstrap for the vllm-torchtpu-dev (dev/experimental) pipeline.
#
# Unlike scripts/bootstrap.sh (which uploads the four CI pipelines and gates on
# [skip ci] / docs-only diffs), this uploads a single, branch-owned pipeline
# file so you can run one-off experiments without touching CI.
#
# Configure the Buildkite pipeline's "Steps" to run:
#   bash .buildkite/scripts/bootstrap_dev.sh

set -euo pipefail

# Resolve the absolute directory path of the current script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared pipeline config file (priority constants +
# upload_with_priority).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

# Resolve and publish Version Metadata for developer pipeline
VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH_OVERRIDE:-$(get_vllm_commit_hash)}"
buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
echo "Using vLLM commit: ${VLLM_COMMIT_HASH}"

TORCH_TPU_WHEEL_GCS_URI="${TORCH_TPU_WHEEL_GCS_URI:-${WHEEL_GCS_URL:-${WHEEL_GCS_PATH:-}}}"
if [[ -n "${TORCH_TPU_WHEEL_GCS_URI}" ]]; then
  export TORCH_TPU_WHEEL_GCS_URI
  buildkite-agent meta-data set "TORCH_TPU_WHEEL_GCS_URI" "${TORCH_TPU_WHEEL_GCS_URI}"
  echo "Using torch-tpu wheel override: ${TORCH_TPU_WHEEL_GCS_URI}"
  buildkite-agent annotate --style info --context torch-tpu-wheel-override \
    "Overriding \`torch-tpu\` wheel from: \`${TORCH_TPU_WHEEL_GCS_URI}\`"
fi
if [[ -n "${TORCH_TPU_OVERRIDE_SOURCE:-}" ]]; then
  export TORCH_TPU_OVERRIDE_SOURCE
  buildkite-agent meta-data set "TORCH_TPU_OVERRIDE_SOURCE" "${TORCH_TPU_OVERRIDE_SOURCE}"
  echo "Using torch-tpu override source: ${TORCH_TPU_OVERRIDE_SOURCE}"
fi

# Allow running the full CI suite (or selected CI_LANES) from the dev pipeline
# when RUN_FULL_SUITE=1 or CI_LANES is specified without a custom DEV_PIPELINE_FILE.
if [[ "${RUN_FULL_SUITE:-0}" == "1" || ( -n "${CI_LANES:-}" && -z "${DEV_PIPELINE_FILE:-}" ) ]]; then
  echo "--- Delegating to full CI bootstrap (bootstrap.sh)"
  exec bash "${SCRIPT_DIR}/bootstrap.sh"
fi

# Dev builds are experiments — they must never outrank real CI on a shared
# queue. Default to PRIORITY_DEFAULT (1); override per build with
# DEV_JOB_PRIORITY if an experiment needs to jump ahead of nightlies only.
JOB_PRIORITY="${DEV_JOB_PRIORITY:-$PRIORITY_DEFAULT}"
export JOB_PRIORITY
buildkite-agent meta-data set "JOB_PRIORITY" "$JOB_PRIORITY"
# No WORKLOAD_PRIORITY either: on kube an unclassed workload is ordered at 0,
# ahead of the nightlies and behind every other kind of build.
echo "--- Dev build priority: ${JOB_PRIORITY}"

# Which pipeline to upload. Defaults to the demo pipeline_dev.yml; set
# DEV_PIPELINE_FILE in the build's env to run a different dev pipeline
# without editing this file.
DEV_PIPELINE_FILE="${DEV_PIPELINE_FILE:-.buildkite/pipeline_dev.yml}"

if [[ ! -f "${DEV_PIPELINE_FILE}" ]]; then
  echo "ERROR: dev pipeline file '${DEV_PIPELINE_FILE}' not found on branch" \
       "'${BUILDKITE_BRANCH:-unknown}' at commit ${BUILDKITE_COMMIT:-unknown}." \
       "Create it on your branch, or point DEV_PIPELINE_FILE at an existing file."
  exit 1
fi

echo "--- :pipeline: Uploading ${DEV_PIPELINE_FILE}"
upload_with_priority "${DEV_PIPELINE_FILE}" "$JOB_PRIORITY"

echo "--- Buildkite Dev Bootstrap Finished"
