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

# Priority constants for pipeline jobs.
# Post-merge > Pre-merge > Integration pipeline > Benchmark > Other/Default > Nightly
export PRIORITY_POST_MERGE=11
export PRIORITY_PRE_MERGE=6
export PRIORITY_INTEGRATION=4
export PRIORITY_BENCHMARK=3
export PRIORITY_DEFAULT=2
export PRIORITY_NIGHTLY=1
export PRIORITY_KERNEL_TUNING=-9

# Implemented dynamic job prioritization by injecting integers during upload
upload_with_priority() {
  local yaml_file=$1
  local JOB_PRIORITY=$2
  echo "--- :pipeline: Uploading $yaml_file with priority ${JOB_PRIORITY:-$PRIORITY_DEFAULT}"
  {
    echo "priority: ${JOB_PRIORITY:-$PRIORITY_DEFAULT}";
    cat "$yaml_file";
  } | buildkite-agent pipeline upload
}

# ---------------------------------------------------------------------------
# vLLM last known good (LKG) resolution
# ---------------------------------------------------------------------------
# The vLLM commit this repo builds against is pinned in pyproject.toml, and
# that pin is the only source of truth -- CI, the prod image and a plain
# `pip install` all read it. The integration pipeline instead builds vLLM main
# HEAD and, if the whole suite passes, promotes that commit into the pin (see
# update_lkg_version.sh).
#
# The bootstrap step resolves the ref once and publishes it as the
# VLLM_COMMIT_HASH build metadata, so the build step and every downstream test
# step agree on one vLLM and therefore one image tag.

export VLLM_REPO_URL="https://github.com/vllm-project/vllm.git"
export CI_IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"

# The one pipeline that tests upstream HEAD rather than the pin, and so is the
# only one allowed to promote. Every other pipeline builds the pinned LKG.
export VLLM_HEAD_PIPELINE_SLUG="vllm-torchtpu-integration"

_PIPELINE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# The LKG ref currently pinned in pyproject.toml.
lkg_vllm_ref() {
  bash "${_PIPELINE_REPO_ROOT}/docker/vllm_pin.sh" read "${_PIPELINE_REPO_ROOT}/pyproject.toml"
}

# True in the pipeline that tracks upstream vLLM main.
tracks_vllm_head() {
  [[ "${BUILDKITE_PIPELINE_SLUG:-}" == "${VLLM_HEAD_PIPELINE_SLUG}" ]]
}

# Which upstream vLLM this build should test: main HEAD in the integration
# pipeline, the pyproject.toml pin everywhere else.
# VLLM_COMMIT_HASH_OVERRIDE forces a specific ref for one-off manual builds.
resolve_vllm_ref() {
  if [[ -n "${VLLM_COMMIT_HASH_OVERRIDE:-}" ]]; then
    printf '%s\n' "${VLLM_COMMIT_HASH_OVERRIDE}"
    return 0
  fi
  if tracks_vllm_head; then
    git ls-remote "${VLLM_REPO_URL}" HEAD | awk '{ print $1 }'
    return 0
  fi
  lkg_vllm_ref
}

# Resolve the vLLM ref for this build and publish it to build metadata.
# Called once, from the bootstrap step.
publish_vllm_ref() {
  local ref
  local track="lkg"
  tracks_vllm_head && track="upstream main HEAD"
  ref="$(resolve_vllm_ref)"
  if [[ -z "${ref}" ]]; then
    echo "[FATAL] Could not resolve a vLLM ref (tracking ${track})." >&2
    return 1
  fi
  buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${ref}"
  echo "--- vLLM ref for this build: ${ref} (tracking ${track})"
}

# The vLLM ref this build was pinned to, as published by the bootstrap step.
build_vllm_ref() {
  local ref="${VLLM_COMMIT_HASH:-}"
  if [[ -z "${ref}" ]] && command -v buildkite-agent >/dev/null 2>&1; then
    ref="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")"
  fi
  if [[ -z "${ref}" ]]; then
    if [[ -n "${BUILDKITE:-}" ]]; then
      # Inside a real build an empty ref means bootstrap never ran (or failed).
      # Falling back to the pin here would let an upstream-HEAD build silently
      # test -- and then promote -- the LKG it was meant to replace.
      echo "[FATAL] VLLM_COMMIT_HASH metadata is unset in a Buildkite build." >&2
      echo "[FATAL] The bootstrap step must call publish_vllm_ref first." >&2
      return 1
    fi
    ref="$(lkg_vllm_ref)"
  fi
  printf '%s\n' "${ref}"
}

# Docker tags allow [A-Za-z0-9_.-]; git refs may contain '/' and more.
sanitize_docker_tag() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '-'
}

# The CI image for this build, tagged with BOTH the vllm-torchtpu commit and
# the vLLM ref. Without the vLLM component the LKG and upstream-HEAD pipelines
# produce the same tag for the same repo commit and can overwrite each other in
# the registry, so a test could silently run the wrong vLLM.
ci_image_tag() {
  if [[ -n "${VLLM_TORCHTPU_IMAGE_TAG:-}" ]]; then
    printf '%s\n' "${VLLM_TORCHTPU_IMAGE_TAG}"
    return 0
  fi
  local ref
  ref="$(build_vllm_ref)" || return 1
  printf '%s:%s-%s\n' \
    "${CI_IMAGE_REPO}" "${BUILDKITE_COMMIT:-latest}" "$(sanitize_docker_tag "${ref}")"
}

# Fail fast if an image does not contain the vLLM it was built for. Guards
# against a mis-set build arg, a stale tag, or two pipelines racing on one tag.
#
# /etc/vllm-revision holds every identifier the installer recorded for the
# installed vLLM -- the requested ref, the resolved commit, the version -- one
# per line, because which of those a given installer keeps in direct_url.json
# is not guaranteed. A match on any line means the right vLLM is installed.
verify_image_vllm() {
  local image="$1" expected="$2" actual rc=0
  actual="$(docker run --rm --entrypoint cat "${image}" /etc/vllm-revision 2>&1)" || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "[FATAL] Could not read /etc/vllm-revision from ${image} (exit ${rc}):" >&2
    echo "${actual}" >&2
    return 1
  fi
  if ! grep -Fxq "${expected}" <<<"${actual}"; then
    echo "[FATAL] ${image} expects vLLM '${expected}', but it identifies as:" >&2
    echo "${actual}" >&2
    echo "[FATAL] Refusing to test or promote the wrong vLLM." >&2
    return 1
  fi
  echo "Verified ${image} contains vLLM ${expected}."
}
