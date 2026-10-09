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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

TARGET="${1:-dev}"
IMAGE_SUFFIX=""
if [ "${TARGET}" = "omni" ]; then
  IMAGE_SUFFIX="-omni"
elif [ "${TARGET}" != "dev" ]; then
  echo "[FATAL] Unsupported target '${TARGET}'; expected dev or omni." >&2
  exit 1
fi

IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"
# Resolve the target vLLM commit SHA from build metadata
VLLM_COMMIT_HASH="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")"
if [ -z "${VLLM_COMMIT_HASH}" ]; then
  echo "[FATAL] VLLM_COMMIT_HASH metadata is empty; bootstrap.sh did not set it." >&2
  exit 1
fi

# Include vLLM commit SHA in the image tag for registry isolation
IMAGE_TAG="${IMAGE_REPO}:${COMMIT_HASH}-${VLLM_COMMIT_HASH}${IMAGE_SUFFIX}"

# Layer cache: every build reuses the layers of the latest main build's image
# whose inputs are unchanged, so usually only the layers from `COPY . .` on are
# rebuilt, pushed and pulled. Only main post-merge builds move the tag (below),
# so a PR's layers never become every other build's cache.
CACHE_REF="${IMAGE_REPO}:buildcache${IMAGE_SUFFIX}"
# DOCKER_CACHE_FROM (build env) reads the layers from another image instead,
# e.g. an earlier build's, to try the cache on a branch without moving the tag.
CACHE_FROM="${DOCKER_CACHE_FROM:-${CACHE_REF}}"
CACHE_ARGS=(--cache-epoch "$(date -u +%G-W%V)")
if docker manifest inspect "${CACHE_FROM}" >/dev/null 2>&1; then
  CACHE_ARGS+=(--cache-from "${CACHE_FROM}")
else
  echo "No ${CACHE_FROM} yet; building without the layer cache."
fi

echo "--- Building Docker Image: ${IMAGE_TAG} (target: ${TARGET})"
# Run build_image.sh and pass VLLM_COMMIT_HASH directly as a build argument
./docker/build_image.sh --target "${TARGET}" -t "${IMAGE_TAG}" -c "${VLLM_COMMIT_HASH}" "${CACHE_ARGS[@]}"

# Publish image tag metadata for downstream test steps
buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX:-}" "${IMAGE_TAG}"

echo "--- Pushing Docker Image to Registry"
docker push "${IMAGE_TAG}"

# The integration pipeline also builds main, but against vLLM HEAD; its layers
# would make the next pinned build miss from the dependency install on.
if [[ "${BUILDKITE_PIPELINE_SLUG:-}" == "vllm-torchtpu-ci" \
      && "${BUILDKITE_BRANCH:-}" == "main" \
      && "${BUILDKITE_PULL_REQUEST:-false}" == "false" \
      && -z "${VLLM_COMMIT_HASH_OVERRIDE:-}" ]]; then
  echo "--- Moving ${CACHE_REF} to this image"
  docker tag "${IMAGE_TAG}" "${CACHE_REF}"
  docker push "${CACHE_REF}"
  docker rmi "${CACHE_REF}" >/dev/null 2>&1 || true
fi

echo "--- Cleaning up local built image"
docker rmi "${IMAGE_TAG}" || true

echo "--- Done setup_docker_env.sh"
