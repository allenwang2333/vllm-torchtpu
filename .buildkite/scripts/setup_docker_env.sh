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

# Resolve optional torch-tpu wheel override from env or Buildkite metadata
TORCH_TPU_WHEEL_GCS_URI="${TORCH_TPU_WHEEL_GCS_URI:-${WHEEL_GCS_URL:-${WHEEL_GCS_PATH:-}}}"
if [ -z "${TORCH_TPU_WHEEL_GCS_URI}" ] && command -v buildkite-agent >/dev/null 2>&1; then
  TORCH_TPU_WHEEL_GCS_URI="$(buildkite-agent meta-data get "TORCH_TPU_WHEEL_GCS_URI" --default "" 2>/dev/null || true)"
fi

TORCH_TPU_OVERRIDE_SOURCE="${TORCH_TPU_OVERRIDE_SOURCE:-}"
if [ -z "${TORCH_TPU_OVERRIDE_SOURCE}" ] && command -v buildkite-agent >/dev/null 2>&1; then
  TORCH_TPU_OVERRIDE_SOURCE="$(buildkite-agent meta-data get "TORCH_TPU_OVERRIDE_SOURCE" --default "" 2>/dev/null || true)"
fi

OVERRIDE_TAG_SUFFIX=""
OVERRIDE_KEY="${TORCH_TPU_WHEEL_GCS_URI:-${TORCH_TPU_OVERRIDE_SOURCE:-}}"
if [ -n "${OVERRIDE_KEY}" ]; then
  OVERRIDE_HASH="$(printf '%s' "${OVERRIDE_KEY}" | sha256sum | cut -c1-8)"
  OVERRIDE_TAG_SUFFIX="-whl-${OVERRIDE_HASH}"
fi

# Include vLLM commit SHA (and optional wheel override hash) in the image tag for registry isolation
IMAGE_TAG="${IMAGE_REPO}:${COMMIT_HASH}-${VLLM_COMMIT_HASH}${OVERRIDE_TAG_SUFFIX}${IMAGE_SUFFIX}"

BASE_IMAGE_TO_USE="${BASE_IMAGE_OVERRIDE:-}"
if [ -n "${BASE_IMAGE_TO_USE}" ] && [ -n "${IMAGE_SUFFIX}" ] && [[ "${BASE_IMAGE_TO_USE}" != *"${IMAGE_SUFFIX}" ]]; then
  BASE_IMAGE_TO_USE="${BASE_IMAGE_TO_USE}${IMAGE_SUFFIX}"
fi

if [ -n "${OVERRIDE_KEY}" ] && [ -n "${BASE_IMAGE_TO_USE}" ]; then
  echo "--- Pulling pre-built base image: ${BASE_IMAGE_TO_USE}"
  docker pull -q "${BASE_IMAGE_TO_USE}"
else
  echo "--- Building Docker Image: ${IMAGE_TAG} (target: ${TARGET})"
  # Run build_image.sh and pass VLLM_COMMIT_HASH directly as a build argument
  ./docker/build_image.sh --target "${TARGET}" -t "${IMAGE_TAG}" -c "${VLLM_COMMIT_HASH}"
  BASE_IMAGE_TO_USE="${IMAGE_TAG}"
fi

if [ -n "${TORCH_TPU_WHEEL_GCS_URI}" ]; then
  echo "--- Overriding torch-tpu wheel in ${IMAGE_TAG} (base: ${BASE_IMAGE_TO_USE}) from ${TORCH_TPU_WHEEL_GCS_URI}"
  ./docker/build_override_image.sh -b "${BASE_IMAGE_TO_USE}" -t "${IMAGE_TAG}" -w "${TORCH_TPU_WHEEL_GCS_URI}"
elif [ -n "${TORCH_TPU_OVERRIDE_SOURCE}" ]; then
  echo "--- Overriding torch-tpu wheel in ${IMAGE_TAG} (base: ${BASE_IMAGE_TO_USE}) from ${TORCH_TPU_OVERRIDE_SOURCE}"
  ./docker/build_override_image.sh -b "${BASE_IMAGE_TO_USE}" -t "${IMAGE_TAG}" -o "${TORCH_TPU_OVERRIDE_SOURCE}"
fi

# Publish image tag metadata for downstream test steps
buildkite-agent meta-data set "CI_IMAGE_TAG${IMAGE_SUFFIX:-}" "${IMAGE_TAG}"

echo "--- Pushing Docker Image to Registry"
docker push "${IMAGE_TAG}"

echo "--- Cleaning up local built image"
docker rmi "${IMAGE_TAG}" || true

echo "--- Done setup_docker_env.sh"
