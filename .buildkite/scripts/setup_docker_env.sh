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

UPREV_TORCH_TPU="${UPREV_TORCH_TPU:-0}"
IMAGE_REPO="us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu"
COMMIT_HASH="${BUILDKITE_COMMIT:-latest}"

if [ "${UPREV_TORCH_TPU}" = "1" ]; then
  IMAGE_TAG="${IMAGE_REPO}:${COMMIT_HASH}-uprev"
  BUILD_EXTRA_ARGS=(--uprev-torchtpu)
  echo "--- Building Uprev Docker Image (Latest torch-tpu): ${IMAGE_TAG}"
else
  IMAGE_TAG="${IMAGE_REPO}:${COMMIT_HASH}"
  BUILD_EXTRA_ARGS=()
  echo "--- Building Docker Image: ${IMAGE_TAG}"
fi

# Run the build_image.sh script with target dev
./docker/build_image.sh --target dev -t "${IMAGE_TAG}" "${BUILD_EXTRA_ARGS[@]}"

# If uprev mode, record installed torch-tpu version into Buildkite metadata and annotate
if [ "${UPREV_TORCH_TPU}" = "1" ]; then
  echo "--- Extracting installed torch-tpu version"
  INSTALLED_TORCH_TPU=$(docker run --rm "${IMAGE_TAG}" python3 -c "import importlib.metadata; print(importlib.metadata.version('torch-tpu'))" 2>/dev/null || true)
  if [ -n "${INSTALLED_TORCH_TPU}" ]; then
    echo "Installed torch-tpu version: ${INSTALLED_TORCH_TPU}"
    if command -v buildkite-agent >/dev/null 2>&1; then
      buildkite-agent meta-data set "TORCH_TPU_VERSION" "${INSTALLED_TORCH_TPU}" || true
      buildkite-agent annotate ":rocket: Uprev Target: Running against **torch-tpu==${INSTALLED_TORCH_TPU}**" --style "info" || true
    fi
  fi
fi

echo "--- Pushing Docker Image to Registry"
docker push "${IMAGE_TAG}"

echo "--- Cleaning up local built image"
docker rmi "${IMAGE_TAG}" || true

echo "--- Done setup_docker_env.sh"
