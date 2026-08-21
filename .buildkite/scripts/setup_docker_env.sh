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

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

VLLM_REF="$(build_vllm_ref)"
IMAGE_TAG="$(ci_image_tag)"

echo "--- Building Docker Image: ${IMAGE_TAG} (vLLM ${VLLM_REF})"
# Run the existing build_image.sh script with target dev
./docker/build_image.sh --target dev -t "${IMAGE_TAG}" --vllm-ref "${VLLM_REF}"

# Prove the image holds the vLLM this build resolved before anything is pushed,
# tested against, or promoted off the back of it.
echo "--- Verifying the built image contains vLLM ${VLLM_REF}"
verify_image_vllm "${IMAGE_TAG}" "${VLLM_REF}"

echo "--- Pushing Docker Image to Registry"
docker push "${IMAGE_TAG}"

echo "--- Cleaning up local built image"
docker rmi "${IMAGE_TAG}" || true

echo "--- Done setup_docker_env.sh"
