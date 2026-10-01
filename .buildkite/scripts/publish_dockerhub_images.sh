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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"

# Run by nightly_releases.yml, which logs in to Docker Hub first. As in
# publish_nightly_images.sh, the branch decides the tags. main pushes
# nightly-YYYYMMDD-<commit>-<vllm> and moves nightly. release/X.Y.Z pushes
# YYYYMMDD-X.Y.Z and moves X.Y.Z, so a release build never touches nightly.
BRANCH="${BUILDKITE_BRANCH:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
DATE="$(date +%Y%m%d)"
# The vLLM the nightly tests ran against, set by bootstrap.sh.
VLLM_COMMIT_HASH="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
case "${BRANCH}" in
  main)
    DATE_TAG="nightly-${DATE}-${BUILDKITE_COMMIT:0:7}-${VLLM_COMMIT_HASH:0:7}"
    LATEST_TAG="nightly"
    ;;
  release/*)
    VERSION="${BRANCH#release/}"
    # Starting with a digit keeps it from colliding with nightly or latest.
    if [[ ! "${VERSION}" =~ ^[0-9][A-Za-z0-9_.-]*$ ]]; then
      echo "ERROR: branch '${BRANCH}' does not name a version usable as an image tag; expected release/X.Y.Z." >&2
      exit 1
    fi
    DATE_TAG="${DATE}-${VERSION}"
    LATEST_TAG="${VERSION}"
    ;;
  *)
    echo "ERROR: Docker Hub images are published from main or release/X.Y.Z, not '${BRANCH}'." >&2
    exit 1
    ;;
esac

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

IMAGE_NAME="${DOCKERHUB_REPO:?DOCKERHUB_REPO must be set in nightly_releases.yml}"
DATE_IMAGE="${IMAGE_NAME}:${DATE_TAG}"
LATEST_IMAGE="${IMAGE_NAME}:${LATEST_TAG}"

cd "${REPO_ROOT}"

echo "--- :docker: Building prod image: ${DATE_IMAGE}"
./docker/build_image.sh --target prod -c "${VLLM_COMMIT_HASH}" -t "${DATE_IMAGE}"

echo "--- :docker: Tagging as ${LATEST_IMAGE}"
docker tag "${DATE_IMAGE}" "${LATEST_IMAGE}"

echo "--- :docker: Pushing ${DATE_IMAGE} and ${LATEST_IMAGE} to Docker Hub"
docker push "${DATE_IMAGE}"
docker push "${LATEST_IMAGE}"

echo "--- Cleaning up locally built image tags"
docker rmi "${DATE_IMAGE}" "${LATEST_IMAGE}" || true

echo "--- Successfully built and published ${DATE_IMAGE} and ${LATEST_IMAGE}"
