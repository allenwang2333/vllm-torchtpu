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

# The branch decides the tags. main pushes nightly-YYYYMMDD and moves latest.
# release/X.Y.Z pushes YYYYMMDD-X.Y.Z and moves X.Y.Z, so a release nightly
# never touches latest. The Buildkite pipeline runs this script from whichever
# branch its schedule names, so a release branch cut from main after this
# change needs only a new schedule, not a code change.
#
# With PUBLISH_DOCKERHUB=1 (set by nightly_releases.yml and release_images.yml
# in vllm-torchtpu-ci, which log in to Docker Hub first) it builds only the
# DOCKERHUB_TARGET image the yml names (dev for nightly, prod for release) and
# pushes it to DOCKERHUB_REPO instead: main pushes
# nightly-YYYYMMDD-<commit>-<vllm> and moves nightly; release/X.Y.Z pushes
# X.Y.Z and moves latest.
BRANCH="${BUILDKITE_BRANCH:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
# TODO(test-only): for a CI dry run, act as main or as release/0.0.0-test
# depending on which step runs. Remove before merging.
if [[ "${BRANCH}" == "jefft/docker_image_ci" ]]; then
  [[ "${BUILDKITE_STEP_KEY:-}" == "publish_dockerhub_release" ]] && BRANCH="release/0.0.0-test" || BRANCH="main"
fi
DATE="$(date +%Y%m%d)"
PUBLISH_DOCKERHUB="${PUBLISH_DOCKERHUB:-0}"
VLLM_COMMIT_HASH=""
if [[ "${PUBLISH_DOCKERHUB}" == "1" ]]; then
  # The vLLM the CI tests ran against, set by bootstrap.sh.
  VLLM_COMMIT_HASH="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH")"
fi
case "${BRANCH}" in
  main)
    DATE_TAG="nightly-${DATE}"
    LATEST_TAG="latest"
    DOCKERHUB_TAG="nightly-${DATE}-${BUILDKITE_COMMIT:0:7}-${VLLM_COMMIT_HASH:0:7}"
    DOCKERHUB_LATEST_TAG="nightly"
    ;;
  release/*)
    VERSION="${BRANCH#release/}"
    # Starting with a digit keeps it from colliding with latest or nightly-*.
    if [[ ! "${VERSION}" =~ ^[0-9][A-Za-z0-9_.-]*$ ]]; then
      echo "ERROR: branch '${BRANCH}' does not name a version usable as an image tag; expected release/X.Y.Z." >&2
      exit 1
    fi
    DATE_TAG="${DATE}-${VERSION}"
    LATEST_TAG="${VERSION}"
    DOCKERHUB_TAG="${VERSION}"
    DOCKERHUB_LATEST_TAG="latest"
    ;;
  *)
    echo "ERROR: nightly images are published from main or release/X.Y.Z, not '${BRANCH}'." >&2
    exit 1
    ;;
esac

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

REGISTRY="${NIGHTLY_REGISTRY:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu}"
cd "${REPO_ROOT}"

if [[ "${PUBLISH_DOCKERHUB}" == "1" ]]; then
  DOCKERHUB_REPO="${DOCKERHUB_REPO:?DOCKERHUB_REPO must be set when PUBLISH_DOCKERHUB=1}"
  DATE_TAG="${DOCKERHUB_TAG}"
  LATEST_TAG="${DOCKERHUB_LATEST_TAG}"
  TARGETS=("${DOCKERHUB_TARGET:?DOCKERHUB_TARGET must be set when PUBLISH_DOCKERHUB=1}")
else
  # Build and push variants in order of dependency (ci -> dev, prod) so BuildKit layer cache is reused
  TARGETS=("ci" "dev" "prod")
fi

for TARGET in "${TARGETS[@]}"; do
  if [[ "${PUBLISH_DOCKERHUB}" == "1" ]]; then
    IMAGE_NAME="${DOCKERHUB_REPO}"
  else
    IMAGE_NAME="${REGISTRY}/vllm-torchtpu-${TARGET}"
  fi
  DATE_IMAGE="${IMAGE_NAME}:${DATE_TAG}"
  LATEST_IMAGE="${IMAGE_NAME}:${LATEST_TAG}"

  echo "--- :docker: Building Nightly Image (${TARGET}): ${DATE_IMAGE}"
  # An empty -c (GAR mode) makes build_image.sh use the pyproject.toml pin.
  ./docker/build_image.sh --target "${TARGET}" -c "${VLLM_COMMIT_HASH}" -t "${DATE_IMAGE}"

  echo "--- :docker: Tagging as ${LATEST_IMAGE}"
  docker tag "${DATE_IMAGE}" "${LATEST_IMAGE}"

  if [[ "${PUBLISH_DOCKERHUB}" == "1" ]]; then
    echo "--- :docker: Pushing ${DATE_IMAGE} and ${LATEST_IMAGE} to Docker Hub"
    # TODO(test-only): Docker Hub pushes disabled for a CI dry run. Restore
    # before merging.
    echo "[SIMULATED] docker push ${DATE_IMAGE}"
    echo "[SIMULATED] docker push ${LATEST_IMAGE}"

    echo "--- Cleaning up locally built image tags for ${TARGET}"
    docker rmi "${DATE_IMAGE}" "${LATEST_IMAGE}" || true
    continue
  fi

  # Step 1: Tag and push extra legacy torchtpu-vllm images for backward compatibility
  # so existing consumers continue working smoothly until migration is finalized.
  EXTRA_IMAGE_NAME="${REGISTRY}/torchtpu-vllm-${TARGET}"
  EXTRA_DATE_IMAGE="${EXTRA_IMAGE_NAME}:${DATE_TAG}"
  EXTRA_LATEST_IMAGE="${EXTRA_IMAGE_NAME}:${LATEST_TAG}"

  echo "--- :docker: Tagging extra legacy images for backward compatibility (${EXTRA_DATE_IMAGE}, ${EXTRA_LATEST_IMAGE})"
  docker tag "${DATE_IMAGE}" "${EXTRA_DATE_IMAGE}"
  docker tag "${DATE_IMAGE}" "${EXTRA_LATEST_IMAGE}"

  echo "--- :docker: Pushing ${DATE_IMAGE} and ${LATEST_IMAGE} to Artifact Registry"
  docker push "${DATE_IMAGE}"
  docker push "${LATEST_IMAGE}"

  echo "--- :docker: Pushing extra images ${EXTRA_DATE_IMAGE} and ${EXTRA_LATEST_IMAGE} to Artifact Registry"
  docker push "${EXTRA_DATE_IMAGE}"
  docker push "${EXTRA_LATEST_IMAGE}"

  echo "--- Cleaning up locally built image tags for ${TARGET}"
  docker rmi "${DATE_IMAGE}" "${LATEST_IMAGE}" "${EXTRA_DATE_IMAGE}" "${EXTRA_LATEST_IMAGE}" || true
done

echo "--- Successfully built and published all nightly images!"
