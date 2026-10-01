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

# Builds the prod image and publishes it to Docker Hub.
#
# Called by nightly_releases.yml, which limits it to the NIGHTLY=1 builds of
# main and release/X.Y.Z in vllm-torchtpu-ci and logs in to Docker Hub before
# this script runs.
#
# As in publish_nightly_images.sh, the branch decides the tags, so a release
# branch cut from main after this change needs only a NIGHTLY=1 schedule on
# vllm-torchtpu-ci, not a code change:
#
#   main            ${DOCKERHUB_REPO}:nightly                                moves
#                   ${DOCKERHUB_REPO}:nightly-YYYYMMDD-<torchtpu7>-<vllm7>   never reused
#   release/X.Y.Z   ${DOCKERHUB_REPO}:X.Y.Z                                  moves
#                   ${DOCKERHUB_REPO}:YYYYMMDD-X.Y.Z                         never reused
#
# Release branches use the same tags publish_nightly_images.sh pushes to
# Artifact Registry: cutting release/X.Y.Z is the release, so X.Y.Z follows the
# branch head. A release build never touches nightly, and nothing here moves
# latest.
#
# <vllm7> is the vLLM commit read back from the built image, so the tag names
# the code inside it even when pyproject.toml pins a tag such as v0.29.0.
#
# Environment:
#   DOCKERHUB_REPO    target repository, e.g. <org>/vllm-torchtpu (required)
#   DRY_RUN           1 builds, checks and tags the image but does not push
#   VLLM_COMMIT_HASH  vLLM ref to build when run outside Buildkite; inside
#                     Buildkite it comes from the meta-data set by bootstrap.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"

DRY_RUN="${DRY_RUN:-0}"
DOCKERHUB_REPO="${DOCKERHUB_REPO:-}"

# Fail before the build, not after it, when the repository is not configured.
if [ -z "${DOCKERHUB_REPO}" ] || [[ "${DOCKERHUB_REPO}" == *REPLACE_WITH* ]]; then
  echo "[FATAL] DOCKERHUB_REPO is not set (got '${DOCKERHUB_REPO}'); set it in nightly_releases.yml." >&2
  exit 1
fi

BRANCH="${BUILDKITE_BRANCH:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)}"
VERSION=""
case "${BRANCH}" in
  main)
    ;;
  release/*)
    VERSION="${BRANCH#release/}"
    # Starting with a digit keeps it from colliding with nightly or latest.
    if [[ ! "${VERSION}" =~ ^[0-9][A-Za-z0-9_.-]*$ ]]; then
      echo "ERROR: branch '${BRANCH}' does not name a version usable as an image tag; expected release/X.Y.Z." >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: Docker Hub images are published from main or release/X.Y.Z, not '${BRANCH}'." >&2
    exit 1
    ;;
esac

# Build the same vLLM the nightly tests ran against.
if [ -n "${BUILDKITE:-}" ]; then
  VLLM_COMMIT_HASH="$(buildkite-agent meta-data get "VLLM_COMMIT_HASH" --default "")"
fi
VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH:-}"
if [ -z "${VLLM_COMMIT_HASH}" ]; then
  echo "[FATAL] VLLM_COMMIT_HASH is empty; bootstrap.sh did not set it." >&2
  exit 1
fi

TORCHTPU_COMMIT="${BUILDKITE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
DATE="$(date -u +%Y%m%d)"

# Built under a local name; the public tags are added only after the checks
# pass. The vllm-torchtpu repository name is one cleanup_docker.sh removes, so
# anything left behind by a killed job is cleared by the next build.
LOCAL_IMAGE="vllm-torchtpu:dockerhub-nightly-${TORCHTPU_COMMIT:0:12}"
PUBLIC_TAGS=()

cleanup() {
  docker rmi "${LOCAL_IMAGE}" ${PUBLIC_TAGS[@]+"${PUBLIC_TAGS[@]}"} >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "--- Cleaning up old Docker images and cache"
bash "${SCRIPT_DIR}/cleanup_docker.sh"

echo "--- :docker: Building prod image (${BRANCH} @ ${TORCHTPU_COMMIT:0:7}, vLLM ${VLLM_COMMIT_HASH})"
cd "${REPO_ROOT}"
./docker/build_image.sh --target prod -c "${VLLM_COMMIT_HASH}" -t "${LOCAL_IMAGE}"

echo "--- Checking the built image"
# Reads package metadata only, so it runs on a CPU agent without importing
# torch_tpu. vcs_info comes from the PEP 610 direct_url.json recorded when vLLM
# is installed from git.
IMAGE_INFO="$(docker run --rm "${LOCAL_IMAGE}" python -c '
import json
from importlib import metadata

vllm = metadata.distribution("vllm")
vcs = json.loads(vllm.read_text("direct_url.json") or "{}").get("vcs_info", {})
print(metadata.version("vllm_torchtpu"))
print(vllm.version)
print(vcs.get("commit_id", ""))
print(vcs.get("requested_revision", ""))
')"
mapfile -t INFO <<< "${IMAGE_INFO}"
TORCHTPU_VERSION="${INFO[0]:-}"
VLLM_VERSION="${INFO[1]:-}"
VLLM_RESOLVED="${INFO[2]:-}"
VLLM_REQUESTED="${INFO[3]:-}"
echo "vllm_torchtpu ${TORCHTPU_VERSION}, vllm ${VLLM_VERSION} (requested '${VLLM_REQUESTED}', commit '${VLLM_RESOLVED}')"

if [ -n "${VLLM_RESOLVED}" ]; then
  # Same guard as tpu-inference's verify_image_vllm: never publish an image
  # whose vLLM differs from the one the tests used.
  if [ "${VLLM_REQUESTED}" != "${VLLM_COMMIT_HASH}" ] && [[ "${VLLM_RESOLVED}" != "${VLLM_COMMIT_HASH}"* ]]; then
    echo "[FATAL] ${LOCAL_IMAGE} contains vLLM ${VLLM_RESOLVED} (requested '${VLLM_REQUESTED}'), expected ${VLLM_COMMIT_HASH}." >&2
    exit 1
  fi
  VLLM_SHORT="${VLLM_RESOLVED:0:7}"
else
  echo "WARNING: the image does not record the vLLM commit; tagging with the ref '${VLLM_COMMIT_HASH}' instead." >&2
  VLLM_SHORT="${VLLM_COMMIT_HASH//[^A-Za-z0-9_.-]/-}"
fi

if [ -n "${VERSION}" ]; then
  # release/X.Y.Z: same tags as publish_nightly_images.sh.
  DATE_TAG="${DATE}-${VERSION}"
  MOVING_TAG="${VERSION}"
else
  DATE_TAG="nightly-${DATE}-${TORCHTPU_COMMIT:0:7}-${VLLM_SHORT}"
  MOVING_TAG="nightly"
fi
# The dated tag is pushed first, so the moving tag never points at an image
# that has no dated tag of its own.
PUBLIC_TAGS=("${DOCKERHUB_REPO}:${DATE_TAG}" "${DOCKERHUB_REPO}:${MOVING_TAG}")

for TAG in "${PUBLIC_TAGS[@]}"; do
  docker tag "${LOCAL_IMAGE}" "${TAG}"
done

if [ "${DRY_RUN}" = "1" ] || [ "${DRY_RUN}" = "true" ]; then
  echo "--- DRY_RUN=${DRY_RUN}: skipping push of ${PUBLIC_TAGS[*]}"
  SUMMARY="Dry run, nothing pushed. Would publish: \`${PUBLIC_TAGS[*]}\`"
  STYLE="info"
else
  for TAG in "${PUBLIC_TAGS[@]}"; do
    echo "--- :docker: Pushing ${TAG}"
    docker push "${TAG}"
  done
  SUMMARY="Published \`${PUBLIC_TAGS[*]}\` from ${BRANCH} (vLLM ${VLLM_COMMIT_HASH})"
  STYLE="success"
fi

if [ -n "${BUILDKITE:-}" ]; then
  buildkite-agent annotate --style "${STYLE}" --context dockerhub-nightly ":docker: ${SUMMARY}"
fi

echo "--- Done: ${SUMMARY}"
