#!/bin/bash
set -euo pipefail

# Determine directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${VLLM_OMNI_DIR}/.." && pwd)"

# Default settings
DEFAULT_TAG="us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/vllm-omni/vllm-omni-tpu:speechmaxxxing-v6"
DEFAULT_BASE="us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/vllm-omni/vllm-omni-tpu:speechmaxxxing-v5"
IMAGE_TAG="${DEFAULT_TAG}"
BASE_IMAGE="${DEFAULT_BASE}"
PUSH=false

show_help() {
  cat << 'HELP'
==============================================================================
Build Script for vLLM-Omni Cloud TPU Docker Image
==============================================================================
Usage:
  bash scratch/build_omni_image.sh [options]

Options:
  -t, --image-tag <tag>     Destination image tag
                            (default: us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/vllm-omni/vllm-omni-tpu:speechmaxxxing-v3)
  -b, --base-image <image>  Base Docker image to build on top of
                            (default: us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/vllm-omni/vllm-omni-tpu:speechmaxxxing-v3)
  --push                    Push the built image to Google Artifact Registry
  -h, --help                Show this help message
==============================================================================
HELP
}

# Parse command line flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    -b|--base-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    --push)
      PUSH=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
done

echo "============================================================"
echo " Building vLLM-Omni TPU Docker Image"
echo "============================================================"
echo " Image Tag:     ${IMAGE_TAG}"
echo " Base Image:    ${BASE_IMAGE}"
echo " Workspace:     ${WORKSPACE_DIR}"
echo " Dockerfile:    ${SCRIPT_DIR}/Dockerfile.omni"
echo " Push on build: ${PUSH}"
echo "============================================================"

# Check for gcloud token (needed if building from base or authenticating)
ACCESS_TOKEN="${ACCESS_TOKEN:-}"
if [ -z "${ACCESS_TOKEN}" ]; then
  if command -v gcloud >/dev/null 2>&1; then
    ACCESS_TOKEN="$(gcloud auth print-access-token 2>/dev/null || true)"
  fi
fi

DOCKER_BUILD_ARGS=(
  --progress=plain
  -f "${SCRIPT_DIR}/Dockerfile.omni"
  --build-arg BASE_IMAGE="${BASE_IMAGE}"
  -t "${IMAGE_TAG}"
)

if [ -n "${ACCESS_TOKEN}" ]; then
  export ACCESS_TOKEN
  DOCKER_BUILD_ARGS+=(--secret "id=gcloud_token,env=ACCESS_TOKEN")
fi

echo "===> Running docker build..."
DOCKER_BUILDKIT=1 docker build "${DOCKER_BUILD_ARGS[@]}" "${WORKSPACE_DIR}"

echo ""
echo "===> Successfully built image: ${IMAGE_TAG}"

if [ "${PUSH}" = true ]; then
  echo "===> Configuring Docker authentication for Artifact Registry..."
  gcloud auth configure-docker us-central1-docker.pkg.dev --quiet 2>/dev/null || true
  echo "===> Pushing image to Artifact Registry: ${IMAGE_TAG}..."
  docker push "${IMAGE_TAG}"
  echo "===> Image pushed successfully!"
fi
