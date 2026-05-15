#!/bin/bash
set -eu -o history -o allexport

# This script builds and pushes torchtpu-vllm docker images for nightly runs.
# It assumes it is run from the repository root.

REGISTRY="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container"
DOCKER_TAG_DATE="nightly-$(date +%Y%m%d)"
UPLOAD_IMAGE_TO_GCR=${UPLOAD_IMAGE_TO_GCR:-true}

echo "===> Building torchtpu-vllm base/ci/dev/prod images..."

for target in base ci dev prod; do
  bash docker/build_image.sh --target "$target" --torch-tpu-registry -t "${REGISTRY}/torchtpu-vllm-${target}:latest"
done

if [[ "${UPLOAD_IMAGE_TO_GCR}" == "true" ]]; then
  echo "===> Pushing images to Artifact Registry..."

  for target in base ci dev prod; do
    IMAGE_NAME="torchtpu-vllm-${target}"
    LATEST_TAG="${REGISTRY}/${IMAGE_NAME}:latest"
    DATE_TAG="${REGISTRY}/${IMAGE_NAME}:${DOCKER_TAG_DATE}"

    echo "Tagging $IMAGE_NAME for date specifically: $DATE_TAG"
    docker tag "$LATEST_TAG" "$DATE_TAG"

    echo "Pushing $IMAGE_NAME to Artifact Registry..."
    docker push "$DATE_TAG"
    docker push "$LATEST_TAG"
  done
  echo "===> Done!"
fi
