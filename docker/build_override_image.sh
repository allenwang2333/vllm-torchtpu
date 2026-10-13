#!/bin/bash
set -e

usage() {
  cat <<USAGE
Usage: ./docker/build_override_image.sh [options]

Builds a vllm-torchtpu override variant image by reinstalling torch-tpu from an
override container image, a GCS wheel URI, or a local wheel file/dir.

Options:
  -t, --image-tag TAG          Tag for the built image (default: vllm-torchtpu-local-override)
  -b, --base-image IMAGE       Base vllm-torchtpu image to override (default: vllm-torchtpu-local)
  -o, --override-source SRC    Override container image, GCS URI, or local wheel path
  -w, --wheel-uri URI          GCS URI or local file/directory containing .whl file(s)
  -h, --help                   Show this help and exit

Examples:
  # Override from a container image containing /opt/wheels/*.whl:
  ./docker/build_override_image.sh -b vllm-torchtpu-local:dev -t vllm-torchtpu-local:dev-override

  # Override from a GCS wheel URI:
  ./docker/build_override_image.sh -b vllm-torchtpu-local:dev -t vllm-torchtpu-local:dev-override \\
    -w gs://ci-wheels-tpu/torch_tpu-0.2.0-cp312-cp312-linux_x86_64.whl

  # Override from a local wheel file:
  ./docker/build_override_image.sh -b vllm-torchtpu-local:dev -t vllm-torchtpu-local:dev-override \\
    -w /path/to/torch_tpu.whl
USAGE
}

# Default values
IMAGE_TAG="vllm-torchtpu-local-override"
BASE_IMAGE="vllm-torchtpu-local"
OVERRIDE_SOURCE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu:nightly-latest"
WHEEL_URI=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    -b|--base-image)
      BASE_IMAGE="$2"
      shift 2
      ;;
    -o|--override-source)
      OVERRIDE_SOURCE="$2"
      shift 2
      ;;
    -w|--wheel-uri)
      WHEEL_URI="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Auto-detect if -o/--override-source was given a GCS URI or local wheel path/dir
if [ -z "${WHEEL_URI}" ]; then
  if [[ "${OVERRIDE_SOURCE}" == gs://* || "${OVERRIDE_SOURCE}" == *.whl || -d "${OVERRIDE_SOURCE}" || -f "${OVERRIDE_SOURCE}" ]]; then
    WHEEL_URI="${OVERRIDE_SOURCE}"
  fi
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../" && pwd)"
STAGED_WHEELS="${REPO_ROOT}/.staged_override_wheels"

CLEANUP_STAGED_WHEELS=false
cleanup() {
  if [ "$CLEANUP_STAGED_WHEELS" = true ]; then
    rm -rf "${STAGED_WHEELS}"
    CLEANUP_STAGED_WHEELS=false
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CLEANUP_STAGED_WHEELS=true
rm -rf "${STAGED_WHEELS}"
mkdir -p "${STAGED_WHEELS}"

USE_LOCAL_WHEELS="0"
if [ -n "${WHEEL_URI}" ]; then
  USE_LOCAL_WHEELS="1"
  # Avoid pulling the remote override container image when installing from a staged wheel
  OVERRIDE_SOURCE="${BASE_IMAGE}"
  echo "===> Staging override wheel(s) from: ${WHEEL_URI}"
  if [[ "${WHEEL_URI}" == gs://* ]]; then
    if command -v gcloud >/dev/null 2>&1; then
      gcloud storage cp "${WHEEL_URI}" "${STAGED_WHEELS}/"
    elif command -v gsutil >/dev/null 2>&1; then
      gsutil cp "${WHEEL_URI}" "${STAGED_WHEELS}/"
    else
      echo "ERROR: Neither gcloud nor gsutil is available to download ${WHEEL_URI}" >&2
      exit 1
    fi
  elif [ -d "${WHEEL_URI}" ]; then
    cp "${WHEEL_URI}"/*.whl "${STAGED_WHEELS}/"
  elif [ -f "${WHEEL_URI}" ]; then
    cp "${WHEEL_URI}" "${STAGED_WHEELS}/"
  else
    echo "ERROR: Wheel URI or path not found: ${WHEEL_URI}" >&2
    exit 1
  fi

  if ! compgen -G "${STAGED_WHEELS}/*.whl" > /dev/null; then
    echo "ERROR: No .whl files were staged in ${STAGED_WHEELS} from ${WHEEL_URI}" >&2
    exit 1
  fi
fi

echo "===> Building vllm-torchtpu local-wheel-override variant image..."
echo "Variant Image Tag: $IMAGE_TAG"
echo "Base Image: $BASE_IMAGE"
if [ "${USE_LOCAL_WHEELS}" = "1" ]; then
  echo "Wheel URI: $WHEEL_URI"
else
  echo "Override Source: $OVERRIDE_SOURCE"
fi

# Run docker build with BuildKit
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  -f "${SCRIPT_DIR}/Dockerfile.override" \
  -t "${IMAGE_TAG}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg OVERRIDE_SOURCE="${OVERRIDE_SOURCE}" \
  --build-arg USE_LOCAL_WHEELS="${USE_LOCAL_WHEELS}" \
  "${REPO_ROOT}"

cleanup
echo "===> Done!"
