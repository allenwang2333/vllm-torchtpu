#!/bin/bash
#SBATCH --job-name=k3-mla-mixed-tune
#SBATCH --partition=dev-2x2x1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00

set -euo pipefail
ulimit -c 0

source /etc/profile.d/shared-env.sh
source /mnt/nfs-share-disk/homes/opentpu_jiongxuan/envs/vllm-torchtpu/bin/activate

: "${TUNING_SOURCE_ROOT:?TUNING_SOURCE_ROOT must identify the staged source tree}"
: "${TUNING_OUTPUT:?TUNING_OUTPUT must identify the JSONL output path}"
: "${SOURCE_COMMIT:?SOURCE_COMMIT must be set}"
: "${KERNEL_SHA256:?KERNEL_SHA256 must be set}"

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=tpu

cd "${TUNING_SOURCE_ROOT}"
actual_kernel_sha256="$(sha256sum src/vllm_torchtpu/kernels/mla/v2/kernel.py | awk '{print $1}')"
if [[ "${actual_kernel_sha256}" != "${KERNEL_SHA256}" ]]; then
  echo "Kernel digest mismatch: expected=${KERNEL_SHA256} actual=${actual_kernel_sha256}" >&2
  exit 2
fi

python scripts/slurm/k3-mla-tuning/k3_mla_mixed_tuner.py \
  --output "${TUNING_OUTPUT}" \
  "$@"
