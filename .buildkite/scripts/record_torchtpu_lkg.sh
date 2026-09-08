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

cd "${REPO_ROOT}"

echo "--- All critical uprev tests passed! Recording Holographic LKG..."

TORCH_TPU_VER=$(buildkite-agent meta-data get "TORCH_TPU_VERSION" 2>/dev/null || echo "${1:-unknown}")
echo "Verified Stable Torch-TPU Version: ${TORCH_TPU_VER}"

if [ "${TORCH_TPU_VER}" != "unknown" ]; then
  # 1. Update simple version shortcut
  echo "${TORCH_TPU_VER}" > .buildkite/torchtpu_lkg.version

  # 2. Update holographic dependency manifest
  TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  BUILD_URL="${BUILDKITE_BUILD_URL:-unknown}"

  cat <<EOF > .buildkite/torchtpu_lkg.yaml
# ==============================================================================
# Buildkite Automated Torch-TPU Holographic LKG Manifest
# Automatically updated by .buildkite/scripts/record_torchtpu_lkg.sh
# ==============================================================================

current_lkg:
  status: "VERIFIED_PASS"
  verified_at: "${TIMESTAMP}"
  buildkite_build_url: "${BUILD_URL}"
  components:
    torch_tpu: "${TORCH_TPU_VER}"
  test_coverage_summary:
    tpu_runner: PASS
    speculative_decoding: PASS
    multichip_collectives: PASS
    disagg_pcp4_dp4: PASS
    offline_inference_smoke: PASS
EOF

  echo "Holographic LKG updated successfully at .buildkite/torchtpu_lkg.yaml"

  if command -v buildkite-agent >/dev/null 2>&1; then
    buildkite-agent annotate ":white_check_mark: All critical tests passed! Verified stable Holographic LKG: **torch-tpu==${TORCH_TPU_VER}**" --style "success" || true
  fi
fi
