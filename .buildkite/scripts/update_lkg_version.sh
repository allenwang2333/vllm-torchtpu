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

# Promote the vLLM commit this build tested to "last known good" by updating
# the pin in pyproject.toml on main.
#
# Runs as the final step of the integration pipeline, once every test step has
# passed (see .buildkite/pipeline_promote.yml). From that commit on, every
# other pipeline -- and every `pip install` of this package -- builds against
# the newly promoted vLLM.
#
# REPORT-ONLY BY DEFAULT. Out of the box this resolves and validates the
# promotion, rewrites pyproject.toml in the ephemeral build workspace, prints
# the exact diff that would land, and stops there. Nothing is committed and
# nothing is pushed, so the loop can be exercised end to end before anyone
# grants CI write access to this repo.
#
# Set LKG_PROMOTE_PUSH=1 to actually commit and push. That path needs the
# GitHub App used by the cpu-queue agents (ci-infra,
# terraform/gcp_old/tpu-inference/modules/ci_cpu) to have write access to this
# repo; it installs a git credential helper system-wide that mints a
# short-lived installation token per git operation, so `git push` works with no
# token handling here.
#
# Usage:
#   update_lkg_version.sh [<vllm-commit-sha>]
# The SHA defaults to the VLLM_COMMIT_HASH build metadata published by the
# bootstrap step, which is the vLLM the suite actually ran against.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

TARGET_BRANCH="${LKG_TARGET_BRANCH:-main}"
PYPROJECT="${REPO_ROOT}/pyproject.toml"
VLLM_PIN="${REPO_ROOT}/docker/vllm_pin.sh"
MAX_ATTEMPTS=3

annotate() {
  local style="$1" body="$2"
  if command -v buildkite-agent >/dev/null 2>&1; then
    buildkite-agent annotate "${body}" --style "${style}" --context "lkg-promote" || true
  fi
  echo "${body}"
}

NEW_REF="${1:-$(build_vllm_ref)}"

# Only a resolved upstream commit may be promoted. A tag or branch name would
# make the pin float, which is the exact failure mode LKG exists to prevent.
if [[ ! "${NEW_REF}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[FATAL] Expected a full 40-character vLLM commit SHA, got '${NEW_REF}'." >&2
  exit 1
fi

for (( attempt = 1; attempt <= MAX_ATTEMPTS; attempt++ )); do
  # Promote onto the current tip of main, not onto whatever this build checked
  # out, so a long test run cannot revert commits that landed while it ran.
  git fetch origin "${TARGET_BRANCH}"
  git checkout -f -B "${TARGET_BRANCH}" "origin/${TARGET_BRANCH}"

  CURRENT_REF="$(bash "${VLLM_PIN}" read "${PYPROJECT}")"
  if [[ "${CURRENT_REF}" == "${NEW_REF}" ]]; then
    annotate info ":arrows_counterclockwise: vLLM LKG is already \`${NEW_REF}\`; nothing to promote."
    exit 0
  fi

  bash "${VLLM_PIN}" write "${NEW_REF}" "${PYPROJECT}"
  if git diff --quiet -- "${PYPROJECT}"; then
    annotate warning ":warning: vLLM pin rewrite produced no change; skipping promotion."
    exit 0
  fi

  if [[ "${LKG_PROMOTE_PUSH:-0}" != "1" ]]; then
    echo "--- :mag: Promotion that would land on ${TARGET_BRANCH}"
    git --no-pager diff -- "${PYPROJECT}"
    annotate success \
      ":white_check_mark: Suite green on vLLM \`${NEW_REF}\`. Would promote LKG \`${CURRENT_REF}\` → \`${NEW_REF}\` (report-only; set LKG_PROMOTE_PUSH=1 to commit)."
    exit 0
  fi

  git config user.name "Buildkite Bot"
  git config user.email "buildkite-bot@users.noreply.github.com"
  git add -- "${PYPROJECT}"

  # [skip ci] keeps this commit from re-triggering the post-merge pipeline,
  # which would test the vLLM we just finished testing (bootstrap.sh honours it).
  git commit -s -m "[skip ci] Promote vLLM LKG to ${NEW_REF}" \
    -m "Promoted by ${BUILDKITE_BUILD_URL:-a local run} after a green integration build. Previous LKG: ${CURRENT_REF}."

  if git push origin "HEAD:${TARGET_BRANCH}"; then
    annotate success ":arrow_up: vLLM LKG promoted \`${CURRENT_REF}\` → \`${NEW_REF}\`"
    exit 0
  fi

  echo "Push rejected (attempt ${attempt}/${MAX_ATTEMPTS}); ${TARGET_BRANCH} moved. Retrying on the new tip."
done

annotate error ":x: Failed to promote vLLM LKG to \`${NEW_REF}\` after ${MAX_ATTEMPTS} attempts."
exit 1
