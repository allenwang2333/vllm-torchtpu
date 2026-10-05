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
#
# ---------------------------------------------------------------------------
# Verify PyPI / TestPyPI Google Cloud OIDC (Trusted Publisher) flow and
# upload a dummy wheel.
#
# Environment variables (all optional):
#   PYPI_BASE_URL       Target PyPI base URL (default: https://test.pypi.org)
#   TEST_PYPI_PACKAGE   Throwaway dummy package name matching TestPyPI
#                       Pending/Trusted Publisher configuration
#                       (default: tmp-oidc-sandbox-pkg-7f9a2c)
# ---------------------------------------------------------------------------

set -euo pipefail
# Disable command tracing unconditionally so tokens never leak to build logs.
set +x
# Restrict all temporary files/directories to owner-only (0700 / 0600).
umask 077

PYPI_BASE_URL="${PYPI_BASE_URL:-https://test.pypi.org}"
PYPI_BASE_URL="${PYPI_BASE_URL%/}"
TEST_PYPI_PACKAGE="${TEST_PYPI_PACKAGE:-tmp-oidc-sandbox-pkg-7f9a2c}"
PKG_MODULE_NAME="${TEST_PYPI_PACKAGE//-/_}"
DUMMY_VERSION="0.0.${BUILDKITE_BUILD_NUMBER:-1}.dev$(date +%s)"

WorkDir="$(mktemp -d)"
cleanup() {
  rm -rf "${WorkDir}"
}
trap cleanup EXIT

echo "--- :python: Setting up isolated virtualenv"
python3 -m venv "${WorkDir}/venv"
# shellcheck source=/dev/null
source "${WorkDir}/venv/bin/activate"
python -m pip install --quiet --upgrade pip id build twine

echo "--- :key: Step 1: Querying OIDC audience from ${PYPI_BASE_URL}"
audience_resp="$(curl -fsSL "${PYPI_BASE_URL}/_/oidc/audience")"
oidc_audience="$(jq -r '.audience' <<< "${audience_resp}")"
if [[ -z "${oidc_audience}" || "${oidc_audience}" == "null" ]]; then
  echo "ERROR: Failed to parse audience from response: ${audience_resp}" >&2
  exit 1
fi
echo "Target PyPI URL : ${PYPI_BASE_URL}"
echo "Expected Audience: ${oidc_audience}"

echo "--- :google: Step 2: Minting GCP OIDC ID token via Metadata Server"
# `python -m id <audience>` queries the GCE metadata server for an OIDC JWT.
oidc_token="$(python -m id "${oidc_audience}")"
if [[ -z "${oidc_token}" ]]; then
  echo "ERROR: Failed to obtain OIDC token from GCP Metadata Server." >&2
  exit 1
fi

# Only log non-identifying structural claims (never log the JWT, email, sub, or
# GCP project/instance identifiers in public CI logs).
echo "Verified GCP OIDC Token Structural Claims (redacted identity):"
python -m id "${oidc_audience}" -d | jq '{
  iss: .iss,
  aud: .aud,
  email_verified: .email_verified,
  has_email_claim: (.email != null and .email != ""),
  has_sub_claim: (.sub != null and .sub != "")
}'

echo "--- :package: Step 3: Building dummy wheel (${TEST_PYPI_PACKAGE}==${DUMMY_VERSION})"
pkg_dir="${WorkDir}/dummy_pkg"
mkdir -p "${pkg_dir}/${PKG_MODULE_NAME}"

cat <<EOF > "${pkg_dir}/pyproject.toml"
[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "${TEST_PYPI_PACKAGE}"
version = "${DUMMY_VERSION}"
description = "Throwaway dummy package to verify GCP OIDC Trusted Publishing"
readme = "README.md"
requires-python = ">=3.10"
EOF

cat <<EOF > "${pkg_dir}/README.md"
# ${TEST_PYPI_PACKAGE}
Temporary throwaway package for verifying GCP OIDC Trusted Publisher workflow.
EOF

cat <<EOF > "${pkg_dir}/${PKG_MODULE_NAME}/__init__.py"
__version__ = "${DUMMY_VERSION}"
EOF

python -m build --wheel --outdir "${pkg_dir}/dist" "${pkg_dir}"
ls -lh "${pkg_dir}/dist/"

echo "--- :lock: Step 4: Exchanging GCP OIDC token for short-lived PyPI API token"
mint_resp_file="${WorkDir}/mint_resp.json"

# Pipe OIDC token via stdin into jq and curl (--data-binary @-) so the token
# never appears in any process argv (/proc/<pid>/cmdline or `ps aux`).
http_code="$(jq -Rs '{token: rtrimstr("\n")}' <<< "${oidc_token}" | \
  curl -sS -o "${mint_resp_file}" -w "%{http_code}" \
    -X POST "${PYPI_BASE_URL}/_/oidc/mint-token" \
    -H "Content-Type: application/json" \
    --data-binary @-)"

# Immediately clear the raw OIDC JWT from memory.
unset oidc_token

if [[ "${http_code}" != "200" ]]; then
  echo "Mint-token returned HTTP ${http_code}:" >&2
  # Only print error fields (.message / .errors), never any '.token' field.
  jq '{message: .message, errors: .errors}' "${mint_resp_file}" >&2 || true
  echo "" >&2
  echo "======================================================================" >&2
  echo "NOTE: GCP OIDC token generation succeeded, but ${PYPI_BASE_URL} rejected" >&2
  echo "the publisher lookup (HTTP ${http_code})." >&2
  echo "Once you register a Pending Publisher on ${PYPI_BASE_URL}/manage/account/publishing/" >&2
  echo "with PyPI Project Name '${TEST_PYPI_PACKAGE}' and your GCP Service Account," >&2
  echo "re-run this build to complete the wheel upload." >&2
  echo "======================================================================" >&2
  exit 1
fi

api_token="$(jq -r '.token // empty' "${mint_resp_file}")"
rm -f "${mint_resp_file}"
if [[ -z "${api_token}" ]]; then
  echo "ERROR: HTTP 200 from mint-token but no '.token' field found." >&2
  exit 1
fi
echo "Successfully minted short-lived PyPI API token (redacted)."

echo "--- :rocket: Step 5: Uploading dummy wheel to ${PYPI_BASE_URL}/legacy/"
TWINE_USERNAME="__token__" \
TWINE_PASSWORD="${api_token}" \
  python -m twine upload \
    --repository-url "${PYPI_BASE_URL}/legacy/" \
    "${pkg_dir}/dist/"*.whl

unset api_token

echo "--- :white_check_mark: PyPI GCP OIDC verification succeeded!"
echo "View package at: ${PYPI_BASE_URL}/project/${TEST_PYPI_PACKAGE}/${DUMMY_VERSION}/"
