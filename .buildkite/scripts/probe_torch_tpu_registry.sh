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

# Enumerate torch-tpu nightlies (and torch builds) in the private registry
# and print each nightly's Requires-Dist torch pin, to plan the SC2-halt
# dependency bisect (docs/sc2-sparsecore-halt-2026-08.md on branch
# mhhua/sc2-sparsecore-halt-evidence).

set -euo pipefail

if command -v gcloud >/dev/null 2>&1; then
  TOKEN="$(gcloud auth print-access-token)"
else
  TOKEN="$(curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
    | sed -nE 's/.*"access_token"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p')"
fi

REG="https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple"

echo "--- torch versions in registry"
curl -su "oauth2accesstoken:${TOKEN}" "${REG}/torch/" \
  | grep -oE 'torch-[0-9][^<"#]*\.whl' | sort -u | tail -30 || true

echo "--- torch-tpu wheel filenames (dev202607xx onward)"
curl -su "oauth2accesstoken:${TOKEN}" "${REG}/torch-tpu/" > /tmp/tt_index.html
grep -oE 'torch_tpu-0\.1\.1\.dev20260[78][0-9]{9,}[^<"#]*\.whl' /tmp/tt_index.html | sort -u || true

echo "--- per-nightly torch requirement (PEP 658 metadata)"
for v in $(grep -oE '0\.1\.1\.dev20260[78][0-9]{9,}' /tmp/tt_index.html | sort -u); do
  url=$(grep -oE "href=\"[^\"]*torch_tpu-${v}[^\"]*\.whl[^\"]*\"" /tmp/tt_index.html \
        | head -1 | sed -e 's/^href="//' -e 's/"$//' -e 's/#.*//')
  if [ -z "${url}" ]; then
    continue
  fi
  case "${url}" in
    http*) full="${url}" ;;
    *) full="${REG}/torch-tpu/${url}" ;;
  esac
  req=$(curl -sfu "oauth2accesstoken:${TOKEN}" "${full}.metadata" 2>/dev/null \
        | grep -iE '^Requires-Dist: torch([ =<>!~]|$)' | head -2 | tr '\n' ';' || true)
  echo "${v} -> ${req:-METADATA-UNAVAILABLE (${full})}"
done
