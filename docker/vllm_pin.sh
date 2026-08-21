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

# Read or rewrite the upstream vLLM git ref pinned in pyproject.toml.
#
# That pin is the single source of truth for which vLLM commit this repo builds
# against -- the "last known good" (LKG) revision. Everything that resolves a
# vLLM ref goes through this script so the substitution lives in exactly one
# place:
#
#   * .buildkite/scripts/update_lkg_version.sh promotes a green upstream commit
#     by committing a new pin to main.
#   * docker/Dockerfile overrides the pin for the duration of one build, so the
#     integration pipeline can test vLLM HEAD without touching main.
#   * .buildkite/scripts/configs/pipeline_config.sh reads the pin to decide
#     which vLLM every other pipeline builds and tags its image with.
#
# Usage:
#   vllm_pin.sh read  [pyproject.toml]
#   vllm_pin.sh write <ref> [pyproject.toml]
#
# `write` with an empty <ref> is a deliberate no-op: the Dockerfile always
# passes the build arg, and an unset arg means "keep whatever is pinned".

set -euo pipefail

# The dependency line this script owns, e.g.
#     "vllm @ git+https://github.com/vllm-project/vllm.git@v0.27.0",
# The ref is everything between the '@' and the closing quote.
_PIN_RE='"vllm @ git\+https://github\.com/vllm-project/vllm\.git@'

_default_pyproject() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  printf '%s\n' "${script_dir}/../pyproject.toml"
}

# Print the currently pinned ref. Fails if the pin is missing, which is the
# signal that pyproject.toml was reformatted out from under this script.
pin_read() {
  local file="$1" ref
  ref="$(sed -nE "s#^[[:space:]]*${_PIN_RE}([^\"]+)\".*#\1#p" "${file}" | head -n1)"
  if [[ -z "${ref}" ]]; then
    echo "[FATAL][vllm_pin] No vLLM git pin found in ${file}." >&2
    echo "[FATAL][vllm_pin] Expected a dependency of the form" >&2
    echo "[FATAL][vllm_pin]   \"vllm @ git+https://github.com/vllm-project/vllm.git@<ref>\"" >&2
    return 1
  fi
  printf '%s\n' "${ref}"
}

pin_write() {
  local file="$1" ref="$2" tmp written

  # The ref is interpolated into a sed replacement, so reject anything that
  # could escape it. Real refs are commit SHAs or tags.
  if [[ ! "${ref}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
    echo "[FATAL][vllm_pin] Refusing to write malformed vLLM ref '${ref}'." >&2
    return 1
  fi

  tmp="$(mktemp)"
  sed -E "s#^([[:space:]]*${_PIN_RE})[^\"]+(\".*)#\1${ref}\2#" "${file}" >"${tmp}"
  mv "${tmp}" "${file}"

  # Read back rather than trusting sed: a silently unmatched pattern would
  # otherwise ship an image built against the wrong vLLM.
  written="$(pin_read "${file}")"
  if [[ "${written}" != "${ref}" ]]; then
    echo "[FATAL][vllm_pin] Wrote '${ref}' to ${file} but read back '${written}'." >&2
    return 1
  fi
  echo "[vllm_pin] ${file} now pins vLLM ${ref}"
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    read)
      pin_read "${2:-$(_default_pyproject)}"
      ;;
    write)
      local ref="${2:-}"
      if [[ -z "${ref}" ]]; then
        echo "[vllm_pin] No override ref given; keeping the pin in pyproject.toml."
        return 0
      fi
      pin_write "${3:-$(_default_pyproject)}" "${ref}"
      ;;
    *)
      echo "Usage: $0 read [pyproject.toml]" >&2
      echo "       $0 write <ref> [pyproject.toml]" >&2
      return 1
      ;;
  esac
}

main "$@"
