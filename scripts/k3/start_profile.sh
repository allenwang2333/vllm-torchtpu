#!/usr/bin/env bash
# Start the configured profiler on an existing vLLM server. The current TPU
# worker starts its raw torch profiler immediately; profile_isl1_osl1k.sh
# enforces the delay/capture window and calls stop_profile.sh to flush it.
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 SERVER_IP" >&2
  exit 2
fi
SERVER_IP="$1"
SERVER_PORT="${SERVER_PORT:-8000}"
BASE_URL="http://${SERVER_IP}:${SERVER_PORT}"

echo "Starting vLLM profiler at ${BASE_URL%/}/start_profile"
curl --fail-with-body --silent --show-error \
  --request POST \
  "${BASE_URL%/}/start_profile"
echo
