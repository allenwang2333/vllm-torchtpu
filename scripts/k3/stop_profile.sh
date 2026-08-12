#!/usr/bin/env bash
# Stop and flush the configured profiler on an existing vLLM server.
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 SERVER_IP" >&2
  exit 2
fi
SERVER_IP="$1"
SERVER_PORT="${SERVER_PORT:-8000}"
BASE_URL="http://${SERVER_IP}:${SERVER_PORT}"

echo "Stopping vLLM profiler at ${BASE_URL%/}/stop_profile"
curl --fail-with-body --silent --show-error \
  --request POST \
  "${BASE_URL%/}/stop_profile"
echo
