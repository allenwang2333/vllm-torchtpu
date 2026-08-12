#!/usr/bin/env bash
# Stop and flush the configured profiler on an existing vLLM server.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Stopping vLLM profiler at ${BASE_URL%/}/stop_profile"
curl --fail-with-body --silent --show-error \
  --request POST \
  "${BASE_URL%/}/stop_profile"
echo
