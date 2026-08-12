#!/usr/bin/env bash
# Start the configured profiler on an existing vLLM server. The server's
# profiler-config.max_iterations setting controls when capture stops.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Starting vLLM profiler at ${BASE_URL%/}/start_profile"
curl --fail-with-body --silent --show-error \
  --request POST \
  "${BASE_URL%/}/start_profile"
echo
