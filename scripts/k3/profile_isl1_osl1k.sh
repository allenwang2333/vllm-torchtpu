#!/usr/bin/env bash
# Warm up, then capture two exact ISL=1 / OSL=1000 requests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if (( $# != 1 )); then
  echo "Usage: $0 SERVER_IP" >&2
  exit 2
fi
SERVER_IP="$1"
SERVER_PORT="${SERVER_PORT:-8000}"
BASE_URL="http://${SERVER_IP}:${SERVER_PORT}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/profile_isl1_osl1k}"
PROF_DIR="${PROF_DIR:-$REPO_ROOT/results/kimi-k3-profile}"

iteration_count() {
  curl --fail --silent --show-error "${BASE_URL%/}/metrics" |
    awk '/^vllm:iteration_tokens_total_count/ { total += $NF; found = 1 }
         END { if (!found) exit 1; printf "%.0f\n", total }'
}

wait_for_iteration() {
  local target="$1"
  while (( $(iteration_count) < target )); do
    if ! kill -0 "$request_pid" 2>/dev/null; then
      wait "$request_pid" || true
      echo "Request ended before engine iteration $target" >&2
      exit 1
    fi
    sleep 0.01
  done
}

request_pid=
profiler_started=0
cleanup() {
  if (( profiler_started )); then
    "$SCRIPT_DIR/stop_profile.sh" "$SERVER_IP" || true
  fi
  if [[ -n "$request_pid" ]] && kill -0 "$request_pid" 2>/dev/null; then
    kill "$request_pid" 2>/dev/null || true
    wait "$request_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUTPUT_DIR"
curl --fail --silent --show-error "${BASE_URL%/}/health" >/dev/null

for warmup in 1 2; do
  echo "Warmup $warmup/2"
  OUTPUT_FILE="$OUTPUT_DIR/warmup-${warmup}-response.json" \
    "$SCRIPT_DIR/curl_isl1_osl1k.sh" "$SERVER_IP"
done

for capture in 1 2; do
  echo "Profile capture $capture/2"
  baseline="$(iteration_count)"
  OUTPUT_FILE="$OUTPUT_DIR/capture-${capture}-response.json" \
    "$SCRIPT_DIR/curl_isl1_osl1k.sh" "$SERVER_IP" &
  request_pid=$!

  wait_for_iteration "$((baseline + 3))"
  profiler_started=1
  "$SCRIPT_DIR/start_profile.sh" "$SERVER_IP"
  profile_start="$(iteration_count)"
  wait_for_iteration "$((profile_start + 15))"
  "$SCRIPT_DIR/stop_profile.sh" "$SERVER_IP"
  profiler_started=0

  wait "$request_pid"
  request_pid=
done

echo "Profiles: $PROF_DIR"
