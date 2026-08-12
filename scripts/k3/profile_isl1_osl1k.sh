#!/usr/bin/env bash
# Run two ISL=1 / OSL=1000 benchmarks while collecting bounded TPU traces.
#
# TorchTPU's profiler is explicitly stopped after the configured iteration
# window so every capture is flushed before the next one starts.
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8000}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/perf_isl1_osl1k_profiled}"
PROF_DIR="${PROF_DIR:-$REPO_ROOT/results/kimi-k3-profile}"
NUM_CAPTURES="${NUM_CAPTURES:-2}"
PROFILE_DELAY_ITERATIONS="${PROFILE_DELAY_ITERATIONS:-3}"
PROFILE_MAX_ITERATIONS="${PROFILE_MAX_ITERATIONS:-15}"
POLL_INTERVAL="${POLL_INTERVAL:-0.05}"
TRACE_ROOT="$PROF_DIR/plugins/profile"
TRACE_LABEL="${TRACE_LABEL:-$(date +%Y%m%d-%H%M%S)}"

iteration_count() {
  curl --fail --silent --show-error "${BASE_URL%/}/metrics" |
    awk '/^vllm:iteration_tokens_total_count/ { total += $NF }
         END { printf "%.0f\n", total }'
}

wait_for_iteration() {
  local target="$1"
  local current
  while true; do
    current="$(iteration_count)"
    if (( current >= target )); then
      return
    fi
    if ! kill -0 "$bench_pid" 2>/dev/null; then
      wait "$bench_pid"
      echo "Benchmark exited before profiler target iteration ${target}" >&2
      exit 1
    fi
    sleep "$POLL_INTERVAL"
  done
}

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TRACE_ROOT"

for ((capture = 1; capture <= NUM_CAPTURES; capture++)); do
  echo "Profile capture ${capture}/${NUM_CAPTURES}"
  baseline="$(iteration_count)"
  capture_trace_dir="$TRACE_ROOT/${TRACE_LABEL}-capture-${capture}"
  if [[ -e "$capture_trace_dir" ]]; then
    echo "Trace destination already exists: $capture_trace_dir" >&2
    exit 1
  fi

  declare -A trace_dirs_before=()
  while IFS= read -r trace_dir; do
    trace_dirs_before["$trace_dir"]=1
  done < <(find "$TRACE_ROOT" -mindepth 1 -maxdepth 1 -type d -print)

  BASE_URL="$BASE_URL" \
    OUTPUT_DIR="$OUTPUT_DIR/capture-${capture}" \
    "$SCRIPT_DIR/bench_isl1_osl1k.sh" "$@" &
  bench_pid=$!
  profiler_started=0

  cleanup_capture() {
    if (( profiler_started )); then
      BASE_URL="$BASE_URL" "$SCRIPT_DIR/stop_profile.sh" || true
    fi
    if kill -0 "$bench_pid" 2>/dev/null; then
      kill "$bench_pid" 2>/dev/null || true
      wait "$bench_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_capture INT TERM EXIT

  wait_for_iteration "$((baseline + PROFILE_DELAY_ITERATIONS))"
  BASE_URL="$BASE_URL" "$SCRIPT_DIR/start_profile.sh"
  profiler_started=1

  profile_start="$(iteration_count)"
  wait_for_iteration "$((profile_start + PROFILE_MAX_ITERATIONS))"
  BASE_URL="$BASE_URL" "$SCRIPT_DIR/stop_profile.sh"
  profiler_started=0

  new_trace_dirs=()
  while IFS= read -r trace_dir; do
    if [[ -z "${trace_dirs_before[$trace_dir]+x}" ]]; then
      new_trace_dirs+=("$trace_dir")
    fi
  done < <(find "$TRACE_ROOT" -mindepth 1 -maxdepth 1 -type d -print)
  if (( ${#new_trace_dirs[@]} == 0 )); then
    echo "Profiler stopped without creating an XPlane trace directory" >&2
    exit 1
  fi

  mkdir "$capture_trace_dir"

  for trace_dir in "${new_trace_dirs[@]}"; do
    trace_files=("$trace_dir"/*)
    if (( ${#trace_files[@]} == 0 )); then
      echo "Empty profiler trace directory: $trace_dir" >&2
      exit 1
    fi
    for trace_file in "${trace_files[@]}"; do
      trace_name="${trace_file##*/}"
      if [[ -e "$capture_trace_dir/$trace_name" ]]; then
        echo "Duplicate trace file: $capture_trace_dir/$trace_name" >&2
        exit 1
      fi
      mv "$trace_file" "$capture_trace_dir/"
    done
    rmdir "$trace_dir"
  done

  trace_count="$(find "$capture_trace_dir" -maxdepth 1 -type f -name '*.xplane.pb' | wc -l)"
  if (( trace_count != 32 )); then
    echo "Expected 32 rank traces in $capture_trace_dir, found $trace_count" >&2
    exit 1
  fi
  echo "Trace: $capture_trace_dir ($trace_count ranks)"

  wait "$bench_pid"
  trap - INT TERM EXIT
done
