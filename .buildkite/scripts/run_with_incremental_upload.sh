#!/bin/bash
# Run a long command with results uploaded as they appear.
#
#   run_with_incremental_upload.sh <command...>
#
# Exists so a pipeline step does not have to carry shell state in YAML (where
# Buildkite's `$VAR` interpolation would eat it). See
# incremental_artifact_upload.sh for why per-cell upload matters.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/incremental_artifact_upload.sh" \
  "${INCREMENTAL_WATCH_DIR:-/mnt/disks/persist/perf_eval_results}" \
  "${INCREMENTAL_DEST_DIR:-perf_eval_results}" \
  "${INCREMENTAL_INTERVAL:-60}" &
uploader=$!
trap 'kill "$uploader" 2>/dev/null || true' EXIT INT TERM

"$@"
status=$?

kill "$uploader" 2>/dev/null || true
exit "$status"
