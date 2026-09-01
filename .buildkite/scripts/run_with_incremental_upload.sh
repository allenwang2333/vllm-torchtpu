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

# Pass the watch dir EMPTY unless a caller names one. The uploader treats a
# non-empty $1 as "watch exactly this", so defaulting it here pinned every job
# to the single-host root -- and on the multihost path, where the results live
# under $HOME/persist, it scanned a directory that does not exist and uploaded
# nothing, silently, for the whole of every pod job. Found by W1 on BK #329
# (`bk artifacts download` returned "No artifacts found" an hour in). Empty
# lets the uploader's own discovery run, which already knows both roots.
bash "${SCRIPT_DIR}/incremental_artifact_upload.sh" \
  "${INCREMENTAL_WATCH_DIR:-}" \
  "${INCREMENTAL_DEST_DIR:-perf_eval_results}" \
  "${INCREMENTAL_INTERVAL:-60}" &
uploader=$!
trap 'kill "$uploader" 2>/dev/null || true' EXIT INT TERM

"$@"
status=$?

kill "$uploader" 2>/dev/null || true
exit "$status"
