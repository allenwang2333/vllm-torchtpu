#!/bin/bash
# Upload results as each one appears, instead of only when the job ends.
#
# A long benchmark job that uploads in its `artifact_paths` post-command step
# loses everything if it is cancelled, wedged or killed. Run this in the
# background next to the job and every completed cell is already retrievable:
#
#   bash .buildkite/scripts/incremental_artifact_upload.sh &
#   UP=$!; trap 'kill $UP 2>/dev/null' EXIT
#   bash .buildkite/scripts/run_multihost.sh ...
#
# The container writes into the persistent-disk directory that is bind-mounted
# at /perf_eval_results, so this watches the host side of that mount. Files are
# re-uploaded when their mtime advances, which is what makes a growing
# server.log useful while the job is still running.
set -uo pipefail

# The two harnesses do not agree on where the results bind-mount lives:
# run_in_docker.sh uses /mnt/disks/persist, run_multihost.sh uses $HOME/persist
# (`PERSIST_ROOT="${HOME}/persist"`). Watching one hard-coded path silently
# uploads nothing on the other -- so watch every candidate that exists.
WATCH_DIRS=()
if [ "$#" -ge 1 ] && [ -n "${1:-}" ]; then
  WATCH_DIRS=("$1")
else
  for _root in "/mnt/disks/persist" "${HOME}/persist" "/var/lib/buildkite-agent/persist"; do
    [ -d "${_root}/perf_eval_results" ] && WATCH_DIRS+=("${_root}/perf_eval_results")
  done
fi
WATCH_DIR="${WATCH_DIRS[0]:-/mnt/disks/persist/perf_eval_results}"
DEST_DIR="${2:-perf_eval_results}"
INTERVAL="${3:-60}"
# The harness wipes the persistent results dir on the way in; wait past that so
# the first scan cannot upload the previous job's leftovers.
START_DELAY="${4:-180}"
STATE="$(mktemp)"
trap 'rm -f "$STATE"' EXIT

command -v buildkite-agent >/dev/null || {
  echo "incremental_artifact_upload: no buildkite-agent on PATH; not uploading"
  exit 0
}

sleep "$START_DELAY"
echo "incremental_artifact_upload: watching ${WATCH_DIRS[*]:-$WATCH_DIR}"
while true; do
  for WATCH_DIR in "${WATCH_DIRS[@]:-$WATCH_DIR}"; do
  [ -d "$WATCH_DIR" ] || continue
  while IFS= read -r -d '' src; do
    rel="${src#"$WATCH_DIR"/}"
    stamp=$(stat -c '%Y %s' "$src" 2>/dev/null) || continue
    prev=$(grep -F "|$rel|" "$STATE" 2>/dev/null | tail -1 | cut -d'|' -f1)
    [ "$prev" = "$stamp" ] && continue
    mkdir -p "$DEST_DIR/$(dirname "$rel")"
    cp -f "$src" "$DEST_DIR/$rel" 2>/dev/null || continue
    if buildkite-agent artifact upload "$DEST_DIR/$rel" >/dev/null 2>&1; then
      echo "$stamp|$rel|" >> "$STATE"
      echo "incremental_artifact_upload: $rel"
    fi
  done < <(find "$WATCH_DIR" -type f -print0 2>/dev/null)
  done
  sleep "$INTERVAL"
done
