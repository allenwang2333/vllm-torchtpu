#!/usr/bin/env bash
# Symlink each rank's decode xplane into its own session directory, so
# llar_chip_skew.py gets genuinely per-rank data.
#
# Required because xprof caches combined op-stats keyed on the session
# *directory*, not the file: reading individual paths from one session returns
# the same rank-combined aggregate every time, silently. Symlinks give each file
# a distinct session path without copying gigabytes.
#
#   scripts/debug/stage_chipiso.sh <trace-root> <out-dir>
set -euo pipefail
SRC="${1:?usage: stage_chipiso.sh <trace-root> <out-dir>}"
OUT="${2:?usage: stage_chipiso.sh <trace-root> <out-dir>}"
rm -rf "${OUT}"
n=0
# Only the full captures: the profiler leaves some ranks with ~1.5 MB partial
# planes that carry no HLO stats at all.
while IFS= read -r f; do
  r=$(basename "$f" | sed 's/_.*//')
  mkdir -p "${OUT}/${r}/plugins/profile/s1"
  ln -sf "$(realpath "$f")" "${OUT}/${r}/plugins/profile/s1/$(basename "$f")"
  n=$((n + 1))
done < <(find "${SRC}" -name "*.xplane.pb" -size +50M)
echo "staged ${n} rank(s) from ${SRC} -> ${OUT}"
