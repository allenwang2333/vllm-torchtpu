#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reduce every xprof phase capture inside this container into /tmp/xpr, which
# the caller then tars to its stdout. Runs on the *worker* hosts of a multihost
# slice, whose /perf_eval_results is container-local and about to be deleted.
set -uo pipefail

SRC_ROOT="${1:-/perf_eval_results}"
DST_ROOT="${2:-/tmp/xpr}"
REDUCE="${REDUCE:-/root/torchtpu-vllm/scripts/profiling/xprof_reduce.py}"

rm -rf "${DST_ROOT:?}"
mkdir -p "$DST_ROOT"
echo "host $(hostname): reducing captures under $SRC_ROOT"
while IFS= read -r pdir; do
  cap_dir="$(dirname "$(dirname "$pdir")")"
  phase="$(basename "$cap_dir")"
  out="$DST_ROOT/$phase"
  echo "  $cap_dir -> $out"
  python3 "$REDUCE" --root "$cap_dir" --out "$out" \
      --slim none --meta-from none || true
  mkdir -p "$out/stats"
  while IFS= read -r -d '' f; do
    rank="$(basename "$(dirname "$f")")"
    mkdir -p "$out/stats/$rank"
    cp "$f" "$out/stats/$rank/" || true
  done < <(find "$cap_dir" -name 'batch_composition_stats_*.json' -print0 2>/dev/null)
done < <(find "$SRC_ROOT" -type d -path '*/plugins/profile' 2>/dev/null | sort)
find "$DST_ROOT" -type f | wc -l | xargs echo "files to ship:"
du -sh "$DST_ROOT" 2>/dev/null || true
exit 0
