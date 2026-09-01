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

# Get an xprof capture off a TPU agent.
#
# A 93-layer / 32-rank decode capture is ~10 GB of xplane.pb. These nodes have
# read-only GCS scope, so `gsutil cp` out is not an option, and 10 GB is not an
# artifact upload either. `scripts/profiling/xprof_reduce.py` turns it into a
# few tens of MB that still reproduces the full per-op table; this uploads that
# *first*, then the raw traces only if they fit under RAW_TRACE_CAP_MB.
#
# Usage: k3_profile_collect.sh <results_dir> [tag]
set -uo pipefail

RESULTS_DIR="${1:?usage: k3_profile_collect.sh <results_dir> [tag]}"
TAG="${2:-k3}"
RAW_TRACE_CAP_MB="${RAW_TRACE_CAP_MB:-1500}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REDUCE="${REPO_DIR}/scripts/profiling/xprof_reduce.py"
OUT_ROOT="$(dirname "$RESULTS_DIR")"

say() { echo "--- [collect] $*"; }
upload() {  # paths are relative to the job workspace; never fail the job
  command -v buildkite-agent >/dev/null 2>&1 || return 0
  buildkite-agent artifact upload "$1" || true
}

if [ ! -d "$RESULTS_DIR" ]; then
  say "no results dir at $RESULTS_DIR -- nothing to collect"
  exit 0
fi

# --- 1. census before we touch anything ------------------------------------
CENSUS="$RESULTS_DIR/trace_census.txt"
{
  echo "# xplane inventory, $(date -u +%FT%TZ)"
  find "$RESULTS_DIR" -name '*.xplane.pb' -printf '%s\t%p\n' 2>/dev/null | sort -rn
  echo "# totals"
  du -sb "$RESULTS_DIR" 2>/dev/null
  df -h . 2>/dev/null | tail -2
} > "$CENSUS"
say "census: $(grep -c 'xplane.pb' "$CENSUS" || echo 0) xplane files"
head -5 "$CENSUS"

# --- 2. reduce every phase dir this host produced --------------------------
PY=(python3)
command -v python3 >/dev/null 2>&1 || PY=(docker run --rm -v "$PWD:$PWD" -w "$PWD" \
  "${VLLM_TORCHTPU_IMAGE_TAG:-python:3.12-slim}" python3)

# A phased capture lands in <profile>/<phase>/plugins/profile/<ts>/; the
# unphased one in <profile>/plugins/profile/<ts>/. Discover the capture roots
# rather than assume, so a run that silently fell back still gets collected.
while IFS= read -r pdir; do
  cap_dir="$(dirname "$(dirname "$pdir")")"
  phase="$(basename "$cap_dir")"
  say "reducing $phase ($cap_dir)"
  "${PY[@]}" "$REDUCE" --root "$cap_dir" \
      --out "$RESULTS_DIR/xprof/$phase" --slim rank0 --meta-from rank0 || true
  # The rank sandboxes hold the per-step batch composition stats, which are the
  # proof the window is decode-only. They are tiny and must not be lost.
  mkdir -p "$RESULTS_DIR/xprof/$phase/stats"
  while IFS= read -r -d '' f; do
    rank="$(basename "$(dirname "$f")")"
    mkdir -p "$RESULTS_DIR/xprof/$phase/stats/$rank"
    cp "$f" "$RESULTS_DIR/xprof/$phase/stats/$rank/" || true
  done < <(find "$cap_dir" -name 'batch_composition_stats_*.json' -print0 2>/dev/null)
  gzip -f "$RESULTS_DIR/xprof/$phase"/slim/*.slim.xplane.pb 2>/dev/null || true
done < <(find "$RESULTS_DIR" -type d -path '*/plugins/profile' 2>/dev/null | sort)

# --- 3. the small bundle, uploaded before anything else can go wrong -------
# Everything the analysis needs except the raw traces: reduced planes, the
# per-step batch composition stats, the bench result JSONs, config.json, logs.
BUNDLE="$OUT_ROOT/${TAG}_xprof_reduced.tgz"
tar czf "$BUNDLE" -C "$RESULTS_DIR" --exclude=./profile . || true
say "reduced bundle $(du -h "$BUNDLE" 2>/dev/null | cut -f1)"
upload "$BUNDLE"

# --- 4. raw traces only if they fit ----------------------------------------
RAW_MB=$(du -sm "$RESULTS_DIR/profile" 2>/dev/null | cut -f1)
RAW_MB=${RAW_MB:-0}
if [ "$RAW_MB" -gt 0 ] && [ "$RAW_MB" -le "$RAW_TRACE_CAP_MB" ]; then
  say "raw traces are ${RAW_MB} MB (cap ${RAW_TRACE_CAP_MB}) -- keeping them"
  tar czf "$OUT_ROOT/${TAG}_raw_traces.tgz" -C "$RESULTS_DIR" profile || true
  upload "$OUT_ROOT/${TAG}_raw_traces.tgz"
else
  say "raw traces are ${RAW_MB} MB, over the ${RAW_TRACE_CAP_MB} MB cap -- dropped"
fi
rm -rf "${RESULTS_DIR:?}/profile"

say "left for artifact_paths:"
du -sh "$OUT_ROOT"/* 2>/dev/null || true
exit 0
