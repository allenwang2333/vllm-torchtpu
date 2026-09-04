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

# Run the Kimi-K3 decode-only phased capture and collect it from EVERY host of
# the slice -- including the hosts whose container is about to be deleted.
#
# Usage: k3r5_decodeprof_run.sh [tp32|sliced8] [inner command ...]
#
# With no inner command it runs the decode-profile config. With one, it runs
# that instead and still collects every host's capture -- which is how the
# ladder gets its folded-in c=1 window off the pod. A capture that nothing
# collects is a capture that dies with the slice, and the ladder harness runs
# INSIDE the container where it cannot reach the worker hosts.
#
# Two things make this more than a one-liner:
#
#  1. On a Ray slice only the head container has /perf_eval_results bind-mounted.
#     The other 24 ranks write their capture inside a container that
#     run_multihost.sh's cleanup() removes on exit. The config therefore points
#     the profiler at /root/.cache/huggingface, which IS bind-mounted on every
#     host, so after the slice is gone each host still has its own traces on its
#     own disk and this script sshs out and reduces them in place.
#  2. A capture is worth more than the run that produced it. Buildkite skips the
#     rest of a step's commands once one fails, so collection cannot be a second
#     command in the yml -- a server that died after the decode window still
#     leaves a usable trace, and this script collects on every exit path.
set -uo pipefail

RIG="${1:-${BUILDKITE_K3_RIG:-tp32}}"
shift || true
INNER=("$@")
CONFIG="${CONFIG:-kimi-k3-tp32-ep-best-lo-profile}"
BUILD="${BUILDKITE_BUILD_NUMBER:-local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REDUCE="${REPO_DIR}/scripts/profiling/xprof_reduce.py"
RESULTS="${K3_RESULTS_DIR:-perf_eval_results/k3r5-decodeprof-${RIG}}"
XPROF_OUT="${RESULTS}/xprof"

# The container reads this; both container drivers forward BUILDKITE_* verbatim
# and neither forwards anything else we could name.
export BUILDKITE_K3_RIG="$RIG"

say() { echo "--- [k3r5] $*"; }
upload() {
  command -v buildkite-agent >/dev/null 2>&1 || return 0
  buildkite-agent artifact upload "$1" || true
}

# --- 1. run -----------------------------------------------------------------
mkdir -p "$RESULTS"
case "$RIG" in
  tp32)
    HOST_PROFILE_DIR="${HOME}/hf_home/k3r5_profile/${K3_PROFILE_SUBDIR:-tp32}/${BUILD}"
    say "TP32 pod run; per-host capture dir ${HOST_PROFILE_DIR}"
    if [ "${#INNER[@]}" -gt 0 ]; then
      bash "${SCRIPT_DIR}/run_multihost.sh" "${INNER[@]}"
    else
      bash "${SCRIPT_DIR}/run_multihost.sh" bash ./scripts/vllm/benchmarking/run_benchmarks.sh \
          --config "$CONFIG" --results-dir "/${RESULTS}"
    fi
    rc=$?
    ;;
  sliced8)
    # Single host: /perf_eval_results is bound and copied back for us, so the
    # capture is already in the workspace when the container exits.
    HOST_PROFILE_DIR="perf_eval_results/k3r5_profile/sliced8/${BUILD}"
    say "sliced8 single-host rehearsal; capture dir ${HOST_PROFILE_DIR}"
    if [ "${#INNER[@]}" -gt 0 ]; then
      bash "${SCRIPT_DIR}/run_in_docker.sh" "${INNER[@]}"
    else
      bash "${SCRIPT_DIR}/run_in_docker.sh" bash ./scripts/vllm/benchmarking/run_benchmarks.sh \
          --config "$CONFIG" --results-dir "/${RESULTS}"
    fi
    rc=$?
    ;;
  *)
    echo "ERROR: rig must be tp32 or sliced8 (got '$RIG')" >&2
    exit 2
    ;;
esac
say "benchmark driver exited ${rc}; collecting the capture regardless"

# --- 2. reduce whatever this host produced ----------------------------------
PY=(python3)
command -v python3 >/dev/null 2>&1 || PY=(docker run --rm -v "$PWD:$PWD" -w "$PWD" \
  "${VLLM_TORCHTPU_IMAGE_TAG:-python:3.12-slim}" python3)

reduce_tree() {  # <capture root> <output root> <label>
  local src="$1" dst="$2" label="$3" pdir cap phase
  [ -d "$src" ] || { say "no capture tree at $src ($label)"; return 0; }
  mkdir -p "$dst"
  {
    echo "# xplane inventory for ${label}, $(date -u +%FT%TZ)"
    find "$src" -name '*.xplane.pb' -printf '%s\t%p\n' 2>/dev/null | sort -rn
    du -sb "$src" 2>/dev/null
    df -h . 2>/dev/null | tail -2
  } > "${dst}/trace_census.txt"
  say "$label: $(grep -c 'xplane.pb' "${dst}/trace_census.txt" || echo 0) xplane files"
  # ONE OUTPUT DIRECTORY PER CAPTURE TIMESTAMP, not per phase. The profiler is
  # armed once per benchmark cell, so two concurrencies produce two sibling
  # <ts> dirs under <phase>/plugins/profile/ whose files have IDENTICAL names
  # (rank0_<host>.xplane.pb in both). xprof_reduce keys its output on that
  # basename alone, so reducing the phase dir writes both armings to the same
  # planes/<base>__<plane>.json.gz and the second silently overwrites the
  # first: build 367 shipped a 98 MB bundle that had lost its entire c=1
  # window, with no error anywhere. Reduce each <ts> separately.
  local tsdir ts
  while IFS= read -r tsdir; do
    [ -d "$tsdir" ] || continue
    ts="$(basename "$tsdir")"
    phase="$(basename "$(dirname "$(dirname "$tsdir")")")"
    say "  reducing ${label}/${phase}/${ts}"
    "${PY[@]}" "$REDUCE" --root "$tsdir" --out "${dst}/${phase}/${ts}" --slim rank0 --meta-from rank0 || true
    gzip -f "${dst}/${phase}/${ts}"/slim/*.slim.xplane.pb 2>/dev/null || true
  done < <(find "$src" -type d -path '*/plugins/profile/*' -mindepth 3 -maxdepth 3 2>/dev/null | sort)
  # The per-step batch composition stats stay in the rank sandboxes and are the
  # proof the window is decode-only. They are tiny, they cover every arming, and
  # they must not be lost.
  while IFS= read -r pdir; do
    cap="$(dirname "$(dirname "$pdir")")"
    phase="$(basename "$cap")"
    mkdir -p "${dst}/${phase}/stats"
    while IFS= read -r -d '' f; do
      mkdir -p "${dst}/${phase}/stats/$(basename "$(dirname "$f")")"
      cp "$f" "${dst}/${phase}/stats/$(basename "$(dirname "$f")")/" || true
    done < <(find "$cap" -name 'batch_composition_stats_*.json' -print0 2>/dev/null)
  done < <(find "$src" -type d -path '*/plugins/profile' 2>/dev/null | sort)
}

reduce_tree "$HOST_PROFILE_DIR" "${XPROF_OUT}/host0" "host0"

# --- 3. the small bundle, uploaded before anything else can go wrong --------
BUNDLE="perf_eval_results/k3r5_${RIG}_xprof_reduced.tgz"
tar czf "$BUNDLE" -C "$RESULTS" . 2>/dev/null || true
say "reduced bundle $(du -h "$BUNDLE" 2>/dev/null | cut -f1)"
upload "$BUNDLE"

# --- 4. the other hosts of the slice ----------------------------------------
if [ "$RIG" = "tp32" ]; then
  SSH_USER="${SSH_USER:-$(whoami)}"
  SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o UserKnownHostsFile=/dev/null
            -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -i ~/.ssh/id_rsa)
  if [ -z "${WORKER_IPS:-}" ]; then
    ZONE="${ZONE:-$(curl -s -H 'Metadata-Flavor: Google' \
      'http://metadata.google.internal/computeMetadata/v1/instance/zone' | awk -F/ '{print $NF}')}"
    TPU_NAME="${TPU_NAME:-$(curl -s -H 'Metadata-Flavor: Google' \
      'http://metadata.google.internal/computeMetadata/v1/instance/description' 2>/dev/null || echo '')}"
    if [ -n "$TPU_NAME" ] && [ -n "$ZONE" ]; then
      ALL_IPS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone "$ZONE" \
        --format='value(networkEndpoints[].ipAddress)' 2>/dev/null)
      ALL_IPS="${ALL_IPS//;/ }"; ALL_IPS="${ALL_IPS//,/ }"
      # shellcheck disable=SC2206
      IPS=($ALL_IPS)
      WORKER_IPS=$(IFS=, ; echo "${IPS[*]:1}")
    fi
  fi
  IFS=',' read -r -a WORKERS <<< "${WORKER_IPS:-}"
  say "worker hosts to collect from: ${WORKER_IPS:-<none discovered>}"
  widx=0
  for ip in "${WORKERS[@]}"; do
    [ -n "$ip" ] || continue
    widx=$((widx + 1))
    tgz="perf_eval_results/k3r5_${RIG}_xprof_host${widx}.tgz"
    # The container wrote as root into the bind mount; default umask leaves the
    # files world-readable, so the agent user can reduce them without sudo.
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "cat > ~/xprof_reduce.py" < "$REDUCE" || continue
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "
      set -u
      SRC=\"\$HOME/hf_home/k3r5_profile/${K3_PROFILE_SUBDIR:-tp32}/${BUILD}\"
      DST=/tmp/k3r5_xpr_${BUILD}
      rm -rf \"\$DST\"; mkdir -p \"\$DST\"
      [ -d \"\$SRC\" ] || { echo \"host${widx}: no capture at \$SRC\" >&2; exit 0; }
      find \"\$SRC\" -name '*.xplane.pb' -printf '%s\t%p\n' 2>/dev/null | sort -rn > \"\$DST/trace_census.txt\"
      find \"\$SRC\" -mindepth 3 -maxdepth 3 -type d -path '*/plugins/profile/*' | sort | while read -r t; do
        ts=\$(basename \"\$t\"); phase=\$(basename \$(dirname \$(dirname \"\$t\")))
        python3 ~/xprof_reduce.py --root \"\$t\" --out \"\$DST/\$phase/\$ts\" --slim none --meta-from none >&2 || true
      done
      find \"\$SRC\" -type d -path '*/plugins/profile' | sort | while read -r p; do
        cap=\$(dirname \$(dirname \"\$p\")); phase=\$(basename \"\$cap\")
        mkdir -p \"\$DST/\$phase/stats\"
        find \"\$cap\" -name 'batch_composition_stats_*.json' -print0 | while IFS= read -r -d '' f; do
          r=\$(basename \$(dirname \"\$f\")); mkdir -p \"\$DST/\$phase/stats/\$r\"; cp \"\$f\" \"\$DST/\$phase/stats/\$r/\" || true
        done
      done
      cd \"\$DST\" && tar czf - . " > "$tgz" 2>/dev/null || true
    say "  host${widx} (${ip}): $(du -h "$tgz" 2>/dev/null | cut -f1)"
    upload "$tgz"
    # Leave nothing behind: the boot disk is 100 GB and the raw traces are GBs.
    # Keep the raw traces unless asked to drop them. Build 367's bundle lost a
    # whole window to a silent overwrite and the raws had already been deleted,
    # which turned a re-reduce into a re-run of the pod.
    if [ "${K3_DROP_RAW:-0}" = "1" ]; then
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" \
        "rm -rf /tmp/k3r5_xpr_${BUILD} ~/xprof_reduce.py; sudo -n rm -rf \"\$HOME/hf_home/k3r5_profile/${K3_PROFILE_SUBDIR:-tp32}/${BUILD}\" 2>/dev/null || true" || true
    else
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "rm -rf /tmp/k3r5_xpr_${BUILD} ~/xprof_reduce.py" || true
    fi
  done
  if [ "${K3_DROP_RAW:-0}" = "1" ]; then
    sudo -n rm -rf "${HOME}/hf_home/k3r5_profile/${K3_PROFILE_SUBDIR:-tp32}/${BUILD}" 2>/dev/null || true
  else
    say "raw traces kept at ${HOME}/hf_home/k3r5_profile/${K3_PROFILE_SUBDIR:-tp32}/${BUILD} (K3_DROP_RAW=1 to drop)"
  fi
fi

say "collected:"
du -sh perf_eval_results/* 2>/dev/null || true
exit "$rc"
