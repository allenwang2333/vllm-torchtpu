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

# Measures line coverage of vllm_torchtpu over tests/, running one pytest
# process per test directory (the selected test files directly in it). If that
# process is killed by a signal, each of the directory's files is run again in
# a process of its own, so a crash loses only the crashing file's coverage.
#
# Usage: run_coverage_split.sh <name> <pytest -m expression> <timeout_s> \
#          [include regex] [exclude regex]
#
# The optional regexes (grep -E) pick which collected test files this run
# covers, so one marker selection can be split across several jobs.
#
# Writes /perf_eval_results/coverage_<name>.dat.<index>[.<suffix>] data files
# and /perf_eval_results/coverage_<name>_runs.txt, which lists every pytest
# run with its exit code, duration and target. Always exits 0 once tests have
# been collected; failures are reported in the log and in that list.

set -uo pipefail

NAME="$1"
MARKERS="$2"
TIMEOUT="$3"
INCLUDE="${4:-.}"
EXCLUDE="${5:-}"
OUT=/perf_eval_results
RUNS="${OUT}/coverage_${NAME}_runs.txt"
run_index=0

# run_pytest <label> <test file>...: one pytest process with its own data file.
# Sets rc to pytest's exit code.
run_pytest() {
  local label="$1"
  shift
  run_index=$((run_index + 1))
  echo "--- [coverage:${NAME}] run ${run_index}: ${label} ($# files)"
  local start
  start=$(date +%s)
  COVERAGE_FILE="${OUT}/coverage_${NAME}.dat.$(printf '%04d' "${run_index}")" \
    python3 -m pytest -sv -rfE --timeout="${TIMEOUT}" -m "${MARKERS}" \
      --cov=vllm_torchtpu --cov-report= "$@" < /dev/null
  rc=$?
  echo "${rc} $(( $(date +%s) - start ))s ${label}" >> "${RUNS}"
  if [[ "${rc}" -ge 128 ]]; then
    echo "[coverage:${NAME}] CRASH: pytest killed by signal $((rc - 128)) in ${label}"
  elif [[ "${rc}" -ne 0 && "${rc}" -ne 5 ]]; then
    echo "[coverage:${NAME}] pytest exit ${rc} in ${label}"
  fi
}

if [[ ! -f .coveragerc ]]; then
  echo "[coverage] .coveragerc missing in the image; branch/parallel/subprocess settings would be silently dropped"
  exit 1
fi

echo "--- [coverage:${NAME}] Collecting test files for -m '${MARKERS}'"
python3 -m pytest --collect-only -q --continue-on-collection-errors \
  -m "${MARKERS}" tests/ > /tmp/collect.txt 2>&1
grep -oE '^tests/[^:]+\.py::' /tmp/collect.txt | sed 's/::$//' | sort -u \
  | grep -E "${INCLUDE}" | { if [[ -n "${EXCLUDE}" ]]; then grep -vE "${EXCLUDE}"; else cat; fi; } \
  > /tmp/files.txt
grep -E '^ERROR ' /tmp/collect.txt || true
total=$(wc -l < /tmp/files.txt)
xargs -n1 dirname < /tmp/files.txt | sort -u > /tmp/dirs.txt
echo "[coverage:${NAME}] ${total} test files in $(wc -l < /tmp/dirs.txt) directories have selected tests"
if [[ "${total}" -eq 0 ]]; then
  tail -50 /tmp/collect.txt
  exit 1
fi

: > "${RUNS}"
while read -r dir; do
  mapfile -t files < <(awk -v d="${dir}" '{ f = $0; sub(/\/[^\/]*$/, "", f); if (f == d) print }' /tmp/files.txt)
  run_pytest "${dir}/" "${files[@]}"
  if [[ "${rc}" -ge 128 ]]; then
    echo "[coverage:${NAME}] re-running the ${#files[@]} files of ${dir}/ one per process"
    for f in "${files[@]}"; do
      run_pytest "${f}" "${f}"
    done
  fi
done < /tmp/dirs.txt

echo "--- [coverage:${NAME}] Runs (exit code, time, target)"
cat "${RUNS}"
echo "[coverage:${NAME}] runs: ${run_index}," \
  "ok: $(awk '$1 == 0' "${RUNS}" | wc -l)," \
  "test failures: $(awk '$1 == 1' "${RUNS}" | wc -l)," \
  "crashed: $(awk '$1 >= 128' "${RUNS}" | wc -l)," \
  "other: $(awk '$1 != 0 && $1 != 1 && $1 < 128' "${RUNS}" | wc -l)"
exit 0
