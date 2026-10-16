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

# Measures line coverage of vllm_torchtpu over tests/, running pytest once per
# test file so each file gets a fresh process. A file that crashes its pytest
# process loses only its own coverage data.
#
# Usage: run_coverage_per_file.sh <name> <pytest -m expression> <timeout_s>
#
# Writes /perf_eval_results/coverage_<name>.dat.<index>[.<suffix>] data files
# and /perf_eval_results/coverage_<name>_files.txt, which lists every file
# with its pytest exit code. Always exits 0; failures are reported in the log
# and in that list.

set -uo pipefail

NAME="$1"
MARKERS="$2"
TIMEOUT="$3"
OUT=/perf_eval_results
RESULTS="${OUT}/coverage_${NAME}_files.txt"

if [[ ! -f .coveragerc ]]; then
  echo "[coverage] .coveragerc missing in the image; branch/parallel/subprocess settings would be silently dropped"
  exit 1
fi

echo "--- [coverage:${NAME}] Collecting test files for -m '${MARKERS}'"
python3 -m pytest --collect-only -q --continue-on-collection-errors \
  -m "${MARKERS}" tests/ > /tmp/collect.txt 2>&1
grep -oE '^tests/[^:]+\.py' /tmp/collect.txt | sort -u > /tmp/files.txt
grep -E '^ERROR ' /tmp/collect.txt || true
total=$(wc -l < /tmp/files.txt)
echo "[coverage:${NAME}] ${total} test files have selected tests"
if [[ "${total}" -eq 0 ]]; then
  tail -50 /tmp/collect.txt
  exit 1
fi

: > "${RESULTS}"
i=0
while read -r f; do
  i=$((i + 1))
  echo "--- [coverage:${NAME}] (${i}/${total}) ${f}"
  start=$(date +%s)
  COVERAGE_FILE="${OUT}/coverage_${NAME}.dat.$(printf '%04d' "${i}")" \
    python3 -m pytest -v -rfE --timeout="${TIMEOUT}" -m "${MARKERS}" \
      --cov=vllm_torchtpu --cov-report= "${f}" < /dev/null
  rc=$?
  secs=$(( $(date +%s) - start ))
  echo "${rc} ${secs}s ${f}" >> "${RESULTS}"
  if [[ "${rc}" -ge 128 ]]; then
    echo "[coverage:${NAME}] CRASH: pytest killed by signal $((rc - 128)) in ${f}; this file's in-process coverage is lost"
  elif [[ "${rc}" -ne 0 && "${rc}" -ne 5 ]]; then
    echo "[coverage:${NAME}] pytest exit ${rc} in ${f}"
  fi
done < /tmp/files.txt

echo "--- [coverage:${NAME}] Per-file results (exit code, time, file)"
cat "${RESULTS}"
echo "[coverage:${NAME}] files: ${total}," \
  "ok: $(awk '$1 == 0' "${RESULTS}" | wc -l)," \
  "test failures: $(awk '$1 == 1' "${RESULTS}" | wc -l)," \
  "crashed: $(awk '$1 >= 128' "${RESULTS}" | wc -l)," \
  "other: $(awk '$1 != 0 && $1 != 1 && $1 < 128' "${RESULTS}" | wc -l)"
exit 0
