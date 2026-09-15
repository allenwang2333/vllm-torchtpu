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
#
# Materialize the KDA microbenchmark's arms and score them in one job.
#
# Four arms, all built from the image's own source tree:
#   golden         the kernel as it ships -- the control, and the denominator
#                  of every ratio
#   golden-aa      a second copy of golden, measured on the same clock. Golden
#                  against itself is the only way to know what a ratio of 1.0
#                  looks like on this machine today.
#   wrong-kernel   golden with one k-block dropped from the chunked solve
#   pessimal-tile  golden with the prefill tile size forced from 64 to 16
#
# The last two are injected faults: a benchmark nobody has caught detecting a
# fault is a benchmark nobody should trust.

set -euo pipefail

REPO="${REPO:-/root/torchtpu-vllm}"
ARMS="${ARMS:-/tmp/kda-arms}"
OUT="${OUT:-/perf_eval_results/kda}"
REPLICATE="${REPLICATE:-1}"

mkdir -p "${OUT}"
rm -rf "${ARMS}"

for arm in golden golden-aa wrong-kernel pessimal-tile; do
  mkdir -p "${ARMS}/${arm}"
  cp -a "${REPO}/src" "${ARMS}/${arm}/src"
  # Bytecode from the image's install would land in the tree hash and differ
  # between arms for reasons that have nothing to do with the kernel.
  find "${ARMS}/${arm}" -name __pycache__ -type d -prune -exec rm -rf {} +
done

patch -p1 -s -d "${ARMS}/wrong-kernel" < "${REPO}/bench/faults/wrong_kernel.patch"
patch -p1 -s -d "${ARMS}/pessimal-tile" < "${REPO}/bench/faults/pessimal_tile.patch"

# A patch that did not apply is a fault arm that is secretly golden, which is
# the one failure this whole exercise exists to catch.
grep -q "0 < c_b < r_b" \
  "${ARMS}/wrong-kernel/src/vllm_torchtpu/kernels/gdn/v3/compute_kda.py"
grep -q "mixed_tile_size: int = 16" \
  "${ARMS}/pessimal-tile/src/vllm_torchtpu/kernels/gdn/v3/wrapper.py"

echo "--- scoring replicate ${REPLICATE} on $(python3 -c 'import jax; print(jax.devices())')" >&2

python3 "${REPO}/bench/kda_scorer.py" \
  --job-id "${BUILDKITE_JOB_ID:-local}-r${REPLICATE}" \
  --arm "golden=${ARMS}/golden:control" \
  --arm "golden-aa=${ARMS}/golden-aa" \
  --arm "wrong-kernel=${ARMS}/wrong-kernel" \
  --arm "pessimal-tile=${ARMS}/pessimal-tile" \
  --config prefill1024 --config decode1 \
  > "${OUT}/scoring-r${REPLICATE}.json"

echo "--- payload" >&2
cat "${OUT}/scoring-r${REPLICATE}.json"
