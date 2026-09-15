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
# Seven arms, all built from the image's own source tree:
#   golden          the kernel as it ships -- the control, and the denominator
#                   of every ratio
#   golden-aa       a second copy of golden, measured on the same clock. Golden
#                   against itself is the only way to know what a ratio of 1.0
#                   looks like on this machine today.
#   wrong-kernel    golden with the last 8 channels dropped from every q/k
#                   vector -- wrong on both the chunked and the recurrent path
#   subtle-kernel   the same fault at one channel of 128 instead of eight. It
#                   is what says how small a wrong answer this oracle still
#                   separates from golden's own fp32-vs-fp64 gap.
#   pessimal-tile   golden with the prefill tile size forced from 64 to 16. It
#                   changes the chunked path only, which is also what makes it
#                   the demonstration of what the harness does with a candidate
#                   that leaves one frozen config's kernel untouched.
#   pessimal-pipe   golden with the pipeline's buffer count forced from 2 to 1,
#                   which changes both paths
#   pessimal-decode the decode path's two batched head contractions split into
#                   one dot per head -- the same arithmetic in more
#                   instructions, and the only fault aimed at the config that
#                   carries this workload's call weight
#
# All but the first two are injected faults: a benchmark nobody has caught
# detecting a fault is a benchmark nobody should trust.

set -euo pipefail

REPO="${REPO:-/root/torchtpu-vllm}"
ARMS="${ARMS:-/tmp/kda-arms}"
OUT="${OUT:-/perf_eval_results/kda}"
REPLICATE="${REPLICATE:-1}"

mkdir -p "${OUT}"
rm -rf "${ARMS}"

ARM_NAMES="golden golden-aa wrong-kernel subtle-kernel pessimal-tile \
pessimal-pipe pessimal-decode"

for arm in ${ARM_NAMES}; do
  mkdir -p "${ARMS}/${arm}"
  cp -a "${REPO}/src" "${ARMS}/${arm}/src"
  # Bytecode from the image's install would land in the tree hash and differ
  # between arms for reasons that have nothing to do with the kernel.
  find "${ARMS}/${arm}" -name __pycache__ -type d -prune -exec rm -rf {} +
done

patch -p1 -s -d "${ARMS}/wrong-kernel" < "${REPO}/bench/faults/wrong_kernel.patch"
patch -p1 -s -d "${ARMS}/subtle-kernel" < "${REPO}/bench/faults/subtle_kernel.patch"
patch -p1 -s -d "${ARMS}/pessimal-tile" < "${REPO}/bench/faults/pessimal_tile.patch"
patch -p1 -s -d "${ARMS}/pessimal-pipe" < "${REPO}/bench/faults/pessimal_pipeline.patch"
patch -p1 -s -d "${ARMS}/pessimal-decode" \
  < "${REPO}/bench/faults/pessimal_decode.patch"

# A patch that did not apply is a fault arm that is secretly golden, which is
# the one failure this whole exercise exists to catch.
grep -q "out.shape\[-1\] - 8" \
  "${ARMS}/wrong-kernel/src/vllm_torchtpu/kernels/gdn/v3/compute_kda.py"
grep -q "out.shape\[-1\] - 1" \
  "${ARMS}/subtle-kernel/src/vllm_torchtpu/kernels/gdn/v3/compute_kda.py"
grep -q "mixed_tile_size: int = 16" \
  "${ARMS}/pessimal-tile/src/vllm_torchtpu/kernels/gdn/v3/wrapper.py"
grep -q "num_buffers: int = 1" \
  "${ARMS}/pessimal-pipe/src/vllm_torchtpu/kernels/gdn/v3/config.py"
grep -q "for h in range(cfgs.num_v_heads)" \
  "${ARMS}/pessimal-decode/src/vllm_torchtpu/kernels/gdn/v3/compute_kda.py"

echo "--- scoring replicate ${REPLICATE} on $(python3 -c 'import jax; print(jax.devices())')" >&2

python3 "${REPO}/bench/kda_scorer.py" \
  --job-id "${BUILDKITE_JOB_ID:-local}-r${REPLICATE}" \
  --arm "golden=${ARMS}/golden:control" \
  --arm "golden-aa=${ARMS}/golden-aa" \
  --arm "wrong-kernel=${ARMS}/wrong-kernel" \
  --arm "subtle-kernel=${ARMS}/subtle-kernel" \
  --arm "pessimal-tile=${ARMS}/pessimal-tile" \
  --arm "pessimal-pipe=${ARMS}/pessimal-pipe" \
  --arm "pessimal-decode=${ARMS}/pessimal-decode" \
  --config prefill1024 --config decode1 \
  --rounds "${ROUNDS:-3}" \
  > "${OUT}/scoring-r${REPLICATE}.json"

echo "--- payload" >&2
cat "${OUT}/scoring-r${REPLICATE}.json"
