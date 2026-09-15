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
# The end-to-end half of scoping the KDA kernel: what the workload does today,
# how much of a step the kernel is, and whether the microbenchmark tracks it.
#
# Three phases, each a serve + bench pair. `serve` and `bench` are the
# reproduce recipe VERBATIM in every phase; only the phase's own additions
# differ, and they are listed here rather than folded into the command:
#
#   baseline   nothing added. This is the number the campaign is measured
#              against, and the only one quotable as the workload's throughput.
#   profile    the torch profiler armed on the server and `--profile` on the
#              bench, so the run writes an xplane trace. Profiling costs
#              ~11% TTFT, so this phase's own latencies are NOT the baseline.
#   perturbed  the prefill tile size forced from 64 to 16 in the served source
#              -- the same fault the microbenchmark measures as an arm. If the
#              micro number moves and this one does not, the microbenchmark is
#              not a model of the workload.
#   nullkernel the two KDA kernel launches replaced by a zero of the same
#              shape, so the step pays everything except this kernel. The
#              model's output is nonsense and its latency is the floor: the
#              difference from baseline is the whole of what optimizing this
#              kernel could ever win. Nothing after this phase is golden.

set -uo pipefail

REPO="${REPO:-/root/torchtpu-vllm}"
OUT="${OUT:-/perf_eval_results/kda-e2e}"
MODEL="${MODEL:-moonshotai/Kimi-Linear-48B-A3B-Instruct}"
PORT="${PORT:-8000}"
PHASES="${PHASES:-baseline profile perturbed}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-3600}"

mkdir -p "${OUT}"

start_server() {  # $1 = phase, $@ = extra serve args
  local phase="$1"; shift
  echo "--- [${phase}] starting the server" >&2
  vllm serve "${MODEL}" \
    --tensor-parallel-size=8 \
    --data-parallel-size=1 \
    --enable-expert-parallel \
    --max-model-len=8192 \
    --max-num-batched-tokens=4096 \
    --max-num-seqs=256 \
    --gpu-memory-utilization=0.85 \
    --kv-cache-dtype=fp8 \
    --attention-backend CUSTOM \
    --trust-remote-code \
    --block-size 256 \
    --compilation-config '{"compile_sizes":[1,8,16,32,64,128,256,1024,2048,4096]}' \
    --port "${PORT}" \
    "$@" > "${OUT}/server-${phase}.log" 2>&1 &
  SERVER_PID=$!

  local waited=0
  until curl -sf "http://localhost:${PORT}/health" > /dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "+++ [${phase}] the server died after ${waited}s" >&2
      tail -n 60 "${OUT}/server-${phase}.log" >&2
      return 1
    fi
    if [ "${waited}" -ge "${READY_TIMEOUT_S}" ]; then
      echo "+++ [${phase}] the server was not ready in ${waited}s" >&2
      tail -n 60 "${OUT}/server-${phase}.log" >&2
      stop_server
      return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  echo "--- [${phase}] server ready after ${waited}s" >&2
}

stop_server() {
  pkill -f "vllm serve" || true
  sleep 20
  rm -f /tmp/libtpu_lockfile*
}

run_bench() {  # $1 = phase, $@ = extra bench args
  local phase="$1"; shift
  echo "--- [${phase}] benchmarking" >&2
  vllm bench serve \
    --backend vllm \
    --model "${MODEL}" \
    --trust-remote-code \
    --host localhost \
    --port "${PORT}" \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --random-range-ratio 0.0 \
    --num-prompts 32 \
    --max-concurrency 1 \
    --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
    --temperature 0 \
    --seed 42 \
    --save-result \
    --result-filename "${OUT}/kimi_linear_1k_128_c1-${phase}.json" \
    "$@" 2>&1 | tee "${OUT}/bench-${phase}.log"
}

status=0

for phase in ${PHASES}; do
  case "${phase}" in
    baseline)
      start_server baseline && run_bench baseline || status=1
      stop_server
      ;;
    profile)
      start_server profile \
        --profiler-config.profiler torch \
        --profiler-config.torch_profiler_dir "${OUT}/profile" \
        --profiler-config.ignore_frontend true \
        && run_bench profile --profile || status=1
      stop_server
      find "${OUT}/profile" -name '*.xplane.pb' -printf '%s %p\n' >&2 || true
      ;;
    nullkernel)
      patch -p1 -s -d "${REPO}" < "${REPO}/bench/faults/null_kernel.patch"
      grep -q "Ceiling probe" \
        "${REPO}/src/vllm_torchtpu/kernels/gdn/v3/wrapper.py"
      start_server nullkernel && run_bench nullkernel || status=1
      stop_server
      # Put the kernel back, so a later phase in the same job is not measuring
      # this one's patch.
      patch -R -p1 -s -d "${REPO}" < "${REPO}/bench/faults/null_kernel.patch"
      ;;
    perturbed)
      # The served source is the image's own copy, so this patches the kernel
      # the server is about to import. The phase order matters: nothing after
      # this point is golden.
      patch -p1 -s -d "${REPO}" < "${REPO}/bench/faults/pessimal_tile.patch"
      grep -q "mixed_tile_size: int = 16" \
        "${REPO}/src/vllm_torchtpu/kernels/gdn/v3/wrapper.py"
      start_server perturbed && run_bench perturbed || status=1
      stop_server
      ;;
    *)
      echo "+++ unknown phase ${phase}" >&2; status=1 ;;
  esac
done

exit "${status}"
