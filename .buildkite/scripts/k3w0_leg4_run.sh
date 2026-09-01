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

# Round-3 leg 4: two profiled server lifetimes, one v7x-32 agent.
#
#   A  kimi-k3-tp32-ep-w0ladder  compile_sizes [1,2,4,8,8192]   (no hier)
#   B  kimi-k3-tp32-ep-w0hier    compile_sizes [16,8192] + hier EP
#
# Two `run_multihost.sh` INVOCATIONS, not two cells of one. W1: Ray's
# `RayWorkerProc.initialize_worker` applies the driver's env with
# `os.environ.setdefault`, so a value baked into the worker containers at
# `docker run` wins and would pin EVERY cell of a shared container set to the
# same arm -- the ladder cell would silently get hierarchical EP too. Separate
# invocations mean separate `docker run`, so the export below reaches lifetime
# B's containers and cannot reach A's.
#
# A runs first: it is the novel measurement, and it is the one whose result
# decides an upstream default. B is one env var from BK #327, which is already
# taken, so B costs one lifetime rather than two.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

disk() { printf '[disk] %s %s\n' "$1" \
    "$(df -P "${HOME}/persist" / 2>/dev/null | awk 'NR>1{printf "%s:%s free(%s used) ", $6, $4, $5}')"; }

cells() {  # results dir, tag -- every number on stdout before it is anywhere else
  python3 - "$1" "$2" <<'PYCELL' 2>/dev/null || echo "[cell] $2 no result json"
import json, os, re, sys
d, cid = sys.argv[1], sys.argv[2]
for fn in sorted(os.listdir(d) if os.path.isdir(d) else []):
    if not re.match(r"^isl\d+_osl\d+_c\d+(\.warmup\d+)?\.json$", fn):
        continue
    try:
        j = json.load(open(os.path.join(d, fn)))
    except Exception:
        continue
    print(f"[cell] {cid} {fn} TPOT {j.get('median_tpot_ms'):.4f} "
          f"TTFT {j.get('median_ttft_ms'):.1f} ITL {j.get('median_itl_ms'):.4f} "
          f"tok/s {j.get('total_token_throughput'):.2f} completed {j.get('completed')}")
PYCELL
}

verdict() {  # results dir -- the arm's own positive controls, on stdout
  local d="perf_eval_results/$1"
  grep -aom1 "compile_sizes': \[[^]]*\]" "$d/server.log" 2>/dev/null | sed 's/^/[bucket] /' \
    || echo "[bucket] compile_sizes NOT FOUND"
  grep -aoh 'padded_total_num_scheduled_tokens.: [0-9]*' "$d/server.log" 2>/dev/null \
    | sed 's/.*: //' | sort -un | tr '\n' ',' | sed 's/^/[bucket] padded_tokens seen: /;s/$/\n/'
  grep -hoE 'MoE expert bank on this rank: w13 \([0-9, ]+\) \(ep_size=[0-9]+, moe_tp_size=[0-9]+[^)]*\)' \
    "$d/server.log" 2>/dev/null | sort | uniq -c | sed 's/^/[hier] bank /'
  grep -hoE '\[sharded-ep-load\] ep_size=[0-9]+ [^:]*: this rank keeps [0-9]+/[0-9]+' \
    "$d/server.log" 2>/dev/null | sed -E 's/ep_rank=[0-9]+/ep_rank=*/' | sort | uniq -c \
    | sed 's/^/[hier] loader /'
  grep -c 'falling back to flat expert parallelism' "$d/server.log" 2>/dev/null \
    | sed 's/^/[hier] fallback lines: /'
  grep -am1 '^\[cmd\] vllm serve' "$d/server.log" 2>/dev/null | sed 's/^/[serve] /' \
    || grep -am1 'vllm serve ' "$d/server.log" 2>/dev/null | sed 's/^/[serve] /'
  # Boot cost, first-class: a config that is faster per token and minutes
  # slower to start is not shippable, and which of the two it is IS the result.
  grep -aoE 'Server ready on port [0-9]+ \([0-9]+s\)' "$d/server.log" 2>/dev/null \
    | sed 's/^/[boot] /'
  grep -aom1 'Using token paddings: \[[^]]*\]' "$d/server.log" 2>/dev/null \
    | sed 's/^/[boot] /'
  printf '[boot] compiled graph variants (distinct compile-size shapes): %s\n' \
    "$(grep -aoE 'artifact_compile_range_[0-9]+_[0-9]+' "$d/server.log" 2>/dev/null \
       | sort -u | wc -l)"
}

run_one() {  # config, tag
  echo "--- [leg4] LIFETIME $2 : $1 : $(date -u +%H:%M:%S)"
  bash "${SCRIPT_DIR}/run_multihost.sh" bash ./scripts/vllm/benchmarking/run_eval_flow.sh \
      --config "$1" --results-dir "/perf_eval_results/$1"
  local rc=$?
  echo "--- [leg4] $1 driver exited ${rc}; collecting regardless"
  verdict "$1"
  cells "perf_eval_results/$1" "$1"
  bash "${SCRIPT_DIR}/k3_profile_collect.sh" "perf_eval_results/$1" "$2" || true
  disk "after $1"
  return "$rc"
}

disk start
unset TPU_MOE_HIERARCHICAL_EP
run_one kimi-k3-tp32-ep-w0ladder k3lad
a_rc=$?

export TPU_MOE_HIERARCHICAL_EP=1
run_one kimi-k3-tp32-ep-w0hier k3hier
b_rc=$?

echo "--- [leg4] ladder exited ${a_rc}, hier exited ${b_rc}"
exit "$a_rc"
