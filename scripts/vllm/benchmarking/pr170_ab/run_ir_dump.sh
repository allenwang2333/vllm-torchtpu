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

# Capture post-optimization XLA HLO from the real server, for the two MoE
# permute paths, so the compiled graphs can be diffed offline.
#
# The arm matrix pinned the conc-32 gsm8k collapse on
# ONEHOT_MOE_PERMUTE_THRESHOLD=32768 (one-hot permute) with CUSTOM RPA and
# per-layer KV. This script serves the same model twice -- once each way --
# with XLA dumping on, and packages the after-optimizations HLO.
#
#   onehot  ONEHOT_MOE_PERMUTE_THRESHOLD=32768 USE_MOE_SPARSE_CORE=1 (bad)
#   gather  no MoE envs                                              (good)
#
# It also runs a short gsm8k on each server. That is a validity check, not a
# score: if the onehot mode comes back CLEAN with dumping enabled, then the
# dump changed the compilation and the captured HLO is not the graph that
# fails -- the whole artifact is then worthless and the log must say so.
#
# Usage: run_ir_dump.sh <onehot|gather>
# Runs inside the CI docker image via .buildkite/scripts/run_in_docker.sh.
set -uo pipefail

MODE="${1:-}"
MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
PORT=8000
RESULTS_DIR="/perf_eval_results"
READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
DUMP_DIR="/tmp/xla_dump_${MODE}"

# The MoE lives inside the torch-compiled forward subgraph, which torch_tpu
# names after its first op. Dumping only that module keeps the dump to a
# handful of files per rank instead of ~25k modules / ~27GB: torch_tpu
# compiles every small op as its own XLA module (hundreds of
# tt_jit_copy__as_strided_inverse, tt_jit_compile_mlir_as_strided, ...).
DUMP_MODULE_RE=".*layernorm_L59C0_fused_add_rms_norm.*"

# Every module of interest carries the MoE grouped-GEMM kernel, whose op_name
# encodes the shape: gmm_v2-g_<local_experts>-m_<num_tokens*topk>-k_... . This
# is the content filter that keeps the MoE subgraph at EVERY token bucket.
MOE_MARKER="gmm_v2-g_"

SERVE_ENV=()
case "$MODE" in
  onehot)
    SERVE_ENV=(ONEHOT_MOE_PERMUTE_THRESHOLD=32768 USE_MOE_SPARSE_CORE=1)
    ;;
  gather)
    SERVE_ENV=()
    ;;
  *)
    echo "=== IRDUMP ERROR: unknown mode '${MODE}' ==="
    echo "usage: run_ir_dump.sh <onehot|gather>"
    exit 1
    ;;
esac

mkdir -p "$RESULTS_DIR"
rm -rf "$DUMP_DIR"
mkdir -p "$DUMP_DIR"
COMMIT="${BUILDKITE_COMMIT:-unknown}"
FAIL=0

# Append so an image-level XLA_FLAGS is preserved rather than clobbered.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_dump_to=${DUMP_DIR} --xla_dump_hlo_as_text --xla_dump_include_timestamp=false --xla_dump_hlo_module_re=${DUMP_MODULE_RE}"

echo "=== IRDUMP START MODE=${MODE} COMMIT=${COMMIT} ENV=[${SERVE_ENV[*]-}] ==="
echo "[irdump] XLA_FLAGS=${XLA_FLAGS}"

# shellcheck disable=SC2317  # invoked indirectly via trap
cleanup() {
  echo "=== IRDUMP cleanup: killing vllm server MODE=${MODE} ==="
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_server() {
  local server_log="$RESULTS_DIR/server_irdump_${MODE}.log"
  echo "=== IRDUMP SERVER START MODE=${MODE} ==="
  env "${SERVE_ENV[@]+"${SERVE_ENV[@]}"}" \
    vllm serve "$MODEL" \
    --max-model-len=9216 \
    --max-num-batched-tokens=8192 \
    --no-enable-prefix-caching \
    --max-num-seqs=512 \
    --gpu-memory-utilization=0.9 \
    --block-size=256 \
    --async-scheduling \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser=qwen3_coder \
    --reasoning-parser=qwen3 \
    --limit-mm-per-prompt='{"image": 0, "video": 0}' \
    --kv-cache-dtype=fp8 \
    --tensor-parallel-size=8 \
    --enable-expert-parallel \
    --attention-backend=CUSTOM \
    --port=$PORT > "$server_log" 2>&1 &
  SERVER_PID=$!
  echo "[irdump] server pid=$SERVER_PID, waiting up to ${READY_TIMEOUT_S}s"

  local waited=0
  until curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "=== IRDUMP ERROR: server died during startup MODE=${MODE} ==="
      tail -100 "$server_log"
      exit 1
    fi
    if [ "$waited" -ge "$READY_TIMEOUT_S" ]; then
      echo "=== IRDUMP ERROR: server not ready after ${READY_TIMEOUT_S}s ==="
      tail -100 "$server_log"
      exit 1
    fi
    sleep 15
    waited=$((waited + 15))
    if [ $((waited % 300)) -eq 0 ]; then
      echo "[irdump] still waiting for server... ${waited}s"
    fi
  done
  echo "=== IRDUMP SERVER READY MODE=${MODE} after ~${waited}s ==="
}

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 10
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 10
}

probe() {
  echo "=== IRDUMP PROBE MODE=${MODE} ==="
  curl -s "http://localhost:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"'"$MODEL"'","prompt":"Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:","max_tokens":64,"temperature":0}' \
    | tee -a "$RESULTS_DIR/probes_irdump_${MODE}.json"
  echo ""
  echo "=== IRDUMP PROBE MODE=${MODE} END ==="
}

run_gsm8k() {
  echo "=== IRDUMP GSM8K START MODE=${MODE} ==="
  HF_HUB_OFFLINE=0 lm_eval \
    --model local-completions \
    --model_args "model=$MODEL,base_url=http://localhost:$PORT/v1/completions,num_concurrent=32,max_retries=3,tokenized_requests=False" \
    --tasks gsm8k \
    --seed "0,1234,None,1234" \
    --limit 100 \
    --log_samples \
    --output_path "$RESULTS_DIR/gsm8k_irdump_${MODE}" \
    2>&1 | tee "$RESULTS_DIR/gsm8k_irdump_${MODE}.log" || FAIL=1
  echo "=== IRDUMP GSM8K END MODE=${MODE} ==="
}

# Keep exactly the post-optimization HLO of the MoE subgraph.
#
# Two traps this deliberately avoids, both of which destroyed a previous run's
# artifact (build 255: 8596 matched files, then all decode buckets evicted):
#
#   1. XLA emits THREE files per module -- <name>.after_optimizations.txt (the
#      HLO), plus -buffer-assignment.txt and -buffer-assignment-values.txt.
#      The -values variant is ~100MB, two orders of magnitude larger than the
#      HLO, so a substring match on 'after_optimizations' keeps mostly ballast.
#      Match the exact '.after_optimizations.txt' suffix instead.
#
#   2. Selecting "the N largest" selects AGAINST the decode buckets: bucket
#      16/32/64/128 modules are the smallest, prefill 2048/4096/8192 the
#      largest, so size-ranking drops exactly the shapes under investigation.
#      Select by content (the MoE marker), never by size, and take no cap --
#      restricting the dump at the source keeps the set small already.
prune_dumps() {
  echo "=== IRDUMP PRUNE MODE=${MODE} dir=${DUMP_DIR} ==="
  if [ ! -d "$DUMP_DIR" ]; then
    echo "[irdump] dump dir missing -- XLA wrote nothing"
    return
  fi
  echo "[irdump] files before prune: $(find "$DUMP_DIR" -type f | wc -l)"
  echo "[irdump] bytes before prune: $(du -sb "$DUMP_DIR" | cut -f1)"

  # 1. exact suffix only -- drops the two buffer-assignment variants.
  find "$DUMP_DIR" -type f ! -name '*.after_optimizations.txt' -delete 2>/dev/null

  local kept
  kept=$(find "$DUMP_DIR" -type f | wc -l)
  echo "[irdump] .after_optimizations.txt files: ${kept} bytes: $(du -sb "$DUMP_DIR" | cut -f1)"
  if [ "$kept" -eq 0 ]; then
    echo "[irdump] WARNING: no .after_optimizations.txt files matched; the dump"
    echo "[irdump] naming may differ in this XLA build -- nothing to package"
    return
  fi

  # 2. content filter: keep only modules carrying the MoE grouped-GEMM kernel.
  find "$DUMP_DIR" -type f -print0 \
    | xargs -0 grep -L "$MOE_MARKER" 2>/dev/null \
    | while IFS= read -r victim; do rm -f "$victim"; done

  kept=$(find "$DUMP_DIR" -type f | wc -l)
  echo "[irdump] modules containing ${MOE_MARKER}: ${kept} bytes: $(du -sb "$DUMP_DIR" | cut -f1)"
  if [ "$kept" -eq 0 ]; then
    echo "[irdump] WARNING: no module contained ${MOE_MARKER}; either the MoE"
    echo "[irdump] subgraph was not dumped (check --xla_dump_hlo_module_re="
    echo "[irdump] ${DUMP_MODULE_RE}) or the kernel naming changed"
    return
  fi
}

# Per-file manifest that PROVES which token buckets are in the artifact.
# gmm_v2's m_<N> is num_tokens*topk, and [B,4096] is the hidden-state shape at
# bucket B, so a non-zero count on [32,4096] is direct evidence that the
# bucket-32 graph -- the one that garbles -- was actually captured.
print_manifest() {
  echo "=== IRDUMP MANIFEST MODE=${MODE} ==="
  echo "[irdump] columns: bytes file gmm_shape n32=[32,4096] n64=[64,4096]"
  local f n32 n64 gmm
  while IFS= read -r f; do
    n32=$(grep -c '\[32,4096\]' "$f" 2>/dev/null || true)
    n64=$(grep -c '\[64,4096\]' "$f" 2>/dev/null || true)
    gmm=$(grep -oE "${MOE_MARKER}[0-9]+-m_[0-9]+" "$f" 2>/dev/null | head -1)
    echo "[irdump] $(stat -c %s "$f") $(basename "$f") ${gmm:-none} n32=${n32:-0} n64=${n64:-0}"
  done < <(find "$DUMP_DIR" -type f | sort)
  echo "=== IRDUMP MANIFEST MODE=${MODE} END ==="

  local total32
  total32=$(grep -rlc '\[32,4096\]' "$DUMP_DIR" 2>/dev/null | wc -l)
  if [ "$total32" -eq 0 ]; then
    echo "=== IRDUMP WARNING MODE=${MODE}: no module contains [32,4096] --"
    echo "=== the bucket-32 graph was NOT captured; this artifact cannot"
    echo "=== answer the bucket-32 question ==="
  else
    echo "=== IRDUMP BUCKET32 OK MODE=${MODE} modules=${total32} ==="
  fi
}

package_dumps() {
  local tgz="$RESULTS_DIR/xla_dump_${MODE}.tgz"
  if [ -z "$(find "$DUMP_DIR" -type f -print -quit 2>/dev/null)" ]; then
    echo "[irdump] nothing to package"
    return
  fi
  tar czf "$tgz" -C /tmp "xla_dump_${MODE}" || FAIL=1
  echo "[irdump] packaged $(du -sb "$tgz" 2>/dev/null | cut -f1) bytes -> ${tgz}"
}

start_server
probe
run_gsm8k
stop_server

prune_dumps
print_manifest
package_dumps

# The gsm8k score is the validity check on the whole artifact: onehot must
# still be garbage here, otherwise dumping perturbed the compilation.
SCORE_LINE="$(grep -hE "exact_match" "$RESULTS_DIR/gsm8k_irdump_${MODE}.log" 2>/dev/null | tr '\n' ';')"
echo "=== IRDUMP MODE=${MODE} GSM8K=${SCORE_LINE} ==="
echo "=== IRDUMP DONE MODE=${MODE} FAIL=${FAIL} ==="
exit "$FAIL"
