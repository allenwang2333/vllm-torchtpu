#!/usr/bin/env bash
# Diagnostic, not a test. Boots the compilation-cache-e2e model repeatedly
# against one Tier-3 cache, to find out which program roughly 1 boot in 10
# compiles during weight loading (414a3bcdb281ad91_761b6dd1fdb0fa32_*.bin in the
# flaky Test 5 diffs).
#
# DIAG_DUMP_ON_CACHE_MISS=1 makes the worker log the StableHLO, with source
# locations, of every program new to the process, tagged with its cache key.
# Every boot's log goes under $ARTIFACTS_DIR/ccdiag/<host>/ with the keys it
# dumped, one row per boot in summary.tsv, and the dump of the target program
# in target_graph.txt for the boots that built it.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
WARM_BOOTS="${DIAG_WARM_BOOTS:-10}"
TARGET_KEY="414a3bcdb281ad91"
OUT="$(pwd)/${ARTIFACTS_DIR:-perf_eval_results}/ccdiag/$(hostname)"
CACHE_DIR="$(mktemp -d /tmp/ccdiag_XXXXXX)"
TIER3_DIR="${CACHE_DIR}/torch_tpu_tier3"
KNOWN="${CACHE_DIR}/tier3_known.txt"

# The same cache setup as run_compilation_cache_e2e.sh.
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${TIER3_DIR}"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm_cache"
export VLLM_TARGET_DEVICE="tpu"
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_XLA_CHECK_RECOMPILATION=1
export DIAG_DUMP_ON_CACHE_MISS=1

mkdir -p "${OUT}" "${TIER3_DIR}" "${VLLM_CACHE_ROOT}"
: > "${KNOWN}"
printf 'boot\trc\tsecs\tfirst_report_graphs\tcompile_secs\tdumped_keys\ttarget_dumped\tnew_tier3_files\n' \
  > "${OUT}/summary.tsv"

clean_tpu_processes() {
  for d in /proc/[0-9]*; do
    if [ -f "$d/cmdline" ]; then
      pid=$(basename "$d")
      cmd=$(tr "\0" " " < "$d/cmdline" 2>/dev/null || true)
      if echo "$cmd" | grep -qiE "python|vllm|enginecore"; then
        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
          kill -9 "$pid" 2>/dev/null || true
        fi
      fi
    fi
  done
  rm -rf /tmp/libtpu* /tmp/tpu_logs* || true
}

boot() {
  local name=$1
  local dir="${OUT}/${name}"
  mkdir -p "${dir}"
  clean_tpu_processes
  local start=${SECONDS}
  python3 examples/offline_inference.py \
    --model "${MODEL}" \
    --max-model-len 256 --max-num-batched-tokens 256 --max-tokens 8 \
    --prompt "The capital of France is" > "${dir}/log.txt" 2>&1
  local rc=$?
  local secs=$((SECONDS - start))
  # As run_compilation_cache_e2e.sh does, so a late Tier-3 write is counted
  # against the boot that made it.
  sync && sleep 10

  find "${TIER3_DIR}" -type f -name '*.bin' -printf '%P\n' | sort > "${dir}/tier3_after.txt"
  local new
  new=$(comm -13 "${KNOWN}" "${dir}/tier3_after.txt")

  # The first recompilation report covers everything up to the first warmup
  # case, weight loading included: 33 graphs normally, 34 when the extra
  # program was built.
  local graphs compile_secs
  graphs=$(grep -m1 -oE 'Total number of cached graphs: [0-9]+, new: [0-9]+, case: compute_selected_logits' \
    "${dir}/log.txt" | grep -oE '[0-9]+' | head -1)
  compile_secs=$(grep -oE 'first read since the last check, [0-9.]+s compiling' "${dir}/log.txt" \
    | grep -oE '[0-9.]+s' | tr -d s | awk '{s += $1} END {printf "%.3f", s}')

  if [[ "${name}" == cold ]]; then
    cp "${dir}/tier3_after.txt" "${KNOWN}"
    new="($(wc -l < "${KNOWN}") files)"
  elif [[ -n "${new}" ]]; then
    while read -r f; do rm -f "${TIER3_DIR}/${f}"; done <<< "${new}"
  fi

  grep -oE 'due to cache miss for key: .{0,200}' "${dir}/log.txt" | sort -u > "${dir}/keys.txt"
  local target=no
  if grep -q "due to cache miss for key: .*${TARGET_KEY}" "${dir}/log.txt"; then
    target=yes
    # From the target's dump header up to the next program's dump or the next
    # Python log line.
    awk -v key="${TARGET_KEY}" '
      /due to cache miss for key: / { on = index($0, key) > 0 }
      /^\(EngineCore|^(INFO|WARNING|ERROR) / { on = 0 }
      on' "${dir}/log.txt" > "${dir}/target_graph.txt"
  fi
  gzip -f "${dir}/log.txt"
  local row
  row=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "${name}" "${rc}" "${secs}" "${graphs:-?}" \
    "${compile_secs}" "$(wc -l < "${dir}/keys.txt")" "${target}" "$(echo "${new}" | tr '\n' ' ')")
  echo "${row}" >> "${OUT}/summary.tsv"
  echo "=== ${row}"
  return "${rc}"
}

echo "--- Cold boot (${MODEL})"
if ! boot cold; then
  echo "Cold boot failed; last lines:"
  zcat "${OUT}/cold/log.txt.gz" | grep -v 'compilation_cache.cc' | tail -50
  exit 1
fi

for i in $(seq -w 1 "${WARM_BOOTS}"); do
  echo "--- Warm boot ${i}/${WARM_BOOTS}"
  boot "warm${i}" || zcat "${OUT}/warm${i}/log.txt.gz" | grep -v 'compilation_cache.cc' | tail -30
done

echo "--- Summary"
column -t -s $'\t' "${OUT}/summary.tsv" || cat "${OUT}/summary.tsv"
for f in "${OUT}"/*/target_graph.txt; do
  [[ -f "${f}" ]] || continue
  echo "--- ${f#"${OUT}"/} (first 80 of $(wc -l < "${f}") lines)"
  head -80 "${f}"
done
rm -rf "${CACHE_DIR}"
