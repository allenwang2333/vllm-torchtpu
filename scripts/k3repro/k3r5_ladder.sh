#!/bin/bash
# The round-5 batched ladder: several one-variable arms on ONE pod slot.
#
# The pod is a single agent, so this is one build with several server lifetimes,
# not one build per lane. Every arm runs the SAME standing ladder, so any two
# arms are comparable to each other and to #365:
#
#   real weights, gs:// + runai_streamer, TP32+EP32, ISL 8192 / OSL 1024
#   lo: max_num_seqs 8   compile_sizes [1,2,4,8,8192]        c = 1, 2, 4, 8
#   hi: max_num_seqs 32  compile_sizes [1,2,4,8,16,32,8192]  c = 16, 32
#   3 reps per cell inside ONE lifetime (rep 1 --keep-alive, reps 2-3 --host)
#
# c=8 is the knee and the only cell whose rep spread (3.48%) exceeded the 2.05%
# band, so it re-runs at 32 prompts. It does that against the lo lifetime that is
# already up (--host), not a second bring-up.
#
# Baseline to beat, #365, same ladder:
#   TTFT  995.2 / 1449.2 / 3287.4 / 4508.3 / 9154.8 / 6121.8 ms
#   TPOT 18.954 / 19.405 / 21.128 / 28.420 / 47.083 / 82.705 ms   (c = 1/2/4/8/16/32)
#
# Usage: k3r5_ladder.sh [arm ...]      default: every arm in ARMS_DEFAULT
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/../vllm/benchmarking/run_benchmarks.sh"
CLEANUP="${SCRIPT_DIR}/../vllm/benchmarking/cleanup_server.sh"
PROBE="${SCRIPT_DIR}/../vllm/benchmarking/k3_completion_probe.py"
OUT="${OUT:-/perf_eval_results/k3r5-ladder}"
REPS="${REPS:-3}"
ARMS_DEFAULT="base k1 hierep c1scale stacked"
rc=0

# --- the arms, one variable each -------------------------------------------
# Each arm is a list of `NAME=VALUE` exports plus two flags: whether it touches
# weights/loading/layout (-> the sanity probe is mandatory: build 299 reported a
# healthy TPOT for a model with half its experts uninitialised) and whether it
# needs the hierarchical-EP shape gate.
arm_env() {
  case "$1" in
    base)    echo "" ;;
    # Lane K: fp8 the two latent projections (moe.py:86,97 quant_config=None),
    # weight 51.38 -> 25.69 MB on 184 instructions/step.
    k1)      echo "${K3_ARM_K1_ENV:-TPU_MOE_LATENT_PROJ_FP8=1}" ;;
    # Lane I2: hierarchical EP, which needs the loader realignment fix to be in
    # the image -- without it the experts are never initialised.
    hierep)  echo "TPU_MOE_HIERARCHICAL_EP=1" ;;
    # Lane I1: KDA o_proj relayout + E8M0 scale narrowing.
    c1scale) echo "${K3_ARM_C1_ENV:-TPU_KDA_OPROJ_RELAYOUT=1 TPU_MOE_SCALE_E8M0_NARROW=1}" ;;
    stacked) echo "$(arm_env k1) $(arm_env hierep) $(arm_env c1scale)" ;;
    *)       echo "__UNKNOWN__" ;;
  esac
}
arm_touches_weights() { case "$1" in base) return 1 ;; *) return 0 ;; esac; }
arm_needs_hierep_gate() { case "$1" in hierep|stacked) return 0 ;; *) return 1 ;; esac; }

say() { echo "--- [ladder] $*"; }

# --- gates ------------------------------------------------------------------
# The compile cache is NOT keyed on LIBTPU_INIT_ARGS, so an arm that changes a
# libtpu flag would silently run the previous arm's executable. Clearing it also
# makes compilation_seconds meaningful: ~45 s fresh, ~23 s on a hit, and an arm
# that reports 23 s did not recompile.
clear_compile_cache() {
  rm -rf /dev/shm/torch_tpu_cache 2>/dev/null || true
  say "compile cache cleared"
}
report_compile_seconds() {  # <server log>
  local s
  s=$(grep -oE 'compilation_seconds[^0-9]*([0-9]+\.?[0-9]*)' "$1" 2>/dev/null | tail -1)
  say "compile: ${s:-<not reported>}"
}
# Void the arm rather than print a line: hierarchical EP that silently falls
# back is an arm measuring nothing, and it looks like a null result.
hierep_gate() {  # <server log>
  local want='w13 (56, 3072, 1792)' hits fallbacks
  hits=$(grep -c "$want" "$1" 2>/dev/null || echo 0)
  fallbacks=$(grep -ciE 'hierarchical.*(fallback|disabled|not supported)' "$1" 2>/dev/null || echo 0)
  say "hierEP gate: '$want' seen ${hits}x (need all 32 ranks), fallbacks=${fallbacks}"
  if [ "$hits" -lt 32 ] || [ "$fallbacks" -gt 0 ]; then
    say "hierEP GATE FAILED -- arm VOID, its numbers must not be reported"
    return 1
  fi
  return 0
}
sanity_probe() {  # <tag>
  [ -f "$PROBE" ] || { say "no probe script; skipping"; return 0; }
  python3 "$PROBE" --host 127.0.0.1 --port 8000 --model moonshotai/Kimi-K3 \
      --out "${OUT}/$1/completions.json" 2>&1 | tail -5
}

# --- one lifetime ------------------------------------------------------------
run_lifetime() {  # <arm> <config> <cells...>  where a cell is "conc:num_prompts"
  local arm="$1" cfg="$2"; shift 2
  local base="${OUT}/${arm}/${cfg}" first=1 cell conc np r
  for r in $(seq 1 "$REPS"); do
    for cell in "$@"; do
      conc="${cell%%:*}"; np="${cell##*:}"
      if [ "$first" = 1 ]; then
        say "${arm}/${cfg} c=${conc} np=${np} rep ${r} (starts the server, keeps it alive)"
        K3_CONC="$conc" K3_NUM_PROMPTS="$np" bash "$BENCH" --config "$cfg" \
            --results-dir "${base}/c${conc}/rep${r}" --keep-alive || rc=1
        first=0
        report_compile_seconds "${base}/c${conc}/rep${r}/server.log"
        if arm_needs_hierep_gate "$arm"; then
          hierep_gate "${base}/c${conc}/rep${r}/server.log" || { rc=1; return 1; }
        fi
        arm_touches_weights "$arm" && sanity_probe "$arm"
      else
        say "${arm}/${cfg} c=${conc} np=${np} rep ${r} (against the running server)"
        K3_CONC="$conc" K3_NUM_PROMPTS="$np" bash "$BENCH" --config "$cfg" \
            --results-dir "${base}/c${conc}/rep${r}" --host 127.0.0.1 || rc=1
      fi
    done
  done
  say "${arm}/${cfg}: stopping server"
  bash "$CLEANUP" || true
}

for arm in ${*:-$ARMS_DEFAULT}; do
  env_list="$(arm_env "$arm")"
  if [ "$env_list" = "__UNKNOWN__" ]; then say "unknown arm '$arm' -- skipped"; rc=1; continue; fi
  echo ""
  say "================ ARM ${arm} ================"
  say "env: ${env_list:-<none: this is the control>}"
  ( # subshell: one arm's exports must not leak into the next
    for kv in $env_list; do export "${kv?}"; done
    clear_compile_cache
    run_lifetime "$arm" kimi-k3-tp32-ep-best-lo 1:16 2:16 4:16 8:32
    [ "${K3_LADDER_HI:-1}" = "1" ] && run_lifetime "$arm" kimi-k3-tp32-ep-best-hi 16:64 32:64
  )
done

echo ""
say "=== ladder summary (median of the per-rep medians) ==="
python3 - "$OUT" <<'PYEOF'
import glob, json, os, statistics, sys
root = sys.argv[1]
cells = {}
for f in glob.glob(os.path.join(root, "**", "isl8192_osl1024_c*.json"), recursive=True):
    if ".warmup" in f:
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    arm = os.path.relpath(f, root).split(os.sep)[0]
    cells.setdefault((arm, d["max_concurrency"]), []).append(
        (d["median_ttft_ms"], d["median_tpot_ms"], d["output_throughput"], d["completed"]))
base = {c: statistics.median(x[1] for x in v) for (a, c), v in cells.items() if a == "base"}
ref = {1: 18.954, 2: 19.405, 4: 21.128, 8: 28.420, 16: 47.083, 32: 82.705}
print(f"{'arm':9}{'conc':>5}{'reps':>5}{'done':>6}{'TTFT ms':>10}{'TPOT ms':>10}"
      f"{'spread':>8}{'vs base':>9}{'vs #365':>9}")
for key in sorted(cells, key=lambda k: (k[0], k[1])):
    v = cells[key]; a, c = key
    t = [x[0] for x in v]; p = [x[1] for x in v]
    sp = (max(p) - min(p)) / statistics.median(p) * 100 if len(p) > 1 else 0.0
    med = statistics.median(p)
    vb = f"{(med / base[c] - 1) * 100:+.2f}%" if c in base and a != "base" else "-"
    vr = f"{(med / ref[c] - 1) * 100:+.2f}%" if c in ref else "-"
    print(f"{a:9}{c:>5}{len(v):>5}{v[0][3]:>6}{statistics.median(t):>10.1f}"
          f"{med:>10.3f}{sp:>7.2f}%{vb:>9}{vr:>9}")
print("\n#365 same ladder: TPOT 18.954 / 19.405 / 21.128 / 28.420 / 47.083 / 82.705")
print("Spread band from #365 is 2.05%; c=8 was 3.48% at 16 prompts, hence 32 here.")
PYEOF
exit "$rc"
