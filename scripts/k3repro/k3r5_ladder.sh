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
# Cost order, and it is not arbitrary: `base` is first because #365 is on a
# different branch, a different image and a c=8 cell with a different prompt
# count, so without a control on THIS image every other arm is a cross-build
# comparison. The lossless pair is measured before the lossy arm, because
# c1+e8m0 can land on ladder+probe alone while k1 needs an accuracy run.
# c1e8m0 is NOT in the default list: the E8M0 half showed a perfect 36.75 MiB
# operand in HLO while the LLO DMA descriptor stayed at 43,008 granules -- not
# one byte saved, because the scale's degenerate sublane dim pads back up under
# the narrow tiling. Until that lane confirms the granule count falls, a
# "c1e8m0" arm would measure C1 alone and the difference would be reported as
# an interaction. Run `c1` until then, and swap it for `c1e8m0` on confirmation.
# k1 is HELD: in arm C's HLO the fp8 parameter enters as T(32,128)(4,1) while the
# consumer wants T(8,128)(4,1), so a full-tensor retile `copy` lands on the weight
# where the bf16 baseline gets a free bitcast. If that copy is an HBM round-trip it
# is 25.69 R + 25.69 W + 25.69 R = 77.07 MB against the baseline 51.38 MB -- a 1.50x
# REGRESSION, not a win. Both projections are also genuinely M=1 at decode pad 1, so
# XLA may take the matvec off the MXU onto a VPU broadcast/multiply/reduce. Both are
# CPU-checkable in minutes; a ~50 min pod lifetime is the wrong way to find out.
ARMS_DEFAULT="base c1_compactscale hierep stacked"
rc=0

# --- the arms, one variable each -------------------------------------------
# Every lane's change is a SOURCE patch carried on this branch behind its own
# env gate, so one image serves every arm and an arm is still exactly one
# variable. Every gate defaults OFF, which makes `base` byte-identical in
# behaviour to unmodified 76d747572.
#
# The gate must change the compiled graph when it flips. That is why the cache
# is cleared per arm and compilation_seconds is printed with the gate values:
# "the flag was set" and "the program changed" are two different claims and an
# env-gated patch that silently reuses the previous arm's executable is exactly
# the r3 A/B that had to be voided.
K3_GATES="TPU_K3_FP8_LATENT_PROJ K3_KDA_OPROJ_RELAYOUT TPU_MOE_COMPACT_SCALE K3_HIEREP_LOADER_FIX TPU_MOE_HIERARCHICAL_EP"
arm_env() {
  case "$1" in
    base)     echo "" ;;                                    # the control, all gates off
    c1)       echo "K3_KDA_OPROJ_RELAYOUT=1" ;;             # lane I1: KDA o_proj load-time relayout
    # Compact MoE scale. e8m0 over bf16 because it is the same diff for twice the
    # win (43,008 -> 21,504 -> 10,752 granules) and equally lossless: E8M0 encodes
    # powers of two, which both float8_e8m0fnu and bf16 hold exactly, so the
    # arithmetic is unchanged and no accuracy run is owed.
    e8m0)     echo "TPU_MOE_COMPACT_SCALE=e8m0" ;;
    c1e8m0|c1_compactscale)
              echo "K3_KDA_OPROJ_RELAYOUT=1 TPU_MOE_COMPACT_SCALE=e8m0" ;;  # the lossless pair: lands on ladder + probe, no eval owed
    k1)       echo "TPU_K3_FP8_LATENT_PROJ=1" ;;         # lane K: fp8 the two latent projections. LOSSY -- needs an accuracy run
    hierep)   echo "TPU_MOE_HIERARCHICAL_EP=1 K3_HIEREP_LOADER_FIX=1" ;;  # lane I2: env + loader patch
    stacked)  echo "$(arm_env c1_compactscale) $(arm_env k1) $(arm_env hierep)" ;;
    *)        echo "__UNKNOWN__" ;;
  esac
}
# hierEP is two changes wearing one name. The loader fix is a 3-line CORRECTNESS
# bug (BK #299 served a model whose experts were never initialised) and lands on
# its own unit test with no pod time at all; the env flag is a PERFORMANCE lever
# worth -7.3..-9.6% in r3. Reporting them together would make a 3-line bug fix
# look like it needs a pod run to justify.
arm_notes() {
  case "$1" in
    stacked) echo "${K3_ARM_NOTES:-}Expect sub-additive: 77% of the compact-scale win is the STRAGGLER, not the GEMM -- shrinking scale bytes shrinks the spacing of the expert ladder, and the wait is that ladder's spread. hierEP attacks the same spread from the other side, so c1_compactscale x hierep is the pair most likely to overlap; report that pair's shortfall separately from the aggregate. Also: hierEP is a correctness fix (3-line loader realignment, landable on its unit test alone) PLUS a performance flag; the ladder prices only the flag." ;;
    hierep) echo "${K3_ARM_NOTES:-}hierEP is a correctness fix (3-line loader realignment, landable on its unit test alone, no pod time required) PLUS a performance flag (TPU_MOE_HIERARCHICAL_EP). The ladder number here prices only the flag; the fix must not be gated on it." ;;
    *) echo "${K3_ARM_NOTES:-}" ;;
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
# Prints the gate values NEXT TO the compile time on purpose: an arm is only
# honest if both "the flag was set" and "the program was rebuilt" are true.
ARM_COMPILE_SECONDS=""
report_compile_seconds() {  # <server log>
  local s
  s=$(grep -oE 'compilation_seconds[^0-9]*([0-9]+\.?[0-9]*)' "$1" 2>/dev/null | tail -1)
  local g gates=""
  for g in $K3_GATES; do gates="${gates}${g}=${!g:-0} "; done
  ARM_COMPILE_SECONDS="$(printf '%s' "${s:-}" | grep -oE '[0-9]+\.?[0-9]*' | tail -1)"
  say "compile: ${s:-<not reported>} | gates: ${gates}"
}
# Void the arm rather than print a line: hierarchical EP that silently falls
# back is an arm measuring nothing, and it reads as a null result.
#
# The bank-shape assertion is the whole gate. Grepping for the "falling back to
# flat expert parallelism" warning is NOT a second check: that line comes from
# `logger.warning_once`, which defaults to scope="local" -- local rank 0 of each
# host, 4 ranks of 32 -- so its ABSENCE says nothing about the other 28 and a
# silent fallback on 28 ranks would pass. It is still counted and printed,
# because a non-zero count is informative even though a zero is not.
hierep_gate() {  # <server log>
  local want='w13 (56, 3072, 1792) (ep_size=16, moe_tp_size=2)' hits warned
  hits=$(grep -cF "$want" "$1" 2>/dev/null || echo 0)
  warned=$(grep -ciE 'falling back to flat expert parallelism' "$1" 2>/dev/null || echo 0)
  say "hierEP gate: bank shape seen on ${hits}/32 ranks; fallback warnings ${warned} (host-scoped, informative only when > 0)"
  if [ "$hits" -lt 32 ]; then
    say "hierEP GATE FAILED -- arm VOID, its numbers must not be reported"
    return 1
  fi
  [ "$warned" -gt 0 ] && say "hierEP: fallback warned on at least one host -- treat the arm as VOID"
  [ "$warned" -gt 0 ] && return 1
  return 0
}
sanity_probe() {  # <tag>
  [ -f "$PROBE" ] || { say "no probe script; skipping"; return 0; }
  local o="${OUT}/$1/probe.txt"
  python3 "$PROBE" --host 127.0.0.1 --port 8000 --model moonshotai/Kimi-K3 \
      --out "${OUT}/$1/completions.json" 2>&1 | tee "$o" | tail -5
  ARM_WORST_RATIO="$(grep -oE 'worst_distinct_ratio[=: ]+[0-9.]+' "$o" | grep -oE '[0-9.]+$' | tail -1)"
  ARM_SANITY="$(grep -oE 'verdict[=: ]+[A-Z_]+' "$o" | grep -oE '[A-Z_]+$' | tail -1)"
  ARM_SANITY="${ARM_SANITY:-UNKNOWN}"
  say "sanity: ${ARM_SANITY} worst_distinct_ratio=${ARM_WORST_RATIO:-?}"
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
    run_lifetime "$arm" kimi-k3-tp32-ep-best-lo ${K3_CELLS_LO:-1:8 2:8 4:16 8:32}
    # The hi rung needs its OWN lifetime (max_num_seqs 8 -> 32 is a server flag),
    # i.e. a second ~20-50 min bring-up per arm. On a one-agent queue that is the
    # dominant cost, so hi runs only for the arms whose claim reaches c>=16 --
    # by default the control and the stack.
    case " ${K3_ARMS_WITH_HI:-base stacked} " in
      *" $arm "*) run_lifetime "$arm" kimi-k3-tp32-ep-best-hi ${K3_CELLS_HI:-16:32 32:32} ;;
      *) say "hi rung skipped for arm ${arm} (K3_ARMS_WITH_HI='${K3_ARMS_WITH_HI:-base stacked}')" ;;
    esac
    # One file per arm, in the reviewer's contract shape. The reviewer is the
    # single writer to the operator's tracker; lanes emit JSON and never touch it.
    python3 "${SCRIPT_DIR}/k3r5_arm_json.py" "$arm" "${OUT}/${arm}" "${OUT}/ladder/${arm}.json" \
        "sha=${BUILDKITE_COMMIT:-}" "weights=${K3_WEIGHTS:-real}" "gate_env=${env_list}" \
        "compilation_seconds=${ARM_COMPILE_SECONDS:-}" "cache_cleared=true" \
        "sanity_verdict=${ARM_SANITY:-NOT_RUN}" \
        ${ARM_WORST_RATIO:+"worst_distinct_ratio=${ARM_WORST_RATIO}"} \
        "notes=$(arm_notes "$arm")" || say "arm json failed for ${arm}"
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
