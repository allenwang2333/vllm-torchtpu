#!/bin/bash
# Concurrency sweep for Kimi-K3 on the tpu7x-32 pod, REAL weights, run from the
# Ray head container. Possible on stock main only since #539 landed
# `runai_repull`; before that every off-head rank died at worker init resolving
# the head-local Run:AI cache dir (dev build 362).
#
# Two server lifetimes:
#   lo: max_num_seqs 8   -> c = 1, 2, 4, 8   (MLA tuned row hits)
#   hi: max_num_seqs 32  -> c = 16, 32       (tuned row cannot hit; see config)
#
# Three repeats per cell inside ONE server lifetime each: the first pass keeps
# the server alive, the rest bench against it with --host. That measures
# bench-to-bench spread without paying three 1561 GB bring-ups.
#
# Comparison point is main's own committed nightly baseline for the same model
# and shape: baselines/perf/kimi-k3-tp32-ep.baseline.json (TTFT 4038.1 ms,
# TPOT 30.899 ms at c=1). Note that baseline uses 64 prompts and this sweep 16
# at c<=8; both report medians, whose run-to-run spread on this rig is <1%.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/../vllm/benchmarking/run_benchmarks.sh"
CLEANUP="${SCRIPT_DIR}/../vllm/benchmarking/cleanup_server.sh"
OUT="${OUT:-/perf_eval_results/k3-main-real-sweep}"
REPS="${REPS:-3}"
CONFIGS="${CONFIGS:-kimi-k3-tp32-ep-best-lo kimi-k3-tp32-ep-best-hi}"
rc=0

for cfg in $CONFIGS; do
  echo "--- ${cfg}: rep 1/${REPS} (starts the server, then keeps it alive)"
  bash "$BENCH" --config "$cfg" --results-dir "${OUT}/${cfg}/rep1" --keep-alive || rc=1
  for r in $(seq 2 "$REPS"); do
    echo "--- ${cfg}: rep ${r}/${REPS} (against the running server)"
    bash "$BENCH" --config "$cfg" --results-dir "${OUT}/${cfg}/rep${r}" --host 127.0.0.1 || rc=1
  done
  echo "--- ${cfg}: stopping server"
  bash "$CLEANUP" || true
done

echo ""
echo "=== sweep summary (median of the per-rep medians) ==="
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
    cfg = "lo" if "-lo/" in f else "hi"
    cells.setdefault((cfg, d["max_concurrency"]), []).append(
        (d["median_ttft_ms"], d["median_tpot_ms"], d["output_throughput"], d["completed"]))
print(f"{'cell':5}{'conc':>5}{'reps':>5}{'done':>6}{'TTFT ms':>11}{'spread':>8}"
      f"{'TPOT ms':>10}{'spread':>8}{'out tok/s':>11}")
for key in sorted(cells, key=lambda k: (k[0] != "lo", k[1])):
    v = cells[key]
    t = [x[0] for x in v]; p = [x[1] for x in v]; o = [x[2] for x in v]
    sp = lambda xs: (max(xs) - min(xs)) / statistics.median(xs) * 100 if len(xs) > 1 else 0.0
    print(f"{key[0]:5}{key[1]:>5}{len(v):>5}{v[0][3]:>6}{statistics.median(t):>11.1f}"
          f"{sp(t):>7.2f}%{statistics.median(p):>10.3f}{sp(p):>7.2f}%{statistics.median(o):>11.1f}")
print()
print("main's committed nightly baseline, same model/shape, c=1, 64 prompts:")
print("  TTFT 4038.1 ms   TPOT 30.899 ms   out 28.7 tok/s")
PYEOF
exit "$rc"
