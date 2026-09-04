#!/bin/bash
# Concurrency sweep for Kimi-K3 on the tpu7x-32 pod, DUMMY weights, run from
# the Ray head container. Two server lifetimes:
#
#   lo: max_num_seqs 8   -> c = 1, 2, 4, 8    (MLA tuned row hits)
#   hi: max_num_seqs 32  -> c = 16, 32        (tuned row cannot hit; see config)
#
# Three repeats per cell, all inside ONE server lifetime per config: the first
# pass keeps the server alive, the next two bench against it with --host. That
# measures bench-to-bench spread without paying three bring-ups, and it is what
# the 2.05% noise band on this rig was derived from.
#
# Dummy weights are used deliberately and their validity is bounded: a matched
# pair at TP32 put dummy within +1.8% TTFT / -1.3% TPOT of real at c=1, but an
# earlier pair measured dummy -22% off real at decode pad 8. So the c=1 row here
# stands in for real; the c>1 rows are dummy-only numbers and must be labelled
# as such.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/../vllm/benchmarking/run_benchmarks.sh"
CLEANUP="${SCRIPT_DIR}/../vllm/benchmarking/cleanup_server.sh"
OUT="${OUT:-/perf_eval_results/k3-main-dummy-sweep}"
REPS="${REPS:-3}"
rc=0

for cfg in kimi-k3-tp32-ep-dummy-sweep-lo kimi-k3-tp32-ep-dummy-sweep-hi; do
  echo "--- ${cfg}: rep 1/${REPS} (starts the server, then keeps it alive)"
  bash "$BENCH" --config "$cfg" --results-dir "${OUT}/${cfg}/rep1" --keep-alive || rc=1
  for r in $(seq 2 "$REPS"); do
    echo "--- ${cfg}: rep ${r}/${REPS} (against the running server)"
    bash "$BENCH" --config "$cfg" --results-dir "${OUT}/${cfg}/rep${r}" \
         --host 127.0.0.1 || rc=1
  done
  echo "--- ${cfg}: stopping server"
  bash "$CLEANUP" || true
done

echo ""
echo "=== sweep summary (median of the per-rep medians) ==="
python3 - "$OUT" <<'PYEOF'
import glob, json, os, statistics, sys, re
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
        (d["median_ttft_ms"], d["median_tpot_ms"], d["output_throughput"]))
print(f"{'cfg':4} {'conc':>4} {'reps':>4} {'TTFT ms':>10} {'spread':>7} "
      f"{'TPOT ms':>9} {'spread':>7} {'out tok/s':>10}")
for (cfg, c) in sorted(cells, key=lambda k: (k[0] != "lo", k[1])):
    v = cells[(cfg, c)]
    t = [x[0] for x in v]; p = [x[1] for x in v]; o = [x[2] for x in v]
    sp = lambda xs: (max(xs) - min(xs)) / statistics.median(xs) * 100 if len(xs) > 1 else 0.0
    print(f"{cfg:4} {c:>4} {len(v):>4} {statistics.median(t):>10.1f} {sp(t):>6.2f}% "
          f"{statistics.median(p):>9.3f} {sp(p):>6.2f}% {statistics.median(o):>10.1f}")
PYEOF
exit "$rc"
