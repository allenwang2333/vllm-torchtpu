#!/bin/bash
# One fan-out shard of the gaps probe.  Usage: bash gaps_shard.sh <plan>
#
# `<plan>` is a key of `gaps_common.PLANS`: e1, e2a, e2b, e3.  One plan per
# chip; the split is documented next to PLANS.
#
# Order matters and there is no `set -e`.  The module read-back runs first and
# is FATAL: `sparsecore_topk.py` is absent from this image entirely and
# `streamindex_topk.py` has a different md5 inside it, so without the
# `--map-dir` of the checkout's `src/` the job imports stale code, exits 0 and
# silently benchmarks the wrong kernel.  `bench_gaps_trace.py` re-checks the
# md5s itself and refuses to measure on a mismatch; this block exists so the
# failure is legible in the log rather than only in a JSON.
#
# The trace step needs the side venv at $PY_PROF (jax/jaxlib 0.11.0).  The
# image's pinned 0.10.2 SIGSEGVs against libtpu 0.0.44.1 on the first
# `jax.profiler.trace` call: `PLUGIN_Profiler_Api size: expected 80, got 104`.
PLAN=${1:-e1}
M=${M:-/root/kw/variants/merged}
OUT=${CDK_OUTPUT_DIR:-/tmp}/gaps
mkdir -p "$OUT"
PY_SYS=${PY_SYS:-python}
PY_PROF=${PY_PROF:-/tmp/j11/bin/python}
TRACE_TIMEOUT=${TRACE_TIMEOUT:-9000}

echo "===== PLAN $PLAN BEGIN $(date -u +%H:%M:%S) ====="
echo "PY_SYS=$PY_SYS PY_PROF=$PY_PROF M=$M OUT=$OUT"

echo "----- module read-back (plan $PLAN) -----"
$PY_SYS - <<PYEOF
import hashlib, os, sys
sys.path.insert(0, "$M")
import gaps_common as GC
import merged_common as MC
mods = GC.load_all_modules()
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc
import vllm_torchtpu
print("vllm_torchtpu.__file__ =", vllm_torchtpu.__file__)
want = {
    "streamindex_topk.py": "1e221f10cbf3573aa0d2ae78c3463532",
    "sparsecore_topk.py": "8bc9e25b8701a53a97704564b9fbc658",
    "bench_streamidx_topk.py": "e8e492d13e59c4a1675cc7a98c5a0813",
    "streamindex_topk_merged.py": "5bc0578427412d56209ef8b5d764fcae",
    "streamindex_topk_cond.py": "951981dbae9e1e8d58709f574eb4abea",
}
bad = 0
for m in [mods["incumbent"], mods["merged"], mods["cond"], sc,
          MC.load_instrument()]:
    b = os.path.basename(m.__file__)
    h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
    w = want.get(b)
    ok = w is None or w == h
    bad += not ok
    print(f"MODULE {m.__file__} md5={h} expected={w} "
          f"{'OK' if ok else 'MISMATCH'}")
print("plan", "$PLAN", "->",
      [(c.name, len(a), r) for c, a, r in GC.PLANS["$PLAN"]])
sys.exit(1 if bad else 0)
PYEOF
RC_MD5=$?
echo "----- read-back rc=$RC_MD5 -----"
if [ "$RC_MD5" != "0" ]; then
  echo "[plan $PLAN] FATAL: on-pod module md5 mismatch; not measuring."
  exit "$RC_MD5"
fi

if ! $PY_PROF -c "import jax, vllm_torchtpu; print('prof venv jax', jax.__version__)"; then
  echo "[plan $PLAN] j11 venv unusable; the trace step is expected to SIGSEGV"
  PY_PROF=$PY_SYS
fi

echo "===== PLAN $PLAN STEP trace BEGIN $(date -u +%H:%M:%S) ====="
timeout "$TRACE_TIMEOUT" $PY_PROF "$M/bench_gaps_trace.py" \
    --plan "$PLAN" --outdir "$OUT/trace" ${TRACE_ARGS:---iters 20 --chunk 5}
RC_TRACE=$?
echo "===== PLAN $PLAN STEP trace END rc=$RC_TRACE $(date -u +%H:%M:%S) ====="

echo "===== PLAN $PLAN SUMMARY ====="
printf "md5   rc=%s\ntrace rc=%s\n" "$RC_MD5" "$RC_TRACE"
echo "xplanes from this plan:"
find "$OUT/trace" -path "*$PLAN-*" -name "*.xplane.pb" 2>/dev/null | wc -l
exit "$RC_TRACE"
