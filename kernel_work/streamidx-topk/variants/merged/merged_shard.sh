#!/bin/bash
# One fan-out shard of the merged-op config sweep.
#
# Usage: bash merged_shard.sh <shard> <num_shards>
#
# Two steps, in this order and for this reason:
#
#   1. `correct` -- `merged` vs `inc` on top-k indices, on the image's pinned
#      jax 0.10.2.  No profiler is involved, so it cannot SIGSEGV, and it runs
#      FIRST so a wall-clock limit later cannot cost us the correctness
#      evidence.
#   2. `trace`   -- one `*.xplane.pb` per (config, arm), from a side venv
#      holding jax/jaxlib 0.11.0.  The image's 0.10.2 SIGSEGVs against libtpu
#      0.0.44.1 on the first `jax.profiler.trace` call
#      (`PLUGIN_Profiler_Api size: expected 80, got 104`).
#
# No `set -e`: a failed step is a result and must not take the other with it.
# Both probes flush their JSON after every config, so a step killed by its
# timeout still leaves everything it finished under $CDK_OUTPUT_DIR.
SHARD=${1:-0}
NSHARDS=${2:-4}
M=${M:-/root/kw/variants/merged}
OUT=${CDK_OUTPUT_DIR:-/tmp}/merged
mkdir -p "$OUT"
PY_SYS=${PY_SYS:-python}
PY_PROF=${PY_PROF:-/tmp/j11/bin/python}
CORRECT_TIMEOUT=${CORRECT_TIMEOUT:-5400}
TRACE_TIMEOUT=${TRACE_TIMEOUT:-9000}

echo "===== SHARD $SHARD/$NSHARDS BEGIN $(date -u +%H:%M:%S) ====="
echo "PY_SYS=$PY_SYS PY_PROF=$PY_PROF M=$M OUT=$OUT"

echo "----- module read-back (shard $SHARD) -----"
$PY_SYS - <<PYEOF
import hashlib, os, sys
sys.path.insert(0, "$M")
import merged_common as MC
mods = [MC.load_incumbent(), MC.load_merged(), MC.load_instrument()]
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as sc
mods.append(sc)
import vllm_torchtpu
print("vllm_torchtpu.__file__ =", vllm_torchtpu.__file__)
for m in mods:
    print(f"MODULE {m.__file__} md5="
          f"{hashlib.md5(open(m.__file__,'rb').read()).hexdigest()}")
print("EXPECT streamindex_topk.py         md5=1e221f10cbf3573aa0d2ae78c3463532")
print("EXPECT sparsecore_topk.py          md5=8bc9e25b8701a53a97704564b9fbc658")
print("EXPECT bench_streamidx_topk.py     md5=e8e492d13e59c4a1675cc7a98c5a0813")
print("configs:", [c.name for c in MC.CONFIGS])
PYEOF
echo "----- read-back rc=$? -----"

echo "===== SHARD $SHARD STEP correct BEGIN $(date -u +%H:%M:%S) ====="
timeout "$CORRECT_TIMEOUT" $PY_SYS "$M/correctness_merged.py" \
    --shard "$SHARD" --num-shards "$NSHARDS" \
    --out "$OUT/correctness.shard$SHARD.json" ${CORRECT_ARGS:-}
RC_CORRECT=$?
echo "===== SHARD $SHARD STEP correct END rc=$RC_CORRECT $(date -u +%H:%M:%S) ====="

if ! $PY_PROF -c "import jax, vllm_torchtpu; print('prof venv jax', jax.__version__)"; then
  echo "[shard $SHARD] j11 venv unusable; the trace step is expected to SIGSEGV"
  PY_PROF=$PY_SYS
fi

echo "===== SHARD $SHARD STEP trace BEGIN $(date -u +%H:%M:%S) ====="
timeout "$TRACE_TIMEOUT" $PY_PROF "$M/bench_merged_trace.py" \
    --shard "$SHARD" --num-shards "$NSHARDS" \
    --outdir "$OUT/trace" ${TRACE_ARGS:---iters 20 --chunk 5}
RC_TRACE=$?
echo "===== SHARD $SHARD STEP trace END rc=$RC_TRACE $(date -u +%H:%M:%S) ====="

echo "===== SHARD $SHARD SUMMARY ====="
printf "correct rc=%s\ntrace   rc=%s\n" "$RC_CORRECT" "$RC_TRACE"
find "$OUT" -name "*.shard$SHARD.json" | head
echo "xplanes from this shard:"
find "$OUT/trace" -name "*.xplane.pb" 2>/dev/null | wc -l
exit 0
