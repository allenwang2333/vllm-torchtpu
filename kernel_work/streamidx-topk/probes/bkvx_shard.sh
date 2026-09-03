#!/bin/bash
# The extended `bkv_p` sweep + the f64 oracle, in ONE job on the 4-chip pod.
#
# Single **unpinned** process, deliberately -- the same reason `bdt_shard.sh`
# gives: these absolutes have to be comparable with the baseline of record in
# `probes/baseline-device-time.md`, and a fanned-out wall clock is contended.
# The cost is wall clock; the benefit is that the numbers mean something.
#
# Steps, in this order and for these reasons:
#
#   1. `env`    -- md5 read-back of both kernel modules AND the instrument.
#      A `--map-dir` that did not take is otherwise completely silent, and an
#      earlier job in this exercise measured the image's older kernel behind a
#      green JobStatus with rc=1 on all four shards. A green JobStatus is not
#      evidence your code ran.
#   2. `chain`  -- `probe_bkvp_extended.py` on the image's pinned jax 0.10.2.
#      No profiler is involved, so this step cannot SIGSEGV. It carries both
#      the primary timing result AND the f64 correctness oracle, and it runs
#      first so a wall-clock limit on the trace step cannot cost either.
#      Sections flush their JSON after every arm, so a killed step still
#      leaves every arm it finished under $CDK_OUTPUT_DIR.
#   3. `trace`  -- `probe_bkvx_trace.py` from a side venv holding jax/jaxlib
#      0.11.0. The image's pinned 0.10.2 SIGSEGVs against libtpu 0.0.44.1
#      (`PLUGIN_Profiler_Api size: expected 80, got 104`).
#
# The unit test is NOT re-run here: `bdt_shard.sh` ran
# `tests/kernels/test_dsv4_streamindex_topk.py` unmodified against these exact
# module md5s in job j-79a410b8 and it collected 20 and passed 20. Re-running
# it would add nothing and would cost this job the pod for longer. The
# correctness evidence this job adds is the f64 oracle in section B, which is
# the thing the unit test cannot give: the suite parametrises `bkv_p` only
# where `page_size * bkv_p == 128`, so it never runs a production `bkv_p` at
# page size 1024.
#
# No `set -e`: a failed step is a result and must not take the others with it.
P=${P:-/root/probes}
export BDT_INSTRUMENTS=${BDT_INSTRUMENTS:-/root/instruments}
OUT=${CDK_OUTPUT_DIR:-/tmp}/bkvx
mkdir -p "$OUT"
PY_SYS=${PY_SYS:-python}
CHAIN_LE9_TIMEOUT=${CHAIN_LE9_TIMEOUT:-11400}
CHAIN_GT9_TIMEOUT=${CHAIN_GT9_TIMEOUT:-5400}
TRACE_TIMEOUT=${TRACE_TIMEOUT:-4200}
declare -A RC

step() {  # step <name> <cmd...>
  local name=$1; shift
  echo "===== BKVX STEP $name BEGIN $(date -u +%H:%M:%S) ====="
  "$@"
  RC[$name]=$?
  echo "===== BKVX STEP $name END rc=${RC[$name]} $(date -u +%H:%M:%S) ====="
}

# ---- 1. env + md5 read-back --------------------------------------------
echo "===== BKVX ENV $(date -u) ====="
$PY_SYS -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"
pip list 2>/dev/null | grep -iE "^(jax|jaxlib|libtpu|ml.dtypes|numpy)" || true
echo "CDK_OUTPUT_DIR=$CDK_OUTPUT_DIR"
ls -la "$P" | head -40
echo "----- kernel module read-back -----"
$PY_SYS - <<'PYEOF'
import hashlib, os, sys
sys.path.insert(0, os.environ.get("BDT_INSTRUMENTS", "/root/instruments"))
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk
import bench_streamidx_topk as B
for m in (streamindex_topk, sparsecore_topk, B):
    h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
    print(f"MODULE {m.__file__} md5={h}")
print("EXPECT streamindex_topk.py    md5=1e221f10cbf3573aa0d2ae78c3463532")
print("EXPECT sparsecore_topk.py     md5=8bc9e25b8701a53a97704564b9fbc658")
print("EXPECT bench_streamidx_topk.py md5=e8e492d13e59c4a1675cc7a98c5a0813")
import ml_dtypes, numpy
print("ml_dtypes", ml_dtypes.__version__, "numpy", numpy.__version__)
PYEOF
RC[env]=$?

# ---- 2. chain slope + the f64 oracle (the primary result) --------------
#
# TWO processes, split at p = pages_per_seq = 9, and the split is not cosmetic.
# For p > 9 the kernel's page walk indexes `page_indices[batch_idx * 9 + kv_p]`
# with `kv_p` running to p, and `page_indices` is exactly `64 * 9 = 576` long,
# so the last sequences read past the end of the array. Those positions are
# masked to -inf and the OUTPUT is still correct, but the DMA source page
# number is garbage and the fetch itself may fault. Running the p > 9 region in
# its own process means a fault there costs only that region: the p <= 9 run
# has already written its JSON, including the whole f64 oracle.
#
# `p = 2` is forced into the oracle's coverage in both runs regardless of
# --p-min, because it is the reference sigma and the incumbent comparison are
# defined against.
step chain_le9 timeout "$CHAIN_LE9_TIMEOUT" $PY_SYS \
     "$P/probe_bkvp_extended.py" --name bkvp-extended-le9 \
     --p-max 9 --skip A2 ${BKVX_CHAIN_ARGS:-}

step chain_gt9 timeout "$CHAIN_GT9_TIMEOUT" $PY_SYS \
     "$P/probe_bkvp_extended.py" --name bkvp-extended-gt9 \
     --p-min 10 --skip D,E ${BKVX_CHAIN_ARGS:-}

# ---- 3. hlo_stats, from the jax 0.11 side venv -------------------------
echo "===== BKVX STEP j11_venv BEGIN ====="
VENV=/tmp/j11_bkvx
uv venv --system-site-packages "$VENV" && \
  uv pip install --python "$VENV/bin/python" -q jax==0.11.0 jaxlib==0.11.0
RC[j11_venv]=$?
PY_PROF=$VENV/bin/python
if ! $PY_PROF -c "import jax, vllm_torchtpu; print('side venv jax', jax.__version__)"; then
  echo "[bkvx] j11 venv unusable; falling back to the image python (the trace"
  echo "[bkvx] step is then expected to SIGSEGV -- reported as such)"
  PY_PROF=$PY_SYS
  RC[j11_venv]=98
fi
echo "===== BKVX STEP j11_venv END rc=${RC[j11_venv]} ====="

# Split at p = 9 for the same reason the chain step is split: a faulting DMA
# in the p > 9 arms must not take the p <= 9 traces with it. `dec_p2` is the
# control and is traced first.
step trace_le9 timeout "$TRACE_TIMEOUT" $PY_PROF "$P/probe_bkvx_trace.py" \
     --outdir "$OUT/trace" \
     ${BKVX_TRACE_ARGS:---iters 50 --chunk 10 --bkvp 2,1,3,5,9}

step trace_gt9 timeout "$TRACE_TIMEOUT" $PY_PROF "$P/probe_bkvx_trace.py" \
     --outdir "$OUT/trace" \
     ${BKVX_TRACE_GT9_ARGS:---iters 50 --chunk 10 --bkvp 10,16}

# ---- summary ------------------------------------------------------------
echo "===== BKVX STEP TABLE ====="
for k in env chain_le9 chain_gt9 j11_venv trace_le9 trace_gt9; do
  printf "%-12s rc=%s\n" "$k" "${RC[$k]:-notrun}"
done
echo "===== BKVX ARTIFACTS ====="
du -sh "$OUT" 2>/dev/null
find "${CDK_OUTPUT_DIR:-/tmp}" -name "bkvp-extended*.json" | head -10
find "$OUT" -name "*.json" | head -40
find "$OUT" -name "*.xplane.pb" | head -40
exit 0
