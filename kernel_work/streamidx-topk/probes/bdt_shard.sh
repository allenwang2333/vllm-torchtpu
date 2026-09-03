#!/bin/bash
# The corrected device-time baseline, in ONE job on the whole 4-chip pod.
#
# Everything runs sequentially in a single **unpinned** process, deliberately.
# `probes/P7-bkvp-and-spill.md` §A4.1 could only report ratios because its four
# shards contended; this run has to produce absolutes comparable with
# `bench_prod.json`, so it does not fan out. And the brief requires Task 2's
# `bkv_p` arms to sit in the same process as Task 1's baseline, so they can be
# read against their own controls rather than across jobs. The cost is wall
# clock; the benefit is that the numbers mean something.
#
# Steps, in this order and for these reasons:
#
#   1. `env`     -- md5 read-back of both kernel modules AND the test file.
#      The image at ec4bcecf predates the kernel modules, so they are mapped
#      in over its checkout; a `--map-dir` that did not take is otherwise
#      completely silent, and an earlier job in this exercise measured the
#      image's older kernel behind a green JobStatus with rc=1 on all four
#      shards. Guard copied from `probes/r2_shard.sh`.
#   2. `pytest`  -- `tests/kernels/test_dsv4_streamindex_topk.py`, UNMODIFIED.
#      Cheap, and it runs early so a wall-clock limit later cannot cost us the
#      correctness evidence. Expect **20 passed**. Collecting 13 means the
#      image's stale test file is being run, which is a red flag and not a
#      pass; collecting 0 exits 5 and pytest calls that a failure.
#   3. `chain`   -- Task 1 and Task 2 by chain slope, on the image's pinned
#      jax 0.10.2. No profiler is involved, so this step cannot SIGSEGV. It is
#      the primary result and runs before the trace step for that reason.
#   4. `trace`   -- Task 1 and Task 2 by `hlo_stats`, from a side venv holding
#      jax/jaxlib 0.11.0 (the image's 0.10.2 SIGSEGVs against libtpu 0.0.44.1,
#      `PLUGIN_Profiler_Api size: expected 80, got 104`). Same pattern as
#      `probes/r2_shard.sh`.
#
# Each timed step is wrapped in `timeout` so one overrun cannot cost the
# others, and the probes flush their JSON after every arm, so a step that is
# killed still leaves every arm it finished under `$CDK_OUTPUT_DIR`.
#
# No `set -e`: a failed step is a result and must not take the others with it.
P=${P:-/root/probes}
export BDT_INSTRUMENTS=${BDT_INSTRUMENTS:-/root/instruments}
OUT=${CDK_OUTPUT_DIR:-/tmp}/bdt
mkdir -p "$OUT"
PY_SYS=${PY_SYS:-python}
CHECKOUT=${CHECKOUT:-/root/cloud-devkit/vllm-torchtpu}
CHAIN_TIMEOUT=${CHAIN_TIMEOUT:-9000}
TRACE_TIMEOUT=${TRACE_TIMEOUT:-5400}
declare -A RC

step() {  # step <name> <cmd...>
  local name=$1; shift
  echo "===== BDT STEP $name BEGIN $(date -u +%H:%M:%S) ====="
  "$@"
  RC[$name]=$?
  echo "===== BDT STEP $name END rc=${RC[$name]} $(date -u +%H:%M:%S) ====="
}

# ---- 1. env + md5 read-back --------------------------------------------
echo "===== BDT ENV $(date -u) ====="
$PY_SYS -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"
pip list 2>/dev/null | grep -iE "^(jax|jaxlib|libtpu)" || true
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
print("EXPECT streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532")
print("EXPECT sparsecore_topk.py  md5=8bc9e25b8701a53a97704564b9fbc658")
print("prod shapes:", sorted(s.name for s in B.SHAPES
                             if s.name.startswith("prod")))
PYEOF
RC[env]=$?
echo "----- test file read-back -----"
md5sum "$CHECKOUT/tests/kernels/test_dsv4_streamindex_topk.py"
echo "EXPECT test_dsv4_streamindex_topk.py md5=becf9f23ace9424a7d67cd3b34de9afd"
echo "EXPECT 20 collected -- 13 means a STALE test file, a red flag not a pass"

# ---- 2. the unmodified unit test ---------------------------------------
echo "===== BDT STEP pytest_deps BEGIN ====="
uv pip install --system -q pytest pytest-mock pytest-timeout parameterized
RC[pytest_deps]=$?
echo "===== BDT STEP pytest_deps END rc=${RC[pytest_deps]} ====="

step pytest bash -c "cd $CHECKOUT && python -m pytest -q -p no:cacheprovider \
    tests/kernels/test_dsv4_streamindex_topk.py 2>&1 \
    | tee $OUT/pytest.log; exit \${PIPESTATUS[0]}"

# ---- 3. chain slope (the primary result) -------------------------------
step chain timeout "$CHAIN_TIMEOUT" $PY_SYS \
     "$P/probe_baseline_device_time.py" ${BDT_CHAIN_ARGS:-}

# ---- 4. hlo_stats, from the jax 0.11 side venv -------------------------
echo "===== BDT STEP j11_venv BEGIN ====="
VENV=/tmp/j11_bdt
uv venv --system-site-packages "$VENV" && \
  uv pip install --python "$VENV/bin/python" -q jax==0.11.0 jaxlib==0.11.0
RC[j11_venv]=$?
PY_PROF=$VENV/bin/python
if ! $PY_PROF -c "import jax, vllm_torchtpu; print('side venv jax', jax.__version__)"; then
  echo "[bdt] j11 venv unusable; falling back to the image python (the trace"
  echo "[bdt] step is then expected to SIGSEGV -- reported as such)"
  PY_PROF=$PY_SYS
  RC[j11_venv]=98
fi
echo "===== BDT STEP j11_venv END rc=${RC[j11_venv]} ====="

step trace timeout "$TRACE_TIMEOUT" $PY_PROF "$P/probe_bdt_trace.py" \
     --outdir "$OUT/trace" \
     ${BDT_TRACE_ARGS:---iters 100 --iters-task2 50 --chunk 10}

# ---- summary ------------------------------------------------------------
echo "===== BDT STEP TABLE ====="
for k in env pytest_deps pytest chain j11_venv trace; do
  printf "%-12s rc=%s\n" "$k" "${RC[$k]:-notrun}"
done
echo "===== BDT PYTEST SUMMARY LINE ====="
tail -3 "$OUT/pytest.log" 2>/dev/null
echo "===== BDT ARTIFACTS ====="
du -sh "$OUT" 2>/dev/null
find "$OUT" -name "*.json" | head -40
find "$OUT" -name "*.xplane.pb" | head -40
exit 0
