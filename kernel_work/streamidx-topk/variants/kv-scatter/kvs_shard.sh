#!/bin/bash
# One fan-out shard of the KV-scatter relayout A/B.
#
#   bash kvs_shard.sh <step> [args...]
#
# Steps:
#   hlo     -- `--mode layout` then `--mode hlo` over every (config, arm).
#              Compiler only; runs on the image's pinned jax 0.10.2, which
#              cannot SIGSEGV because no profiler is involved.  This is the
#              pass condition (does the reshape become a bitcast), so it runs
#              on its own chip and does not share a shard with anything that
#              could time out first.
#   correct -- byte-equality and top-k equality.  Also jax 0.10.2.
#   trace   -- one `*.xplane.pb` per (config, arm), from a side venv holding
#              jax/jaxlib 0.11.0.  The image's 0.10.2 SIGSEGVs on the first
#              `jax.profiler.trace` call against libtpu 0.0.44.1
#              (`PLUGIN_Profiler_Api size: expected 80, got 104`).
#              Extra args are passed through, so a shard can take one config.
#
# No `set -e`: a failed step is a result and must not take the rest of the log
# with it.  Every probe flushes its JSON as it goes, so a step killed by its
# timeout still leaves what it finished under $CDK_OUTPUT_DIR.
#
# Every step first reads back the md5 of the kernel modules it will import.
# The image ships a DIFFERENT `streamindex_topk.py` and NO `sparsecore_topk.py`
# at all, so a job whose `--map-dir` did not take imports stale code, exits 0
# and measures the wrong thing.  `kvs_common.report_md5s` compares against the
# checkout's values and every probe refuses to run on a mismatch.

STEP=${1:-hlo}
shift 2>/dev/null
KW=${KW:-/root/kw}
VD=$KW/variants/kv-scatter
REPO=/root/cloud-devkit/vllm-torchtpu
PY_SYS=${PY_SYS:-python}
OUT=${CDK_OUTPUT_DIR:-/tmp}/kvs
LABEL=${TPU_FANOUT_SHARD:-x}
mkdir -p "$OUT"

echo "===== kv-scatter shard $LABEL step=$STEP $(date -u +%H:%M:%S) ====="
$PY_SYS -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
md5sum "$VD"/*.py "$VD"/*.sh
echo "----- on-pod kernel md5 read-back -----"
md5sum "$REPO"/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py \
       "$REPO"/src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py \
       "$REPO"/src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py
echo "EXPECT streamindex_topk.py md5=1e221f10cbf3573aa0d2ae78c3463532"
echo "EXPECT sparsecore_topk.py  md5=8bc9e25b8701a53a97704564b9fbc658"

cd "$REPO" || exit 9
RC=0

run () {
  local tag=$1; shift
  echo "----- kv-scatter $LABEL $tag BEGIN $(date -u +%H:%M:%S) -----"
  "$@" 2>&1
  local r=$?
  echo "----- kv-scatter $LABEL $tag END rc=$r $(date -u +%H:%M:%S) -----"
  [ $r -ne 0 ] && RC=$((RC + 1))
  return $r
}

case "$STEP" in
  hlo)
    run layout timeout "${HLO_TIMEOUT:-3600}" $PY_SYS "$VD/probe_kvs_hlo.py" \
        --mode layout --root "$OUT"
    run hlo    timeout "${HLO_TIMEOUT:-3600}" $PY_SYS "$VD/probe_kvs_hlo.py" \
        --mode hlo --root "$OUT" "$@"
    ;;
  correct)
    run correct timeout "${CORRECT_TIMEOUT:-5400}" $PY_SYS \
        "$VD/probe_kvs_correct.py" --json "$OUT/correct.json" "$@"
    ;;
  trace)
    VENV=/tmp/j11_kvs_$LABEL
    echo "----- j11 venv $VENV -----"
    uv venv --system-site-packages "$VENV" && \
      uv pip install --python "$VENV/bin/python" -q jax==0.11.0 jaxlib==0.11.0
    PY_PROF=$VENV/bin/python
    if ! $PY_PROF -c "import jax, vllm_torchtpu; print(jax.__version__)"; then
      echo "[kvs] j11 venv unusable; the trace step is expected to SIGSEGV"
      PY_PROF=$PY_SYS
      RC=$((RC + 1))
    fi
    # Per-shard trace dir.  Two trace shards run concurrently and
    # `bench_kvs_trace.py` writes one `trace_meta.json` per invocation; a shared
    # dir would have the second shard overwrite the first shard`s metadata.
    # The xplane key the reader uses is the `<config>__<arm>` directory just
    # above `plugins/`, so an extra level above it changes nothing.
    run trace timeout "${TRACE_TIMEOUT:-9000}" $PY_PROF \
        "$VD/bench_kvs_trace.py" --outdir "$OUT/trace/s$LABEL" \
        --iters "${ITERS:-20}" --chunk "${CHUNK:-5}" "$@"
    ;;
  *)
    echo "unknown step $STEP"; exit 2
    ;;
esac

echo "===== kv-scatter shard $LABEL step=$STEP ARTIFACTS ====="
find "$OUT" -maxdepth 2 -name "*.json" | head -20
find "$OUT" -name "*.hlo.txt" | head -20
find "$OUT" -name "*.xplane.pb" 2>/dev/null | wc -l
echo "===== kv-scatter shard $LABEL step=$STEP combined rc=$RC ====="
exit $RC
