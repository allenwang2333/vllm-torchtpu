#!/bin/bash
# Option B runner shard entry point.  One argument: the arm
# (plan | correct | hostcost | trace | all).
#
# Exists so the `cdk` RUN_CMD stays free of nested quoting -- RUN_CMD is a
# single-quoted YAML scalar that may not contain a single quote.
#
# No `set -e`: a failed arm is a result and must not take the rest of the log
# with it.  The combined rc is accumulated and reported at the end.
#
# ONE unpinned process on the whole 4-chip pod.  `trace` and `hostcost` are
# timings, and a fanned-out capture is contended.
#
# Every arm first reads back the md5 of the kernel modules it will import.
# The image ships a DIFFERENT `streamindex_topk.py` and NO `sparsecore_topk.py`
# at all, so a job whose `--map-dir` did not take imports stale code, exits 0
# and benchmarks the wrong thing.  `runner_common.report_md5s` compares against
# the checkout's values and `probe_runner`/`bench_runner_trace` refuse to run
# on a mismatch.

ARM=${1:-all}
KW=${KW:-/root/kw}
VD=$KW/variants/option-b-runner
REPO=/root/cloud-devkit/vllm-torchtpu
PY=${PY:-python}
# jax 0.11.0 side venv: the image's pinned 0.10.2 SIGSEGVs the first time
# `jax.profiler.trace` is called against libtpu 0.0.44.1
# (PLUGIN_Profiler_Api size: expected 80, got 104).
PY_PROF=${PY_PROF:-/tmp/j11/bin/python}
OUT=${CDK_OUTPUT_DIR:-/tmp}/obrunner
mkdir -p "$OUT"

echo "===== option-b-runner shard $ARM ====="
$PY -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__, jax.devices())"
md5sum "$VD"/*.py "$VD"/*.sh

cd "$REPO" || exit 9
RC=0

run () {
  echo "----- option-b-runner arm $1 BEGIN -----"
  shift
  "$@" 2>&1
  local r=$?
  echo "----- option-b-runner arm END rc=$r -----"
  [ $r -ne 0 ] && RC=$((RC + 1))
  return $r
}

case "$ARM" in
  plan)
    run plan $PY "$VD/probe_runner.py" --arm plan --json "$OUT/plan.json"
    ;;
  correct)
    run correct $PY "$VD/probe_runner.py" --arm correct \
        --json "$OUT/correct.json"
    ;;
  hostcost)
    run hostcost $PY "$VD/probe_runner.py" --arm hostcost \
        --trials "${TRIALS:-9}" --json "$OUT/hostcost.json"
    ;;
  trace)
    run trace $PY_PROF "$VD/bench_runner_trace.py" \
        --iters "${ITERS:-20}" --outdir "$OUT/trace"
    ;;
  all)
    run plan     $PY "$VD/probe_runner.py" --arm plan --json "$OUT/plan.json"
    run correct  $PY "$VD/probe_runner.py" --arm correct \
        --json "$OUT/correct.json"
    run hostcost $PY "$VD/probe_runner.py" --arm hostcost \
        --trials "${TRIALS:-9}" --json "$OUT/hostcost.json"
    run trace    $PY_PROF "$VD/bench_runner_trace.py" \
        --iters "${ITERS:-20}" --outdir "$OUT/trace"
    ;;
  *)
    echo "unknown arm $ARM"; exit 2
    ;;
esac

echo "===== option-b-runner shard $ARM combined rc=$RC ====="
exit $RC
