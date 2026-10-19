#!/bin/bash
# Job body for scripts/vllm/benchmark/d_leg_bench.py in a simple-run-command pod.
# Usage: bash scripts/vllm/benchmark/d_leg_job.sh <tag> <d_leg_bench args...>
set -o pipefail
TAG=$1; shift
W=/workspace/vllm-torchtpu
for d in /root/torchtpu-vllm/src /root/cloud-devkit/vllm-torchtpu/src; do
  [ -d "$d" ] && cp -rf $W/src/vllm_torchtpu "$d/" || true
done
export PYTHONPATH="$W/src:$W${PYTHONPATH:+:$PYTHONPATH}"
OUT=${CDK_OUTPUT_DIR:-/tmp/out}/$TAG
mkdir -p $OUT
cd $W
python3 -u scripts/vllm/benchmark/d_leg_bench.py --tag $TAG --out-dir $OUT "$@" 2>&1 | tee $OUT/bench.log
rc=$?
echo "D_LEG_EXIT=$rc"
sleep 60  # let workers finish writing traces
if [ -d $OUT/xprof ]; then
  for d in $OUT/xprof/*/; do
    [ -d "$d" ] || continue
    python3 -u scripts/vllm/benchmark/xplane_ops.py "$d" --steps ${PROFILE_STEPS:-6} --max-files 1 \
      --json $OUT/$(basename $d)_ops.json 2>&1 | tee $OUT/$(basename $d)_ops.txt | head -400
  done
  find $OUT/xprof -type f | head -50
fi
echo "D_LEG_DONE rc=$rc"
exit $rc
