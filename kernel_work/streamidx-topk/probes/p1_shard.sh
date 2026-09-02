#!/bin/bash
# P1 shard: every /xprof-llo capture recipe for stage A's decode kernel.
#
#   1. trace-side capture, kernel arm (bkv_p=2) and reference arm (bkv_p=1)
#   2. dump-side capture,  same two arms  (LLO IR text + Mosaic MLIR)
#   3. periodic GFC counter sampling at two intervals, so the 16-bit wrap can
#      be diffed out -- this is the only *measured* half; everything from the
#      dump-side tools is static compile-time analysis.
#
# No `set -e`: a failed recipe is a result and must not take the others with
# it.  Every step's rc lands in the table at the end; read that, not the shard
# exit code.
P=${P:-/root/probes}
OUT=${CDK_OUTPUT_DIR:-/tmp}/p1
mkdir -p "$OUT"
PY=${PY:-python}
declare -A RC

step() {  # step <name> <cmd...>
  local name=$1; shift
  echo "===== P1 STEP $name BEGIN ====="
  "$@"
  RC[$name]=$?
  echo "===== P1 STEP $name END rc=${RC[$name]} ====="
}

step trace_stage_a $PY "$P/probe_p1_capture.py" --mode trace --arm stage_a \
     --outdir "$OUT/trace_stage_a" --iters 50
step trace_ref     $PY "$P/probe_p1_capture.py" --mode trace --arm ref \
     --outdir "$OUT/trace_ref" --iters 50
step dump_stage_a  $PY "$P/probe_p1_capture.py" --mode dump  --arm stage_a \
     --outdir "$OUT/dump_stage_a"
step dump_ref      $PY "$P/probe_p1_capture.py" --mode dump  --arm ref \
     --outdir "$OUT/dump_ref"

# ---- periodic GFC counter sampling ---------------------------------------
# The client needs the `xprof` pip package, which must NOT be installed over
# the image's jax -- hence its own venv.
echo "===== P1 STEP xpcap_venv BEGIN ====="
uv venv /tmp/xpcap && uv pip install --python /tmp/xpcap/bin/python -q xprof
RC[xpcap_venv]=$?
echo "===== P1 STEP xpcap_venv END rc=${RC[xpcap_venv]} ====="

for IV in 20 1000; do
  SDIR="$OUT/sample_iv$IV"
  rm -rf "$SDIR"; mkdir -p "$SDIR"
  echo "===== P1 STEP sample_iv$IV BEGIN ====="
  $PY "$P/probe_p1_capture.py" --mode server --seconds 45 --port 9012 \
      --outdir "$SDIR/workload" &
  SRV=$!
  # wait for the workload to compile, warm and open the port
  for _ in $(seq 1 90); do
    [ -f "$SDIR/workload/READY" ] && break
    sleep 2
  done
  if [ -f "$SDIR/workload/READY" ]; then
    sleep 3
    /tmp/xpcap/bin/python "$P/probe_p1_sample_client.py" "$IV" "$SDIR/logdir" \
        localhost:9012 2000
    RC[sample_iv$IV]=$?
  else
    echo "[p1] workload never reported READY; skipping capture"
    RC[sample_iv$IV]=99
  fi
  wait $SRV
  echo "===== P1 STEP sample_iv$IV END rc=${RC[sample_iv$IV]} ====="
done

echo "===== P1 STEP TABLE ====="
for k in trace_stage_a trace_ref dump_stage_a dump_ref xpcap_venv \
         sample_iv20 sample_iv1000; do
  printf "%-16s rc=%s\n" "$k" "${RC[$k]:-notrun}"
done
echo "===== P1 ARTIFACT SIZES ====="
du -sh "$OUT"/* 2>/dev/null
find "$OUT" -name "*.xplane.pb" | head -40
exit 0
