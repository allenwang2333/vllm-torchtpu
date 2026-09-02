#!/bin/bash
# P1 shard: every /xprof-llo capture recipe for stage A's decode kernel.
#
#   1. dump-side capture  -- LLO IR text + Mosaic MLIR.  A *compiler* flag, so
#      it needs no profiler and runs on the image's pinned jax.
#      Two arms: the kernel (bkv_p=2) and a reference (bkv_p=1, same bytes,
#      half the DMA granule).  /xprof-llo will not accept a single kernel's
#      absolute numbers.
#   2. trace-side capture -- the `Tensor Core` bundle-marker line.  Needs
#      `jax.profiler.trace`, which SIGSEGVs on jaxlib 0.10.2 against libtpu
#      0.0.44.1 (PLUGIN_Profiler_Api 80 vs 104 bytes), so it runs from a
#      side venv holding jax/jaxlib 0.11.0 over the image's site-packages.
#      The side venv exists so the *other three shards* keep the image's
#      pinned jax and stay comparable to the step-1 baseline.
#   3. periodic GFC counter sampling at two intervals.  This is the ONLY
#      measured half; everything the dump-side tools produce is static
#      compile-time analysis.  Two intervals because the GFC counters are
#      16 bit and wrap in ~34 us at 100% of a unit.
#
# No `set -e`: a failed recipe is a result and must not take the others with
# it.  Every step's rc lands in the table at the end; read that table, not the
# shard's exit code.
P=${P:-/root/probes}
OUT=${CDK_OUTPUT_DIR:-/tmp}/p1
mkdir -p "$OUT"
PY_SYS=${PY_SYS:-python}
declare -A RC

step() {  # step <name> <cmd...>
  local name=$1; shift
  echo "===== P1 STEP $name BEGIN ====="
  "$@"
  RC[$name]=$?
  echo "===== P1 STEP $name END rc=${RC[$name]} ====="
}

echo "===== P1 ENV ====="
$PY_SYS -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"
pip list 2>/dev/null | grep -iE "^(jax|jaxlib|libtpu)" || true

# --------------------------------------------------------------------------
# 1. dump-side (no profiler needed)
# --------------------------------------------------------------------------
step dump_stage_a $PY_SYS "$P/probe_p1_capture.py" --mode dump --arm stage_a \
     --outdir "$OUT/dump_stage_a"
step dump_ref     $PY_SYS "$P/probe_p1_capture.py" --mode dump --arm ref \
     --outdir "$OUT/dump_ref"

# --------------------------------------------------------------------------
# 2/3. profiler-dependent steps need jax 0.11.0
# --------------------------------------------------------------------------
echo "===== P1 STEP j11_venv BEGIN ====="
uv venv --system-site-packages /tmp/j11 && \
  uv pip install --python /tmp/j11/bin/python -q jax==0.11.0 jaxlib==0.11.0
RC[j11_venv]=$?
PY_PROF=/tmp/j11/bin/python
if ! $PY_PROF -c "import jax, vllm_torchtpu; print(jax.__version__)"; then
  echo "[p1] j11 venv unusable; falling back to the image python (the trace"
  echo "[p1] and sampling steps are then expected to SIGSEGV -- reported as such)"
  PY_PROF=$PY_SYS
  RC[j11_venv]=98
fi
echo "===== P1 STEP j11_venv END rc=${RC[j11_venv]} ====="

step trace_stage_a $PY_PROF "$P/probe_p1_capture.py" --mode trace \
     --arm stage_a --outdir "$OUT/trace_stage_a" --iters 50
step trace_ref     $PY_PROF "$P/probe_p1_capture.py" --mode trace \
     --arm ref --outdir "$OUT/trace_ref" --iters 50

echo "===== P1 STEP xpcap_venv BEGIN ====="
# The capture client needs the `xprof` pip package, which must NOT be
# installed over the image -- hence a clean venv of its own.
uv venv /tmp/xpcap && uv pip install --python /tmp/xpcap/bin/python -q xprof
RC[xpcap_venv]=$?
echo "===== P1 STEP xpcap_venv END rc=${RC[xpcap_venv]} ====="

for IV in 20 1000; do
  SDIR="$OUT/sample_iv$IV"
  rm -rf "$SDIR"; mkdir -p "$SDIR"
  echo "===== P1 STEP sample_iv$IV BEGIN ====="
  $PY_PROF "$P/probe_p1_capture.py" --mode server --seconds 60 --port 9012 \
      --outdir "$SDIR/workload" &
  SRV=$!
  for _ in $(seq 1 120); do
    [ -f "$SDIR/workload/READY" ] && break
    sleep 2
  done
  if [ -f "$SDIR/workload/READY" ]; then
    sleep 3
    /tmp/xpcap/bin/python "$P/probe_p1_sample_client.py" "$IV" \
        "$SDIR/logdir" localhost:9012 2000
    RC[sample_iv$IV]=$?
  else
    echo "[p1] workload never reported READY; skipping capture"
    RC[sample_iv$IV]=99
  fi
  wait $SRV
  echo "===== P1 STEP sample_iv$IV END rc=${RC[sample_iv$IV]} ====="
done

# --------------------------------------------------------------------------
# Pack the dumps: one arm measured ~939 MB of LLO text elsewhere, and text
# compresses ~10x, so this is the difference between a fast sync-outputs and
# a slow one.
# --------------------------------------------------------------------------
echo "===== P1 STEP pack BEGIN ====="
for d in dump_stage_a dump_ref; do
  if [ -d "$OUT/$d" ]; then
    du -sh "$OUT/$d"
    tar czf "$OUT/$d.tgz" -C "$OUT" "$d" && rm -rf "$OUT/$d"
  fi
done
RC[pack]=$?
echo "===== P1 STEP pack END rc=${RC[pack]} ====="

echo "===== P1 STEP TABLE ====="
for k in dump_stage_a dump_ref j11_venv trace_stage_a trace_ref xpcap_venv \
         sample_iv20 sample_iv1000 pack; do
  printf "%-16s rc=%s\n" "$k" "${RC[$k]:-notrun}"
done
echo "===== P1 ARTIFACT SIZES ====="
du -sh "$OUT"/* 2>/dev/null
echo "===== P1 XPLANE FILES ====="
find "$OUT" -name "*.xplane.pb" 2>/dev/null | head -40
exit 0
