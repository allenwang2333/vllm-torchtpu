#!/bin/bash
# Option D measurement shard: one or more arms of the `hlo_stats` pull.
#
#   $@ = arms, run in sequence on this shard's one chip
#        (pre1 | pre21 | dcond1 | dcond21 | d1 | d21)
#
# Structure copied from `probes/r2_shard.sh` for the same reasons:
#   1. `--mode hlo` first -- compiler only, no profiler, cannot SIGSEGV on the
#      image's jax 0.10.2. If the trace dies we still have the static half.
#   2. `--mode trace` from a side venv holding jax/jaxlib 0.11.0, because
#      jaxlib 0.10.2's profiler client SIGSEGVs against libtpu 0.0.44.1.
#
# No `set -e`: a failed arm must not take the others with it.
export D_VARIANT=${D_VARIANT:-/root/optd}
export D_INSTRUMENTS=${D_INSTRUMENTS:-/root/instruments}
OUT=${CDK_OUTPUT_DIR:-/tmp}/optd
LABEL=${TPU_FANOUT_SHARD:-0}
PY_SYS=${PY_SYS:-python}
mkdir -p "$OUT"
declare -A RC

bash "$D_VARIANT/d_readback.sh" "$PY_SYS"

echo "===== OPTD/$LABEL STEP j11_venv BEGIN ====="
VENV=/tmp/j11_optd_$LABEL
uv venv --system-site-packages "$VENV" && \
  uv pip install --python "$VENV/bin/python" -q jax==0.11.0 jaxlib==0.11.0
RC[j11_venv]=$?
PY_PROF=$VENV/bin/python
if ! $PY_PROF -c "import jax, vllm_torchtpu; print(jax.__version__)"; then
  echo "[optd] j11 venv unusable; the trace step is expected to SIGSEGV"
  PY_PROF=$PY_SYS
  RC[j11_venv]=98
fi
echo "===== OPTD/$LABEL STEP j11_venv END rc=${RC[j11_venv]} ====="

for ARM in "$@"; do
  echo "===== OPTD/$LABEL ARM $ARM STEP hlo BEGIN ====="
  $PY_SYS "$D_VARIANT/probe_d_hlo_stats.py" --arm "$ARM" --mode hlo \
      --outdir "$OUT/${ARM}_hlo"
  RC[${ARM}_hlo]=$?
  echo "===== OPTD/$LABEL ARM $ARM STEP hlo END rc=${RC[${ARM}_hlo]} ====="

  echo "===== OPTD/$LABEL ARM $ARM STEP trace BEGIN ====="
  $PY_PROF "$D_VARIANT/probe_d_hlo_stats.py" --arm "$ARM" --mode trace \
      --outdir "$OUT/${ARM}_trace"
  RC[${ARM}_trace]=$?
  echo "===== OPTD/$LABEL ARM $ARM STEP trace END rc=${RC[${ARM}_trace]} ====="
done

echo "===== OPTD/$LABEL TABLE ====="
FAIL=0
for k in "${!RC[@]}"; do
  printf "%-16s rc=%s\n" "$k" "${RC[$k]}"
  [ "$k" = "j11_venv" ] || [ "${RC[$k]}" = "0" ] || FAIL=$((FAIL + 1))
done
echo "===== OPTD/$LABEL ARTIFACTS ====="
du -sh "$OUT"/*_hlo "$OUT"/*_trace 2>/dev/null
find "$OUT" -name "*.xplane.pb" 2>/dev/null | head -20
echo "===== OPTD/$LABEL failed_steps=$FAIL ====="
exit $FAIL
