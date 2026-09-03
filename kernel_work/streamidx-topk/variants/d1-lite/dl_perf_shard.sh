#!/bin/bash
# D1-lite measurement shard.
#
#   $@ = one or more specs `<tag>:<layers>:<ctx>:<arm>`, run in sequence on
#        this shard's one chip.  `<tag>` becomes the config key in the off-pod
#        table; the trace directory is `<tag>__<arm>`.
#
# Structure copied from `option-d-hoist/d_perf_shard.sh` for the same reasons:
#   1. `--mode hlo` first -- compiler only, no profiler, cannot SIGSEGV on the
#      image's jax 0.10.2. If the trace dies we still have the static half.
#   2. `--mode trace` from a side venv holding jax/jaxlib 0.11.0, because
#      jaxlib 0.10.2's profiler client SIGSEGVs against libtpu 0.0.44.1
#      (`PLUGIN_Profiler_Api` struct size 80 vs 104).
#
# No `set -e`: a failed arm must not take the others with it.
export DL_VARIANT=${DL_VARIANT:-/root/d1lite}
export DL_INSTRUMENTS=${DL_INSTRUMENTS:-/root/instruments}
OUT=${CDK_OUTPUT_DIR:-/tmp}/d1lite
LABEL=${TPU_FANOUT_SHARD:-0}
PY_SYS=${PY_SYS:-python}
mkdir -p "$OUT"
declare -A RC

bash "$DL_VARIANT/dl_readback.sh" "$PY_SYS"

echo "===== D1LITE/$LABEL STEP j11_venv BEGIN ====="
VENV=/tmp/j11_dl_$LABEL
uv venv --system-site-packages "$VENV" && \
  uv pip install --python "$VENV/bin/python" -q jax==0.11.0 jaxlib==0.11.0
RC[j11_venv]=$?
PY_PROF=$VENV/bin/python
if ! $PY_PROF -c "import jax, vllm_torchtpu; print(jax.__version__)"; then
  echo "[d1lite] j11 venv unusable; the trace step is expected to SIGSEGV"
  PY_PROF=$PY_SYS
  RC[j11_venv]=98
fi
echo "===== D1LITE/$LABEL STEP j11_venv END rc=${RC[j11_venv]} ====="

for SPEC in "$@"; do
  IFS=: read -r TAG LAYERS CTX ARM <<< "$SPEC"
  KEY="${TAG}__${ARM}"
  COMMON=(--arm "$ARM" --layers "$LAYERS" --ctx "$CTX" --tag "$TAG" --root "$OUT")

  echo "===== D1LITE/$LABEL $KEY STEP hlo BEGIN ====="
  $PY_SYS "$DL_VARIANT/probe_d1lite.py" "${COMMON[@]}" --mode hlo
  RC[${KEY}_hlo]=$?
  echo "===== D1LITE/$LABEL $KEY STEP hlo END rc=${RC[${KEY}_hlo]} ====="

  echo "===== D1LITE/$LABEL $KEY STEP trace BEGIN ====="
  $PY_PROF "$DL_VARIANT/probe_d1lite.py" "${COMMON[@]}" --mode trace
  RC[${KEY}_trace]=$?
  echo "===== D1LITE/$LABEL $KEY STEP trace END rc=${RC[${KEY}_trace]} ====="
done

echo "===== D1LITE/$LABEL TABLE ====="
FAIL=0
for k in "${!RC[@]}"; do
  printf "%-44s rc=%s\n" "$k" "${RC[$k]}"
  [ "$k" = "j11_venv" ] || [ "${RC[$k]}" = "0" ] || FAIL=$((FAIL + 1))
done
echo "===== D1LITE/$LABEL ARTIFACTS ====="
du -sh "$OUT"/hlo/* "$OUT"/traces/* 2>/dev/null
find "$OUT" -name "*.xplane.pb" 2>/dev/null | head -40
echo "===== D1LITE/$LABEL failed_steps=$FAIL ====="
exit $FAIL
