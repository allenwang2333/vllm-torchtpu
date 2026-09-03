#!/bin/bash
# Composed-candidate measurement shard.
#
#   $@ = one or more specs `<config>:<arm>[:<rep>]`, run in sequence, all in
#        ONE python process on this shard's one chip.  ~90 specs across four
#        shards; JAX import + TPU init costs more than a compile, so the probe
#        loops rather than the shell.
#
# Order matters here:
#   1. build the jax 0.11.0 side venv.  jaxlib 0.10.2's profiler client
#      SIGSEGVs against libtpu 0.0.44.1 (`PLUGIN_Profiler_Api` struct size 80
#      vs 104), so every traced run happens under 0.11.0.
#   2. `readback.sh` under that same interpreter, and **stop if it fails**.
#      It exits non-zero when the mapped-in kernels are not the expected md5s;
#      measuring the image's stale copy would exit 0 and report a number for
#      the wrong kernel, which is the failure mode this whole job is guarded
#      against.
#   3. the specs, last, so the shard's rc is the measured command's.
#
# No `set -e`: a dead spec must not take the rest of the shard with it; the
# probe records a per-spec rc and returns the count of failures.
export CX_VARIANT=${CX_VARIANT:-/root/kw/variants/composed}
export CX_INSTRUMENTS=${CX_INSTRUMENTS:-/root/kw/instruments}
OUT=${CDK_OUTPUT_DIR:-/tmp}/composed
LABEL=${TPU_FANOUT_SHARD:-0}
PY_SYS=${PY_SYS:-python}
mkdir -p "$OUT"

echo "===== CX/$LABEL STEP j11_venv BEGIN ====="
VENV=/tmp/j11_cx_$LABEL
uv venv --system-site-packages "$VENV" && \
  uv pip install --python "$VENV/bin/python" -q jax==0.11.0 jaxlib==0.11.0
RC_VENV=$?
PY_PROF=$VENV/bin/python
if ! $PY_PROF -c "import jax, vllm_torchtpu; print(jax.__version__)"; then
  echo "[cx] j11 venv unusable; tracing under the image jax is expected to"
  echo "[cx] SIGSEGV, so this shard will almost certainly produce nothing."
  PY_PROF=$PY_SYS
  RC_VENV=98
fi
echo "===== CX/$LABEL STEP j11_venv END rc=$RC_VENV ====="

echo "===== CX/$LABEL STEP readback BEGIN ====="
bash "$CX_VARIANT/readback.sh" "$PY_PROF"
RC_READBACK=$?
echo "===== CX/$LABEL STEP readback END rc=$RC_READBACK ====="
if [ "$RC_READBACK" != "0" ]; then
  echo "===== CX/$LABEL ABORT: provenance gate failed, refusing to measure ====="
  echo "===== CX/$LABEL failed_specs=ABORT ====="
  exit 90
fi

echo "===== CX/$LABEL STEP specs BEGIN n=$# ====="
$PY_PROF "$CX_VARIANT/probe_composed.py" --root "$OUT" "$@"
RC_SPECS=$?
echo "===== CX/$LABEL STEP specs END rc=$RC_SPECS ====="

echo "===== CX/$LABEL ARTIFACTS ====="
du -sh "$OUT" 2>/dev/null
find "$OUT" -name "*.xplane.pb" 2>/dev/null | wc -l | \
  xargs -I{} echo "xplane_files={}"
find "$OUT/hlo" -name "census.json" 2>/dev/null | wc -l | \
  xargs -I{} echo "census_files={}"
echo "===== CX/$LABEL rc=$RC_SPECS ====="
exit $RC_SPECS
