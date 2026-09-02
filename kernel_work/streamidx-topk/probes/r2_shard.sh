#!/bin/bash
# R2 shard: one arm of the nightly-vs-AgentX `hlo_stats` pull.
#
#   $1 = arm    (nightly | agentx | nightly16)
#   $2 = label  (optional; names the output dir, so the same arm can be run
#                twice on two chips as a contention control)
#
# Two steps, deliberately in this order:
#   1. `--mode hlo`   -- lower+compile and write the optimized HLO plus a
#      static instruction census. A compiler-only step: no profiler, so it runs
#      on the image's pinned jax 0.10.2 and cannot SIGSEGV. If step 2 dies we
#      still have the static half.
#   2. `--mode trace` -- `jax.profiler.trace`, which SIGSEGVs on jaxlib 0.10.2
#      against libtpu 0.0.44.1 (PLUGIN_Profiler_Api 80 vs 104 bytes). It runs
#      from a per-shard side venv holding jax/jaxlib 0.11.0 over the image's
#      site-packages, exactly as `probes/p1_shard.sh` did.
#
# No `set -e`: a failed step is a result and must not take the other with it.
ARM=${1:?usage: r2_shard.sh <nightly|agentx|nightly16> [label]}
LABEL=${2:-$ARM}
P=${P:-/root/probes}
export R2_INSTRUMENTS=${R2_INSTRUMENTS:-/root/instruments}
OUT=${CDK_OUTPUT_DIR:-/tmp}/r2
mkdir -p "$OUT"
PY_SYS=${PY_SYS:-python}
declare -A RC

step() {  # step <name> <cmd...>
  local name=$1; shift
  echo "===== R2/$LABEL STEP $name BEGIN ====="
  "$@"
  RC[$name]=$?
  echo "===== R2/$LABEL STEP $name END rc=${RC[$name]} ====="
}

echo "===== R2/$LABEL ENV ====="
$PY_SYS -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"
pip list 2>/dev/null | grep -iE "^(jax|jaxlib|libtpu)" || true
# The image at ec4bcecf predates both kernel modules, so they are mapped in
# over its checkout. Read back which file actually gets imported and its md5 --
# a --map-dir that did not take is otherwise silent, and the whole probe would
# then measure the image's older kernel.
$PY_SYS - <<PYEOF
import hashlib
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk
for m in (streamindex_topk, sparsecore_topk):
    h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
    print(f"MODULE {m.__file__} md5={h}")
    print(f"        enable_early_exit in signature: "
          f"{'enable_early_exit' in open(m.__file__).read()}")
PYEOF

step hlo $PY_SYS "$P/probe_r2_hlo_stats.py" --arm "$ARM" --mode hlo \
     --outdir "$OUT/${LABEL}_hlo"

echo "===== R2/$LABEL STEP j11_venv BEGIN ====="
VENV=/tmp/j11_$LABEL
uv venv --system-site-packages "$VENV" && \
  uv pip install --python "$VENV/bin/python" -q jax==0.11.0 jaxlib==0.11.0
RC[j11_venv]=$?
PY_PROF=$VENV/bin/python
if ! $PY_PROF -c "import jax, vllm_torchtpu; print(jax.__version__)"; then
  echo "[r2] j11 venv unusable; falling back to the image python (the trace"
  echo "[r2] step is then expected to SIGSEGV -- reported as such)"
  PY_PROF=$PY_SYS
  RC[j11_venv]=98
fi
echo "===== R2/$LABEL STEP j11_venv END rc=${RC[j11_venv]} ====="

step trace $PY_PROF "$P/probe_r2_hlo_stats.py" --arm "$ARM" --mode trace \
     --iters 200 --chunk 10 --outdir "$OUT/${LABEL}_trace"

echo "===== R2/$LABEL STEP TABLE ====="
for k in hlo j11_venv trace; do
  printf "%-10s rc=%s\n" "$k" "${RC[$k]:-notrun}"
done
echo "===== R2/$LABEL ARTIFACTS ====="
du -sh "$OUT"/${LABEL}_* 2>/dev/null
find "$OUT/${LABEL}_trace" -name "*.xplane.pb" 2>/dev/null | head -20
exit 0
