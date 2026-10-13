#!/usr/bin/env bash
# Entry point of a gdn/v3 serve-scorer job (Qwen3.5-397B-A17B-FP8) on one tpu7x 2x2x1 node.
#
# The stage directory (built by stage.sh) holds:
#   scorer/            this directory, exported from the scorer's revision
#   arms/<name>/       one exported source tree per arm
#   arms.manifest      "<name> <expected git tree>" per arm
# Environment:
#   ARMS            space-separated arm names in run order; ":control" marks the control
#   WEIGHTS_DIR     a directory holding the model's Hugging Face snapshot files
#   EXPECT_JAX      the jax (and jaxlib) version the image must hold
#   EXPECT_LIBTPU   the libtpu version the image must hold
#   CANDIDATE       optional; recorded in the output
#   SCORER_FLAGS    scorer flags, e.g. "--tolerance 0.5"
# The scorer's single JSON value is the last thing written to stdout, and is
# also saved as score.json in the job's output directory.
set -euo pipefail
exec 3>&1 1>&2
if [ -n "${CDK_OUTPUT_DIR:-}" ]; then
  mkdir -p "$CDK_OUTPUT_DIR"
  exec 2> >(tee -a "$CDK_OUTPUT_DIR/progress.log" >&2) 1>&2
fi

STAGE=${STAGE:-/mapped/stage}
WORK=${WORK:-/scratch}
MODEL=Qwen/Qwen3.5-397B-A17B-FP8
: "${ARMS:?}" "${WEIGHTS_DIR:?}" "${EXPECT_JAX:?}" "${EXPECT_LIBTPU:?}"

echo "=== gdn/v3 397B serve scorer job ${CDK_JOB_ID:-local} on $(hostname) ==="
python3 - "$EXPECT_JAX" "$EXPECT_LIBTPU" <<'PY'
import sys
from importlib.metadata import version
want = {"jax": sys.argv[1], "jaxlib": sys.argv[1], "libtpu": sys.argv[2]}
for dist in ("vllm", "torch", "torch-tpu", "tpu-sync-torch", "transformers"):
    print(f"image {dist} {version(dist)}")
bad = [f"{d} {version(d)} (want {v})" for d, v in want.items() if version(d) != v]
for d, v in want.items():
    print(f"image {d} {version(d)}")
if bad:
    sys.exit("IMAGE PIN MISMATCH: " + ", ".join(bad))
PY

# Weights onto local disk, laid out as an offline Hugging Face cache so the
# recipe's model id resolves unchanged. The snapshot name is a digest of the
# checkpoint's tensor index, so it names this checkpoint and no other.
export HF_HOME=$WORK/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
REPO_DIR=$HF_HOME/hub/models--${MODEL//\//--}
REV=$(sha1sum "$WEIGHTS_DIR/model.safetensors.index.json" | cut -c1-40)
SNAP=$REPO_DIR/snapshots/$REV
mkdir -p "$SNAP" "$REPO_DIR/refs" "$WORK/weights"
echo -n "$REV" > "$REPO_DIR/refs/main"
t0=$(date +%s)
find "$WEIGHTS_DIR" -maxdepth 1 -type f -printf '%f\n' \
  | xargs -P 32 -I{} cp "$WEIGHTS_DIR/{}" "$WORK/weights/{}"
want=$(find "$WEIGHTS_DIR" -maxdepth 1 -type f | wc -l)
got=$(find "$WORK/weights" -maxdepth 1 -type f | wc -l)
[ "$want" = "$got" ] || { echo "weights: copied $got of $want files"; exit 1; }
for f in "$WORK"/weights/*; do ln -sf "$f" "$SNAP/$(basename "$f")"; done
echo "weights rev $REV: $got files, $(du -sh "$WORK/weights" | cut -f1) staged in $(( $(date +%s) - t0 ))s"

# Arms onto local disk, each checked against the tree it was exported from.
mkdir -p "$WORK/arms"
args=()
for spec in $ARMS; do
  name=${spec%%:*}
  cp -a "$STAGE/arms/$name" "$WORK/arms/$name"
  want=$(awk -v n="$name" '$1==n{print $2}' "$STAGE/arms.manifest")
  got=$(python3 -c "import sys; sys.path.insert(0, '$STAGE/scorer'); import serve_scorer as s; from pathlib import Path; print(s.git_tree_hash(Path('$WORK/arms/$name')))")
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "arm $name: tree $got, manifest says ${want:-nothing}"; exit 1; }
  echo "arm $name: tree $got"
  if [ "$spec" = "$name" ]; then args+=(--arm "$name=$WORK/arms/$name")
  else args+=(--arm "$name=$WORK/arms/$name:${spec#*:}"); fi
done

# A hung TPU leaves the job Running with nothing to show for it; say so.
( while sleep 60; do
    grep -rlsF 'Stall detected' /tmp/tpu_logs >/dev/null && { echo "STALL pod=$HOSTNAME"; break; }
  done ) &

export KE_CACHE_ROOT=$WORK/cache
OUT=$WORK/out
# shellcheck disable=SC2086
python3 "$STAGE/scorer/serve_scorer.py" "${args[@]}" --out-dir "$OUT" \
  ${CANDIDATE:+--candidate "$CANDIDATE"} ${SCORER_FLAGS:-} > "$WORK/score.json" && rc=0 || rc=$?
echo "scorer exit $rc"

# Keep the evidence: logs, per-sweep results, captures and traces (no caches).
if [ -n "${CDK_OUTPUT_DIR:-}" ]; then
  cp "$WORK/score.json" "$CDK_OUTPUT_DIR/score.json" || true
  (cd "$WORK" && tar czf "$CDK_OUTPUT_DIR/out.tgz" out score.json) || true
fi
[ "$rc" -eq 0 ] && cat "$WORK/score.json" >&3
exit "$rc"
