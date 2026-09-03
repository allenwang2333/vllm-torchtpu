#!/bin/bash
# Option B shard entry point.  One argument: the shard name.
#
# Exists so the `cdk` RUN_CMD stays free of nested quoting -- RUN_CMD is a
# single-quoted YAML scalar that may not contain a single quote.
#
# No `set -e`: a failed arm is a result and must not take the shard's log with
# it.  The last command in each branch is the one whose exit code is reported.
#
# Every shard first reads back the md5 of the two kernel modules it will
# import.  An earlier job in this exercise silently ran the *image's* older
# kernel behind a green JobStatus; `probes/r2_shard.sh` added this guard and
# this reuses it.

ARM=$1
KW=${KW:-/root/kw}
VD=$KW/variants/option-b-pipeline
REPO=/root/cloud-devkit/vllm-torchtpu
PY=${PY:-python}

echo "===== option-b shard $ARM ====="
$PY - <<PYEOF
import hashlib, pathlib
import jax
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk
print("jax", jax.__version__)
for m in (streamindex_topk, sparsecore_topk):
    h = hashlib.md5(pathlib.Path(m.__file__).read_bytes()).hexdigest()
    print(f"MODULE {m.__file__} md5={h}")
PYEOF

case "$ARM" in
  pytest-b|pytest-control)
    cd "$REPO" || exit 9
    # Both md5s, per the brief: the image ships a stale 16-test copy of this
    # file and a green `16 passed` would be a stale-file failure, not a pass.
    echo "--- test file md5 (checkout, mapped in) ---"
    md5sum tests/kernels/test_dsv4_streamindex_topk.py
    echo "--- test file md5 (image copy, if a different path exists) ---"
    $PY - <<'PYEOF'
import hashlib, pathlib, importlib.util, sys
seen = set()
for base in sys.path + ["/root/cloud-devkit/vllm-torchtpu", "/workspace",
                        "/app", "/usr/local/lib"]:
    p = pathlib.Path(base) / "tests/kernels/test_dsv4_streamindex_topk.py"
    try:
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            print("FOUND", p, hashlib.md5(p.read_bytes()).hexdigest())
    except OSError:
        pass
for p in pathlib.Path("/").glob("*/vllm*/tests/kernels/test_dsv4_streamindex_topk.py"):
    if p.resolve() not in seen:
        seen.add(p.resolve())
        print("FOUND", p, hashlib.md5(p.read_bytes()).hexdigest())
print("distinct copies on disk:", len(seen))
PYEOF
    PYTHONPATH="$VD:$PYTHONPATH" OPTION_B_G="${OPTION_B_G:-1}" \
      OPTION_B="$([ "$ARM" = pytest-control ] && echo 0 || echo 1)" \
      $PY -m pytest -q -p no:cacheprovider \
      -p option_b_patch -m "not nightly and not multichip" \
      tests/kernels/test_dsv4_streamindex_topk.py
    ;;
  correctness)
    cd "$REPO" || exit 9
    $PY "$VD/probe_correctness.py"
    ;;
  plan_anchors)
    cd "$REPO" || exit 9
    $PY "$VD/probe_flip3.py" --arm plan
    echo "----- plan rc=$? -----"
    $PY "$VD/probe_flip3.py" --arm anchors
    ;;
  *)
    cd "$REPO" || exit 9
    $PY "$VD/probe_flip3.py" --arm "$ARM"
    ;;
esac
RC=$?
echo "===== option-b shard $ARM rc=$RC ====="
exit $RC
