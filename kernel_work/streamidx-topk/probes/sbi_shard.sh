#!/bin/bash
# Follow-up to `bdt_shard.sh` (job j-79a410b8): resolve the ONE arm where the
# two independent device-time methods disagreed.
#
# Everything else in the corrected baseline agreed to within 2 %. Stage B did
# not: chain slope 336.48 us vs `hlo_stats` 431.15 us at prefill, 28.77 vs
# 31.73 at decode. The brief's rule is that such a disagreement IS the result
# and gets investigated rather than resolved by preference, so this job runs
# the same method (chain slope) on two different inputs -- synthetic N(0, 1)
# scores and stage A's real output -- and lets the input be the only variable.
#
# One unpinned process, no fan-out: this has to produce absolutes comparable
# with the baseline job's, and a contended shard cannot. No `set -e`.
P=${P:-/root/probes}
export BDT_INSTRUMENTS=${BDT_INSTRUMENTS:-/root/instruments}
OUT=${CDK_OUTPUT_DIR:-/tmp}/sbi
mkdir -p "$OUT"
PY_SYS=${PY_SYS:-python}

echo "===== SBI ENV $(date -u) ====="
$PY_SYS -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"

# md5 read-back. The image at ec4bcecf predates these modules and they are
# mapped in over its checkout; a --map-dir that silently did not take is how an
# earlier job in this exercise measured the image's OLD kernel behind a green
# JobStatus. Guard copied from `probes/r2_shard.sh`.
echo "----- kernel module read-back -----"
$PY_SYS - <<'PYEOF'
import hashlib, os, sys
sys.path.insert(0, os.environ.get("BDT_INSTRUMENTS", "/root/instruments"))
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk
import bench_streamidx_topk as B
for m in (streamindex_topk, sparsecore_topk, B):
    print(f"MODULE {m.__file__} md5="
          f"{hashlib.md5(open(m.__file__, 'rb').read()).hexdigest()}")
print("EXPECT streamindex_topk.py   md5=1e221f10cbf3573aa0d2ae78c3463532")
print("EXPECT sparsecore_topk.py    md5=8bc9e25b8701a53a97704564b9fbc658")
print("EXPECT bench_streamidx_topk.py md5=e8e492d13e59c4a1675cc7a98c5a0813")
PYEOF
echo "----- probe read-back -----"
md5sum "$P/probe_stage_b_input.py" "$P/_stage_a.py"

echo "===== SBI STEP probe BEGIN $(date -u +%H:%M:%S) ====="
timeout "${SBI_TIMEOUT:-3600}" $PY_SYS "$P/probe_stage_b_input.py"
RC=$?
echo "===== SBI STEP probe END rc=$RC $(date -u +%H:%M:%S) ====="

echo "===== SBI ARTIFACTS ====="
find "${CDK_OUTPUT_DIR:-/tmp}" -name "stage-b-input*.json" 2>/dev/null
exit 0
