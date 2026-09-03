#!/bin/bash
# Job 2 of the extended `bkv_p` sweep: the two sections job 1 could not answer.
#
# Job j-6a35b078 ran sections A, A2, B, C, D and E and answered the headline
# question (p = 9 is the measured minimum at ctx 8704, and the exactness story
# does not survive).  It left two gaps, both of which the coordinator asked for
# by name:
#
#   F -- **where it stops, and whether raising `vmem_limit_bytes` moves the
#        wall.**  Section A2 saw p = 40 rejected against a 67108864-byte limit,
#        which is the *physical* 64 MiB TensorCore VMEM and not a scoped
#        default -- but "the message named 64 MiB" is an observation, not a
#        controlled result.  F sweeps the scoped limit from 16 MiB to 256 MiB
#        at each p and reports which limit binds first.  It was written after
#        job 1 was packaged and so has never run.
#
#   G -- **is the optimum exactly `P = cdiv(ctx, page_size)`, and what does a
#        fixed p = 9 cost on a short sequence.**  Job 1 inferred `argmin = P`
#        from three ctx buckets and never ran `p = 8` at ctx 8192, the one
#        bucket where the prediction was untested.  G1 brackets the argmin
#        directly at seven ctx; G2 runs the **full op** at short ctx, which is
#        what decides whether a constant is admissible at all.
#
# Single unpinned process, same as job 1, so the absolutes stay comparable with
# `probes/baseline-device-time.md`.  Both sections are cheap: job 1's whole
# le9 chain step (about 50 arms) took 5 min 12 s.
#
# No trace step.  Job 1 captured seven xplanes at ctx 8704 and those are what
# the independent `hlo_stats` confirmation is read from; nothing in F or G
# changes the arms they cover.
#
# No `set -e`: a failed step is a result and must not take the others with it.
P=${P:-/root/probes}
export BDT_INSTRUMENTS=${BDT_INSTRUMENTS:-/root/instruments}
OUT=${CDK_OUTPUT_DIR:-/tmp}/bkvx2
mkdir -p "$OUT"
PY_SYS=${PY_SYS:-python}
FG_TIMEOUT=${FG_TIMEOUT:-9000}
declare -A RC

step() {  # step <name> <cmd...>
  local name=$1; shift
  echo "===== BKVX2 STEP $name BEGIN $(date -u +%H:%M:%S) ====="
  "$@"
  RC[$name]=$?
  echo "===== BKVX2 STEP $name END rc=${RC[$name]} $(date -u +%H:%M:%S) ====="
}

# ---- 1. env + md5 read-back --------------------------------------------
# A `--map-dir` that did not take is otherwise completely silent, and an
# earlier job in this exercise measured the image's older kernel behind a green
# JobStatus.  A green JobStatus is not evidence your code ran.
echo "===== BKVX2 ENV $(date -u) ====="
$PY_SYS -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"
echo "CDK_OUTPUT_DIR=$CDK_OUTPUT_DIR"
$PY_SYS - <<'PYEOF'
import hashlib, os, sys
sys.path.insert(0, os.environ.get("BDT_INSTRUMENTS", "/root/instruments"))
sys.path.insert(0, "/root/probes")
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk
import bench_streamidx_topk as B
import _stage_a as SA
for m in (streamindex_topk, sparsecore_topk, B, SA):
    h = hashlib.md5(open(m.__file__, "rb").read()).hexdigest()
    print(f"MODULE {m.__file__} md5={h}")
print("EXPECT streamindex_topk.py     md5=1e221f10cbf3573aa0d2ae78c3463532")
print("EXPECT sparsecore_topk.py      md5=8bc9e25b8701a53a97704564b9fbc658")
print("EXPECT bench_streamidx_topk.py md5=e8e492d13e59c4a1675cc7a98c5a0813")
print("EXPECT _stage_a.py             md5=eea6a51bb40a2bbf7cf5b39571753b62")
import jax
print("device", jax.devices()[0].device_kind, "n", len(jax.devices()))
PYEOF
RC[env]=$?

# ---- 2. sections F and G ------------------------------------------------
#
# ONE process, not split at p = 9 the way job 1 was.  The split existed because
# for p > pages_per_seq = 9 the page walk indexes `page_indices[b*9 + kv_p]`
# past the 576-entry array and the resulting DMA source page is garbage, which
# *might* fault.  Job 1 settled that empirically: p = 10, 11, 12, 14, 16, 20
# and 24 all ran to completion and returned numbers.  F's timed sub-part covers
# p in {9,12,16} and G never exceeds 9, so nothing here is riskier than what
# already ran.  F's compile-only part touches p up to 48 and by construction
# never executes a DMA at all.
step fg timeout "$FG_TIMEOUT" $PY_SYS \
     "$P/probe_bkvp_extended.py" --name bkvp-extended-fg \
     --skip A,A2,B,C,D,E ${BKVX2_ARGS:-}

# ---- summary ------------------------------------------------------------
echo "===== BKVX2 STEP TABLE ====="
for k in env fg; do printf "%-8s rc=%s\n" "$k" "${RC[$k]:-notrun}"; done
echo "===== BKVX2 ARTIFACTS ====="
find "${CDK_OUTPUT_DIR:-/tmp}" -name "bkvp-extended*.json" | head -10
exit 0
