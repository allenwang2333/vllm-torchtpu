#!/bin/bash
# Provenance read-back, shared by every shard of the composed-candidate job.
#
# The pinned image's kernels are STALE: its `streamindex_topk.py` differs from
# the checkout and it has no `sparsecore_topk.py` at all.  Both are supplied by
# `--map-dir <checkout>/src:tpu-job:/root/cloud-devkit/vllm-torchtpu/src`.
# A green `JobStatus` is not evidence that the mapped-in code is what ran -- an
# earlier job in this exercise was green with every shard at rc=1 because the
# image's own `src` shadowed the mapped-in kernels on PYTHONPATH.
#
# So this script imports the modules the way the probe will import them, prints
# the file that actually resolved and its md5, and **exits non-zero on a
# mismatch** so the shard refuses to measure stale code rather than reporting a
# number for it.
#
#   streamindex_topk.py  md5=1e221f10cbf3573aa0d2ae78c3463532
#   sparsecore_topk.py   md5=8bc9e25b8701a53a97704564b9fbc658
#
# It also prints a `difflib` diff of `streamindex_topk_c.kernel` against the
# incumbent's `kernel`.  The composed module's central claim is that not one
# line of Mosaic changed except the `q_layout` branch; the diff puts that on the
# record in the job log instead of leaving it as an assertion in a docstring.
PY=${1:-python}
EXPECT_SI=1e221f10cbf3573aa0d2ae78c3463532
EXPECT_SC=8bc9e25b8701a53a97704564b9fbc658

echo "===== CX PROVENANCE ====="
$PY -c "import jax, jaxlib; print(f'jax {jax.__version__} jaxlib {jaxlib.__version__}')"
pip list 2>/dev/null | grep -iE "^(jax|jaxlib|libtpu) " || true

EXPECT_SI=$EXPECT_SI EXPECT_SC=$EXPECT_SC $PY - <<'PYEOF'
import difflib
import hashlib
import inspect
import os
import sys

sys.path.insert(0, os.environ.get("CX_VARIANT", "/root/kw/variants/composed"))
sys.path.insert(0, os.environ.get("CX_INSTRUMENTS", "/root/kw/instruments"))

import vllm_torchtpu
from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk, streamindex_topk

import bench_streamidx_topk as B
import streamindex_topk_composed as CX

print("PKG vllm_torchtpu", vllm_torchtpu.__file__)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


bad = 0
for m in (streamindex_topk, sparsecore_topk, CX, CX._c, B):
    print(f"MODULE {m.__file__} md5={md5(m.__file__)}")

for mod, want, label in ((streamindex_topk, os.environ["EXPECT_SI"],
                          "streamindex_topk.py"),
                         (sparsecore_topk, os.environ["EXPECT_SC"],
                          "sparsecore_topk.py")):
    got = md5(mod.__file__)
    ok = got == want
    print(f"MD5GATE {label} got={got} want={want} ok={ok}")
    if not ok:
        bad += 1

print("EARLY_EXIT_IN_SIG",
      "enable_early_exit" in open(streamindex_topk.__file__).read())

# Identity of everything the composed module imports rather than copies.
print("HELPER_IDENTITY _effective_row_lengths is incumbent's:",
      CX._effective_row_lengths is streamindex_topk._effective_row_lengths)
print("HELPER_IDENTITY prepare_q_inputs is incumbent's:",
      CX.prepare_q_inputs is streamindex_topk.prepare_q_inputs)
print("HELPER_IDENTITY prepare_index_weights is incumbent's:",
      CX.prepare_index_weights is streamindex_topk.prepare_index_weights)
print("KERNEL_IDENTITY composed.kernel is option-c's kernel:",
      CX.kernel is CX._c.kernel)
print("KERNEL_IDENTITY composed.kernel is incumbent's kernel:",
      CX.kernel is streamindex_topk.kernel)

# The gate, as the module itself reports it.
print("LAYOUTS head_major_mixed=False", CX.describe_layouts(False))
print("LAYOUTS head_major_mixed=True ", CX.describe_layouts(True))
for arm in ("inc", "ctl", "c", "dl", "c_dl"):
    fn, kw = CX.for_arm(arm)
    print(f"ARM {arm:<5} fn={fn.__module__ if hasattr(fn, '__module__') else fn}"
          f".{getattr(fn, '__name__', fn)} kw={kw}")

# The Mosaic diff.  Expected: only the `q_layout` parameter, the branch it
# guards, and the transposes inside it.
inc_src = inspect.getsource(streamindex_topk.kernel).splitlines()
c_src = inspect.getsource(CX._c.kernel).splitlines()
diff = list(difflib.unified_diff(inc_src, c_src, "incumbent.kernel",
                                 "option_c.kernel", lineterm="", n=2))
print(f"KERNEL_DIFF lines_incumbent={len(inc_src)} lines_option_c={len(c_src)} "
      f"diff_lines={len(diff)}")
print("===== KERNEL DIFF BEGIN =====")
for line in diff:
    print(line)
print("===== KERNEL DIFF END =====")

print("BENCH_CONSTS k=%d cr=%d page=%d bkvp=%s nqpb=%s early_exit=%s mml=%d "
      "max_num_seqs=%d H=%d D=%d" %
      (B.K_IDX, B.COMPRESSION_RATIO, B.PAGE_SIZE, B.NUM_KV_PAGES_PER_BLOCK,
       B.NUM_QUERIES_PER_BLOCK, B.ENABLE_EARLY_EXIT, B.PROD_MAX_MODEL_LEN,
       B.PROD_MAX_NUM_SEQS, B.H_IDX, B.D_IDX))

print(f"MD5GATE_FAILURES {bad}")
sys.exit(1 if bad else 0)
PYEOF
RC=$?
echo "===== CX PROVENANCE END rc=$RC ====="
exit $RC
