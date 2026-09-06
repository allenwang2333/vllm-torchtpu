#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The repo suites that cover the ported code, run BEFORE and AFTER in one job.
#
#   shards 0-2  AFTER  -- the integrated tree, which PYTHONPATH already puts
#                         first at $REPO/src.
#   shard 3     BEFORE -- a copy of the same tree with the two changed files
#                         reverted to their 097fb5c4 contents (byte-identical
#                         to origin/main), on PYTHONPATH ahead of $REPO/src.
#
# Same pytest invocation, same test files, same image, same pod. The only
# variable between shard 3 and shards 0-2 is the two source files, which is
# what makes the diff attributable.
set -u
SHARD="$1"
REPO=/root/cloud-devkit/vllm-torchtpu
D="$REPO/kernel_work/streamindex_topk/probes/src_port"

T_TOPK="tests/kernels/test_dsv4_streamindex_topk.py"
T_MID="tests/layers/vllm/test_dsv4_indexer_op.py tests/layers/vllm/test_tpu_indexer.py tests/kernels/test_mla_kv_cache_utils.py"
T_E2E="tests/layers/common/test_sparse_mla_attention.py tests/distributed/test_raiden_layout_fingerprint.py"

PYTEST="python3 -m pytest -q -rA --timeout=1200 -p no:cacheprovider"
MARK="not nightly and not multichip"

cd "$REPO" || exit 2
echo "===== shard $SHARD ====="
date -u

if [ "$SHARD" = "3" ]; then
  BASE=/root/baseline_src
  rm -rf "$BASE"
  cp -a "$REPO/src" "$BASE"
  cp "$D/baseline_streamindex_topk.py" \
     "$BASE/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py"
  cp "$D/baseline_mla_attention_op.py" \
     "$BASE/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py"
  export PYTHONPATH="$BASE:${PYTHONPATH:-}"
  echo "--- BEFORE arm: reverted files in $BASE ---"
  md5sum "$BASE/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py" \
         "$BASE/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py"
else
  echo "--- AFTER arm: integrated tree ---"
  md5sum "$REPO/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py" \
         "$REPO/src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py"
fi

echo "--- which module actually resolves, and what it carries ---"
python3 - <<PYEOF
import inspect
import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as m
import vllm_torchtpu.layers.vllm.custom_ops.mla_attention_op as c
fn = getattr(m.streamindex_topk, "__wrapped__", m.streamindex_topk)
print("kernel   =", m.__file__)
print("callsite =", c.__file__)
print("has kv_len_bucket =", "kv_len_bucket" in inspect.signature(fn).parameters)
src = inspect.getsource(c.VllmTPUSparseAttnIndexer._build_streamidx_op)
print("num_kv_pages_per_block line =",
      [l.strip() for l in src.splitlines() if "num_kv_pages_per_block" in l])
print("kv_len_bucket lines =",
      [l.strip() for l in src.splitlines() if "kv_len_bucket" in l])
PYEOF

rc=0
case "$SHARD" in
  0) $PYTEST -m "$MARK" $T_TOPK || rc=$? ;;
  1) $PYTEST -m "$MARK" $T_MID  || rc=$? ;;
  2) $PYTEST -m "$MARK" $T_E2E  || rc=$? ;;
  3) $PYTEST -m "$MARK" $T_TOPK $T_MID $T_E2E || rc=$? ;;
  *) echo "unknown shard $SHARD"; exit 2 ;;
esac

echo "===== shard $SHARD rc=$rc ====="
date -u
exit $rc
