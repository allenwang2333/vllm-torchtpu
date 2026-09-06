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
# One shard of the Option B (Right-Size) fan-out. Exists so that RUN_CMD stays
# a short, quote-free string: the recipe interpolates RUN_CMD into a
# single-quoted YAML scalar and the fan-out adds another level of nesting on
# top of that.
#
# Usage: bash shard.sh <0|1|2|3>
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
OUT="${CDK_OUTPUT_DIR:-/tmp}/right-size"
mkdir -p "$OUT"

echo "=== shard $1 ==="
python3 -c "import vllm_torchtpu, sys; print('vllm_torchtpu.__file__ =', vllm_torchtpu.__file__)"
python3 -c "import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as m; print('incumbent kernel =', m.__file__)"
echo "variant kernel = $HERE/streamindex_topk_rs.py"
md5sum "$HERE/streamindex_topk_rs.py" \
       /root/cloud-devkit/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py \
       /workspace/kernel_work/streamindex_topk/instruments/microbench.py \
       /workspace/kernel_work/streamindex_topk/instruments/test_correctness.py \
       /workspace/kernel_work/streamindex_topk/instruments/shapes.py

rc=0
case "$1" in
  0)
    # The four step-1 regimes the brief names. Option B is expected to do
    # almost nothing here -- every one of them runs kv_len within one page of
    # max_model_len -- and they are reported either way. The incumbent column
    # is also the cross-check against report.md's step-1 numbers.
    python3 "$HERE/run_perf.py" \
      --regimes decode_prod,decode_64k,mixed_prod,mixed_64k \
      --early-exit both --json "$OUT/perf_step1.json" || rc=$?
    ;;
  1)
    # The confound shape -- B's headline -- plus the SparseCore attribution and
    # the coverage that stands under it.
    python3 "$HERE/run_perf.py" --regimes mixed_confound \
      --early-exit both --json "$OUT/perf_confound.json" || rc=$?
    python3 "$HERE/run_topk_width.py" \
      --regimes mixed_confound,mixed_64k,mixed_prod,decode_64k \
      --json "$OUT/topk_width.json" || rc=$?
    python3 "$HERE/run_sc_coverage.py" --json "$OUT/sc_coverage.json" || rc=$?
    ;;
  2)
    python3 "$HERE/run_correctness.py" \
      --regimes decode_prod,decode_64k,mixed_prod,mask_decode_8000 \
      --json "$OUT/correctness_a.json" || rc=$?
    ;;
  3)
    python3 "$HERE/run_correctness.py" \
      --regimes mixed_confound,mask_mixed_8000 --stale \
      --json "$OUT/correctness_b.json" || rc=$?
    ;;
  *)
    echo "unknown shard $1"; exit 2 ;;
esac

echo "=== shard $1 rc=$rc ==="
exit $rc
