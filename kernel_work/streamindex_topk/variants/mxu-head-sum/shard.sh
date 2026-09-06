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
# Variant F -- `mxu-head-sum`. One shard per chip.
#
#   0  step_probe.py   the step-level decomposition and every F formulation
#   1  bench_ab.py     the four-regime A/B with a drift control
#   2  correctness.py  the standard gate on incumbent AND variant
#   3  llo_f.py        what operand types the two matmuls reach the MXU with
#
# Every absolute time in shards 0 and 1 is contended -- four shards share one
# host's HBM and 16 CPUs. Ratios taken INSIDE one shard are the trustworthy
# quantity and every conclusion is built on one.
set -u
D=/workspace/kernel_work/streamindex_topk/variants/mxu-head-sum
O="${CDK_OUTPUT_DIR:-/tmp}"
S="${1:?shard index}"

echo "===== shard $S starting at $(date -Is) ====="

case "$S" in
0)
  python3 "$D/step_probe.py" --json "$O/step_probe.json"
  ;;
1)
  python3 "$D/bench_ab.py" \
    --regimes decode_prod,decode_64k,mixed_prod,mixed_64k \
    --dtype bf16 --json "$O/bench_ab.json"
  ;;
2)
  python3 "$D/correctness.py" \
    --regimes decode_prod,mixed_prod,decode_64k,mixed_32k,mixed_64k \
    --control-regimes decode_prod,mixed_prod --dtype bf16
  ;;
3)
  for IMPL in incumbent variant; do
    echo "----- llo: $IMPL -----"
    python3 "$D/llo_f.py" --impl "$IMPL" --regime mixed_prod \
      --dump-root /tmp/llo_f
    # ~1.1 GiB per capture; the counts are printed above, the bytes are not
    # worth keeping and definitely not worth syncing.
    rm -rf "/tmp/llo_f/$IMPL-mixed_prod"
  done
  ;;
*)
  echo "unknown shard $S" >&2
  exit 2
  ;;
esac
RC=$?
echo "===== shard $S finished rc=$RC at $(date -Is) ====="
exit $RC
