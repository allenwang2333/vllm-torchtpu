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
# The Option B bucket-correctness gate, re-run against the INTEGRATED src/
# tree. Same gate, same oracle, same negative controls; only the module under
# test moves (see run_src_port_correctness.py).
#
# Every shape here is `bucket > kv_len` -- the shipping configuration, and the
# only one with unwritten lanes for the negative control to catch.
#
#   shard 0  kv 24576 -> bucket 32768  decode + mixed   at (4,2,4)   [Option E]
#   shard 1  kv  8000 -> bucket 16384  decode + mixed   at (4,2,4)
#   shard 2  kv 49152 -> bucket 65536  decode + mixed   at (4,2,4)
#            65536 is exactly `static_kv_len_bucket(max_model_len >= 65536)`,
#            i.e. the value the integrated call site will actually ship.
#   shard 3  the same three kv lengths at the pre-E (2,2,2) tuple, so a red
#            result can be attributed to B or to E rather than to "the port".
set -u
SHARD="$1"
REPO=/root/cloud-devkit/vllm-torchtpu
D="$REPO/kernel_work/streamindex_topk/probes/src_port"
OUT="${CDK_OUTPUT_DIR:-/tmp}/src_port"
mkdir -p "$OUT"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

ARMS=incumbent,right_size,poison,lanecount,neg_control,neg_control_unwritten

echo "===== shard $SHARD ====="
date -u
python3 -c "import numpy, jax, sys; print(numpy.__version__, jax.__version__, sys.version)"
python3 -c "import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as m; print(9999, m.__file__)"

echo "--- md5 of everything that must not have moved ---"
md5sum "$REPO/kernel_work/streamindex_topk/instruments/shapes.py" \
       "$REPO/kernel_work/streamindex_topk/instruments/test_correctness.py" \
       "$REPO/kernel_work/streamindex_topk/instruments/inputs.py" \
       "$REPO/kernel_work/streamindex_topk/probes/bucket_correctness/run_bucket_correctness.py" \
       "$D/baseline_streamindex_topk.py" \
       "$D/run_src_port_correctness.py" \
       "$REPO/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py"

echo "--- the ported src/ kernel is the one PYTHONPATH resolves ---"
grep -c "kv_len_bucket" "$REPO/src/vllm_torchtpu/kernels/deepseek_v4/streamindex_topk.py"
grep -n "num_kv_pages_per_block=" "$REPO/src/vllm_torchtpu/layers/vllm/custom_ops/mla_attention_op.py"

rc=0
case "$SHARD" in
  0)
    python3 "$D/run_src_port_correctness.py" --bkv 4,2,4 \
      --bucket-mode ladder --arms "$ARMS" \
      --regimes bc_a_decode,bc_a_mixed --json "$OUT/s0_e_a.json" || rc=$?
    ;;
  1)
    python3 "$D/run_src_port_correctness.py" --bkv 4,2,4 \
      --bucket-mode ladder --arms "$ARMS" \
      --regimes bc_b_decode,bc_b_mixed --json "$OUT/s1_e_b.json" || rc=$?
    ;;
  2)
    python3 "$D/run_src_port_correctness.py" --bkv 4,2,4 \
      --bucket-mode ladder --arms "$ARMS" \
      --regimes bc_c_decode,bc_c_mixed --json "$OUT/s2_e_c.json" || rc=$?
    ;;
  3)
    python3 "$D/run_src_port_correctness.py" --bkv 2,2,2 \
      --bucket-mode ladder --arms "$ARMS" \
      --regimes bc_a_decode,bc_a_mixed --json "$OUT/s3_base_a.json" || rc=$?
    python3 "$D/run_src_port_correctness.py" --bkv 2,2,2 \
      --bucket-mode ladder --arms "$ARMS" \
      --regimes bc_b_decode,bc_b_mixed --json "$OUT/s3_base_b.json" || rc=$?
    python3 "$D/run_src_port_correctness.py" --bkv 2,2,2 \
      --bucket-mode ladder --arms "$ARMS" \
      --regimes bc_c_decode,bc_c_mixed --json "$OUT/s3_base_c.json" || rc=$?
    ;;
  *)
    echo "unknown shard $SHARD"; exit 2 ;;
esac

echo "===== shard $SHARD rc=$rc ====="
date -u
exit $rc
