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
# Option B correctness at `kv_len_bucket > kv_len` -- the configuration it
# ships in and the only one with unwritten lanes. One shard per TPU chip.
#
#   shard 0  kv 24576 / bucket 32768   decode + mixed, ladder then exact
#   shard 1  kv  8000 / bucket 16384   decode + mixed, ladder then exact
#   shard 2  kv 49152 / bucket 65536   MIXED (the heaviest oracle), ladder+exact
#   shard 3  kv 49152 / bucket 65536   decode, ladder+exact
#            kv  8000 / bucket  9216   decode + mixed -- the bucket that is NOT
#                                      a multiple of bkv_sz, so the masked
#                                      partial store is EMITTED while unwritten
#                                      lanes exist.
#
# `exact` reproduces variants/right-size/run_correctness.py's binding
# (kv_len_bucket == kv_len) and is here only as the control that shows the two
# configurations side by side. It is expected to measure ZERO unwritten lanes.
#
# Nothing under kernel_work/streamindex_topk/instruments/ is written.
set -u
SHARD="$1"
D=/workspace/kernel_work/streamindex_topk/probes/bucket_correctness
OUT="${CDK_OUTPUT_DIR:-/tmp}/bucket_correctness"
mkdir -p "$OUT"

export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

LADDER_ARMS=incumbent,right_size,poison,lanecount,neg_control,neg_control_unwritten
EXACT_ARMS=incumbent,right_size,poison,lanecount,neg_control

echo "===== shard $SHARD ====="
date -u
python3 -c "import numpy, jax, sys; print(numpy.__version__, jax.__version__, sys.version)"
python3 -c "import vllm_torchtpu.kernels.deepseek_v4.streamindex_topk as m; print('incumbent kernel =', m.__file__)"
echo "--- md5 of everything that must not have moved ---"
md5sum /workspace/kernel_work/streamindex_topk/instruments/shapes.py \
       /workspace/kernel_work/streamindex_topk/instruments/test_correctness.py \
       /workspace/kernel_work/streamindex_topk/instruments/inputs.py \
       /workspace/kernel_work/streamindex_topk/variants/right-size/streamindex_topk_rs.py \
       "$D/run_bucket_correctness.py"

rc=0
case "$SHARD" in
  0)
    python3 "$D/run_bucket_correctness.py" --bucket-mode ladder \
      --arms "$LADDER_ARMS" --regimes bc_a_decode,bc_a_mixed \
      --json "$OUT/s0_ladder.json" || rc=$?
    python3 "$D/run_bucket_correctness.py" --bucket-mode exact \
      --arms "$EXACT_ARMS" --regimes bc_a_decode,bc_a_mixed \
      --json "$OUT/s0_exact.json" || rc=$?
    ;;
  1)
    python3 "$D/run_bucket_correctness.py" --bucket-mode ladder \
      --arms "$LADDER_ARMS" --regimes bc_b_decode,bc_b_mixed \
      --json "$OUT/s1_ladder.json" || rc=$?
    python3 "$D/run_bucket_correctness.py" --bucket-mode exact \
      --arms "$EXACT_ARMS" --regimes bc_b_decode,bc_b_mixed \
      --json "$OUT/s1_exact.json" || rc=$?
    ;;
  2)
    python3 "$D/run_bucket_correctness.py" --bucket-mode ladder \
      --arms "$LADDER_ARMS" --regimes bc_c_mixed \
      --json "$OUT/s2_ladder.json" || rc=$?
    python3 "$D/run_bucket_correctness.py" --bucket-mode exact \
      --arms "$EXACT_ARMS" --regimes bc_c_mixed \
      --json "$OUT/s2_exact.json" || rc=$?
    ;;
  3)
    python3 "$D/run_bucket_correctness.py" --bucket-mode ladder \
      --arms "$LADDER_ARMS" --regimes bc_c_decode \
      --json "$OUT/s3_ladder_c.json" || rc=$?
    python3 "$D/run_bucket_correctness.py" --bucket-mode exact \
      --arms "$EXACT_ARMS" --regimes bc_c_decode \
      --json "$OUT/s3_exact_c.json" || rc=$?
    python3 "$D/run_bucket_correctness.py" --bucket-mode ladder \
      --arms "$LADDER_ARMS" --regimes bc_d_decode,bc_d_mixed \
      --json "$OUT/s3_ladder_d.json" || rc=$?
    ;;
  *)
    echo "unknown shard $SHARD"; exit 2 ;;
esac

echo "===== shard $SHARD rc=$rc ====="
date -u
exit $rc
