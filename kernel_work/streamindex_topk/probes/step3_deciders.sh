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
# The four shards of step 3's first decider job, one per chip.
#
# This exists as a file rather than as four inline strings because `RUN_CMD` is
# a single-quoted YAML scalar that may not contain a single quote, and the
# fan-out adds a second level of quoting on top of that. `--tuples
# "2,2,2;4,4,4;8,4,4;8,8,8"` cannot be written safely at that depth. The skill
# says as much: put the shards in a script and call it by index.
#
# Usage: bash step3_deciders.sh <shard index 0..3>
#
# Shard 0 of the TASK (the decode:mixed step mix at the A/B shape) takes no
# chip -- it is a derivation from the scheduler's chunking arithmetic and is
# written up in step3_decider_RESULT.md, labelled `modelled`.

set -u

P=/workspace/kernel_work/streamindex_topk/probes
OUT="${CDK_OUTPUT_DIR:-/tmp}/step3"
mkdir -p "$OUT"

case "${1:?shard index required}" in

  # bkv_p = 8 has never been value-checked, and (8,4,4) has never been run at
  # all. `--mode correct` is the check; `--mode control` is the negative
  # control that proves the check is not vacuous. A control that does not fire
  # invalidates the correctness arm above it, so both run here, adjacent.
  0)
    python3 $P/bkv_p_prod.py --mode correct --tuples "8,4,4" \
      --regimes decode_64k,mixed_prod
    rc_a=$?
    python3 $P/bkv_p_prod.py --mode control --tuples "8,4,4" \
      --regimes decode_64k,mixed_prod
    rc_b=$?
    echo "shard0: correct rc=$rc_a  control rc=$rc_b"
    exit $(( rc_a | rc_b ))
    ;;

  # The (8,4,4) timing arm, plus (2,2,2) as the same-job baseline every delta
  # in this shard is taken against, plus (4,4,4) and (8,8,8) as the two arms
  # that bracket it. Both regimes are in one shard on purpose: the deltas are
  # then all against baselines measured on the same chip in the same job.
  1)
    python3 $P/bkv_p_prod.py --mode sweep \
      --tuples "2,2,2;4,4,4;8,4,4;8,8,8" \
      --regimes decode_64k,mixed_prod \
      --json "$OUT/shard1_bkv_p_sweep.json"
    ;;

  # What a TensorCore-side top-2048 costs. Option D's one unmodelled term.
  2)
    python3 $P/tc_topk.py --json "$OUT/shard2_tc_topk.json"
    ;;

  # Is the MIXED band VPU-saturated? A dynamic K-ladder, not more static LLO.
  3)
    python3 $P/vpu_mxu_strike.py --json "$OUT/shard3_vpu_mxu.json"
    ;;

  *)
    echo "unknown shard index: $1" >&2
    exit 2
    ;;
esac
